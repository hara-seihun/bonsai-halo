#!/usr/bin/env python3
"""Analyse a tools/batch_compare run: throughput per mode, logit agreement against mode 0, and
agreement against whichever matched reference each mode is supposed to reproduce.

    python3 tools/batch_compare.py tools/batch-compare/results/<tag>            # markdown report
    python3 tools/batch_compare.py .../run.json --json report.json              # machine readable too
    python3 tools/batch_compare.py .../run.json --reference-pairs 6:1,7:3,8:2   # matched-family check

Inputs are what the driver wrote: run.json plus one float32 logits dump per (quality shape, mode),
row-major [rows][vocab]. Nothing is recomputed on the GPU here.

Mode 0 stays the reference for every table, because that is the path we ship and it is the only
comparison that shows what a mode costs in output terms. A4 is faster and it changes logits; pairing
it against its own numerical family must not hide that, so the pair tables are additional, never a
replacement.

Pairs are never guessed. An automatic mode runs the deployed path at four rows or fewer and the
sliced path from five to 31, so pairing mode 10 with mode 3 is only meaningful at 32 rows and above;
that is what the row conditions in the pair syntax are for.

The report states its own sample size. Three rounds of eight steps and eight teacher-forced rows are
a smoke test: they catch a mode that is broken or obviously slower, and they are not evidence that a
mode's output quality is acceptable.
"""

import argparse
import json
import os
import re
import statistics
import sys

import numpy as np

REF_MODE = 0


# ---------------------------------------------------------------- reference pairs

PAIR_HELP = """\
MODE:REF pairs, comma separated, each optionally scoped with @ conditions joined by +:

  6:1,7:3,8:2                     compare each optimized mode with the family it reproduces
  9:5                             both are automatic and dispatch identically at every row count
  10:3@rows>=32,11:2@rows>=32     only where the automatic mode actually reaches the module
  10:7@section=quality+shape=prefill

Conditions: rows OP N with OP in >= <= > < =, shape=prefill|decode|multistep, and
section=timing|quality|multistep. An unscoped pair applies everywhere both modes were measured.
rows is rows per pass for prefill, streams for decode and multistep, and row count for quality.
"""

OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "=": lambda a, b: a == b,
    "==": lambda a, b: a == b,
}


class Pair:
    def __init__(self, spec):
        self.spec = spec.strip()
        head, _, cond = self.spec.partition("@")
        m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", head)
        if not m:
            raise SystemExit(f"--reference-pairs: cannot read {spec!r}; expected MODE:REF[@conditions]")
        self.mode, self.ref = int(m.group(1)), int(m.group(2))
        if self.mode == self.ref:
            raise SystemExit(f"--reference-pairs: {spec!r} pairs a mode with itself")
        self.conds = []
        for c in (x for x in cond.split("+") if x.strip()):
            c = c.strip()
            mrows = re.fullmatch(r"rows\s*(>=|<=|==|=|>|<)\s*(\d+)", c)
            mkey = re.fullmatch(r"(shape|section)\s*=\s*([A-Za-z]+)", c)
            if mrows:
                self.conds.append(("rows", mrows.group(1), int(mrows.group(2))))
            elif mkey:
                self.conds.append((mkey.group(1), "=", mkey.group(2)))
            else:
                raise SystemExit(f"--reference-pairs: cannot read condition {c!r} in {spec!r}")

    def applies(self, section, shape, rows):
        for key, op, val in self.conds:
            if key == "rows":
                if rows is None or not OPS[op](rows, val):
                    return False
            elif key == "shape":
                if shape != val:
                    return False
            elif key == "section":
                if section != val:
                    return False
        return True

    def __str__(self):
        return self.spec


def parse_pairs(values):
    pairs = []
    for v in values or []:
        for part in v.split(","):
            if part.strip():
                pairs.append(Pair(part))
    return pairs


def pairs_for(pairs, section, shape, rows, present):
    """Applicable pairs whose two modes were both measured here, plus the ones that were not."""
    used, missing = [], []
    for p in pairs:
        if not p.applies(section, shape, rows):
            continue
        if p.mode in present and p.ref in present:
            used.append(p)
        elif p.mode in present or p.ref in present:
            missing.append(p)
    return used, missing


def suggested_pairs(run):
    """A pair string the operator could pass, derived from which modes this run measured and how the
    driver recorded the automatic dispatch. Printed as a suggestion; never applied on its own."""
    modes = set(run.get("modes", []))
    # what each reimplementation is intended to reproduce; 12 keeps the original scheduling, so its
    # reference is the deployed path itself
    family = {6: 1, 7: 3, 8: 2, 12: 0, 13: 0, 15: 0}
    auto = run.get("auto_dispatch", {}) or {}
    wide_of = {int(k): v.get("rows_ge_32") for k, v in auto.items() if int(k) < 16}
    out = [f"{mode}:{ref}" for mode, ref in sorted(family.items()) if mode in modes and ref in modes]
    for mode in sorted(wide_of):
        if mode not in modes:
            continue
        # An automatic mode that reaches the same numerical family as another automatic mode is
        # comparable at every row count, because they dispatch identically below 32 rows too.
        peers = [o for o in sorted(wide_of)
                 if o != mode and o in modes
                 and family.get(wide_of[o], wide_of[o]) == family.get(wide_of[mode], wide_of[mode])]
        if peers:
            if mode > peers[0]:
                out.append(f"{mode}:{peers[0]}")
            continue
        # Otherwise pair it with the family it should reproduce, only where it dispatches there.
        for ref in (family.get(wide_of[mode]), wide_of[mode]):
            if ref in modes:
                out.append(f"{mode}:{ref}@rows>=32")
                break
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return ",".join(uniq)


# ---------------------------------------------------------------- loading

def load_run(path):
    if os.path.isdir(path):
        path = os.path.join(path, "run.json")
    with open(path) as f:
        return json.load(f), os.path.dirname(os.path.abspath(path)), os.path.abspath(path)


def logits(run_dir, entry, rows, vocab):
    fn = entry.get("logits_file")
    if not fn:
        return None
    p = os.path.join(run_dir, fn)
    if not os.path.exists(p):
        return None
    want = rows * vocab * 4
    have = os.path.getsize(p)
    if have != want:
        raise SystemExit(f"{p}: {have} bytes, expected {want} for [{rows}][{vocab}] float32")
    return np.memmap(p, dtype=np.float32, mode="r").reshape(rows, vocab)


# ---------------------------------------------------------------- timing

def collect_timings(run):
    """(case, config, rows) -> mode -> list of per-round measurement dicts, in round order.

    `rows` is how many rows a single pass of that workload carries, which is what decides how an
    automatic mode dispatches: rows per pass for prefill, one row per stream for decode."""
    out = {}
    for rnd in run.get("rounds", []):
        for m in rnd["measurements"]:
            if m["case"] == "prefill":
                key = ("prefill", f"{m['rows_per_pass']} rows/pass, {m['tokens']} tokens", m["rows_per_pass"])
            else:
                key = ("decode", f"{m['streams']} streams x {m['steps']} steps", m["streams"])
            out.setdefault(key, {}).setdefault(m["mode"], []).append(m)
    return out


def rate(m):
    return m["tokens_per_s"] if m["case"] == "prefill" else m["aggregate_tokens_per_s"]


def spread(values):
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    return (max(values) - min(values)) / med if med else 0.0


def timing_report(run, out, pairs):
    timings = collect_timings(run)
    if not timings:
        out.append("No timed rounds in this run.\n")
        return {}
    summary = {}
    for (case, config, rows), by_mode in sorted(timings.items()):
        out.append(f"### {case}: {config}\n")
        ref = by_mode.get(REF_MODE)
        ref_med = statistics.median([rate(m) for m in ref]) if ref else None
        if case == "prefill":
            some = ref[0] if ref else next(iter(by_mode.values()))[0]
            policy = some["logits_policy"]
            # "last pass" is not "last token": the final pass carries a full row block, and the
            # lm_head runs on every row of it, for the reference mode and the new modes alike.
            rows_pp, tokens, passes = some["rows_per_pass"], some["tokens"], some["passes"]
            last_rows = tokens - rows_pp * (passes - 1)
            logit_rows = 0 if policy == "none" else last_rows if policy == "last-pass-only" else tokens
            out.append(f"Logits (lm_head) computed on: {policy}, which here is {logit_rows} logit rows of "
                       f"{tokens} tokens, not one token: the final pass carries {last_rows} rows and the lm_head "
                       f"runs on all of them. Every mode pays that same cost, but it is a real part of these "
                       f"numbers and it differs between the {rows_pp}-row and other row settings. "
                       f"Token count is the document's real tokens.\n")
        else:
            some = next(iter(by_mode.values()))[0]
            out.append(f"Fixed {some['steps']} greedy steps per stream, EOS emitted and fed back "
                       f"(streams are never stopped early), setup and prompt prefill excluded from the timing.\n")
        out.append("| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |")
        out.append("|---|---:|---|---:|---:|---|---:|---:|")
        for mode in sorted(by_mode):
            ms = by_mode[mode]
            rates = [rate(m) for m in ms]
            walls = [m["wall_s"] for m in ms]
            med = statistics.median(rates)
            rel = f"{med / ref_med:.3f}x" if ref_med else "n/a"
            name = run.get("mode_names", {}).get(str(mode), str(mode))
            # per mode, not pooled: a mode that ran at a lower clock must be visible on its own row
            clocks = [m["telemetry"]["gfx_hz"]["mean"] / 1e6 for m in ms if (m.get("telemetry") or {}).get("gfx_hz")]
            power = [m["telemetry"]["power_uw"]["mean"] / 1e6 for m in ms if (m.get("telemetry") or {}).get("power_uw")]
            clock_cell = f"{statistics.mean(clocks):.0f}" if clocks else "-"
            power_cell = f"{statistics.mean(power):.0f}" if power else "-"
            out.append(f"| {mode} {name} | {med:.1f} | {', '.join(f'{r:.1f}' for r in rates)} | "
                       f"{spread(rates) * 100:.1f}% | {rel} | {', '.join(f'{w:.3f}' for w in walls)} | "
                       f"{clock_cell} | {power_cell} |")
            summary[f"{case}|{config}|{mode}"] = {
                "median_tokens_per_s": med, "per_round_tokens_per_s": rates,
                "wall_s": walls, "relative_to_mode0": med / ref_med if ref_med else None,
                "spread_fraction": spread(rates),
                "gfx_mhz_mean": statistics.mean(clocks) if clocks else None,
                "board_w_mean": statistics.mean(power) if power else None,
            }
        used, missing = pairs_for(pairs, "timing", case, rows, set(by_mode))
        if used:
            out.append("\nAgainst the matched reference for each mode, at this shape:\n")
            out.append("| pair | mode tok/s | reference tok/s | mode / reference |")
            out.append("|---|---:|---:|---:|")
            for p in used:
                a = statistics.median([rate(m) for m in by_mode[p.mode]])
                b = statistics.median([rate(m) for m in by_mode[p.ref]])
                nm = run.get("mode_names", {})
                out.append(f"| {p.mode} {nm.get(str(p.mode), '')} vs {p.ref} {nm.get(str(p.ref), '')} | "
                           f"{a:.1f} | {b:.1f} | {a / b:.3f}x |")
                summary[f"pair|{case}|{config}|{p.mode}:{p.ref}"] = {
                    "mode_median_tokens_per_s": a, "reference_median_tokens_per_s": b,
                    "ratio": a / b, "spec": str(p),
                }
            out.append("")
        for p in missing:
            out.append(f"Pair `{p}` applies to this shape but only one of its two modes was measured.\n")

        worst = max(spread([rate(m) for m in ms]) for ms in by_mode.values())
        n = len(next(iter(by_mode.values())))
        if worst > 0.05:
            out.append(f"\nRun-to-run spread reaches {worst * 100:.1f}% over {n} rounds. "
                       f"Differences smaller than that are not resolved by this sample.\n")
        else:
            out.append(f"\n{n} rounds per mode; worst spread {worst * 100:.1f}%.\n")

        # pooled clocks and temperature during the timed regions, context only
        tel = []
        for mode in sorted(by_mode):
            for m in by_mode[mode]:
                t = m.get("telemetry") or {}
                if not t.get("available"):
                    continue
                g, temp, p = t.get("gfx_hz"), t.get("temperature_mc"), t.get("power_uw")
                tel.append((mode, g, temp, p))
        if tel:
            gh = [x[1]["mean"] / 1e6 for x in tel if x[1]]
            tc = [x[2]["max"] / 1000 for x in tel if x[2]]
            pw = [x[3]["mean"] / 1e6 for x in tel if x[3]]
            bits = []
            if gh:
                bits.append(f"gfx clock mean {min(gh):.0f}-{max(gh):.0f} MHz across modes")
            if tc:
                bits.append(f"edge temperature max {max(tc):.0f} C")
            if pw:
                bits.append(f"board power mean {min(pw):.0f}-{max(pw):.0f} W")
            out.append("Host sysfs during these regions: " + ", ".join(bits) + ".\n")
    return summary


# ---------------------------------------------------------------- stalled samples

def stall_attribution(h, excess):
    """What the host counters say about a sample that took `excess` seconds longer than its peers."""
    if not h or not h.get("available"):
        return [], "no host diagnostics in this record, so the stall cannot be attributed"
    facts, causes = [], []
    psi = h.get("psi") or {}
    cpu = h.get("cpu_busy_fraction_of_wall")
    blocked = cpu is not None and cpu < 0.2
    if cpu is not None:
        facts.append(f"CPU busy {cpu:.2f} thread-seconds per wall second")
        if blocked:
            causes.append("the process was blocked rather than computing")
    resource_cause = False
    for key, label in (("memory_full_stall_s", "memory reclaim"), ("io_full_stall_s", "blocking I/O"),
                       ("memory_some_stall_s", "memory pressure"), ("io_some_stall_s", "I/O pressure"),
                       ("cpu_some_stall_s", "CPU contention")):
        v = psi.get(key)
        if v:
            facts.append(f"{key.replace('_stall_s', '').replace('_', ' ')} PSI {v:.2f} s")
            if excess > 0 and v > 0.25 * excess:
                causes.append(f"{label} stalled {v:.2f} s of the {excess:.2f} s excess")
                resource_cause = True
    mf = h.get("major_faults")
    if mf:
        facts.append(f"{mf} major page faults")
        causes.append(f"{mf} major page faults, so pages came back from disk or swap")
        resource_cause = True
    ivcs = h.get("involuntary_context_switches")
    if ivcs:
        facts.append(f"{ivcs} involuntary context switches")
    ma = h.get("mem_available_kb")
    if ma and ma.get("before") is not None:
        facts.append(f"MemAvailable {ma['before'] / 1e6:.1f} to {ma['after'] / 1e6:.1f} GB")
    sf = h.get("swap_free_kb")
    if sf and sf.get("delta"):
        facts.append(f"SwapFree moved {sf['delta'] / 1e6:+.2f} GB")
        causes.append("swap was touched during the region")
    if blocked and not resource_cause:
        # Waiting, but not on anything the host can name: no reclaim, no paging, no CPU queue. That
        # leaves the device side, its queue or the driver, which these counters cannot see into.
        causes.append("no host resource stall accounts for it, which points at the device, its queue "
                      "or the driver rather than at the machine being busy")
    verdict = "; ".join(causes) if causes else (
        "none of the host counters account for the excess: not faulting, not preempted, not stalled "
        "on memory or I/O")
    return facts, verdict


def anomaly_report(run, out):
    """Timed samples far slower than their peers, kept in every aggregate and named here."""
    found = []
    for (case, config, _), by_mode in sorted(collect_timings(run).items()):
        for mode, ms in sorted(by_mode.items()):
            med = statistics.median([m["wall_s"] for m in ms])
            for m in ms:
                # a stall only ever inflates, and normal spread here is well under a percent
                if med > 0 and m["wall_s"] > 1.2 * med and m["wall_s"] - med > 0.05:
                    found.append((case, config, mode, m, med))
    if not found:
        return {}
    out.append("### stalled samples\n")
    out.append(f"{len(found)} timed sample(s) ran materially longer than the other rounds of the same mode and "
               "shape. They are kept in every table above and in the raw JSON. A stall is a real thing that "
               "happened to this machine, and dropping it would make the benchmark look steadier than the "
               "machine is. The median is the headline statistic precisely so one stall cannot move it, and the "
               "spread figure is where it shows.\n")
    out.append("| round | mode | shape | wall s | median s | ratio | worst pass/step s | GPU busy % | host CPU busy |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    summary, details = {}, []
    for case, config, mode, m, med in found:
        name = run.get("mode_names", {}).get(str(mode), str(mode))
        arr = m.get("step_wall_s") or m.get("pass_wall_s") or []
        worst = max(arr) if arr else None
        busy = ((m.get("telemetry") or {}).get("busy_percent") or {}).get("mean")
        h = m.get("host") or {}
        cpu = h.get("cpu_busy_fraction_of_wall")
        out.append(f"| {m.get('round')} | {mode} {name} | {case}, {config} | {m['wall_s']:.3f} | {med:.3f} | "
                   f"{m['wall_s'] / med:.2f}x | {'-' if worst is None else f'{worst:.3f}'} | "
                   f"{'-' if busy is None else f'{busy:.0f}'} | "
                   f"{'-' if cpu is None else f'{cpu:.2f}'} |")
        excess = m["wall_s"] - med
        facts, verdict = stall_attribution(h, excess)
        where = ""
        if arr and worst is not None:
            idx = arr.index(worst)
            unit = "step" if m["case"] == "decode" else "pass"
            where = (f" {unit} {idx} of {len(arr)} took {worst:.3f} s, "
                     f"{worst / m['wall_s'] * 100:.0f}% of the region")
        details.append(f"- round {m.get('round')} mode {mode} {name}, {case} {config}: "
                       f"{m['wall_s']:.3f} s against a {med:.3f} s median, {excess:.3f} s excess.{where}")
        if facts:
            details.append(f"  - host during the region: {', '.join(facts)}")
        details.append(f"  - {verdict}")
        summary[f"{case}|{config}|{mode}|round{m.get('round')}"] = {
            "wall_s": m["wall_s"], "median_wall_s": med, "ratio": m["wall_s"] / med, "excess_s": excess,
            "worst_pass_or_step_s": worst, "gpu_busy_percent_mean": busy, "host": h, "verdict": verdict,
        }
    out.append("")
    out.extend(details)
    out.append("")
    if any("no host diagnostics" in v["verdict"] for v in summary.values()):
        out.append("Records without host diagnostics predate them. Re-running with the current driver captures "
                   "rusage, PSI and MemAvailable per measurement, which is what distinguishes memory reclaim, "
                   "paging, CPU contention and blocking I/O from each other.\n")
    return summary


# ---------------------------------------------------------------- multistep state check

def token_matrix_diff(ref, cur):
    """Token-for-token comparison of two multistep runs with the same shape."""
    streams, steps = ref["streams"], ref["steps"]
    stream_match, token_match = 0, 0
    first_step, first_stream = None, None
    for b in range(streams):
        a, c = ref["tokens"][b], cur["tokens"][b]
        same = [x == y for x, y in zip(a, c)]
        token_match += sum(same)
        if all(same):
            stream_match += 1
        else:
            step = same.index(False)
            if first_step is None or step < first_step:
                first_step, first_stream = step, b
    return {
        "streams": streams, "steps": steps,
        "streams_matching": stream_match, "tokens_matching": token_match, "tokens_total": streams * steps,
        "first_divergent_step": first_step, "first_divergent_stream": first_stream,
    }


def multistep_report(run, out, pairs):
    runs = run.get("multistep_runs", [])
    if not runs:
        return {}
    ref = next((r for r in runs if r["mode"] == REF_MODE), None)
    out.append("### multistep state check\n")
    if ref is None:
        out.append(f"No mode {REF_MODE} reference; nothing to compare against.\n")
        return {}
    streams, steps = ref["streams"], ref["steps"]
    out.append(f"{streams} sequences carried {steps} greedy steps from their own prompts, identical inputs per mode. "
               f"Slot 0 length after the run: {ref.get('seq_len_after')}.\n")
    out.append("Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass "
               "difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences "
               "there call for inspection of state, replay, parity and floating evaluation order before blaming "
               "a new FFN map. The approximate modes carry a numeric "
               "difference into every step, so their divergence at any step, early or late, can be numerics alone.\n")
    out.append("| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |")
    out.append("|---|---:|---:|---:|---:|")
    summary = {}
    notes = []
    for r in sorted(runs, key=lambda x: x["mode"]):
        name = run.get("mode_names", {}).get(str(r["mode"]), str(r["mode"]))
        if r["mode"] == REF_MODE:
            out.append(f"| {r['mode']} {name} | reference | reference | - | - |")
            continue
        if r["streams"] != streams or r["steps"] != steps:
            out.append(f"| {r['mode']} {name} | shape differs from reference | | | |")
            continue
        d = token_matrix_diff(ref, r)
        stream_match, token_match = d["streams_matching"], d["tokens_matching"]
        first_step, first_stream = d["first_divergent_step"], d["first_divergent_stream"]
        total = d["tokens_total"]
        out.append(f"| {r['mode']} {name} | {stream_match}/{streams} | {token_match}/{total} | "
                   f"{'-' if first_step is None else first_step} | {'-' if first_stream is None else first_stream} |")
        summary[str(r["mode"])] = d
        if first_step is not None:
            a = ref["tokens"][first_stream]
            c = r["tokens"][first_stream]
            notes.append(f"- mode {r['mode']} stream {first_stream} first differs at step {first_step}: "
                         f"mode 0 {a[first_step]}, mode {r['mode']} {c[first_step]}")
            notes.append(f"  - mode 0 text: {ref['text'][first_stream]!r}")
            notes.append(f"  - mode {r['mode']} text: {r['text'][first_stream]!r}")
    out.append("")
    out.extend(notes)
    if notes:
        out.append("")
    out.append("Greedy decode amplifies a small logit difference into a different token, so a divergence here is not "
               "by itself a quality verdict; read it with the logit tables below.\n")
    if any(v["first_divergent_step"] == 0 for v in summary.values()):
        out.append("At least one mode diverges at step 0, which points at the prompt prefill passes rather than at "
                   "the decode loop.\n")

    by_mode = {r["mode"]: r for r in runs}
    used, missing = pairs_for(pairs, "multistep", "multistep", streams, set(by_mode))
    if used:
        out.append(f"Against the matched reference for each mode, same {streams} prompts and {steps} steps. "
                   f"Each step is a {streams}-row pass, which is what decides how an automatic mode dispatches.\n")
        out.append("| pair | streams matching | tokens matching | first divergent step |")
        out.append("|---|---:|---:|---:|")
        for p in used:
            a, b = by_mode[p.mode], by_mode[p.ref]
            if a["streams"] != b["streams"] or a["steps"] != b["steps"]:
                out.append(f"| {p} | shapes differ | | |")
                continue
            d = token_matrix_diff(b, a)
            out.append(f"| {p.mode} vs {p.ref} | {d['streams_matching']}/{d['streams']} | "
                       f"{d['tokens_matching']}/{d['tokens_total']} | "
                       f"{'-' if d['first_divergent_step'] is None else d['first_divergent_step']} |")
            d["spec"] = str(p)
            summary[f"pair|{p.mode}:{p.ref}"] = d
        out.append("")
    for p in missing:
        out.append(f"Pair `{p}` applies here but only one of its two modes ran the multistep check.\n")
    return summary


# ---------------------------------------------------------------- quality

def row_metrics(ref_row, row, topk=5):
    ref64 = ref_row.astype(np.float64)
    cur64 = row.astype(np.float64)
    nf_ref = int(ref64.size - np.count_nonzero(np.isfinite(ref64)))
    nf_cur = int(cur64.size - np.count_nonzero(np.isfinite(cur64)))
    if nf_ref or nf_cur:
        # Softmax, KL and ranking are all meaningless once a logit is NaN or infinite, and a bare
        # nan in a results table reads as a rounding artefact rather than a broken row. Say so.
        return {
            "finite": False, "nonfinite_ref": nf_ref, "nonfinite_mode": nf_cur,
            "top1_match": False, "ref_top1": None, "mode_top1": None,
        }

    def softmax(x):
        m = x.max()
        e = np.exp(x - m)
        return e / e.sum()

    p = softmax(ref64)
    q = softmax(cur64)
    nz = p > 0
    kl_fwd = float(np.sum(p[nz] * (np.log(p[nz]) - np.log(np.maximum(q[nz], 1e-300)))))
    nzq = q > 0
    kl_rev = float(np.sum(q[nzq] * (np.log(q[nzq]) - np.log(np.maximum(p[nzq], 1e-300)))))

    ref_top = int(np.argmax(ref64))
    cur_top = int(np.argmax(cur64))
    # logits are shift invariant; centre before comparing magnitudes
    d = (cur64 - cur64.mean()) - (ref64 - ref64.mean())
    ref_idx = np.argpartition(-ref64, topk)[:topk]
    cur_idx = np.argpartition(-cur64, topk)[:topk]
    rank_of_ref_top = int(np.sum(cur64 > cur64[ref_top]))

    return {
        "finite": True, "nonfinite_ref": 0, "nonfinite_mode": 0,
        "top1_match": ref_top == cur_top,
        "ref_top1": ref_top,
        "mode_top1": cur_top,
        "ref_top1_prob": float(p[ref_top]),
        "mode_prob_of_ref_top1": float(q[ref_top]),
        "rank_of_ref_top1_in_mode": rank_of_ref_top,
        "kl_ref_to_mode_nats": kl_fwd,
        "kl_mode_to_ref_nats": kl_rev,
        "max_abs_logit_diff_centred": float(np.abs(d).max()),
        "rms_logit_diff_centred": float(np.sqrt(np.mean(d * d))),
        f"top{topk}_overlap": int(len(set(ref_idx.tolist()) & set(cur_idx.tolist()))),
    }


def pair_logit_diff(a, b, block=8):
    """Exact and aggregate difference between two logit dumps of the same shape.

    `exact_mismatches` counts float32 elements whose bit patterns differ, which is the direct test of
    an intended bit-identical reimplementation. RMS and max are on the raw values, uncentred: a pair
    that claims identity has nothing to centre away.

    Non-finite values are counted on both sides and kept out of the verdict. A NaN reproduces its own
    bit pattern exactly, so two identically broken dumps would otherwise score as a clean
    reimplementation; and a NaN difference never reaches RMS, because the subtraction is also NaN."""
    rows, vocab = a.shape
    mismatches, per_row, sq, mx = 0, [], 0.0, 0.0
    nf_a, nf_b = 0, 0
    for i in range(0, rows, block):                       # chunked: a 128-row dump is 127 MB per side
        x, y = np.asarray(a[i:i + block]), np.asarray(b[i:i + block])
        diff_bits = x.view(np.int32) != y.view(np.int32)
        per_row.extend(int(c) for c in diff_bits.sum(axis=1))
        mismatches += int(diff_bits.sum())
        fin = np.isfinite(x) & np.isfinite(y)
        nf_a += int(np.size(x) - np.count_nonzero(np.isfinite(x)))
        nf_b += int(np.size(y) - np.count_nonzero(np.isfinite(y)))
        d = np.where(fin, x.astype(np.float64) - y.astype(np.float64), 0.0)
        sq += float(np.sum(d * d))
        mx = max(mx, float(np.abs(d).max()))
    total = rows * vocab
    finite = total - max(nf_a, nf_b)
    return {
        "rows": rows, "vocab": vocab, "elements": total,
        "exact_mismatches": mismatches,
        "exact_mismatch_fraction": mismatches / total if total else 0.0,
        "rows_with_any_mismatch": sum(1 for c in per_row if c),
        "per_row_exact_mismatches": per_row,
        "rms_logit_diff": float(np.sqrt(sq / finite)) if finite > 0 else None,
        "max_abs_logit_diff": mx,
        "nonfinite_mode": nf_a,
        "nonfinite_reference": nf_b,
        "rms_over_finite_elements": finite,
        "bit_identical": mismatches == 0,
        # identical bits are only a reproduction when the values are numbers
        "clean_reproduction": mismatches == 0 and nf_a == 0 and nf_b == 0,
    }


def quality_pairs_report(run, run_dir, out, group, pairs, shape, rows, vocab, summary):
    by_mode = {e["mode"]: e for e in group}
    used, missing = pairs_for(pairs, "quality", shape, rows, set(by_mode))
    for p in missing:
        out.append(f"Pair `{p}` applies to this group but only one of its two modes was measured.\n")
    if not used:
        return
    out.append("Against the matched reference for each mode, same tokens, same rows:\n")
    out.append("| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    verdicts = []
    for p in used:
        a, b = by_mode[p.mode], by_mode[p.ref]
        top1 = sum(int(x == y) for x, y in zip(a["argmax"], b["argmax"]))
        la, lb = logits(run_dir, a, rows, vocab), logits(run_dir, b, rows, vocab)
        if la is None or lb is None:
            out.append(f"| {p.mode} vs {p.ref} | {top1}/{rows} | dump missing | | | | |")
            summary[f"pair|{shape}|{rows}|{p.mode}:{p.ref}"] = {"top1_match": top1, "rows": rows, "spec": str(p)}
            continue
        d = pair_logit_diff(la, lb)
        rms = "n/a" if d["rms_logit_diff"] is None else f"{d['rms_logit_diff']:.3e}"
        out.append(f"| {p.mode} vs {p.ref} | {top1}/{rows} | {d['exact_mismatches']}/{d['elements']} | "
                   f"{d['rows_with_any_mismatch']}/{rows} | {rms} | {d['max_abs_logit_diff']:.3e} | "
                   f"{d['nonfinite_mode']}/{d['nonfinite_reference']} |")
        d["top1_match"] = top1
        d["spec"] = str(p)
        summary[f"pair|{shape}|{rows}|{p.mode}:{p.ref}"] = d
        nf = f"{d['nonfinite_mode']} non-finite values in mode {p.mode} and {d['nonfinite_reference']} in mode {p.ref}"
        if d["clean_reproduction"]:
            verdicts.append(f"- mode {p.mode} reproduced mode {p.ref} exactly here: all {d['elements']} "
                            f"logit values are finite and have identical bit patterns.")
        elif d["bit_identical"]:
            verdicts.append(f"- mode {p.mode} matched mode {p.ref} bit for bit here, but the dumps are not clean: "
                            f"{nf}. Identical NaN or infinity bits are agreement about a broken result, "
                            f"not a reproduction; this pair has not passed.")
        else:
            verdicts.append(f"- mode {p.mode} did not reproduce mode {p.ref} exactly here: "
                            f"{d['exact_mismatches']} of {d['elements']} logit values differ, "
                            f"RMS {rms} over {d['rms_over_finite_elements']} finite elements, "
                            f"largest {d['max_abs_logit_diff']:.3e}, "
                            f"{d['rows_with_any_mismatch']} of {rows} rows affected"
                            + (f"; {nf}." if d["nonfinite_mode"] or d["nonfinite_reference"] else "."))
    out.append("")
    out.extend(verdicts)
    out.append("")
    out.append("These verdicts are about the shape named above, not about every pass that produced it; see the "
               "scope note at the top of this report.\n")


def quality_report(run, run_dir, out, pairs):
    entries = run.get("quality", [])
    if not entries:
        out.append("No quality pass in this run.\n")
        return {}
    vocab = entries[0]["vocab"]
    summary = {}
    groups = sorted({(e["shape"], e["rows"]) for e in entries})
    for shape, rows in groups:
        group = [e for e in entries if e["shape"] == shape and e["rows"] == rows]
        ref = next((e for e in group if e["mode"] == REF_MODE), None)
        if ref is None:
            out.append(f"### quality, {shape} shape, {rows} rows\n")
            out.append(f"Mode {REF_MODE} did not run in this group, so there is no deployed-path comparison here.\n")
            quality_pairs_report(run, run_dir, out, group, pairs, shape, rows, vocab, summary)
            continue
        ref_lg = logits(run_dir, ref, rows, vocab)
        out.append(f"### quality, {shape} shape, {rows} rows\n")
        bad = [(e["mode"], int(e["nonfinite_logits"]), int(e.get("nonfinite_rows", 0)))
               for e in group if e.get("nonfinite_logits")]
        if bad:
            out.append("Non-finite logits in this group: "
                       + ", ".join(f"mode {m} produced {n} NaN or infinite values across {r} rows" for m, n, r in bad)
                       + ". Every comparison below is over the finite values only, and a mode that emits "
                         "non-finite logits has failed this group whatever its agreement scores say.\n")
        if ref_lg is None:
            out.append("Logits were not dumped (--no-logits-dump); only argmax agreement is available.\n")
        out.append(f"{rows} teacher-forced rows, full {vocab}-entry vocabulary, identical token inputs per mode.")
        if shape == "decode":
            out.append("Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger "
                       "one: the row counts are nested, not independent samples.\n")
        else:
            out.append(f"Rows are document tokens following the same {run.get('quality_inputs', {}).get(str(rows), {}).get('prefill_shape', {}).get('context_tokens', '?')}-token "
                       "context, so a smaller row count is a prefix of a larger one: the row counts are nested, "
                       "not independent samples.\n")
        out.append("| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        notes = []
        for e in sorted(group, key=lambda x: x["mode"]):
            name = run.get("mode_names", {}).get(str(e["mode"]), str(e["mode"]))
            if e["mode"] == REF_MODE:
                out.append(f"| {e['mode']} {name} | reference | 0 | - | - | - | - | - |")
                continue
            argmax_match = sum(int(a == b) for a, b in zip(ref["argmax"], e["argmax"]))
            if ref_lg is None:
                out.append(f"| {e['mode']} {name} | {argmax_match}/{rows} | {rows - argmax_match} | - | - | - | - | - |")
                summary[f"{shape}|{rows}|{e['mode']}"] = {"top1_match": argmax_match, "rows": rows,
                                                          "rows_changed": rows - argmax_match}
                continue
            cur = logits(run_dir, e, rows, vocab)
            if cur is None:
                out.append(f"| {e['mode']} {name} | {argmax_match}/{rows} | {rows - argmax_match} | dump missing | | | | |")
                continue
            per_row = [row_metrics(ref_lg[i], cur[i]) for i in range(rows)]
            good = [m for m in per_row if m["finite"]]
            skipped = len(per_row) - len(good)
            match = sum(int(m["top1_match"]) for m in per_row)
            if not good:
                out.append(f"| {e['mode']} {name} | {match}/{rows} | {rows - match} | "
                           f"every row non-finite | | | | |")
                summary[f"{shape}|{rows}|{e['mode']}"] = {"rows": rows, "top1_match": match,
                                                          "rows_changed": rows - match,
                                                          "rows_nonfinite": skipped, "per_row": per_row}
                continue
            kls = [m["kl_ref_to_mode_nats"] for m in good]
            md = [m["max_abs_logit_diff_centred"] for m in good]
            ov = [m["top5_overlap"] for m in good]
            rk = max(m["rank_of_ref_top1_in_mode"] for m in good)
            mark = f" ({len(good)}/{rows} rows)" if skipped else ""
            out.append(f"| {e['mode']} {name} | {match}/{rows} | {rows - match} | {statistics.mean(kls):.4f}{mark} | {max(kls):.4f} | "
                       f"{max(md):.3f} | {statistics.mean(ov):.1f}/5 | {rk} |")
            summary[f"{shape}|{rows}|{e['mode']}"] = {
                "rows": rows, "top1_match": match, "rows_changed": rows - match,
                "rows_nonfinite": skipped,
                "kl_ref_to_mode_nats": kls, "max_abs_logit_diff_centred": md,
                "top5_overlap": ov, "worst_rank_of_ref_top1": rk,
                "per_row": per_row,
            }
            if skipped:
                notes.append(f"- mode {e['mode']}: {skipped} of {rows} rows carry non-finite logits on one side "
                             f"and are excluded from the aggregates above; the mode has failed those rows.")
            if match < rows:
                for i in [i for i, m in enumerate(per_row) if not m["top1_match"] and m["finite"]]:
                    m = per_row[i]
                    notes.append(f"- mode {e['mode']} row {i}: mode0 picks {m['ref_top1']} (p={m['ref_top1_prob']:.3f}), "
                                 f"mode picks {m['mode_top1']}; mode0's token falls to rank {m['rank_of_ref_top1_in_mode']} "
                                 f"with p={m['mode_prob_of_ref_top1']:.3f}")
        out.append("")
        out.extend(notes)
        if notes:
            out.append("")
        quality_pairs_report(run, run_dir, out, group, pairs, shape, rows, vocab, summary)
    return summary


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="results directory or run.json")
    ap.add_argument("--json", help="also write the computed metrics here")
    ap.add_argument("--out", help="write the markdown report here instead of stdout")
    ap.add_argument("--reference-pairs", action="append", metavar="PAIRS", help=PAIR_HELP)
    ap.add_argument("--no-run-pairs", action="store_true",
                    help="ignore the pairs the driver recorded in run.json")
    args = ap.parse_args()

    run, run_dir, run_path = load_run(args.run)
    pair_source = "--reference-pairs"
    pair_specs = args.reference_pairs
    if not pair_specs and not args.no_run_pairs and run.get("reference_pairs"):
        pair_specs = [run["reference_pairs"]]           # what the operator asked for at run time
        pair_source = "run.json --reference-pairs"
    pairs = parse_pairs(pair_specs)
    out = []
    out.append(f"# Batch mode comparison: {run.get('tag', '?')}\n")
    out.append(f"Source: `{run_path}`, run {run.get('started')} to {run.get('finished')}, "
               f"resident server paused: {run.get('resident_server_paused')}, "
               f"full model layers: {run.get('full_model_layers')}, batch capacity {run.get('batch_capacity')}.")
    out.append(f"Model {run.get('model')}, context {run.get('context')}, slots {run.get('slots')}, "
               f"max rows per pass {run.get('max_rows')}, document {run.get('document')}, "
               f"prompts {run.get('prompts_file')}.")
    warm = "clock ramp skipped, no timed workload" if run.get("warmup_skipped") else f"warmup {run.get('warmup_s', 0):.1f} s"
    out.append(f"{run.get('rounds_requested', len(run.get('rounds', [])))} rounds, mode order reshuffled per round (seed {run.get('seed')}), "
               f"{warm}, single process, state reset before every timed region.")
    mem = run.get("device_memory")
    if mem:
        gb = lambda k: mem[k] / 1e9 if mem.get(k) is not None else None
        line = (f"Device memory: {gb('after_load_bytes'):.2f} GB for the model, "
                f"{gb('after_prepare_batch_bytes'):.2f} GB after "
                f"`prepare_batch({mem.get('batch_rows_prepared')}, {mem.get('modes_mask_hex', mem.get('modes_mask'))})`, "
                f"of which {gb('ffn_batch_module_bytes'):.2f} GB is the batched FFN module.")
        if gb("ffn_batch_weight_budget_bytes") is not None:
            line += (f" Weight images for the modes measured budget at "
                     f"{gb('ffn_batch_weight_budget_bytes'):.2f} GB, against "
                     f"{gb('ffn_batch_weight_bytes_all_modes'):.2f} GB to hold every mode at once.")
        line += (" Each image is a further representation of all 64 layers, held alongside the deployed weights "
                 "for as long as the mode that reads it is available.")
        out.append(line)
        resident = [i for i in mem.get("images", []) if i.get("resident")]
        if resident:
            out.append("Resident weight images: "
                       + ", ".join(f"{i['name']} {i['resident_bytes'] / 1e9:.2f} GB" for i in resident) + ".")
        if mem.get("modes_mask_requests_all_images"):
            out.append("No measured mode enters the batched FFN module, so every weight image was built for "
                       "nothing: a zero mask asks the engine for all of them. Only the shared workspace was "
                       "actually needed here.")
        if mem.get("a4_image_option", "both") != "both":
            out.append(f"A4 was restricted to its {mem['a4_image_option']} weight image, which uses one image at "
                       f"every row count instead of picking per call. That is a throughput-for-bytes trade and it "
                       f"applies to every A4 number below.")
    if run.get("modes"):
        flags = run.get("mode_engine_flags", {})
        listed = ", ".join(f"{m} {run.get('mode_names', {}).get(str(m), '')}"
                           + (f" (`--ffn {flags[str(m)]}`)" if str(m) in flags else "")
                           for m in run["modes"])
        out.append(f"Modes measured: {listed}.")
    if any(m >= 16 for m in run.get("modes", [])):
        out.append("Sequence routes: mode 16 commits without widening; 17 widens without committing; "
                   "18 and 19 do both. Widening applies at 32 or more total rows; committing applies "
                   "at every row count. FFN automatic dispatch is separate from this choice.")
        out.append(f"Wide-sequence layout: `{run.get('sequence_layout', 'staged')}`; "
                   f"operand: `{run.get('sequence_operand', 'int8')}`; "
                   f"token tile width: `{run.get('sequence_tile_width', 'auto')}`. "
                   "`scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; "
                   "it is a different numerical map, not a layout-only option.")
        out.append(f"Resident GDN state: `{run.get('gdn_resident', False)}`; "
                   f"state-row split: `{run.get('gdn_state_split', '4')}`. "
                   "When enabled, committed wide GDN passes keep each sequence's state in registers "
                   "across its tokens. Attention and uncommitted passes retain the sliced route.")
    for skip in run.get("quality_skipped", []):
        out.append(f"Quality group {skip['shape']} shape at {skip['rows']} rows was not measured: {skip['reason']}.")
    if pairs:
        out.append(f"Matched reference pairs from {pair_source}: " + ", ".join(f"`{p}`" for p in pairs) + ".")
        out.append("A pair's scope selects which reported shape the pair is applied to. It does not certify that "
                   "every pass inside that measurement took the matched route. Teacher-forced context, prompt "
                   "prefill and the tail pass of a document can carry fewer than 32 rows, and there an automatic "
                   "mode runs the deployed or sliced path however wide the reported rows are. `9:5` needs no scope "
                   "for a different reason: both modes switch on the same row counts, so they agree pass for pass "
                   "including the narrow ones. Read every other pair as a statement about the shape it names.")
    else:
        hint = suggested_pairs(run)
        if hint:
            out.append("No matched reference pairs were given, so every comparison below is against mode 0. "
                       f"For this run's modes, `--reference-pairs {hint}` adds the family comparisons.")
    out.append("")

    timing = timing_report(run, out, pairs)
    anomalies = anomaly_report(run, out)
    multistep = multistep_report(run, out, pairs)
    quality = quality_report(run, run_dir, out, pairs)

    out.append("## Reading this\n")
    out.append("Every number above comes from the full model through Engine::forward_batch. "
               "Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed "
               "step count, which means a stream that emitted EOS kept going.")
    out.append("Sample size is small by design. The quality table compares a handful of teacher-forced rows, "
               "so it can show a mode is wrong; it cannot show a mode is good enough to deploy. "
               "Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.")
    if pairs:
        out.append("The pair tables answer a narrower question than the mode 0 tables: whether a mode reproduces "
                   "the arithmetic it was written to reproduce. A mode that matches its pair exactly can still "
                   "differ from the deployed path, and a faster approximate mode is still an approximation; the "
                   "mode 0 tables above are where that cost stays visible.")
    out.append("")

    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text.rstrip() + "\n")
        print(args.out)
    else:
        sys.stdout.write(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"timing": timing, "multistep": multistep, "quality": quality,
                       "stalled_samples": anomalies,
                       "reference_pairs": [str(p) for p in pairs], "reference_pairs_source": pair_source,
                       "device_memory": run.get("device_memory"),
                       "quality_skipped": run.get("quality_skipped", [])}, f, indent=1)


if __name__ == "__main__":
    main()
