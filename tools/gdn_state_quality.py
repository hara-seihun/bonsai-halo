#!/usr/bin/env python3
"""Compare a batch_compare run's quality dumps across recurrent-state coordinates.

Read the NLL table first. Logit differences against the fp32 arm are SATURATED in this engine:
every activation quantiser scales a 128-block by its own amax, so any perturbation large enough to
move an amax produces a fixed-size output difference, and a coordinate sixteen times coarser
measures the same KL. Teacher-forced negative log likelihood on the document the rows came from is
an absolute quality number with no such floor, and it is paired per row, so it ranks arms.

    tools/gdn_state_quality.py ../../data/bonsai2/batch-comparison/<tag>
    tools/gdn_state_quality.py <dir> --json out.json

tools/batch_compare.py keys everything on the batch mode, because that is what its tables compare.
A packed state is a second axis over the same mode: same weights, same FFN map, same schedule, one
different storage coordinate for the gated-delta state. This reads the run's quality entries, takes
the fp32 arm of each (shape, rows, mode) group as the reference, and reports what the packed arms do
to the distribution the model actually emits.

KL is D(reference || arm) in nats over the full vocabulary, which is the quantity that says whether
a route changed the model's answer rather than its bits. Greedy agreement is the decision the
serving path makes. Both are reported per row so a single bad row cannot hide in a mean.
"""
import argparse
import json
import math
import os
import sys

import numpy as np


def softmax_logprob(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def nll(logits, targets):
    lg = logits[:len(targets)].astype(np.float64)
    mx = lg.max(axis=1, keepdims=True)
    lp = lg - mx - np.log(np.exp(lg - mx).sum(axis=1, keepdims=True))
    return -lp[np.arange(len(targets)), targets]


def compare(ref, arm):
    lp_ref, lp_arm = softmax_logprob(ref), softmax_logprob(arm)
    p_ref = np.exp(lp_ref)
    kl = (p_ref * (lp_ref - lp_arm)).sum(axis=-1)
    top1_ref, top1_arm = ref.argmax(axis=-1), arm.argmax(axis=-1)
    agree = top1_ref == top1_arm
    k = min(5, ref.shape[-1])
    t5_ref = np.argsort(-ref, axis=-1)[:, :k]
    t5_arm = np.argsort(-arm, axis=-1)[:, :k]
    top5 = np.array([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(t5_ref, t5_arm)])
    d = np.abs(ref.astype(np.float64) - arm.astype(np.float64))
    return {
        "rows": int(ref.shape[0]),
        "greedy_agree": int(agree.sum()),
        "greedy_agree_frac": float(agree.mean()),
        "top5_overlap_mean": float(top5.mean()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "kl_per_row": [float(v) for v in kl],
        "logit_max_abs_delta": float(d.max()),
        "logit_rms_delta": float(math.sqrt((d ** 2).mean())),
        "identical_bits": bool(np.array_equal(ref.view(np.uint32), arm.view(np.uint32))),
    }


def paired(d):
    """Mean of a paired difference with both its naive and its stream-clustered standard error.

    `d` is [scored step][stream]. Treating every prediction as independent is wrong here: a stream's
    state error persists across its own steps, so adjacent predictions of one stream carry the same
    perturbation and the naive standard error understates the spread. The cluster-robust form treats
    each stream as one observation, which is the unit that was independently randomised - a distinct
    document window. Both are reported, because the ratio between them is itself the measurement of
    how persistent the error is.
    """
    flat = d.reshape(-1)
    n = flat.size
    mean = float(flat.mean())
    naive = float(flat.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    dev = d - mean
    csum = dev.sum(axis=0)                      # one total per stream
    clustered = float(math.sqrt((csum ** 2).sum()) / n) if d.shape[1] > 1 else naive
    return mean, naive, clustered


def arm_name(entry):
    """A coordinate and a commit cadence are two axes over the same state, and an arm is both.

    A run that walks --gdn-state and --gdn-defer produces several entries per (shape, rows, mode),
    and keying them on the format alone silently keeps whichever was written last."""
    state = entry.get("gdn_state", "f32")
    defer = entry.get("gdn_defer", 1)
    return state if defer == 1 else f"{state}+d{defer}"


def horizon_report(runs, buckets=4):
    """Rank state coordinates on the shape where their rounding accumulates.

    A packed state is re-rounded every time it crosses memory: once per token per layer in a
    generation step, once per pass in prefill. The teacher-forced window above therefore carries at
    most a few dozen roundings and cannot separate coordinates whose error accumulates from ones
    whose error does not. The horizon workload runs hundreds of single-token steps on fixed tokens
    and scores the distribution along the way, so the difference is paired per (stream, step) and
    can be read as a function of how many roundings have happened.

    Arms may come from separate processes. That is safe here and not in a timing panel: the engine
    is deterministic on fixed inputs, so a quality arm reproduces exactly wherever it runs.
    """
    arms, cfg = {}, None
    for run in runs:
        cfg = run.get("horizon_config", cfg)
        for h in run.get("horizon", []):
            key = (h["mode"], arm_name(h))
            arms.setdefault(key, h)
    out = []
    modes = sorted({m for m, _ in arms})
    for mode in modes:
        ref = arms.get((mode, "f32"))
        if ref is None:
            continue
        steps = [s["step"] for s in ref["scored"]]
        tgt = np.array([s["target"] for s in ref["scored"]])
        base = np.array([s["nll"] for s in ref["scored"]], dtype=np.float64)   # [scored][stream]
        entry = {"mode": mode, "mode_name": ref.get("mode_name"), "streams": ref["streams"],
                 "steps": ref["steps"], "ctx": ref["ctx"], "score_every": ref["score_every"],
                 "distinct_windows": ref.get("distinct_windows"),
                 "predictions": int(base.size), "arms": {}}
        edges = np.array_split(np.arange(len(steps)), buckets)
        for name in ["f32"] + sorted(k for m, k in arms if m == mode and k != "f32"):
            h = arms[(mode, name)]
            if [s["step"] for s in h["scored"]] != steps or not np.array_equal(
                    np.array([s["target"] for s in h["scored"]]), tgt):
                print(f"horizon: arm {name} walked different tokens than f32; not comparable", file=sys.stderr)
                continue
            v = np.array([s["nll"] for s in h["scored"]], dtype=np.float64)
            am = np.array([s["argmax"] for s in h["scored"]])
            rm = np.array([s["argmax"] for s in ref["scored"]])
            mean, naive, clus = paired(v - base)
            spread = float((v - base).std(ddof=1)) if v.size > 1 else 0.0
            a = {"nll": float(v.mean()), "perplexity": float(math.exp(v.mean())),
                 "teacher_top1": float((am == tgt).mean()),
                 "greedy_agree_f32": float((am == rm).mean()),
                 "delta_nll": mean, "delta_nll_stderr": naive,
                 "delta_nll_stderr_clustered": clus,
                 "per_prediction_sd": spread,
                 "by_horizon": []}
            for idx in edges:
                if not len(idx):
                    continue
                bm, _bn, bc = paired(v[idx] - base[idx])
                a["by_horizon"].append({
                    "first_step": int(steps[idx[0]]), "last_step": int(steps[idx[-1]]),
                    "predictions": int((v[idx] - base[idx]).size), "delta_nll": bm,
                    "delta_nll_stderr": bc,
                    "greedy_agree_f32": float((am[idx] == rm[idx]).mean())})
            entry["arms"][name] = a
        out.append(entry)
    return out


def print_horizon(report):
    for e in report:
        print(f"\nhorizon, mode {e['mode']} ({e['mode_name']}), {e['streams']} streams x {e['steps']} "
              f"teacher-forced steps, {e['predictions']} scored predictions"
              + ("" if e.get("distinct_windows") else ", OVERLAPPING document windows"))
        for name, a in e["arms"].items():
            d = "" if name == "f32" else (f"  dNLL {a['delta_nll']:+.5f} +- {a['delta_nll_stderr_clustered']:.5f}"
                                          f" (per-prediction sd {a['per_prediction_sd']:.3f})")
            print(f"    {name:9s} NLL {a['nll']:.5f}  ppl {a['perplexity']:.4f}  "
                  f"top1 {a['teacher_top1']:.4f}  agree(f32) {a['greedy_agree_f32']:.4f}{d}")
        print("    by horizon, stream-clustered error (steps of accumulated state rounding):")
        for name, a in e["arms"].items():
            if name == "f32":
                continue
            cells = "  ".join(f"[{b['first_step']:>4}-{b['last_step']:<4}] {b['delta_nll']:+.5f}+-{b['delta_nll_stderr']:.5f}"
                              for b in a["by_horizon"])
            print(f"      {name:9s} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+", help="results directories or run.json files; horizon arms merge across them")
    ap.add_argument("--json", help="also write the report here")
    args = ap.parse_args()
    paths = [os.path.join(p, "run.json") if os.path.isdir(p) else p for p in args.run]
    path = paths[0]
    runs = [json.loads(open(p).read()) for p in paths]
    run = runs[0]
    base_dir = os.path.dirname(os.path.abspath(path))

    groups = {}
    for q in run.get("quality", []):
        if "logits_file" not in q:
            continue
        key = (q["shape"], q["rows"], q["mode"])
        groups.setdefault(key, {})[arm_name(q)] = q

    report = {"run": path, "revision": run.get("git_revision"), "binary": run.get("binary_sha256"),
              "groups": []}
    for (shape, rows, mode), arms in sorted(groups.items()):
        if "f32" not in arms or len(arms) < 2:
            continue
        ref = np.fromfile(os.path.join(base_dir, arms["f32"]["logits_file"]), dtype=np.float32).reshape(rows, -1)
        g = {"shape": shape, "rows": rows, "mode": mode, "mode_name": arms["f32"].get("mode_name"), "arms": {}}
        for name, q in sorted(arms.items()):
            if name == "f32":
                continue
            a = np.fromfile(os.path.join(base_dir, q["logits_file"]), dtype=np.float32).reshape(rows, -1)
            g["arms"][name] = compare(ref, a)
        report["groups"].append(g)

    # Teacher-forced NLL. A prefill-shaped group feeds rows[i] and predicts rows[i+1], so the run's
    # own recorded inputs are the targets and no tokenizer or document is needed here.
    inputs = run.get("quality_inputs") or {}
    report["nll"] = []
    for (shape, rows, mode), arms in sorted(groups.items()):
        if shape != "prefill" or "f32" not in arms:
            continue
        rec = inputs.get(str(rows), {}).get("prefill_shape", {}).get("rows") or []
        if len(rec) < 2:
            continue
        tgt = np.array(rec[1:])
        base = None
        entry = {"shape": shape, "rows": rows, "mode": mode, "predictions": int(len(tgt)), "arms": {}}
        for name in ["f32"] + sorted(k for k in arms if k != "f32"):
            lg = np.fromfile(os.path.join(base_dir, arms[name]["logits_file"]), dtype=np.float32).reshape(rows, -1)
            v = nll(lg, tgt)
            if base is None:
                base = v
            d = v - base
            entry["arms"][name] = {
                "nll": float(v.mean()), "perplexity": float(math.exp(v.mean())),
                "next_token_top1": float((lg[:len(tgt)].argmax(axis=1) == tgt).mean()),
                "delta_nll": float(d.mean()),
                "delta_nll_stderr": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0,
            }
        report["nll"].append(entry)

    # Greedy continuations, when the decode workload captured them.
    texts = {}
    for rnd in run.get("rounds", []):
        for m in rnd.get("measurements", []):
            if m.get("case") != "decode" or "streams_out" not in m:
                continue
            flat = [t for s in m["streams_out"] for t in s["tokens"]]
            texts.setdefault((m["mode"], m.get("streams"), arm_name(m)), flat)
    cont = []
    for (mode, streams, state), toks in sorted(texts.items()):
        if state == "f32":
            continue
        ref = texts.get((mode, streams, "f32"))
        if ref is None:
            continue
        n = min(len(ref), len(toks))
        same = sum(1 for i in range(n) if ref[i] == toks[i])
        first = next((i for i in range(n) if ref[i] != toks[i]), None)
        cont.append({"mode": mode, "streams": streams, "state": state, "tokens": n,
                     "identical": same == n, "matching": same, "first_divergence": first})
    report["continuations"] = cont
    report["horizon"] = horizon_report(runs)

    for e in report["nll"]:
        print(f"teacher-forced NLL, {e['predictions']} predictions, rows {e['rows']} mode {e['mode']}")
        for name, a in e["arms"].items():
            d = "" if name == "f32" else f"  dNLL {a['delta_nll']:+.5f} +- {a['delta_nll_stderr']:.5f}"
            print(f"    {name:5s} NLL {a['nll']:.5f}  ppl {a['perplexity']:.4f}  top1 {a['next_token_top1']:.3f}{d}")
    for g in report["groups"]:
        print(f"{g['shape']:8s} rows {g['rows']:3d} mode {g['mode']:2d} ({g['mode_name']}) - saturated metric, see the NLL table")
        for name, c in g["arms"].items():
            print(f"    {name:4s}  greedy {c['greedy_agree']}/{c['rows']}  top5 {c['top5_overlap_mean']:.3f}  "
                  f"KL mean {c['kl_mean']:.6f} max {c['kl_max']:.6f}  |dlogit| max {c['logit_max_abs_delta']:.4g} "
                  f"rms {c['logit_rms_delta']:.4g}")
    for c in report["continuations"]:
        print(f"continuation mode {c['mode']} streams {c['streams']} state {c['state']}: "
              f"{c['matching']}/{c['tokens']} tokens match"
              + ("" if c["identical"] else f", first divergence at {c['first_divergence']}"))
    print_horizon(report["horizon"])
    if not report["groups"] and not report["horizon"]:
        print("no paired quality groups: run with --gdn-state 0,N", file=sys.stderr)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(args.json)


if __name__ == "__main__":
    main()
