#!/usr/bin/env python3
"""Compare a batch_compare run's quality dumps across sequence-input activation precisions.

Read the NLL table first, for the reason `tools/gdn_state_quality.py` gives and this axis shares:
every activation quantiser in this engine scales a 128-block by its own amax, so KL against an
exact arm saturates and stops ranking anything. Teacher-forced negative log likelihood on the
document the rows came from is an absolute number, paired per row, and it ranks arms.

    tools/seq_quant_quality.py ../../data/bonsai2/batch-comparison/<tag>
    tools/seq_quant_quality.py <dir> --json out.json

Three things this reports that the mode-keyed report cannot:

  * a8 -> a4 WITHIN one mode, which is the cost of the four-bit input projection on its own,
  * the same arms against mode 0 at a8, which is the absolute distance from the exact engine and
    puts the A4 FFN's published cost beside this one on the same instrument, and
  * a4e against a4, which must be bit-identical: they are the same numerical map, one on the
    eight-bit matrix instruction and one on the four-bit one, and any difference is a kernel bug
    rather than a quality result.

The metric helpers come from tools/gdn_state_quality.py rather than a second copy of the same
arithmetic; only the pivot differs.

And read the HORIZON table before deciding anything. The teacher-forced window above scores one
127-prediction slice of one document, which resolves a difference of about +/-0.04 nats; the arms
this axis has to rank differ by less than that, which is why the four-bit input projection sat as
an explicit route rather than a default for a whole day. The horizon workload walks hundreds of
single-token generation steps per stream on fixed teacher-forced tokens, so an activation
quantiser is re-applied once per token per layer instead of once per pass, the pairing is exact per
(stream, step), and the error bar is stream-clustered because one stream's perturbation persists
across its own steps. Run it with `--only horizon --seq-quant a8,a4e,a4`.
"""
import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gdn_state_quality", os.path.join(_here, "gdn_state_quality.py"))
_gsq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gsq)
compare, nll = _gsq.compare, _gsq.nll

REF_QUANT = "a8"
paired, arm_name = _gsq.paired, _gsq.arm_name


def horizon_report(runs, buckets=4):
    """Rank activation precisions on the shape where a quantiser is applied most often.

    Arms are keyed on (mode, state arm, seq_quant) and compared to `a8` inside their own mode, so
    the four-bit input projection is priced against the eight-bit one on identical tokens. The
    mode 0 reference, when the panel carries one, prices what the A4 FFN this engine already serves
    costs on the very same instrument - which is the only number that makes a verdict here
    actionable rather than aesthetic.

    Arms may come from separate processes: the engine is deterministic on fixed inputs, so a quality
    arm reproduces exactly wherever it runs. The check is the target token list, not trust.
    """
    arms, cfg = {}, None
    for run in runs:
        cfg = run.get("horizon_config", cfg)
        for h in run.get("horizon", []):
            arms.setdefault((h["mode"], arm_name(h), h.get("seq_quant", "a8")), h)
    if not arms:
        return []

    def vec(h):
        return np.array([s["nll"] for s in h["scored"]], dtype=np.float64)   # [scored step][stream]

    out = []
    for mode, state in sorted({(m, s) for m, s, _ in arms}):
        ref = arms.get((mode, state, REF_QUANT))
        if ref is None:
            continue
        steps = [s["step"] for s in ref["scored"]]
        tgt = np.array([s["target"] for s in ref["scored"]])
        base, rm = vec(ref), np.array([s["argmax"] for s in ref["scored"]])
        entry = {"mode": mode, "mode_name": ref.get("mode_name"), "state": state,
                 "streams": ref["streams"], "steps": ref["steps"], "ctx": ref["ctx"],
                 "score_every": ref["score_every"], "distinct_windows": ref.get("distinct_windows"),
                 "predictions": int(base.size), "arms": {}}
        edges = np.array_split(np.arange(len(steps)), buckets)
        for q in [REF_QUANT] + sorted(k for m, s, k in arms if (m, s) == (mode, state) and k != REF_QUANT):
            h = arms[(mode, state, q)]
            if [s["step"] for s in h["scored"]] != steps or not np.array_equal(
                    np.array([s["target"] for s in h["scored"]]), tgt):
                print(f"horizon: arm {q} walked different tokens than {REF_QUANT}; not comparable", file=sys.stderr)
                continue
            v, am = vec(h), np.array([s["argmax"] for s in h["scored"]])
            mean, naive, clus = paired(v - base)
            a = {"nll": float(v.mean()), "perplexity": float(math.exp(v.mean())),
                 "teacher_top1": float((am == tgt).mean()),
                 "greedy_agree_ref": float((am == rm).mean()),
                 "delta_nll": mean, "delta_nll_stderr": naive, "delta_nll_stderr_clustered": clus,
                 "per_prediction_sd": float((v - base).std(ddof=1)) if v.size > 1 else 0.0,
                 "by_horizon": []}
            for idx in edges:
                if not len(idx):
                    continue
                bm, _bn, bc = paired(v[idx] - base[idx])
                a["by_horizon"].append({"first_step": int(steps[idx[0]]), "last_step": int(steps[idx[-1]]),
                                        "predictions": int((v[idx] - base[idx]).size),
                                        "delta_nll": bm, "delta_nll_stderr": bc,
                                        "greedy_agree_ref": float((am[idx] == rm[idx]).mean())})
            entry["arms"][q] = a
        # a4e and a4 are one numerical map on two matrix instructions. On this workload only the
        # scored distribution is kept, so equality of per-prediction NLL is the kernel's acceptance
        # against its own map; a difference here is a kernel defect, not a quality result.
        e4, n4 = arms.get((mode, state, "a4e")), arms.get((mode, state, "a4"))
        if e4 is not None and n4 is not None:
            d = np.abs(vec(e4) - vec(n4))
            entry["map_identity"] = {"predictions": int(d.size), "max_abs_delta_nll": float(d.max()),
                                     "mean_abs_delta_nll": float(d.mean())}
        out.append(entry)
    return out


def print_horizon(report, ffn_cost):
    for e in report:
        print(f"\nhorizon, mode {e['mode']} ({e['mode_name']}), state {e['state']}, {e['streams']} streams x "
              f"{e['steps']} teacher-forced steps, {e['predictions']} scored predictions"
              + ("" if e.get("distinct_windows") else ", OVERLAPPING document windows"))
        for q, a in e["arms"].items():
            d = "" if q == REF_QUANT else (f"  dNLL {a['delta_nll']:+.5f} +- {a['delta_nll_stderr_clustered']:.5f}"
                                           f" (naive {a['delta_nll_stderr']:.5f}, per-prediction sd {a['per_prediction_sd']:.3f})")
            print(f"    {q:8s} NLL {a['nll']:.5f}  ppl {a['perplexity']:.4f}  top1 {a['teacher_top1']:.4f}  "
                  f"agree({REF_QUANT}) {a['greedy_agree_ref']:.4f}{d}")
        for q, a in e["arms"].items():
            if q == REF_QUANT:
                continue
            cells = "  ".join(f"[{b['first_step']:>4}-{b['last_step']:<4}] {b['delta_nll']:+.5f}+-{b['delta_nll_stderr']:.5f}"
                              for b in a["by_horizon"])
            print(f"      {q:8s} by horizon: {cells}")
        if "map_identity" in e:
            m = e["map_identity"]
            print(f"    a4e vs a4 (one map, two instructions): max |dNLL| {m['max_abs_delta_nll']:.3e} "
                  f"over {m['predictions']} predictions")
    if ffn_cost is not None:
        print(f"\nreference on this instrument: mode {ffn_cost['mode']} at {REF_QUANT} against mode "
              f"{ffn_cost['ref_mode']} at {REF_QUANT} is dNLL {ffn_cost['delta_nll']:+.5f} "
              f"+- {ffn_cost['delta_nll_stderr_clustered']:.5f} over {ffn_cost['predictions']} predictions")
        print("    that is what the A4 FFN and the wide route this engine already serves cost against")
        print("    the deployed engine. Read every dNLL above beside it, not beside zero.")


def horizon_route_cost(runs, mode, ref_mode=0):
    """What the already-served route costs against the deployed engine, on the horizon instrument."""
    arms = {}
    for run in runs:
        for h in run.get("horizon", []):
            arms.setdefault((h["mode"], h.get("seq_quant", "a8")), h)
    a, b = arms.get((ref_mode, REF_QUANT)), arms.get((mode, REF_QUANT))
    if a is None or b is None:
        return None
    ta = np.array([s["target"] for s in a["scored"]])
    tb = np.array([s["target"] for s in b["scored"]])
    if not np.array_equal(ta, tb):
        return None
    va = np.array([s["nll"] for s in a["scored"]], dtype=np.float64)
    vb = np.array([s["nll"] for s in b["scored"]], dtype=np.float64)
    mean, naive, clus = paired(vb - va)
    return {"mode": mode, "ref_mode": ref_mode, "predictions": int(va.size), "delta_nll": mean,
            "delta_nll_stderr": naive, "delta_nll_stderr_clustered": clus}


def load(base_dir, entry, rows):
    return np.fromfile(os.path.join(base_dir, entry["logits_file"]), dtype=np.float32).reshape(rows, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+", help="results directories or run.json files; horizon arms merge across them")
    ap.add_argument("--json", help="also write the report here")
    ap.add_argument("--route-mode", type=int, default=19,
                    help="the served route whose distance from --exact-mode is the reference scale")
    ap.add_argument("--exact-mode", type=int, default=0, help="the exact engine's mode")
    args = ap.parse_args()
    paths = [os.path.join(p, "run.json") if os.path.isdir(p) else p for p in args.run]
    runs = [json.loads(open(p).read()) for p in paths]
    path = paths[0]
    run = runs[0]
    base_dir = os.path.dirname(os.path.abspath(path))

    groups = {}
    for q in run.get("quality", []):
        if "logits_file" not in q:
            continue
        groups.setdefault((q["shape"], q["rows"]), {})[(q["mode"], q.get("seq_quant", "a8"))] = q

    report = {"run": paths, "revision": run.get("git_revision"), "binary": run.get("binary_sha256"),
              "groups": [], "nll": [], "identity": [], "horizon": [], "route_cost": None}

    for (shape, rows), arms in sorted(groups.items()):
        exact = next((arms[k] for k in arms if k[0] == 0 and k[1] == REF_QUANT), None)
        g = {"shape": shape, "rows": rows, "within_mode": {}, "against_mode0": {}}
        for (mode, quant), q in sorted(arms.items()):
            base = arms.get((mode, REF_QUANT))
            if base is not None and quant != REF_QUANT:
                g["within_mode"][f"{mode}:{quant}"] = compare(load(base_dir, base, rows), load(base_dir, q, rows))
            if exact is not None and (mode, quant) != (0, REF_QUANT):
                g["against_mode0"][f"{mode}:{quant}"] = compare(load(base_dir, exact, rows), load(base_dir, q, rows))
        report["groups"].append(g)

        # a4e and a4 are one numerical map on two instructions: equality is the kernel's acceptance.
        for mode in sorted({k[0] for k in arms}):
            e, n = arms.get((mode, "a4e")), arms.get((mode, "a4"))
            if e is None or n is None:
                continue
            a, b = load(base_dir, e, rows), load(base_dir, n, rows)
            same = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
            report["identity"].append({"shape": shape, "rows": rows, "mode": mode, "values": int(a.size),
                                       "bit_identical": same, "finite": bool(np.isfinite(a).all()),
                                       "differing_values": int((a.view(np.uint32) != b.view(np.uint32)).sum())})

    inputs = run.get("quality_inputs") or {}
    for (shape, rows), arms in sorted(groups.items()):
        if shape != "prefill":
            continue
        rec = inputs.get(str(rows), {}).get("prefill_shape", {}).get("rows") or []
        if len(rec) < 2:
            continue
        tgt = np.array(rec[1:])
        entry = {"shape": shape, "rows": rows, "predictions": int(len(tgt)), "arms": {}}
        base = None
        for (mode, quant) in sorted(arms):
            v = nll(load(base_dir, arms[(mode, quant)], rows), tgt)
            if base is None:
                base = v          # mode 0, a8: the exact engine on this document
            d = v - base
            entry["arms"][f"{mode}:{quant}"] = {
                "nll": float(v.mean()), "perplexity": float(math.exp(v.mean())),
                "next_token_top1": float((load(base_dir, arms[(mode, quant)], rows)[:len(tgt)].argmax(axis=1) == tgt).mean()),
                "delta_nll": float(d.mean()),
                "delta_nll_stderr": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0,
            }
        report["nll"].append(entry)

    for e in report["nll"]:
        print(f"teacher-forced NLL, {e['predictions']} predictions, {e['shape']} rows {e['rows']}")
        for name, a in e["arms"].items():
            print(f"    mode:quant {name:8s} NLL {a['nll']:.5f}  ppl {a['perplexity']:.4f}  "
                  f"next-token top1 {a['next_token_top1']:.3f}  dNLL {a['delta_nll']:+.5f} +- {a['delta_nll_stderr']:.5f}")
    for g in report["groups"]:
        for title, arms in (("within its own mode at a8", g["within_mode"]), ("against mode 0 at a8", g["against_mode0"])):
            if not arms:
                continue
            print(f"{g['shape']:8s} rows {g['rows']:3d}, {title}")
            for name, c in arms.items():
                print(f"    {name:8s} greedy {c['greedy_agree']}/{c['rows']}  top5 {c['top5_overlap_mean']:.3f}  "
                      f"KL mean {c['kl_mean']:.6f} max {c['kl_max']:.6f}  |dlogit| max {c['logit_max_abs_delta']:.4g}")
    for i in report["identity"]:
        verdict = "IDENTICAL" if i["bit_identical"] else f"DIFFERS in {i['differing_values']} values"
        print(f"a4e vs a4, {i['shape']} rows {i['rows']} mode {i['mode']}: {i['values']} logits {verdict}"
              + ("" if i["finite"] else "  (non-finite values present)"))

    report["horizon"] = horizon_report(runs)
    report["route_cost"] = horizon_route_cost(runs, args.route_mode, args.exact_mode)
    print_horizon(report["horizon"], report["route_cost"])

    if not report["groups"] and not report["horizon"]:
        print("no paired quality groups: run with --seq-quant a8,a4", file=sys.stderr)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(args.json)


if __name__ == "__main__":
    main()
