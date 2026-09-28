#!/usr/bin/env python3
"""Counted work and traffic for one GDN head's observed map, per execution form.

Everything here is counted work divided by a declared peak rate. That is a
budget, not a predicted runtime, and the ratio between two budgets is not a
predicted speedup. The measured anchor below shows the size of the gap: the
landed resident_state<4> kernel takes 4.0x its own budget.

Baseline is the resident serial path: the state is loaded into registers once
per pass, every token of the pass runs through it, and it is written back once
(`resident_state` in kernels/halo_rows.hip). No form here reloads the state per
token, so state DRAM traffic is identical across forms and the counted
difference between forms is instruction slots.

Counting rules, all stated so they can be disputed:
  * one lane-op = one VALU instruction slot for one lane; a wave32 instruction
    is 32 lane-ops. Multiply and FMA both count 1. MAC counts follow the repo's
    convention in docs/throughput-budgets.md (one MAC = one FMA lane-op).
  * cross-lane reduction cost is a property of the state layout, not of the
    algebra. Both the serial recurrence and the block form contract the state
    over the key index twice per row per token; only a layout that keeps that
    index inside a lane removes the reduction. A state row split across L lanes
    costs the DPP stages of warp_sum that span L (device.hpp: 1, 2, 3, 4 and 6
    instructions for L = 2, 4, 8, 16, 32; the last stage is permlanex16 plus an
    add). A lane holding R rows pays that R times, because its rows' partial
    sums sit in different registers.
  * contractions over the token index (the rank-one or rank-c state update) and
    over the sub-chunk index (triangular solve, intra-chunk output) need no
    reduction in any of these layouts.
  * vector lane rate 7.424 T MAC/s = 80 SIMD32 x 32 lanes x 2.9 GHz, the same
    constant docs/throughput-budgets.md uses to price vector work; dual issue
    (VOPD) is not assumed anywhere.
  * DRAM 256 GB/s, per PLAN.md.

    python3 docs/gdn-map/gdn_cost.py [--m 128] [--dump FILE]
"""
import argparse
import sys

D = 128          # key dim per head
DV = 128         # value rows per head
HV = 48          # value heads per recurrent layer
LAYERS = 48      # recurrent layers
CONV_CH = 10240  # q, k and v channels per token per layer
VDIM = 6144      # GDN output floats per token per layer
LANE_RATE = 80 * 32 * 2.9e9      # 7.424e12 lane-ops/s, single issue
DRAM = 256e9

# rocprof, one native 128-row A4 pass, empty prefix, no warmup or previous passes.
# tools/batch-compare/results/resident-kernel-times.json
MEASURED_PASS_MS = 232.172
MEASURED_STATE_MS = 18.293673     # resident_state<4>, summed over 48 recurrent layers
MEASURED_CONV_MS = 1.825704
MEASURED_OUT_MS = 0.938310


DPP_STAGES = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 6}


def reductions(m, lanes_per_row, rows_per_lane, d=D, dv=DV):
    """Lane-ops spent on cross-lane reduction for the two state contractions."""
    waves = (dv / rows_per_lane) * lanes_per_row / 32
    return waves * m * 2 * rows_per_lane * DPP_STAGES[lanes_per_row] * 32


def vgprs(lanes_per_row, rows_per_lane, d=D):
    return (d / lanes_per_row) * rows_per_lane


def serial_source(m, layout=(32, 4), d=D, dv=DV):
    """gdn_token as written: decay, k dot, update, q dot."""
    return {"arithmetic": m * 4 * d * dv,
            "reductions": reductions(m, *layout),
            "row scalar": m * dv * 4}


def serial_identities(m, rescale_period, layout=(32, 4), d=D, dv=DV):
    """Identities A and B: 3 state passes plus an amortised rescale."""
    return {"arithmetic": m * (3 + 1.0 / rescale_period) * d * dv,
            "reductions": reductions(m, *layout),
            "row scalar": m * dv * 5,
            "k.q dot": m * d}       # per token per key head, if not done in the conv phase


def block_wy(m, c, layout=(32, 4), d=D, dv=DV):
    """Sub-chunked WY/triangular form, sub-chunk length c."""
    nsub, rem = divmod(m, c)
    sizes = [c] * nsub + ([rem] if rem else [])
    grams = solve = intra = state = 0
    for s in sizes:
        lower = s * (s - 1) // 2
        causal = s * (s + 1) // 2
        grams += lower * (d + 1) + causal * (d + 1)      # Ntil and Ctil, with decay ratios
        solve += lower * dv + s * dv                     # forward substitution and beta
        intra += causal * dv                             # Ctil D
        state += 2 * s * d * dv                          # K S^T and Q S^T
        state += (s + 1) * d * dv                        # decay multiply then rank-s update
    return {"state contractions": state, "token grams": grams,
            "triangular solve": solve, "intra output": intra,
            "reductions": reductions(m, *layout)}


def deferred_write_traffic(r, d=D, dv=DV, pair_bytes=4):
    """Decode traffic per head per token if the state is kept factored.

    S_t = gamma S_base + sum_i c_i u_i k_i^T. The pending list holds 0..r-1 pairs
    over the cycle, so the average read is (r-1)/2 pairs, not r. The rank-r
    materialisation happens on the token that fills the list, when the base and
    all pending pairs are already in registers from that token's own
    contractions, so it adds a base write and no second read. For a dense state
    array and an unconstrained q_t, o_t = S_t q_t touches every base element and
    the read cannot move; that is a property of this representation, not of the
    map.
    """
    return {"base read": d * dv * 4,
            "pending reads": (r - 1) / 2 * (d + dv) * pair_bytes,
            "pending write": (d + dv) * pair_bytes,
            "amortised base write": d * dv * 4 / r}


def deferred_write_work(r, d=D, dv=DV):
    """Counted lane-ops per head per token for the same scheme."""
    pend = (r - 1) / 2
    return {"base contractions": 2 * d * dv,
            "amortised rank-r update": d * dv,          # r*d*dv every r tokens
            "pending corrections": pend * (2 * d + 2 * dv),
            "coefficients and row scalar": pend + dv * 5}


def total(parts):
    return sum(parts.values())


def fmt(n):
    return f"{n:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=128, help="tokens per pass")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    out = []

    def say(s=""):
        out.append(s)
        print(s)

    m = args.m
    say(f"GDN state phase counted work, one head, d={D}, dv={DV}, m={m} tokens per pass")
    say(f"lane rate {LANE_RATE/1e12:.3f}e12 lane-ops/s, DRAM {DRAM/1e9:.0f} GB/s")
    say("Times below are counted work at that declared rate. They are budgets, not")
    say("predicted runtimes, and a ratio of budgets is not a predicted speedup.")
    say()

    state_us_tok = MEASURED_STATE_MS * 1e3 / 128
    budget_us_tok = total(serial_source(128, (32, 4))) / 128 * HV * LAYERS / LANE_RATE * 1e6
    say("measured anchor, rocprof, one native 128-row A4 pass, empty prefix:")
    say(f"  wall {MEASURED_PASS_MS:.3f} ms; resident_state<4> {MEASURED_STATE_MS:.3f} ms over"
        f" 48 layers, resident_conv {MEASURED_CONV_MS:.3f}, resident_output {MEASURED_OUT_MS:.3f}")
    say(f"  state kernel is {MEASURED_STATE_MS/MEASURED_PASS_MS*100:.2f}% of the pass, "
        f"{state_us_tok:.0f} us/token, against a {budget_us_tok:.1f} us/token budget: "
        f"{state_us_tok/budget_us_tok:.1f}x.")
    say("  The traffic budget for the same kernel is smaller still, so neither counted")
    say("  resource explains the measured time at these rates. What does is not")
    say("  established here, and until it is, no count ratio below predicts a time.")
    say()

    deployed = (32, 4)      # SPLIT=4 as deployed: row over 32 lanes, 4 rows per lane
    narrow = (4, 1)         # row over 4 lanes, one row per lane
    rowwise = (1, 1)        # whole row inside one lane
    forms = [
        ("serial, as in gdn_token", deployed, serial_source(m, deployed)),
        ("serial + identities A,B", deployed, serial_identities(m, 16, deployed)),
        ("block WY c=8", deployed, block_wy(m, 8, deployed)),
        ("block WY c=32", deployed, block_wy(m, 32, deployed)),
        ("serial, as written", narrow, serial_source(m, narrow)),
        ("serial + identities A,B", narrow, serial_identities(m, 16, narrow)),
        ("block WY c=8", narrow, block_wy(m, 8, narrow)),
        ("serial + identities A,B", rowwise, serial_identities(m, 16, rowwise)),
        ("block WY c=8", rowwise, block_wy(m, 8, rowwise)),
    ]
    base = total(forms[0][2])
    say(f"{'form':<26}{'L,R':>8}{'VGPR':>6}{'waves/hd':>10}{'lane-ops/tok':>14}"
        f"{'vs deployed':>13}{'us/token':>10}")
    for name, layout, parts in forms:
        per_tok = total(parts) / m
        us = per_tok * HV * LAYERS / LANE_RATE * 1e6
        L, R = layout
        say(f"{name:<26}{str(L) + ',' + str(R):>8}{vgprs(L, R):>6.0f}"
            f"{DV / R * L / 32:>10.0f}{fmt(per_tok):>14}{base/total(parts):>12.2f}x{us:>10.1f}")
    say()

    say("breakdown, lane-ops per token per head")
    for name, layout, parts in forms:
        items = "  ".join(f"{k} {fmt(v/m)}" for k, v in parts.items())
        say(f"  {name + ' ' + str(layout):<34}{items}")
    say()

    floor = 3 * D * DV
    say(f"Three state contractions per token are unavoidable in these formulas:")
    say(f"  k-side (delta needs g S_(t-1) k_t), q-side (output), rank-m update.")
    say(f"  floor {fmt(floor)} lane-ops/token/head = "
        f"{floor*HV*LAYERS/LANE_RATE*1e6:.1f} us/token over 48 layers.")
    say(f"  deployed serial is {base/m/floor:.2f}x that floor, the narrow layout with")
    say(f"  identities A and B is {total(serial_identities(m,16,narrow))/m/floor:.2f}x, "
        f"block WY c=8 on the same layout {total(block_wy(m,8,narrow))/m/floor:.2f}x.")
    say("  Reductions are a layout property: both forms contract the state over the key")
    say("  index twice per row per token, so the block form does not remove them.")
    say("  The block form counts LOWER than the unmodified deployed serial form")
    say(f"  ({fmt(total(block_wy(m,8,deployed))/m)} against {fmt(base/m)}), because it"
        f" drops the decay pass. Its only")
    say("  adverse comparator is serial + identities A,B at the SAME layout, where it")
    say(f"  counts {fmt(total(block_wy(m,8,narrow))/m)} against "
        f"{fmt(total(serial_identities(m,16,narrow))/m)}: more counted arithmetic, by")
    say("  the token-mixing terms. Scheduling, dependence and matrix units are not")
    say("  represented here, and any of them can reverse that on a real machine.")
    say()

    # ---- traffic ----------------------------------------------------------
    say(f"traffic per recurrent layer per pass of {m} tokens, one sequence, fp32")
    state_bytes = HV * D * DV * 4
    say(f"  state read + write        {2*state_bytes/1e6:8.2f} MB   "
        f"({2*state_bytes/m/1e3:7.1f} kB/token)   same for every form")
    op_bytes = m * (CONV_CH + 2 * HV) * 4
    say(f"  conv output and gates     {op_bytes/1e6:8.2f} MB   ({op_bytes/m/1e3:7.1f} kB/token)")
    o_bytes = m * VDIM * 4
    say(f"  outputs                   {o_bytes/1e6:8.2f} MB   ({o_bytes/m/1e3:7.1f} kB/token)")
    say(f"  serial intermediates          0.00 MB   state stays in registers")
    for c in (8, 16, 32, m):
        inter = HV * (c * DV + 2 * c * c + c * D + c * DV) * 4
        where = "below 64 kB; allocation unimplemented" if inter / HV <= 64e3 else "above 64 kB; tiling/storage unimplemented"
        say(f"  block WY c={c:<3} intermediates {inter/1e6:8.2f} MB   "
            f"{inter/HV/1e3:7.1f} kB per head, {where}")
    say()
    total_bytes = 2 * state_bytes + op_bytes + o_bytes
    say(f"  layer total {total_bytes/1e6:.2f} MB, {total_bytes/m/1e3:.1f} kB/token, "
        f"{total_bytes/m*LAYERS/DRAM*1e6:.1f} us/token over 48 layers at 256 GB/s")
    say()

    # ---- where the pass sits ---------------------------------------------
    say("budgets per token, 48 recurrent layers, both at declared peak rates")
    say("(traffic = state read+write amortised over m, plus this token's operands")
    say(" and outputs; work = VALU slots at the single-issue lane rate)")
    say(f"{'m':>5}{'traffic us':>18}{'deployed work us':>20}{'narrow + A,B work us':>24}")
    for mm in (1, 8, 32, 128):
        tr = (2 * state_bytes / mm + (CONV_CH + 2 * HV) * 4 + VDIM * 4) * LAYERS / DRAM * 1e6
        ser = total(serial_source(mm, deployed)) / mm * HV * LAYERS / LANE_RATE * 1e6
        best = total(serial_identities(mm, 16, narrow)) / mm * HV * LAYERS / LANE_RATE * 1e6
        say(f"{mm:>5}{tr:>18.1f}{ser:>20.1f}{best:>24.1f}")
    say(f"At m=128 the measured state kernel is {state_us_tok:.0f} us/token, above both"
        f" columns.")
    say("At m=1 the traffic column dominates the work column by a factor of tens, and")
    say("PLAN.md's 1,270 us/token decode phase sits near it. Neither observation is a")
    say("proof of what binds either regime.")
    say()
    say("For a dense state array and an unconstrained q_t, o_t = S_t q_t touches every")
    say("element, so the read cannot move within that representation. The write can:")
    say("nothing outside the head reads the state between tokens.")
    say()
    say("decode, one head, one token: keep the state factored, materialise every r")
    say("(pending list averages (r-1)/2 pairs; the rank-r update reuses the base and")
    say(" pairs already read by that token's own contractions)")
    now = 2 * D * DV * 4
    say(f"{'r':>5}{'bytes/token/head':>19}{'vs read+write':>15}{'MB/token/seq':>15}{'us/token/seq':>14}")
    say(f"{'-':>5}{now:>19,}{1.0:>14.2f}x{now*HV*LAYERS/1e6:>15.0f}{now*HV*LAYERS/DRAM*1e6:>14.0f}")
    for r in (4, 8, 16, 32):
        b = total(deferred_write_traffic(r))
        say(f"{r:>5}{b:>19,.0f}{now/b:>14.2f}x{b*HV*LAYERS/1e6:>15.0f}"
            f"{b*HV*LAYERS/DRAM*1e6:>14.0f}")
    b16 = total(deferred_write_traffic(8, pair_bytes=2))
    say(f"{'8*':>5}{b16:>19,.0f}{now/b16:>14.2f}x{b16*HV*LAYERS/1e6:>15.0f}"
        f"{b16*HV*LAYERS/DRAM*1e6:>14.0f}   (* bf16 pending pairs)")
    say()
    say("its counted work, lane-ops per token per head, against 65,536 now:")
    for r in (8,):
        w = deferred_write_work(r)
        say(f"  r={r}: " + "  ".join(f"{k} {fmt(v)}" for k, v in w.items())
            + f"  total {fmt(total(w))}")
    say("So the scheme costs no extra counted arithmetic. The traffic column is a")
    say("declared-rate budget for one phase; it is not a tokens/s prediction, and the")
    say("decode phase's own measured time is not isolated here. bf16 state halves the")
    say("base read independently and composes with this.")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
