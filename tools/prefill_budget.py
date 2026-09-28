#!/usr/bin/env python3
"""Reproducible throughput budgets for this model, not ISA-independent lower bounds.

A MAC is counted as two conventional operations. The matrix issue periods are
explicit assumptions, not promises extracted from the ISA manual. Matrix counts
price the usual dense computation; another exact whole-map construction need
not perform that many MACs. Weight traffic assumes one streaming read per pass.

Shapes come from kernels/halo_kernels.h: D=5120, FF=17408, NLAYER=64,
VOCAB=248320, HV=48 value heads of SS=128, HK=16 key heads, CONV_CH=10240,
VDIM=6144, 16 attention layers with NH=24 query heads, NKV=4 KV heads, HD=256,
Q_OUT=12288 (q and gate interleaved), KV_OUT=1024, ATTN_OUT=6144, RMAX=8 rows
per recurrent/attention pass.

Serial and overlapped arithmetic rows bracket how the *declared* arithmetic
could be scheduled. Neither is a runtime floor: every omitted cost only adds
time, so a real engine can fall below both.
"""
import argparse
import json
import math

D, FF, NLAYER, VOCAB = 5120, 17408, 64, 248320
HV, SS, CONV_CH, VDIM = 48, 128, 10240, 6144
NGDN, NATTN = 48, 16
NH, NKV, HD, Q_OUT, KV_OUT, ATTN_OUT = 24, 4, 256, 12288, 1024, 6144
RMAX = 8
SIMD32, WMMA_MACS = 80, 4096          # gfx1151: 40 CUs x 2 SIMD32; 16x16x16 WMMA
HALO_BYTES_PER_BLOCK, HALO_BLOCK = 28, 128   # 24 qs + 2 qh code bytes + fp16 scale
BLK_TOKEN_FLOATS = 2 * CONV_CH + 2 * HV      # GDN replay record: pre-conv, conv out, alpha, beta
# Achieved ns per WMMA per physical SIMD32 at the largest probe row, from
# kelana research/ffn/batched/arithmetic: f16 11.8495, iu8 11.813, iu4 5.99325.
MEASURED_NS = {"f16": 11.8495, "iu8": 11.813, "iu4": 5.99325}


def budget(batch, context, ghz, bandwidth, trit_gain, lm_rows, rows_per_pass,
           wmma_ns=None, sequences=1):
    if sequences < 1 or batch < sequences or batch % sequences:
        raise ValueError("batch rows must divide evenly among positive independent sequences")
    rows_per_sequence = batch // sequences
    rows_per_pass = min(rows_per_pass, rows_per_sequence)
    ffn = NLAYER * 3 * D * FF
    gdn_proj = NGDN * (CONV_CH * D + VDIM * D + D * VDIM)      # qkv, gate(z), ssm_out
    attn_proj = NATTN * (Q_OUT * D + 2 * KV_OUT * D + D * ATTN_OUT)
    lm = D * VOCAB
    alpha_beta = NGDN * 2 * HV * D                             # bf16, not ternary
    trunk = ffn + gdn_proj + attn_proj
    macs = trunk + lm * lm_rows / batch
    # The full vocabulary matrix is streamed once if this pass asks for logits.
    weight_trits = trunk + (lm if lm_rows else 0)
    stored_bytes = weight_trits * HALO_BYTES_PER_BLOCK // HALO_BLOCK + alpha_beta * 2
    # Maximum-entropy code length for independent uniform trits, same per-block
    # scales: a family bound, not the shortest description of these weights.
    entropy_bytes = weight_trits * (math.log2(3) / 8 + 2 / HALO_BLOCK) + alpha_beta * 2
    # A perfect five-trits-per-byte code with the same scales, for comparison.
    five_trit_bytes = weight_trits * (1 / 5 + 2 / HALO_BLOCK) + alpha_beta * 2
    if wmma_ns:   # achieved rates from the native probe, not capacities
        iu4_mac_s = SIMD32 * WMMA_MACS / (wmma_ns[0] * 1e-9)
        iu8_mac_s = SIMD32 * WMMA_MACS / (wmma_ns[1] * 1e-9)
    else:         # declared service periods at the given clock
        iu4_mac_s = SIMD32 * WMMA_MACS * ghz * 1e9 / 16
        iu8_mac_s = SIMD32 * WMMA_MACS * ghz * 1e9 / 32
    # Non-matrix lanes: one f32 FMA per lane per cycle, no dual issue claimed.
    valu_fma_s = SIMD32 * 32 * ghz * 1e9
    # Independent sequences each carry their own prefix and causal positions.
    keys_avg = context + (rows_per_sequence + 1) / 2
    attn_macs = NATTN * NH * HD * 2 * keys_avg
    # Delta-rule recurrence per head per token: S k, the rank-1 update and S q
    # are three passes over the 128x128 state; the decay is one more multiply.
    gdn_macs = NGDN * HV * SS * SS * 3
    gdn_mults = NGDN * HV * SS * SS
    conv_macs = NGDN * 4 * CONV_CH
    seq_macs = attn_macs + gdn_macs + conv_macs
    valu_s = (seq_macs + gdn_mults) / valu_fma_s
    mixed_s = ffn / iu4_mac_s + (macs - ffn) / iu8_mac_s
    sequence_a4_s = trunk / iu4_mac_s + (macs - trunk) / iu8_mac_s
    state_bytes = NGDN * HV * SS * SS * 4
    gdn_per_token = 2 * state_bytes
    groups_per_sequence = math.ceil(rows_per_sequence / rows_per_pass)
    groups = sequences * groups_per_sequence
    # The replay-sliced schedule is a priced alternative, not every current route.
    # Resident direct-commit routes do not pay this replay-record term.
    replay_bytes = 2 * batch * NGDN * BLK_TOKEN_FLOATS * 4
    replay_macs = gdn_macs
    # Minimal unique K/V operand bytes read by attention, assuming perfect GQA
    # reuse and no rereads. kv_current re-reads the prefix once per row group.
    kv_per_position = NATTN * NKV * HD * 2 * 2
    kv_unique = sequences * kv_per_position * (context + rows_per_sequence)
    kv_current = sequences * kv_per_position * sum(
        context + min(rows_per_sequence, (i + 1) * rows_per_pass)
        for i in range(groups_per_sequence))
    logits_bytes = lm_rows * VOCAB * 4
    per_pass_min_bytes = stored_bytes + 2 * state_bytes * sequences + kv_unique + logits_bytes
    per_pass_schedule_bytes = (stored_bytes + 2 * state_bytes * groups
                               + kv_current + logits_bytes)
    return {
        "batch": batch, "sequences": sequences, "rows_per_sequence": rows_per_sequence,
        "prefix_context": context, "logit_rows": lm_rows,
        "rows_per_recurrent_pass": rows_per_pass,
        "dense_macs_per_token": macs,
        "dense_ops_per_token": 2 * macs,
        "components_macs_per_token": {"ffn": ffn, "gdn_projections": gdn_proj,
            "attention_projections": attn_proj, "lm_head_charged_rows": lm * lm_rows / batch,
            "lm_head_all_rows": lm, "bf16_alpha_beta": alpha_beta,
            "attention_qk_av": attn_macs, "gdn_recurrence": gdn_macs,
            "gdn_state_decay_extra_mults": gdn_mults, "gdn_conv1d": conv_macs},
        "sequence_macs_per_token": seq_macs,
        "gdn_replay_macs_per_token": replay_macs,
        "gdn_replay_record_bytes_per_token": replay_bytes / batch,
        "packed_weight_bytes_per_pass": stored_bytes,
        "entropy_weight_bytes_per_pass": entropy_bytes,
        "five_trit_weight_bytes_per_pass": five_trit_bytes,
        "state_bytes": state_bytes,
        "kv_bytes_per_position": kv_per_position,
        "conditional_compute_tps": {
            "all_iu8": iu8_mac_s / macs,
            "ffn_iu4_other_iu8": 1 / mixed_s,
            "ffn_and_sequence_iu4_head_iu8": 1 / sequence_a4_s,
            "all_iu4": iu4_mac_s / macs,
            "all_iu8_plus_serial_valu": 1 / (macs / iu8_mac_s + valu_s),
            "ffn_iu4_other_iu8_plus_serial_valu": 1 / (mixed_s + valu_s),
            "ffn_and_sequence_iu4_head_iu8_plus_serial_valu": 1 / (sequence_a4_s + valu_s),
            "all_iu4_plus_serial_valu": 1 / (macs / iu4_mac_s + valu_s),
            "hypothetical_native_ternary": iu4_mac_s * trit_gain / macs},
        "conditional_bandwidth_tps": {
            "weights_only": bandwidth * 1e9 * batch / stored_bytes,
            "entropy_weights_only": bandwidth * 1e9 * batch / entropy_bytes,
            "weights_and_per_token_f32_state": bandwidth * 1e9 / (stored_bytes / batch + gdn_per_token),
            "weights_and_row_group_f32_state": bandwidth * 1e9 * batch / (stored_bytes + 2 * state_bytes * groups),
            "weights_and_once_per_chunk_f32_state": bandwidth * 1e9 * batch / (stored_bytes + 2 * state_bytes * sequences),
            "row_group_state_kv_logits_ideal_weight_reuse": bandwidth * 1e9 * batch / per_pass_schedule_bytes,
            "same_plus_replay_records": bandwidth * 1e9 * batch / (per_pass_schedule_bytes + replay_bytes),
            "chunk_state_unique_kv_and_logits": bandwidth * 1e9 * batch / per_pass_min_bytes},
        "hypothetical_joint_tps": min(iu4_mac_s * trit_gain / macs,
            bandwidth * 1e9 * batch / per_pass_min_bytes),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", default="1,32,128,256,1024")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--sequences", type=int, default=1,
                   help="independent sequences sharing each batch; equal rows per sequence")
    p.add_argument("--ghz", type=float, default=2.9)
    p.add_argument("--bandwidth-gbs", type=float, default=256)
    p.add_argument("--ternary-gain", type=float, default=1.0,
                   help="EXTRA hardware hypothesis: MAC/issue gain over IU4, not derived from storage density")
    p.add_argument("--logits", choices=["all", "last-pass", "last", "none"], default="all",
                   help="head rows charged to this pass; last-pass spreads one pass of rows over --prompt-tokens")
    p.add_argument("--prompt-tokens", type=int, default=256,
                   help="prompt length for --logits last-pass, matching the batch comparison")
    p.add_argument("--rows-per-pass", type=int, default=RMAX,
                   help="rows sharing one recurrent/attention pass in the modelled schedule")
    p.add_argument("--wmma-ns", nargs=2, type=float, metavar=("IU4_NS", "IU8_NS"),
                   help=f"achieved ns per WMMA per physical SIMD32 instead of the cycle model, "
                        f"e.g. {MEASURED_NS['iu4']} {MEASURED_NS['iu8']} from the native probe")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    batches = [int(b) for b in a.batches.split(",")]
    if min(batches) < 1 or a.context < 0 or min(a.ghz, a.bandwidth_gbs, a.ternary_gain) <= 0:
        p.error("positive batches, clock, bandwidth and ternary gain required; context must be nonnegative")
    if a.rows_per_pass < 1 or a.prompt_tokens < 1:
        p.error("rows per pass and prompt tokens must be positive")
    if a.sequences < 1 or any(b < a.sequences or b % a.sequences for b in batches):
        p.error("each batch must divide evenly among positive independent sequences")
    if a.sequences > 1 and a.logits in {"last", "last-pass"}:
        p.error("multi-sequence budgets require --logits all or none; final-row policies differ by sequence")

    def rows(b):
        if a.logits == "all":
            return b
        if a.logits == "last":
            return 1
        if a.logits == "none":
            return 0
        return min(b, a.prompt_tokens) * b / a.prompt_tokens  # head on the final pass only
    if a.wmma_ns and min(a.wmma_ns) <= 0:
        p.error("WMMA times must be positive")
    out = [budget(b, a.context, a.ghz, a.bandwidth_gbs, a.ternary_gain, rows(b),
                  min(a.rows_per_pass, b), a.wmma_ns, a.sequences) for b in batches]
    rate_model = (f"achieved probe rates: {a.wmma_ns[0]} ns IU4, {a.wmma_ns[1]} ns IU8 per WMMA "
                  f"per physical SIMD32; achieved, not capacity" if a.wmma_ns else
                  f"declared service periods 16/32 cycles at {a.ghz:g} GHz")
    assumptions = {
        "kind": "conditional conventional-computation budgets, not absolute TPS bounds",
        "clock_ghz": a.ghz, "bandwidth_gbs": a.bandwidth_gbs,
        "matrix_rate_model": rate_model, "measured_ns_per_wmma_per_simd32": MEASURED_NS,
        "simd32": SIMD32, "wmma_macs": WMMA_MACS, "iu4_issue_cycles_assumed": 16,
        "iu8_issue_cycles_assumed": 32, "ternary_gain_assumed": a.ternary_gain,
        "rows_per_recurrent_pass": a.rows_per_pass,
        "independent_sequences": a.sequences,
        "schedule_scope": "Explicit F32 state traffic alternatives: per token, per row group, "
            "and resident across each sequence's rows. Replay is a separate column, not a "
            "universal engine cost. Context means occupied prefix per sequence, not allocation capacity.",
        "weight_format": "HALO alternative: 26 code bytes + 2 FP16-scale bytes per 128 weights",
        "entropy_scope": "log2(3) per trit is the maximum-entropy code length for independent "
            "uniform trits with the same per-block scales. It bounds that representation family, "
            "not the shortest description of these particular trained weights.",
        "omitted_online_cost": ["operand construction", "scale arithmetic", "Hadamard and normalization",
            "bf16 alpha/beta projection issue cost, whose MAC count is reported separately",
            "SiLU", "softmax and transcendentals", "activation traffic",
            "cache pressure", "synchronization and launch latency", "resource conflicts"],
        "schedule_costs_not_resolved_by_this_model": [
            "sliced and wide routes have different weight reuse, token tile widths and state "
            "commit policies; this model grants one weight read per whole pass and does not "
            "predict either route's physical DRAM traffic",
            "partially filled matrix fragments and output-head policy depend on the actual route",
            "packed recurrent states, deferred commits and their reconstruction work are not "
            "modelled by these explicit F32 state columns",
            "the replay column counts records, but repeated recurrence arithmetic is absent "
            "from the one-step compute rows"],
        "not_implied": ["all activations can be quantized to A4 without changing the model",
            "fewer weight bits proportionally increase MAC issue throughput",
            "MAC count is invariant under whole-map rewrites",
            "the matrix and non-matrix phases overlap perfectly, or cannot overlap",
            "serial and overlapped rows bracket the runtime: omitted costs can put a real "
            "engine below both, and none of these rows identifies the binding resource",
            "all model weights must be fetched once per pass under every possible representation"]}
    if a.json:
        print(json.dumps({"assumptions": assumptions, "budgets": out}, indent=2))
    else:
        print(assumptions["kind"])
        print(f"{rate_model}; {a.bandwidth_gbs:g} GB/s, logits={a.logits}, {a.rows_per_pass} rows/pass, "
              f"{a.sequences} sequences, context {a.context}; omitted work is NOT free in real kernels.")
        print("batch | IU8 | FFN-A4 | FFN+seq-A4 | IU4 | IU8+VALU | FFN-A4+VALU | IU4+VALU | weights BW | "
              "+row-group state | +KV/logits/replay | ideal chunk+KV+logits")
        for r in out:
            c, m = r["conditional_compute_tps"], r["conditional_bandwidth_tps"]
            print(f"{r['batch']:5} | {c['all_iu8']:.0f} | {c['ffn_iu4_other_iu8']:.0f} | "
                  f"{c['ffn_and_sequence_iu4_head_iu8']:.0f} | {c['all_iu4']:.0f} | "
                  f"{c['all_iu8_plus_serial_valu']:.0f} | "
                  f"{c['ffn_iu4_other_iu8_plus_serial_valu']:.0f} | "
                  f"{c['all_iu4_plus_serial_valu']:.0f} | "
                  f"{m['weights_only']:.0f} | {m['weights_and_row_group_f32_state']:.0f} | "
                  f"{m['same_plus_replay_records']:.0f} | "
                  f"{m['chunk_state_unique_kv_and_logits']:.0f}")
        print("FFN-A4 prices only FFNs at IU4; FFN+seq-A4 also prices sequence projections at IU4.")
        print("Both retain IU8 head work. These are declared maps, not automatic claims about a serving mode.")
        print("VALU rows add the recurrence and attention products serially; omitted costs can put a")
        print("real engine below every row here, and no row identifies which resource binds.")
        print("A ternary-gain > 1 grants hypothetical hardware; packing density alone supplies no such gain.")


if __name__ == "__main__":
    main()
