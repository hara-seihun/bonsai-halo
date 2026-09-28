#include "engine.h"
#include "batch_route.h"
#include "ffn_batch.h"
#include "batch_profile.hpp"
#include "sequence_batch.h"
#include "head_batch.h"
#include "attn_batch.h"
#include "prep_batch.h"
#include "device_budget.h"
#include <algorithm>
#include <set>

namespace halo {

// How many rows one deployed-FFN slice carries. `mvw_rows` fills sixteen matrix columns and then
// takes one more token tile per sixteen rows against the same weight block, so this is the number
// of times a pass re-reads the 3.74 GB FFN weight stream. It is a schedule, not a numerical map:
// every width computes each output element from the same weights in the same K order with the same
// FP32 chain. Settable so one process can measure two widths against a common clock;
// `HALO_FFN_SLICE` gives the same choice to a process that cannot call the setter.
static int g_ffn_slice_rows = [] {
    const char * v = getenv("HALO_FFN_SLICE");
    const int n = v ? atoi(v) : FMAX;
    return n >= 1 && n <= FMAX ? n : FMAX;
}();
int batch_ffn_slice_rows() { return g_ffn_slice_rows; }
void batch_set_ffn_slice_rows(int rows) { if (rows >= 1 && rows <= FMAX) g_ffn_slice_rows = rows; }

// The row count at which an automatic mode takes the wide schedule; see batch_route.h for what a
// pass below it gives up. The engine shipped with 32 and nobody had measured the band underneath
// it: the wide route's own modules admit at RMAX + 1 - `wide_head` and `wide_attn` both test
// `total > RMAX` - and every wide kernel takes its row count as a parameter, so 32 was the only
// thing holding 9..31 rows on the sliced route. `HALO_WIDE_MIN=32` restores the shipped floor
// exactly; `docs/serve-wide-decode.md` measures every width in the band.
// The default stays where the engine shipped it because crossing it is a numerical change, not a
// scheduling one, and `docs/serve-wide-decode.md` measures both halves of that trade.
static int g_wide_min = [] {
    const char * v = getenv("HALO_WIDE_MIN");
    const int n = v ? atoi(v) : WIDE_MIN_DEFAULT;
    return n >= 1 && n <= PASSMAX ? n : WIDE_MIN_DEFAULT;
}();
int batch_wide_min() { return g_wide_min; }
void batch_set_wide_min(int rows) { if (rows >= 1 && rows <= PASSMAX) g_wide_min = rows; }

// `HALO_ROUTE_TRACE=1` prints the route every pass actually took. An instruction census prices an
// instantiation and the launcher decides which one runs, so the same rule holds one level up: the
// row count decides the schedule, and a panel that assumes the route it asked for can measure a
// different one for a whole afternoon. One line per pass, on stderr, outside any timed region.
static const bool g_route_trace = getenv("HALO_ROUTE_TRACE") != nullptr;

void Engine::prepare_sequence() {
    if (!batch_capacity) throw std::runtime_error("prepare_batch before preparing sequence projections");
    if (!batch_sequence) {
        admit_device_bytes(device_bytes, sequence_batch_required_bytes(host_layers.data(), batch_capacity),
            "prepare_sequence");
        batch_sequence = create_sequence_batch(host_layers.data(), batch_capacity);
        device_bytes += sequence_batch_bytes(batch_sequence);
    }
}

void Engine::prepare_batch(int max_rows, unsigned modes) {
    if (max_rows < 1 || max_rows > PASSMAX) throw std::runtime_error("batch capacity outside 1..PASSMAX");
    if (!modes) modes = FFN_BATCH_ALL;
    modes &= ~BATCH_WORKSPACE_ONLY;
    if (batch_capacity >= max_rows) {
        admit_device_bytes(device_bytes, ffn_batch_prepare_bytes(batch_ffn, modes), "prepare_batch");
        device_bytes += ffn_batch_prepare(batch_ffn, modes);
        return;
    }
    if (batch_capacity) throw std::runtime_error("prepare the maximum batch capacity once before inference");
    if (host_layers.size() != NLAYER) throw std::runtime_error("load model before preparing batch kernels");
    admit_device_bytes(device_bytes,
        ffn_batch_workspace_bytes(max_rows) + ffn_batch_weight_bytes(modes), "prepare_batch");
    batch_ffn = create_ffn_batch(host_layers.data(), signs17408, max_rows, modes);
    device_bytes += ffn_batch_bytes(batch_ffn);
    batch_x = (float*) dmalloc((size_t) max_rows * D * sizeof(float));
    batch_sync = (unsigned*) dmalloc((size_t) (2 * NLAYER + 2) * ((max_rows + RMAX - 1) / RMAX) * 32 * sizeof(unsigned));
    admit_device_bytes(device_bytes, head_batch_required_bytes(max_rows), "prepare_batch head");
    batch_head = create_head_batch(max_rows);
    device_bytes += head_batch_bytes(batch_head);
    admit_device_bytes(device_bytes, attn_batch_required_bytes(max_rows), "prepare_batch attn");
    batch_attn = create_attn_batch(max_rows);
    device_bytes += attn_batch_bytes(batch_attn);
    batch_capacity = max_rows;
    HIP_CHECK_H(hipStreamSynchronize(stream));
}

void Engine::forward_batch(std::vector<Seq*> & seqs, const std::vector<std::vector<int>> & toks,
                          bool with_logits, std::vector<int> * argmax, std::vector<float> * all_logits,
                          LogitRows logit_rows) {
    if (seqs.size() != toks.size() || seqs.empty()) throw std::runtime_error("invalid batch sequence count");
    if ((argmax || all_logits) && !with_logits) throw std::runtime_error("logit outputs requested without logits");
    fwd.k_tiled = attn_coord();   // the K cache coordinate is a process setting; a pass reads it once
    // A caller that wants every logit row is asking for every row, whatever it said about the tail.
    if (all_logits) logit_rows = LogitRows::All;
    if (head_rows_all) logit_rows = LogitRows::All;
    if (batch_mode < 0 || batch_mode > 20) throw std::runtime_error("unknown batch mode");
    if (mtp.loaded) throw std::runtime_error("wide batch inference does not capture the MTP head's final hidden");
    std::set<int> slots;
    std::vector<Seq> final;
    int total = 0;
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (!seqs[i] || seqs[i]->slot < 0 || seqs[i]->slot >= nslots || !slots.insert(seqs[i]->slot).second)
            throw std::runtime_error("invalid or duplicate sequence state slot");
        if (toks[i].empty() || seqs[i]->len + toks[i].size() > (size_t) context)
            throw std::runtime_error("empty sequence or context full");
        for (int t : toks[i]) if (t < 0 || t >= VOCAB) throw std::runtime_error("token outside vocabulary");
        total += toks[i].size();
        final.push_back(*seqs[i]);
    }
    if (total > PASSMAX) throw std::runtime_error("forward_batch accepts at most PASSMAX rows");
    const bool sequence_mode = batch_mode >= 16;
    // The one row count that decides the schedule. `batch_route.h` holds what turns on with it.
    const int wide_min = batch_wide_min();
    const bool wide_sequence = batch_mode >= 17 && total >= wide_min;
    const bool automatic = batch_mode == 5 || (batch_mode >= 9 && batch_mode <= 11) || sequence_mode;
    // Mode 20 takes the wide schedule at the deployed FFN's arithmetic: wide sequence projections,
    // resident GDN, wide prep and the wide head, with the FFN still the engine's own eight-row
    // kernel. It needs no repacked weight image, and it is what separates the wide schedule's gain
    // from the A8/A4 FFN map's.
    const int wide_mode = sequence_mode ? (batch_mode == 19 ? 8 : batch_mode == 20 ? 4 : 7)
                                        : batch_mode == 5 ? 1 : batch_mode - 3;
    const int mode = batch_mode >= 12 && batch_mode <= 15 ? 0
                   : automatic ? (total < wide_min ? (total <= 4 ? 0 : 4) : wide_mode) : batch_mode;
    struct RestoreOptions {
        FwdParams &p; int single_map, commit_state;
        ~RestoreOptions() { p.single_map=single_map; p.commit_state=commit_state; }
    } restore_options{fwd,fwd.single_map,fwd.commit_state};
    fwd.single_map = batch_mode >= 12 && batch_mode <= 15 ? batch_mode - 11 : 0;
    fwd.commit_state = batch_mode == 16 || batch_mode == 18 || batch_mode == 19 || batch_mode == 20;
    if (wide_sequence && !batch_sequence) throw std::runtime_error("prepare_sequence before wide sequence inference");
    if (mode && total > batch_capacity) throw std::runtime_error("prepare_batch before selecting an optimized mode");
    if (argmax) argmax->resize(total);
    if (all_logits) all_logits->resize((size_t) total * VOCAB);
    if (g_route_trace)
        fprintf(stderr, "route: batch_mode %d rows %d seqs %d wide_min %d -> mode %d %s%s\n",
                batch_mode, total, (int) seqs.size(), wide_min, mode,
                wide_sequence ? "wide-sequence" : "sliced-sequence", with_logits ? " +logits" : "");

    std::vector<ResidentSeq> resident_seqs;
    if(wide_sequence&&fwd.commit_state&&sequence_resident_enabled(batch_sequence)) {
        int offset=0;
        for(size_t i=0;i<seqs.size();++i) {
            const Seq &s=*seqs[i];int count=toks[i].size();
            resident_seqs.push_back({offset,count,s.keep,s.parity,s.slot});offset+=count;
        }
    }

    struct Slice {
        FwdParams p;
        int offset;
        std::vector<Seq*> seq;
        std::vector<std::vector<int>> tokens;
    };
    std::vector<Slice> slices;
    // The same rows and row groups the slices carry, addressed across the whole pass, for any
    // phase that runs once for the batch instead of once per slice.
    std::vector<RowInfo> wide_rows;
    std::vector<SeqCtl> wide_groups;
    size_t si = 0, ti = 0;
    for (int offset = 0; offset < total;) {
        Slice slice{fwd, offset, {}, {}};
        auto & p = slice.p;
        p.nrows = p.nseq = 0;
        p.prof = nullptr; p.debug_layers = p.debug_stop = 0;
        // DFlash2 reads the residual stream of five layers for every row it ingests. The wide
        // schedule drives those layers from the host and copies the batch's own residual buffer
        // (below); other sliced modes still capture inside the prep phase, with each slice owning
        // its own rows of `hcap`. Mode 4 also launches layer by layer, so it copies below.
        p.hcap = df.loaded && !wide_sequence && mode != 4 ? hcap + (size_t) offset * NCAP * D : nullptr;
        p.with_logits = with_logits;
        while (si < seqs.size() && p.nrows < RMAX) {
            Seq & s = final[si];
            const int k = std::min((int) toks[si].size() - (int) ti, RMAX - p.nrows);
            p.seqs[p.nseq++] = {p.nrows, k, s.keep, s.parity};
            wide_groups.push_back({offset + p.nrows, k, 0, 0});
            for (int j = 0; j < k; ++j) wide_rows.push_back({toks[si][ti + j], s.slot, s.len + j, 0});
            slice.seq.push_back(seqs[si]);
            slice.tokens.emplace_back(toks[si].begin() + ti, toks[si].begin() + ti + k);
            for (int j = 0; j < k; ++j) p.rows[p.nrows + j] = {toks[si][ti + j], s.slot, s.len + j, 0};
            p.nrows += k;
            s.len += k; s.last_n = k; s.keep = fwd.commit_state ? 0 : k; s.parity ^= 1;
            s.rollbackable = !fwd.commit_state;
            ti += k;
            if (ti == toks[si].size()) { ++si; ti = 0; }
        }
        offset += p.nrows;
        slices.push_back(std::move(slice));
    }
    // Captured features are indexed by the batch row, and only the two schedules above place them
    // there: the wide one by copying the shared residual buffer, the sliced one by giving each
    // slice its own rows. A multi-slice `mode == 0` batch would have every slice write rows 0..7 of
    // `hcap` and the drafter would ingest the last slice five times over.
    if (df.loaded && (total > CAPMAX || (mode == 0 && slices.size() > 1)))
        throw std::runtime_error("drafted batch inference needs the sliced or wide schedule to place captured features");

    auto copy_outputs = [&](const Slice & s) {
        if (argmax) HIP_CHECK_H(hipMemcpyAsync(argmax->data() + s.offset, argmax_out,
            s.p.nrows * sizeof(int), hipMemcpyDeviceToHost, stream));
        if (all_logits) HIP_CHECK_H(hipMemcpyAsync(all_logits->data() + (size_t) s.offset * VOCAB,
            logits, (size_t) s.p.nrows * VOCAB * sizeof(float), hipMemcpyDeviceToHost, stream));
    };
    const bool wide_prep = wide_sequence && sequence_direct_layout(batch_sequence) && sequence_prep_wide_enabled();
    // A sliced mode-4 pass launches one layer at a time. Its kernel-local capture index resets
    // at each launch, so capture all five layer boundaries from the shared residual buffer here.
    const bool capture_wide = df.loaded && (wide_sequence || mode == 4);
    // The wide attention needs the direct operand layout (it writes the whole batch's quantised
    // attention output) and the pass's own row metadata.
    int attn_stride = 0;
    const bool wide_attn = wide_sequence && attn_batch_enabled(batch_attn) && total > RMAX
                        && total <= attn_batch_capacity(batch_attn)
                        && sequence_direct_layout(batch_sequence);
    if (wide_attn) {
        if (attn_batch_group_rows(batch_attn) >= 16) {
            // One group is still consecutive rows of one sequence; merging two of them only moves
            // the boundary a launch reads its key chunk at.
            std::vector<SeqCtl> merged;
            for (const SeqCtl & g : wide_groups) {
                if (!merged.empty()) {
                    SeqCtl & m = merged.back();
                    const RowInfo & tail = wide_rows[m.row0 + m.nrows - 1];
                    const RowInfo & head = wide_rows[g.row0];
                    if (m.nrows + g.nrows <= 16 && tail.seq == head.seq && tail.pos + 1 == head.pos
                        && m.row0 + m.nrows == g.row0) { m.nrows += g.nrows; continue; }
                }
                merged.push_back(g);
            }
            wide_groups.swap(merged);
        }
        for (const SeqCtl & g : wide_groups)
            attn_stride = std::max(attn_stride, wide_rows[g.row0 + g.nrows - 1].pos / ACHUNK + 1);
        attn_batch_begin(batch_attn, wide_rows.data(), total, wide_groups.data(), (int) wide_groups.size(), stream);
    }
    auto * trace = batch_profile ? batch_profiler.get() : nullptr;
    if (trace) trace->reset(stream);
    if (mode == 0) {
        for (auto & s : slices) {
            if (trace) trace->begin("full-slice", -2, s.offset, s.p.nrows, 0, stream);
            forward(s.seq, s.tokens, with_logits, nullptr);
            if (trace) trace->end(stream);
            copy_outputs(s);
        }
    } else {
        // Every cooperative launch gets distinct zeroed counters. A kernel boundary
        // orders attention slices; all tokens meet at the layer's batched FFN.
        const size_t sync_words = (size_t) (2 * NLAYER + 2) * slices.size() * 32;
        HIP_CHECK_H(hipMemsetAsync(batch_sync, 0, sync_words * sizeof(unsigned), stream));
        size_t launch = 0;
        auto params = [&](const Slice & s) {
            FwdParams p = s.p;
            p.x = batch_x + (size_t) s.offset * D;
            p.bar = batch_sync + launch++ * 32;
            p.work = p.bar + 1;
            return p;
        };
        const int ffn_slice_rows = g_ffn_slice_rows;
        auto ffn_params = [&](int off, int k) {
            FwdParams p = slices.front().p;
            p.nrows = k; p.nseq = 1; p.seqs[0] = { 0, k, 0, 0 };
            p.x = batch_x + (size_t) off * D;
            p.bar = batch_sync + launch++ * 32;
            p.work = p.bar + 1;
            return p;
        };
        auto sliced = [&](const Slice & s, int stage) {
            auto p = params(s);
            if (trace) p.prof = trace->begin(stage < 0 ? "embed" : stage == NLAYER ? "head" : "sequence",
                stage, s.offset, p.nrows, stage < 0 ? 2 : stage == NLAYER ? 3 : 6, stream);
            launch_forward_slice(p, stage, stream);
            if (trace) trace->end(stream);
        };
        for (const auto & s : slices) sliced(s, -1);
        for (int layer = 0; layer < NLAYER; ++layer) {
            // DFlash2 reads the residual stream entering layers 6, 20, 34, 48 and 62. Inside the
            // persistent kernel that capture is a float4 store in the prep phase; here the host
            // owns the layer boundary and the whole batch shares one residual buffer, so it is a
            // strided copy of `batch_x` into the row's five feature slots. 13.1 MB of copy per
            // 128-row pass, against a pass that moves several gigabytes of weights.
            if (capture_wide) for (int j = 0; j < NCAP; ++j) if (layer == CAP_LAYERS[j] + 1) {
                if (trace) trace->begin("drafter-capture", layer, 0, total, 0, stream);
                HIP_CHECK_H(hipMemcpy2DAsync(hcap + (size_t) j * D, (size_t) NCAP * D * sizeof(float),
                    batch_x, (size_t) D * sizeof(float), (size_t) D * sizeof(float), (size_t) total,
                    hipMemcpyDeviceToDevice, stream));
                if (trace) trace->end(stream);
            }
            if (wide_sequence) {
                // The input prep is row-parallel, so the whole batch takes one pair of launches
                // instead of one persistent-kernel launch per eight-row slice.
                if (wide_prep) {
                    FwdParams p = fwd;
                    p.nrows = total; p.x = batch_x;
                    sequence_bind(batch_sequence, p, layer, 0, total, 1);
                    const LayerW & L = host_layers[layer];
                    SeqPrepArgs a{ batch_x, L.attn_norm, L.attn_norm_s, L.recurrent ? L.alpha_beta : nullptr,
                                   p.sequence_q, p.sequence_scales, p.ab, p.ninv, total, p.sequence_npad };
                    if (trace) trace->begin("sequence-input-prep", layer, 0, total, 0, stream);
                    launch_sequence_prep(a, stream);
                    if (trace) trace->end(stream);
                } else for (const auto & s : slices) {
                    auto p = params(s);
                    if (trace) p.prof = trace->begin("sequence-input-prep", layer, s.offset, p.nrows, 1, stream);
                    bool direct=sequence_bind(batch_sequence,p,layer,s.offset,total,1);
                    launch_sequence_part(p, layer, 1, stream);
                    if(!direct)sequence_capture(batch_sequence, p, layer, s.offset, total, 1, stream);
                    if (trace) trace->end(stream);
                }
                if (trace) trace->begin("sequence-input-projection", layer, 0, total, 0, stream);
                sequence_project(batch_sequence, layer, total, 1, batch_x, stream);
                if (trace) trace->end(stream);
                // Which route writes this layer's output operand. Only the two wide writers carry
                // the four-bit output coordinate; see sequence_set_output_wide.
                sequence_set_output_wide(batch_sequence,
                    (!resident_seqs.empty() && sequence_resident_layer(batch_sequence, layer))
                    || (wide_attn && !host_layers[layer].recurrent));
                if(!resident_seqs.empty()&&sequence_resident_layer(batch_sequence,layer)) {
                    if(trace)trace->begin("gdn-resident-core",layer,0,total,0,stream);
                    sequence_run_resident(batch_sequence,fwd,layer,resident_seqs.data(),resident_seqs.size(),total,stream);
                    if(trace)trace->end(stream);
                } else if (wide_attn && !host_layers[layer].recurrent) {
                    // Rope and the cache write once for the pass, then scores and combine over row
                    // groups: the same groups the slices would have run, from one grid.
                    const LayerW & L = host_layers[layer];
                    FwdParams pa = fwd;
                    pa.nrows = total; pa.qrot = attn_batch_qrot(batch_attn);
                    sequence_bind(batch_sequence, pa, layer, 0, total, 2);
                    // The three launches are one phase by default so every panel this lane has
                    // taken keeps reading the same way. Split, they separate the two halves of the
                    // chunk-partial buffer: the score units write it, the combine reads it, and
                    // both grow with position for different reasons.
                    const bool asplit = trace && attn_trace_split();
                    if (trace && !asplit) trace->begin("sequence-core", layer, 0, total, 0, stream);
                    if (asplit) trace->begin("sequence-core-pre", layer, 0, total, 0, stream);
                    launch_attn_pre_wide(batch_attn, pa, L.q_norm, L.k_norm, L.kv_slot, total, stream);
                    if (asplit) trace->end(stream);
                    const int gpl = attn_batch_groups_per_launch(batch_attn, attn_stride);
                    for (int g0 = 0; g0 < (int) wide_groups.size(); g0 += gpl) {
                        const int ng = std::min(gpl, (int) wide_groups.size() - g0);
                        const SeqCtl & first = wide_groups[g0], & last = wide_groups[g0 + ng - 1];
                        const int rowbase = first.row0, nrows_b = last.row0 + last.nrows - rowbase;
                        int mc = 1;
                        bool one_row = true;
                        for (int g = g0; g < g0 + ng; ++g) {
                            mc = std::max(mc, wide_rows[wide_groups[g].row0 + wide_groups[g].nrows - 1].pos / ACHUNK + 1);
                            if (wide_groups[g].nrows != 1) one_row = false;
                        }
                        if (asplit) trace->begin("sequence-core-score", layer, rowbase, nrows_b, 0, stream);
                        launch_attn_wide(batch_attn, pa, L.kv_slot, g0, ng, rowbase, mc, attn_stride, one_row, stream);
                        if (asplit) trace->end(stream);
                        FwdParams pc = pa;
                        pc.nrows = nrows_b;
                        sequence_bind(batch_sequence, pc, layer, rowbase, total, 2);
                        if (asplit) trace->begin("sequence-core-combine", layer, rowbase, nrows_b, 0, stream);
                        launch_attn_combine_wide(batch_attn, pc, rowbase, nrows_b, attn_stride, stream);
                        if (asplit) trace->end(stream);
                    }
                    if (trace && !asplit) trace->end(stream);
                } else for (const auto & s : slices) {
                    auto p = params(s);
                    if (trace) p.prof = trace->begin("sequence-core", layer, s.offset, p.nrows, 3, stream);
                    bool direct=sequence_bind(batch_sequence,p,layer,s.offset,total,2);
                    if(!direct)sequence_restore(batch_sequence, p, layer, s.offset, stream);
                    launch_sequence_part(p, layer, 2, stream);
                    if(!direct)sequence_capture(batch_sequence, p, layer, s.offset, total, 2, stream);
                    if (trace) trace->end(stream);
                }
                if (trace) trace->begin("sequence-output-projection", layer, 0, total, 0, stream);
                sequence_project(batch_sequence, layer, total, 2, batch_x, stream);
                if (trace) trace->end(stream);
            } else for (const auto & s : slices) sliced(s, layer);
            if (mode == 4) {
                // The deployed FFN body reads no row metadata, no block cache and no recurrent state,
                // so its slice is not bound by RMAX. Its matrix instruction carries sixteen columns
                // whatever the row count, so eight-row slices visited the whole weight stream twice as
                // often as they had to. Same weights, same K order, same accumulation: same bits.
                for (int off = 0; off < total; off += ffn_slice_rows) {
                    const int k = std::min(ffn_slice_rows, total - off);
                    if (trace) trace->begin("ffn", layer, off, k, 0, stream);
                    launch_ffn_slice(ffn_params(off, k), layer, stream);
                    if (trace) trace->end(stream);
                }
            } else {
                if (trace) trace->begin("ffn", layer, 0, total, 0, stream);
                if (!run_ffn_batch(batch_ffn, layer, mode, total, batch_x, batch_x, stream))
                    throw std::runtime_error("FFN batch module rejected the inference shape or mode");
                if (trace) trace->end(stream);
            }
        }
        // One pass over the 278 MB output image for the whole batch, instead of one per slice.
        const bool wide_head = with_logits && batch_head && total > RMAX && total <= head_batch_capacity(batch_head);
        // The rows the head actually has to produce. A `Tail` pass is prompt ingestion: one row is
        // read and the rest were VOCAB floats nothing loaded. The selection is contiguous because
        // both shapes this engine serves are - one tail row for a prompt pass, every row for a step
        // whose rows are one token each of their own sequence.
        const int head_lo = wide_head && logit_rows == LogitRows::Tail ? total - 1 : 0;
        if (wide_head) {
            for (const auto & s : slices) {
                auto p = params(s);
                head_bind(batch_head, p, s.offset, total);
                // SEQ_PART_HEAD runs one phase and returns without a grid barrier, so nothing
                // writes a closing stamp and a phase count above zero would read a zero. The
                // event bracket already times the launch honestly, and leaving P.prof null keeps
                // the traced and untraced arms running the same code. Found by the
                // bonsai-pass-schedule engineer, whose traced profile hit the missing stamp.
                if (trace) trace->begin("head-prep", NLAYER, s.offset, p.nrows, 0, stream);
                launch_sequence_part(p, NLAYER, SEQ_PART_HEAD, stream);
                if (trace) trace->end(stream);
            }
            if (trace) trace->begin("head-projection", NLAYER, head_lo, total - head_lo, 0, stream);
            run_head_batch(batch_head, fwd, total, head_lo, stream);
            if (trace) trace->end(stream);
            // Rows the head did not produce read back as -1 rather than as whatever the buffer held
            // from an earlier pass, so a caller that reaches past its selection sees that it did.
            if (argmax) {
                if (head_lo) std::fill(argmax->begin(), argmax->end(), -1);
                HIP_CHECK_H(hipMemcpyAsync(argmax->data() + head_lo, head_argmax(batch_head) + head_lo,
                    (size_t) (total - head_lo) * sizeof(int), hipMemcpyDeviceToHost, stream));
            }
            if (all_logits) HIP_CHECK_H(hipMemcpyAsync(all_logits->data(), head_logits(batch_head),
                (size_t) total * VOCAB * sizeof(float), hipMemcpyDeviceToHost, stream));
        } else if (with_logits) for (const auto & s : slices) {
            sliced(s, NLAYER);
            copy_outputs(s);
        }
        for (size_t i = 0; i < seqs.size(); ++i) *seqs[i] = final[i];
    }
    if (argmax || all_logits) HIP_CHECK_H(hipStreamSynchronize(stream));
    if (trace) trace->collect(*this);
}

} // namespace halo
