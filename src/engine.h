#pragma once
#include "gguf.h"
#include "repack.h"
#include "q8.h"
#include "../kernels/halo_kernels.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <random>
#include <stdexcept>
#include <memory>
#include <map>
#define HIP_CHECK_H(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); } while (0)

namespace halo {

struct LayerDev {
    bool recurrent;
    const uint8_t * qkv = nullptr, * z = nullptr, * ssm_out = nullptr;      // GDN
    const uint8_t * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr; // attention
    const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;
    const float * attn_norm = nullptr, * post_norm = nullptr;
    const float * attn_norm_s = nullptr, * post_norm_s = nullptr, * ssm_norm_s = nullptr; // folded with the sign vector
    const float * q_norm = nullptr, * k_norm = nullptr;
    const float * conv_w = nullptr, * ssm_a = nullptr, * ssm_dt = nullptr, * ssm_norm = nullptr;
    const unsigned short * alpha_beta = nullptr; // [96][5120] bf16: 48 alpha rows then 48 beta rows
    int kv_slot = -1;
};

struct FfnBatch;
struct SequenceBatch;
struct HeadBatch;
struct AttnBatch;
struct BatchProfiler;
struct BatchTrace {
    std::string kind;
    int layer, offset, rows;
    double start_ms = 0, end_ms = 0;
    std::vector<double> phase_us;
};

struct Engine {
    Gguf g;
    HaloCache cache;
    std::vector<LayerDev> layers;
    const uint8_t * tok_embd = nullptr, * output = nullptr;
    const float * output_norm = nullptr, * output_norm_s = nullptr;
    std::vector<float> h_signs5120, h_signs6144;
    const float * signs5120 = nullptr, * signs6144 = nullptr, * signs17408 = nullptr;

    // activations
    float * x = nullptr, * xn = nullptr, * tmp = nullptr;
    int8_t * xq = nullptr; float * xs = nullptr; int * xsum = nullptr;
    float * big = nullptr, * zbuf = nullptr, * conv_out = nullptr, * ab = nullptr, * y = nullptr, * gu = nullptr;
    float * qfull = nullptr, * kbuf = nullptr, * vbuf = nullptr, * qrot = nullptr;
    AttnPartial * partials = nullptr;
    float * logits = nullptr;
    int * tok_out = nullptr, * pos = nullptr;
    int * h_ring = nullptr, * d_ring = nullptr; // pinned host ring and its device mapping
    int pending = 0; // steps queued since the last synchronisation
    // state (v1 path uses gdn_state slot 0 and conv_state)
    float * gdn_state = nullptr, * conv_state = nullptr, * conv_ring = nullptr, * obuf = nullptr;
    // persistent multi-row forward kernel
    bool fused = true;
    int nslots = 1;                 // sequence slots allocated
    int context = MAXCTX;
    // 0 deployed; 1/2/3 A8/A4/scaled A8; 4 sliced deployed; 5 auto A8.
    // 6/7/8 optimized A8/scaled A8/A4; 9/10/11 their automatic small-row routes.
    // 12 single-row dense consumer; 13 also splits GDN units; 14 also retiles K reductions.
    // 15 runs the original single-row kernel on its own occupancy grid.
    int batch_mode = 0;
    int batch_capacity = 0;
    bool batch_profile = false;
    std::shared_ptr<BatchProfiler> batch_profiler;
    std::vector<BatchTrace> batch_trace;
    double batch_trace_span_ms = 0;
    void set_batch_profile(bool enabled);
    SequenceBatch * batch_sequence = nullptr;
    void prepare_sequence();
    // Wide vocabulary projection over every row of a batch; null keeps the eight-row slices.
    HeadBatch * batch_head = nullptr;
    AttnBatch * batch_attn = nullptr;      // wide attention launches; null keeps the per-slice phase
    FfnBatch * batch_ffn = nullptr;
    float * batch_x = nullptr;
    unsigned * batch_sync = nullptr;
    std::vector<LayerW> host_layers;
    unsigned * bar = nullptr, * work = nullptr; float * amax_val = nullptr; int * amax_idx = nullptr;
    float * blk_cache = nullptr, * hcap = nullptr, * hfinal = nullptr, * ninv = nullptr;
    int * argmax_out = nullptr;
    FwdParams fwd{};
    // Persistent-kernel workgroup count, -1 = the occupancy grid. Setting it per call is how one
    // process interleaves two grids under one clock; the arms differ only in how units are dealt
    // out to workgroups, so their outputs must agree bit for bit.
    int grid_rows_forced = -1;
    // Releasing the pin has to clear the parameter as well as the override, because the launcher
    // only ever writes `fwd.grid_rows` and never clears it: `set_grid_rows(-1)` used to leave the
    // previous pin in the params struct, so in an arm list the FIRST pinned value won every later
    // cell. `--grid-rows 80,-1` measured grid 80 twice and reported one of them as the occupancy
    // grid, which is a silent wrong answer rather than a missing one.
    void set_grid_rows(int g) { grid_rows_forced = g; if (g < 0) fwd.grid_rows = 0; }
    // Register budget for the persistent kernel, -1 = whatever HALO_PK_OCC said. 0 lets the
    // allocator choose (the budget every published number before docs/rows-grid-occupancy.md was measured
    // on), 1 asks for twelve waves per SIMD32. Both are in the executable, so an A/B runs under one
    // clock, and the arms must agree bit for bit.
    int pk_occ_forced = -2;   // -2 = untouched, -1 = the measured rule, 0/1 = pinned
    void set_pk_occ(int o) { pk_occ_forced = o; fwd.pk_occ = o; }
    // Which attention score instantiation a row group gets: 1 the narrowest that covers it, 0 the
    // wide-only dispatch it replaces. Both arms compute the same bits, so this is a timing axis a
    // process can interleave; every params struct is copied from `fwd`, including the drafter's.
    void set_attn_narrow(int on) { fwd.attn_narrow = on ? 1 : 0; }
    int attn_narrow() const { return fwd.attn_narrow; }
    // Whether a sliding window decides the attention unit list as well as the mask. Only the
    // DFlash2 drafter has a window, so this reaches nothing else; 1 skips the chunks the window
    // excludes and 0 deals them and masks them to zero, which is the schedule that shipped before
    // docs/drafter-window-units.md. Both arms are bit-identical, so this is a timing axis one
    // process interleaves, and `fwd` is copied into the drafter's params on every step.
    void set_attn_deal(int on) { fwd.attn_window_deal = on ? 1 : 0; }
    int attn_deal() const { return fwd.attn_window_deal; }
    // Query heads per score unit on a one-row-group launch. Same argument: both arms compute the
    // same bits, so it is a timing axis a process can interleave.
    void set_attn_hg(int h) { fwd.attn_hg = h > 1 ? h : 1; }
    int attn_hg() const { return fwd.attn_hg; }
    unsigned long long * prof = nullptr;
    void print_profile(); // per-phase timing of the last fused step (requires prof enabled)
    // The same device stamps, read out of `k_dflash` instead of `k_forward_rows`. The drafter is a
    // cooperative kernel with about a hundred grid syncs in it and nothing had ever looked inside
    // one, so the only published account of a drafted step is two numbers, draft and verify.
    // HALO_PROFILE=1 HALO_PROFILE_DRAFT=1 accumulates the segment timeline over every drafted step
    // of a run; it costs one 8 kB copy per step, on a path that already synchronises to read the
    // drafts back, and nothing when it is off.
    std::map<std::string, double> df_prof_us;   // microseconds summed per segment name
    std::map<std::string, long> df_prof_cnt;
    long df_prof_steps = 0;
    double df_prof_total_us = 0;
    void print_draft_profile();
    void reset_draft_profile() { df_prof_us.clear(); df_prof_cnt.clear(); df_prof_steps = 0; df_prof_total_us = 0; }
    __half * kcache = nullptr, * vcache = nullptr;

    // A sequence occupying one state slot. `len` is the position of the next row; `keep` is how many
    // rows of the last pass the next pass commits (replays) into the GDN state.
    struct Seq { int slot = 0; int len = 0; int keep = 0; int parity = 0; int last_n = 0; bool rollbackable = true; };
    // One pass over rows of several sequences. rows[i] positions are assigned from each seq's len.
    // toks[si] are that sequence's tokens for this pass. Fills argmax (per row) when logits.
    void forward(std::vector<Seq *> & seqs, const std::vector<std::vector<int>> & toks, bool logits, std::vector<int> * argmax);
    static constexpr unsigned BATCH_WORKSPACE_ONLY = 1u << 30;
    // modes is a bitmask of module modes; zero prepares every mode. Preparation is outside inference.
    void prepare_batch(int max_rows = 128, unsigned modes = 0);
    // Which rows of a pass the vocabulary head has to produce. Prompt ingestion reads exactly one
    // logit row, the last, and the head is the only phase whose entire cost is output width: at 256
    // rows it spends 21 ms and 127 MB of buffer so that `Engine::prefill` can take `am.back()`.
    // `Tail` asks for that row alone. `All` is what generation needs, because there every row is a
    // different sequence's next token, and what every logit identity panel needs.
    enum class LogitRows { All, Tail };
    void forward_batch(std::vector<Seq *> & seqs, const std::vector<std::vector<int>> & toks,
        bool with_logits, std::vector<int> * argmax = nullptr, std::vector<float> * all_logits = nullptr,
        LogitRows rows = LogitRows::All);
    // Services every `Tail` request as `All`, so one process can time both arms of the row
    // selection against each other. `tools/batch_profile --head-rows` and
    // `tools/batch_compare --head-rows` set it.
    bool head_rows_all = false;
    // Batch mode for prompt ingestion in the ordinary chat, bench and serving paths. Zero keeps the
    // eight-row persistent passes those paths have always used. When it is set and the batch
    // modules are built, `prefill` hands whole blocks of prompt tokens to `forward_batch` instead,
    // which is the only place the wide schedule ever reached before. Generation is untouched: the
    // route ends at the last prompt token.
    int prefill_batch_mode = 0;
    // Batch mode for a generation step that carries several sequences at once. Zero keeps the
    // eight-row persistent pass, which is what a concurrent server ran until this was added: a
    // step of N rows became `ceil(N/8)` complete forwards, each re-reading the whole weight set,
    // so served throughput was flat from eight clients up. `ServeBatch` defaults it to
    // `prefill_batch_mode`, because a served process already builds those modules for ingestion
    // and mode 20 is the wide schedule at the deployed FFN's arithmetic. It changes no numerical
    // map: a row carries its own token, slot and position, and which rows share a pass is a
    // schedule. A single-request step is untouched - it never reaches this.
    int decode_batch_mode = 0;
    // RMAX when nothing arms the wide route, otherwise the rows one prompt pass carries.
    int prefill_width() const;
    // Keep only the first `keep` rows of a sequence's last pass (speculative rejection).
    static void rollback(Seq & s, int keep) {
        if (!s.rollbackable && keep != s.last_n) throw std::runtime_error("cannot reject tokens from a committed pass");
        if (keep < 0 || keep > s.last_n) throw std::runtime_error("invalid accepted prefix");
        s.len -= s.last_n - keep; s.keep = s.rollbackable ? keep : 0; s.last_n = keep;
    }
    void reset_seq(Seq & s);
    Seq seq0;

    // ---- MTP drafter ----
    struct Mtp { Safetensors st; Q8Cache cache; MtpW w{}; float * h_out = nullptr; bool loaded = false; } mtp;
    void load_mtp(const std::string & safetensors_path, int nthreads);
    // one MTP pass: row i = (hidden h_in[i*D], token toks[i]) at MTP position poss[i]; returns argmax per row
    void mtp_forward(Seq & s, const float * h_in, const std::vector<int> & toks, const std::vector<int> & poss, std::vector<int> & argmax);
    struct SpecStats { long drafted = 0, accepted = 0, steps = 0; double t_draft = 0, t_verify = 0; long prompt_tokens = 0, cached_tokens = 0; double t_prefill = 0; };
    template <typename F> void generate_mtp(const std::vector<int> & prompt, int n_max, int n_draft, F on_token, SpecStats & st);

    // ---- DFlash2 drafter ----
    // The drafter's weight coordinate is a served choice, not a numerical one: the target verifies
    // every drafted token, so only the accepted count can move. Both images can be resident at once
    // so an A/B is one process; `use_q4` picks the arm for the next step.
    struct Dflash {
        Safetensors st; Q8Cache cache, cache4; DflashW w{}, w4{}; bool loaded = false, q4_loaded = false, use_q4 = false;
        float * th = nullptr, * xn = nullptr, * dyn = nullptr, * cbuf = nullptr, * abuf = nullptr, * tmpo = nullptr, * hproj_out = nullptr;
        float * cand_v = nullptr, * thr = nullptr, * topv = nullptr; int * cand_i = nullptr, * cand_n = nullptr, * topi = nullptr, * draft_out = nullptr;
    } df;
    // want: bit 0 = Q8 image, bit 1 = Q4 image. Both may be loaded; `df.use_q4` selects.
    // Q4 is the default: docs/drafter-q4.md measures it 10.4% faster on drafted generation with the
    // same output tokens and no acceptance cost over ten workloads.
    enum { DRAFT_Q8 = 1, DRAFT_Q4 = 2 };
    void load_dflash(const std::string & safetensors_path, int nthreads, unsigned want = DRAFT_Q4);
    // The names of `k_dflash`'s segments, in the order its grid syncs close them. `n_ctx > 0` adds
    // the ingest prefix; the block draft is always DF_BLOCK rows. It mirrors the kernel body, so a
    // phase added there needs a name added here - the reader asserts the count it read.
    static std::vector<std::string> dflash_segment_names(int n_ctx);
    // Ingest the captured features of the last target pass's rows [cap_row0, cap_row0 + n_ctx)
    // (positions ctx_pos0 ..) into the drafter cache, then (if drafts) draft a block anchored at
    // `anchor` at position `start`. n_ctx is at most RMAX, so a wide target pass feeds this in
    // chunks; `cap_row0` is the chunk's first row of `hcap`.
    void dflash_step(Seq & s, int n_ctx, int ctx_pos0, int anchor, int start, std::vector<int> * drafts, int cap_row0 = 0);
    // Ingest row i of the last captured target pass into seqs[i]'s drafter cache at position
    // seqs[i]->len - 1, without drafting. A served batched step has one row per sequence, so this
    // keeps every sequence's drafter context current while it shares passes (at most RMAX rows).
    void dflash_ingest(const std::vector<Seq *> & seqs);
    template <typename F> void generate_dflash(const std::vector<int> & prompt, int n_max, F on_token, SpecStats & st);

    // ---- generic generation with a prompt-prefix cache (server use) ----
    struct GenParams { int max_tokens = 1024; float temp = 0.0f; float top_p = 0.95f; int top_k = 20; unsigned seed = 42; bool speculative = true; std::function<bool()> aborted;
                       std::vector<size_t> snapshot_points; }; // prompt prefixes worth keeping (e.g. the system message), besides the prompt itself
    // Runs the prompt (reusing the cached prefix when the prompt extends it), then decodes until
    // on_token returns false, max_tokens or an abort. Greedy decode uses DFlash2 when loaded and
    // `speculative`; sampled decode is plain. The prefix snapshot lives in an extra state area.
    void generate(const std::vector<int> & prompt, const GenParams & params, const std::function<bool(int)> & on_token, SpecStats & st);
    // Prompt-prefix snapshots: GDN state, conv ring and pending block cache of a slot at a pass
    // boundary, in extra state areas past the sequence slots, plus the K/V of every cache slot
    // (target attention, MTP head and drafter layers) for the snapshot's positions in a buffer
    // sized to the snapshot. Without the K/V copy a snapshot silently reads whatever positions the
    // last unrelated request left in the slot's cache and the model answers nonsense.
    static constexpr int SNAP_AREAS = 4;
    struct Snapshot { std::vector<int> tokens; Seq seq; bool valid = false; long last_use = 0;
                      __half * k = nullptr; __half * v = nullptr; size_t kv_tokens = 0; };
    Snapshot snaps[SNAP_AREAS];
    long snap_clock = 0;
    size_t snap_kv_budget = (size_t) 6 << 30; // HALO_SNAPSHOT_KV_GB overrides; K/V per snapshot token is ~90 KB
    size_t snap_kv_used = 0;
    void snapshot_save(const Seq & s, const std::vector<int> & tokens);
    void snapshot_restore(Seq & s, int area);
    int snapshot_find(const std::vector<int> & prompt); // longest strict prefix, or -1
    void prefill(Seq & s, const std::vector<int> & toks, size_t from, int & next, std::vector<int> * drafts);
    void publish_last_logits(int rows);

    hipStream_t stream = nullptr;
    hipGraphExec_t graph_prompt = nullptr, graph_gen = nullptr;
    bool use_graph = true;
    size_t device_bytes = 0;
    int n_pos = 0;
    std::string dump_dir; // when set, forward_token (non-graph) writes activations here
    void dump(const char * name, const float * dev, size_t n);

    void load(const std::string & path, int nthreads, int n_slots = 1, int context_capacity = MAXCTX);
    void reset();
    // Feed one token; returns the argmax of the logits when want_logits, else -1.
    int step(int token, bool want_logits);
    // Copy logits to host (VOCAB floats). Valid after a step with want_logits.
    void get_logits(std::vector<float> & out, int row = 0);
    std::vector<float> read(const float * dev, size_t n);

private:
    void build_forward(bool with_logits);
    void forward_token(bool with_logits);
    void * dmalloc(size_t bytes);
    template <typename T> T * upload(const void * src, size_t bytes);
    const float * upload_f32(const GgufTensor & t);
    const unsigned short * upload_bf16_pair(const GgufTensor & a, const GgufTensor & b);
    const uint8_t * upload_halo(const std::string & name, int64_t N, int64_t K);
    const float * upload_folded(const GgufTensor * norm, const std::vector<float> & signs, bool per_head_128);
};

int sample_from_logits(const std::vector<float> & logits, float temp, int top_k, float top_p, std::mt19937 & rng);

// Rows per deployed-FFN slice in a wide pass: how many times the pass re-reads the FFN weight
// stream. Widths differ only in schedule, never in arithmetic, so a measurement tool can walk them
// as a case axis inside one process. See `src/batch.cpp` and `docs/ffn-slice-width.md`.
int  batch_ffn_slice_rows();
void batch_set_ffn_slice_rows(int rows);

template <typename F>
void Engine::generate_mtp(const std::vector<int> & prompt, int n_max, int n_draft, F on_token, SpecStats & st) {
    Seq & s = seq0;
    std::vector<Seq *> sv { &s };
    std::vector<int> am, dam;
    // prefill in row blocks; after each block, run the MTP over the same rows so its KV cache
    // covers every prompt position: row i pairs h(prompt[i]) with prompt[i+1]
    const int n = (int) prompt.size();
    int next = -1;
    std::vector<float> dummy;
    for (int i = 0; i < n; i += RMAX) {
        const int k = std::min(RMAX, n - i);
        std::vector<std::vector<int>> tv { std::vector<int>(prompt.begin() + i, prompt.begin() + i + k) };
        const bool last = i + k == n;
        forward(sv, tv, last, last ? &am : nullptr);
        if (last) next = am[k - 1];
        std::vector<int> toks(k), poss(k);
        for (int j = 0; j < k; j++) { toks[j] = (i + j + 1 < n) ? prompt[i + j + 1] : next; poss[j] = i + j + 1; }
        mtp_forward(s, hfinal, toks, poss, dam);
    }
    // dam.back() is the first draft token; mtp.h_out row k-1 seeds the chain
    int produced = 0;
    std::vector<int> drafts;
    int chain_row = (n - 1) % RMAX;
    if (!on_token(next)) return;
    produced++;
    for (;;) {
        // draft
        drafts.clear();
        drafts.push_back(dam.back());
        int hrow = chain_row;
        auto tnow = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        double td0 = tnow();
        for (int d = 1; d < n_draft; d++) {
            std::vector<int> t1 { drafts.back() }, p1 { s.len + d }, o1;
            mtp_forward(s, mtp.h_out + (size_t) hrow * D, t1, p1, o1);
            drafts.push_back(o1[0]);
            hrow = 0;
        }
        // verify: rows [next, d1 .. dK]
        std::vector<int> block { next };
        block.insert(block.end(), drafts.begin(), drafts.end());
        std::vector<std::vector<int>> tv { block };
        double tv0 = tnow(); st.t_draft += tv0 - td0;
        forward(sv, tv, true, &am);
        st.t_verify += tnow() - tv0;
        int a = 0;
        while (a < (int) drafts.size() && am[a] == drafts[a]) a++;
        if (getenv("HALO_SPEC_DEBUG")) { fprintf(stderr, "step %ld: next=%d drafts=", st.steps, next); for (int d : drafts) fprintf(stderr, " %d", d); fprintf(stderr, "  argmax="); for (int v : am) fprintf(stderr, " %d", v); fprintf(stderr, "  a=%d\n", a); }
        st.steps++; st.drafted += (long) drafts.size(); st.accepted += a;
        // commit: d1..da and the bonus am[a]
        bool go = true;
        for (int i = 0; i < a && go; i++) { go = on_token(drafts[i]); produced++; if (produced >= n_max) go = false; }
        if (go) { go = on_token(am[a]); produced++; if (produced >= n_max) go = false; }
        rollback(s, a + 1);
        if (!go) break;
        next = am[a];
        // refresh the MTP over the committed rows: (h(row i), token i+1) for i in 0..a, token a+1 = bonus
        std::vector<int> toks(a + 1), poss(a + 1);
        for (int i = 0; i <= a; i++) { toks[i] = i < a ? drafts[i] : am[a]; poss[i] = s.len - (a + 1) + i + 1; }
        double tr0 = tnow();
        mtp_forward(s, hfinal, toks, poss, dam);
        st.t_draft += tnow() - tr0;
        chain_row = a;
    }
}

template <typename F>
void Engine::generate_dflash(const std::vector<int> & prompt, int n_max, F on_token, SpecStats & st) {
    Seq & s = seq0;
    std::vector<Seq *> sv { &s };
    std::vector<int> am, drafts;
    auto tnow = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    int next = -1;
    // The same prompt ingestion the server takes, rather than a second copy of it: `prefill` owns
    // the pass width, the drafter chunking and the wide route, so this path gets all three.
    prefill(s, prompt, 0, next, &drafts);
    fwd.hcap = hcap;   // every verify pass below captures the features the next ingest reads
    int produced = 0;
    if (!on_token(next)) { fwd.hcap = nullptr; return; }
    produced++;
    for (;;) {
        std::vector<int> block { next };
        block.insert(block.end(), drafts.begin(), drafts.end());
        std::vector<std::vector<int>> tv { block };
        double tv0 = tnow();
        forward(sv, tv, true, &am);
        st.t_verify += tnow() - tv0;
        int a = 0;
        while (a < (int) drafts.size() && am[a] == drafts[a]) a++;
        if (getenv("HALO_SPEC_DEBUG")) { fprintf(stderr, "step %ld: anchor=%d drafts=", st.steps, next); for (int d : drafts) fprintf(stderr, " %d", d); fprintf(stderr, "  argmax="); for (int v : am) fprintf(stderr, " %d", v); fprintf(stderr, "  a=%d\n", a); }
        st.steps++; st.drafted += (long) drafts.size(); st.accepted += a;
        bool go = true;
        for (int i = 0; i < a && go; i++) { go = on_token(drafts[i]); produced++; if (produced >= n_max) go = false; }
        if (go) { go = on_token(am[a]); produced++; if (produced >= n_max) go = false; }
        const int start_old = s.len - (int) block.size();
        rollback(s, a + 1);
        if (!go) break;
        next = am[a];
        double td0 = tnow();
        dflash_step(s, a + 1, start_old, next, s.len, &drafts);
        st.t_draft += tnow() - td0;
    }
    fwd.hcap = nullptr;
}

} // namespace halo
