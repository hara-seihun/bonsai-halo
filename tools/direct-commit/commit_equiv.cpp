// Host mirror of the two GDN prefill schedules, to settle what direct commit changes before any
// GPU time is spent on it.
//
// The kernel path being mirrored is ph_gdn_pre and ph_gdn in kernels/halo_rows.hip. The mirror
// keeps the structure that decides floating-point results and addressing: the conv ring
// recurrence, the per-head L2 normalisation with its wave reduction, the gate expressions, the
// delta rule with its per-lane column split and warp_sum association, the blk_cache parity slots,
// and the shared row buffers that several sequences of one pass index through SeqCtl::row0. It
// uses four value heads instead of forty-eight so a scenario runs in a second; the state size, the
// lane split and the reduction order are the real ones.
//
// The claim under test is not "the mirror reproduces the GPU". It is that two schedules of the
// same arithmetic agree:
//
//   replay      pass i leaves the committed state at the prefix its predecessor accepted, and
//               pass i+1 replays pass i's rows from blk_cache before running its own.
//   commit      pass i stores the conv ring after its last row and the state after its last
//               token, and pass i+1 arrives with n_replay = 0.
//
// Both schedules apply the same operations to the same operands in the same order, so agreement
// here is bitwise, not approximate. Any inequality is a real difference in the contract.
//
// Build:  make -C tools/direct-commit
// Run:    tools/direct-commit/commit_equiv           (an argument changes the input seed)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace mirror {

// ---- geometry --------------------------------------------------------------------------------
// SS, the lane split, the parity cache and RMAX are the kernel's. HV and HK are cut down: every
// head runs an independent recurrence, so four of them exercise the same code as forty-eight.
constexpr int SS = 128;                       // state size = head dim
constexpr int HV = 4;                         // value heads (48 in the model)
constexpr int HK = 2;                         // key heads (16 in the model)
constexpr int KDIM = HK * SS;
constexpr int VDIM = HV * SS;
constexpr int CONV_CH = 2 * KDIM + VDIM;
constexpr int BLK_TOKEN_FLOATS = 2 * CONV_CH + 2 * HV;
constexpr int RMAX = 8;
constexpr float NORM_EPS = 1e-6f;
constexpr float OUT_SCALE = 0.08838834764831845f;

// ---- device reduction mirror -----------------------------------------------------------------
// warp_sum's DPP chain: xor 1, xor 2, row_ror 4, row_ror 8, swap the 16-lane halves. The rotations
// leave the lanes holding sums of the same 32 values in different associations, and the kernel
// lets each lane use its own result, so the mirror keeps all 32 rather than broadcasting one.
inline void warp_sum32(float v[32]) {
    float t[32];
    auto snap = [&] { for (int i = 0; i < 32; i++) t[i] = v[i]; };
    snap(); for (int i = 0; i < 32; i++) v[i] = t[i] + t[i ^ 1];
    snap(); for (int i = 0; i < 32; i++) v[i] = t[i] + t[i ^ 2];
    snap(); for (int i = 0; i < 32; i++) { const int row = i & ~15, l = i & 15; v[i] = t[i] + t[row + ((l + 4) & 15)]; }
    snap(); for (int i = 0; i < 32; i++) { const int row = i & ~15, l = i & 15; v[i] = t[i] + t[row + ((l + 8) & 15)]; }
    snap(); for (int i = 0; i < 32; i++) v[i] = t[i] + t[i ^ 16];
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---- weights, state slots, pass description ----------------------------------------------------
struct Layer {
    std::vector<float> conv_w;  // [CONV_CH][4]
    std::vector<float> ssm_a;   // [HV]
    std::vector<float> ssm_dt;  // [HV]
};

// One sequence's durable state: conv_ring, gdn_state and the two parity halves of blk_cache.
struct Slot {
    std::vector<float> ring{ std::vector<float>(3 * CONV_CH, 0.0f) };
    std::vector<float> state{ std::vector<float>((size_t) HV * SS * SS, 0.0f) };
    std::vector<float> cache{ std::vector<float>((size_t) 2 * RMAX * BLK_TOKEN_FLOATS, 0.0f) };
    float * tok(int parity, int i) { return cache.data() + ((size_t) parity * RMAX + i) * BLK_TOKEN_FLOATS; }
    const float * tok(int parity, int i) const { return cache.data() + ((size_t) parity * RMAX + i) * BLK_TOKEN_FLOATS; }
};

// The row buffers of one launch, shared by every sequence in it, exactly as P.big, P.ab, P.ninv
// and P.o are shared.
struct Pass {
    const float * big = nullptr;   // [rows][CONV_CH]
    const float * ab = nullptr;    // [rows][2 * HV]
    const float * ninv = nullptr;  // [rows]
    float * o = nullptr;           // [rows][VDIM]
};
struct SeqCtl { int row0 = 0, nrows = 0, n_replay = 0, parity = 0; };

// ---- ph_gdn_pre --------------------------------------------------------------------------------
// Commit moves one store: the ring is persisted after this pass's rows instead of after the
// replayed prefix. What lands in blk_cache is the same either way.
void gdn_pre(const Layer & L, Slot & s, const Pass & P, const SeqCtl & S, bool commit) {
    std::vector<float> r0(CONV_CH), r1(CONV_CH), r2(CONV_CH), conv(CONV_CH);
    for (int ch = 0; ch < CONV_CH; ch++) { r0[ch] = s.ring[ch]; r1[ch] = s.ring[CONV_CH + ch]; r2[ch] = s.ring[2 * CONV_CH + ch]; }

    for (int t = 0; t < S.n_replay; t++) {
        const float * prev = s.tok(S.parity ^ 1, t);
        for (int ch = 0; ch < CONV_CH; ch++) { const float v = prev[ch]; r0[ch] = r1[ch]; r1[ch] = r2[ch]; r2[ch] = v; }
    }
    if (!commit)
        for (int ch = 0; ch < CONV_CH; ch++) { s.ring[ch] = r0[ch]; s.ring[CONV_CH + ch] = r1[ch]; s.ring[2 * CONV_CH + ch] = r2[ch]; }

    for (int i = 0; i < S.nrows; i++) {
        const int row = S.row0 + i;
        const float * x = P.big + (size_t) row * CONV_CH;
        float * tok = s.tok(S.parity, i);
        for (int ch = 0; ch < CONV_CH; ch++) {
            const float * w = L.conv_w.data() + ch * 4;
            conv[ch] = silu(w[0] * r0[ch] + w[1] * r1[ch] + w[2] * r2[ch] + w[3] * x[ch]);
            tok[ch] = x[ch];
        }
        // q and k channels are L2-normalised per 128-channel head: four waves reduce their own 32
        // channels, then the four partials are added in wave order.
        for (int head = 0; head * SS < 2 * KDIM; head++) {
            float red[4];
            for (int w = 0; w < 4; w++) {
                float lanes[32];
                for (int l = 0; l < 32; l++) { const float v = conv[head * SS + w * 32 + l]; lanes[l] = v * v; }
                warp_sum32(lanes);
                red[w] = lanes[0];
            }
            const float inv = 1.0f / std::fmax(std::sqrt(red[0] + red[1] + red[2] + red[3]), NORM_EPS);
            for (int c = 0; c < SS; c++) conv[head * SS + c] *= inv;
        }
        for (int ch = 0; ch < CONV_CH; ch++) tok[CONV_CH + ch] = conv[ch];
        // The device caches the evaluated gates here, not the raw projections it used to; the
        // reader below takes them as they are. docs/gdn-sliced-gates.md.
        for (int t = 0; t < 2 * HV; t++) {
            const float ab = P.ab[(size_t) row * 2 * HV + t] * P.ninv[row];
            if (t < HV) {
                const float xg = ab + L.ssm_dt[t];
                const float sp = xg > 20.0f ? xg : std::log1p(std::exp(xg));
                tok[2 * CONV_CH + t] = std::exp(L.ssm_a[t] * sp);
            } else {
                tok[2 * CONV_CH + t] = sigmoid(ab);
            }
        }
        for (int ch = 0; ch < CONV_CH; ch++) { r0[ch] = r1[ch]; r1[ch] = r2[ch]; r2[ch] = x[ch]; }
    }
    if (commit)
        for (int ch = 0; ch < CONV_CH; ch++) { s.ring[ch] = r0[ch]; s.ring[CONV_CH + ch] = r1[ch]; s.ring[2 * CONV_CH + ch] = r2[ch]; }
}

// ---- ph_gdn ------------------------------------------------------------------------------------
// Commit moves the state write-back from "after the replayed prefix" to "after the last token".
// The token loop, its operands and its outputs are untouched.
void gdn_state(const Layer & L, Slot & s, const Pass & P, const SeqCtl & S, bool commit) {
    const int ntok = S.n_replay + S.nrows;
    for (int h = 0; h < HV; h++) {
        const int kh = h % HK;
        std::vector<float> m((size_t) SS * 32 * 4);
        auto M = [&](int j, int l, int s2) -> float & { return m[((size_t) j * 32 + l) * 4 + s2]; };
        for (int j = 0; j < SS; j++)
            for (int l = 0; l < 32; l++)
                for (int s2 = 0; s2 < 4; s2++) M(j, l, s2) = s.state[((size_t) h * SS + j) * SS + l + 32 * s2];
        auto store = [&] {
            for (int j = 0; j < SS; j++)
                for (int l = 0; l < 32; l++)
                    for (int s2 = 0; s2 < 4; s2++) s.state[((size_t) h * SS + j) * SS + l + 32 * s2] = M(j, l, s2);
        };

        for (int t = 0; t < ntok; t++) {
            const bool replay = t < S.n_replay;
            const float * tok = replay ? s.tok(S.parity ^ 1, t) : s.tok(S.parity, t - S.n_replay);
            const float g = tok[2 * CONV_CH + h];
            const float beta = tok[2 * CONV_CH + HV + h];
            const float * q = tok + CONV_CH + kh * SS;
            const float * k = tok + CONV_CH + KDIM + kh * SS;
            const float * v = tok + CONV_CH + 2 * KDIM + h * SS;
            float * out = replay || !P.o ? nullptr : P.o + (size_t) (S.row0 + t - S.n_replay) * VDIM;

            for (int j = 0; j < SS; j++) {
                float dot[32];
                for (int l = 0; l < 32; l++) {
                    float p = 0.0f;
                    for (int s2 = 0; s2 < 4; s2++) { M(j, l, s2) *= g; p = std::fma(M(j, l, s2), k[l + 32 * s2], p); }
                    dot[l] = p;
                }
                warp_sum32(dot);
                float op[32];
                for (int l = 0; l < 32; l++) {
                    const float delta = (v[j] - dot[l]) * beta;
                    float a = 0.0f;
                    for (int s2 = 0; s2 < 4; s2++) {
                        M(j, l, s2) = std::fma(delta, k[l + 32 * s2], M(j, l, s2));
                        a = std::fma(M(j, l, s2), q[l + 32 * s2], a);
                    }
                    op[l] = a;
                }
                warp_sum32(op);
                if (out) out[h * SS + j] = op[0] * OUT_SCALE;
            }
            if (!commit && t + 1 == S.n_replay) store();
        }
        if (commit) store();
    }
}

// One launch: every sequence's conv phase, then every sequence's state phase, as the two phases of
// the kernel do either side of a grid barrier.
void run_pass(const Layer & L, std::vector<Slot *> & slots, const Pass & P, const std::vector<SeqCtl> & seqs, bool commit) {
    for (size_t i = 0; i < seqs.size(); i++) gdn_pre(L, *slots[i], P, seqs[i], commit);
    for (size_t i = 0; i < seqs.size(); i++) gdn_state(L, *slots[i], P, seqs[i], commit);
}

// ---- inputs ------------------------------------------------------------------------------------
// Token inputs are keyed by (sequence, absolute position), so both schedules feed identical values
// to identical tokens however the passes are cut.
struct Inputs {
    std::vector<float> big, ab, ninv;
    Inputs(int seq, int ntok, uint32_t seed) : big((size_t) ntok * CONV_CH), ab((size_t) ntok * 2 * HV), ninv(ntok) {
        for (int t = 0; t < ntok; t++) {
            std::mt19937 rng(seed * 2654435761u + (uint32_t) seq * 40503u + (uint32_t) t);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            for (int c = 0; c < CONV_CH; c++) big[(size_t) t * CONV_CH + c] = u(rng);
            for (int c = 0; c < 2 * HV; c++) ab[(size_t) t * 2 * HV + c] = 2.0f * u(rng);
            ninv[t] = 0.5f + 0.25f * u(rng);
        }
    }
};

Layer make_layer(uint32_t seed) {
    Layer L;
    L.conv_w.resize((size_t) CONV_CH * 4);
    L.ssm_a.resize(HV);
    L.ssm_dt.resize(HV);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & w : L.conv_w) w = 0.5f * u(rng);
    for (int h = 0; h < HV; h++) { L.ssm_a[h] = -0.5f - 0.25f * std::fabs(u(rng)); L.ssm_dt[h] = u(rng); }
    return L;
}

// ---- comparison ----------------------------------------------------------------------------------
struct Diff { size_t n_differ = 0; double max_abs = 0.0; };
Diff compare(const std::vector<float> & a, const std::vector<float> & b) {
    Diff d;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        uint32_t x, y;
        std::memcpy(&x, &a[i], 4); std::memcpy(&y, &b[i], 4);
        if (x != y) { d.n_differ++; d.max_abs = std::fmax(d.max_abs, std::fabs((double) a[i] - (double) b[i])); }
    }
    return d;
}
void absorb(Diff & worst, const Diff & d, bool & flag) {
    if (!d.n_differ) return;
    flag = false;
    worst.n_differ += d.n_differ;
    worst.max_abs = std::fmax(worst.max_abs, d.max_abs);
}

// ---- single-sequence scenarios ---------------------------------------------------------------
// A scenario is a list of passes. `commit` is the schedule-under-test's flag for that pass;
// `replay_after_commit` forces n_replay even after a committed pass, the contract violation we
// want to see rather than assume.
struct PassPlan { int rows; bool commit; bool replay_after_commit = false; };
struct Result { bool outputs = true, state = true, ring = true; Diff worst; };

Result run_scenario(const std::vector<PassPlan> & plan, uint32_t seed) {
    const Layer L = make_layer(seed);
    int ntok = 0;
    for (const auto & p : plan) ntok += p.rows;
    const Inputs in(0, ntok + RMAX, seed);   // spare tokens for the reference run's trailing pass

    auto pass_of = [&](int pos, std::vector<float> & o) {
        Pass P;
        P.big = in.big.data() + (size_t) pos * CONV_CH;
        P.ab = in.ab.data() + (size_t) pos * 2 * HV;
        P.ninv = in.ninv.data() + pos;
        P.o = o.data();
        return P;
    };

    // Reference: every pass replays its predecessor's rows and commits nothing. One extra pass is
    // appended so the reference reaches the commit point of the last committed pass under test.
    std::vector<Slot> ref_after;
    std::vector<std::vector<float>> ref_out;
    {
        Slot s;
        int pos = 0, parity = 0, prev = 0;
        std::vector<PassPlan> ref_plan = plan;
        ref_plan.push_back({ 1, false });
        for (const auto & p : ref_plan) {
            std::vector<float> o((size_t) p.rows * VDIM, 0.0f);
            Pass P = pass_of(pos, o);
            std::vector<Slot *> slots{ &s };
            std::vector<SeqCtl> seqs{ SeqCtl{ 0, p.rows, prev, parity } };
            run_pass(L, slots, P, seqs, false);
            ref_out.push_back(std::move(o));
            ref_after.push_back(s);
            pos += p.rows; parity ^= 1; prev = p.rows;
        }
    }

    Result r;
    Slot s;
    int pos = 0, parity = 0, prev = 0;
    bool prev_committed = false;
    for (size_t i = 0; i < plan.size(); i++) {
        const PassPlan & p = plan[i];
        std::vector<float> o((size_t) p.rows * VDIM, 0.0f);
        Pass P = pass_of(pos, o);
        std::vector<Slot *> slots{ &s };
        std::vector<SeqCtl> seqs{ SeqCtl{ 0, p.rows, (prev_committed && !p.replay_after_commit) ? 0 : prev, parity } };
        run_pass(L, slots, P, seqs, p.commit);

        absorb(r.worst, compare(ref_out[i], o), r.outputs);
        // A committed pass's state must equal the reference state one pass later: that is the
        // point at which the reference has replayed exactly these tokens and no others.
        const Slot & ref = ref_after[p.commit ? i + 1 : i];
        absorb(r.worst, compare(ref.state, s.state), r.state);
        absorb(r.worst, compare(ref.ring, s.ring), r.ring);

        pos += p.rows; parity ^= 1; prev = p.rows; prev_committed = p.commit;
    }
    return r;
}

// ---- several sequences in one pass ------------------------------------------------------------
// Two sequences share a launch, its row buffers and its output buffer, and differ in row0, slot
// and row count. Each must produce exactly what it produces alone.
Result multi_sequence(uint32_t seed) {
    const Layer L = make_layer(seed);
    const std::vector<std::pair<int, int>> shape = { { 5, 3 }, { 3, 5 }, { 1, 7 }, { 4, 4 }, { 8, 0 } };  // rows for A, B
    int na = 0, nb = 0;
    for (const auto & p : shape) { na += p.first; nb += p.second; }
    const Inputs ia(0, na, seed), ib(1, nb, seed + 7u);

    Result r;
    Slot alone_a, alone_b, both_a, both_b;
    int pa = 0, pb = 0, parity = 0;
    for (const auto & p : shape) {
        const int rows = p.first + p.second;
        // Shared buffers: sequence A occupies rows [0, p.first), B occupies [p.first, rows).
        std::vector<float> big((size_t) rows * CONV_CH), ab((size_t) rows * 2 * HV), ninv(rows), o((size_t) rows * VDIM, 0.0f);
        for (int i = 0; i < p.first; i++) {
            std::memcpy(&big[(size_t) i * CONV_CH], &ia.big[(size_t) (pa + i) * CONV_CH], CONV_CH * 4);
            std::memcpy(&ab[(size_t) i * 2 * HV], &ia.ab[(size_t) (pa + i) * 2 * HV], 2 * HV * 4);
            ninv[i] = ia.ninv[pa + i];
        }
        for (int i = 0; i < p.second; i++) {
            const int row = p.first + i;
            std::memcpy(&big[(size_t) row * CONV_CH], &ib.big[(size_t) (pb + i) * CONV_CH], CONV_CH * 4);
            std::memcpy(&ab[(size_t) row * 2 * HV], &ib.ab[(size_t) (pb + i) * 2 * HV], 2 * HV * 4);
            ninv[row] = ib.ninv[pb + i];
        }
        Pass shared{ big.data(), ab.data(), ninv.data(), o.data() };
        std::vector<Slot *> slots;
        std::vector<SeqCtl> seqs;
        if (p.first) { slots.push_back(&both_a); seqs.push_back({ 0, p.first, 0, parity }); }
        if (p.second) { slots.push_back(&both_b); seqs.push_back({ p.first, p.second, 0, parity }); }
        run_pass(L, slots, shared, seqs, true);

        // The same rows, each sequence on its own launch with its own buffers.
        std::vector<float> oa((size_t) std::max(p.first, 1) * VDIM, 0.0f), ob((size_t) std::max(p.second, 1) * VDIM, 0.0f);
        if (p.first) {
            Pass Pa{ ia.big.data() + (size_t) pa * CONV_CH, ia.ab.data() + (size_t) pa * 2 * HV, ia.ninv.data() + pa, oa.data() };
            std::vector<Slot *> sa{ &alone_a };
            std::vector<SeqCtl> ca{ SeqCtl{ 0, p.first, 0, parity } };
            run_pass(L, sa, Pa, ca, true);
            std::vector<float> got(o.begin(), o.begin() + (size_t) p.first * VDIM);
            absorb(r.worst, compare(oa, got), r.outputs);
            absorb(r.worst, compare(alone_a.state, both_a.state), r.state);
            absorb(r.worst, compare(alone_a.ring, both_a.ring), r.ring);
        }
        if (p.second) {
            Pass Pb{ ib.big.data() + (size_t) pb * CONV_CH, ib.ab.data() + (size_t) pb * 2 * HV, ib.ninv.data() + pb, ob.data() };
            std::vector<Slot *> sb{ &alone_b };
            std::vector<SeqCtl> cb{ SeqCtl{ 0, p.second, 0, parity } };
            run_pass(L, sb, Pb, cb, true);
            std::vector<float> got(o.begin() + (size_t) p.first * VDIM, o.end());
            absorb(r.worst, compare(ob, got), r.outputs);
            absorb(r.worst, compare(alone_b.state, both_b.state), r.state);
            absorb(r.worst, compare(alone_b.ring, both_b.ring), r.ring);
        }
        pa += p.first; pb += p.second; parity ^= 1;
    }
    return r;
}

} // namespace mirror

int main(int argc, char ** argv) {
    using namespace mirror;
    const uint32_t seed = argc > 1 ? (uint32_t) std::strtoul(argv[1], nullptr, 10) : 20260921u;
    std::printf("Direct-commit GDN equivalence, host mirror of ph_gdn_pre and ph_gdn\n");
    std::printf("seed %u, %d heads of %d state rows, %d conv channels, comparison is bitwise\n\n", seed, HV, SS, CONV_CH);

    struct Case { const char * name; std::vector<PassPlan> plan; bool expect_equal; };
    const std::vector<Case> cases = {
        { "first pass commits from empty state",        { { 8, true } }, true },
        { "eight-row prefill, four passes",             { { 8, true }, { 8, true }, { 8, true }, { 8, true } }, true },
        { "ragged pass sizes with a short tail",        { { 8, true }, { 5, true }, { 8, true }, { 3, true }, { 1, true } }, true },
        { "one-row passes, shorter than the ring",      { { 1, true }, { 1, true }, { 1, true }, { 2, true } }, true },
        { "uncommitted pass, then a committing pass",   { { 8, false }, { 8, true } }, true },
        { "old, commit, old, old",                      { { 8, false }, { 6, true }, { 8, false }, { 4, false } }, true },
        { "commit, old, commit, commit",                { { 8, true }, { 5, false }, { 7, true }, { 8, true } }, true },
        { "one-row commit between eight-row passes",    { { 8, true }, { 1, true }, { 8, true } }, true },
        { "violation: a committed pass replayed again", { { 8, true }, { 8, true, true } }, false },
    };

    int failures = 0;
    std::printf("%-44s %-7s %-7s %-7s  %s\n", "scenario", "rows", "state", "ring", "verdict");
    for (const auto & c : cases) {
        const Result r = run_scenario(c.plan, seed);
        const bool equal = r.outputs && r.state && r.ring;
        const bool ok = equal == c.expect_equal;
        failures += !ok;
        std::printf("%-44s %-7s %-7s %-7s  %s", c.name,
                    r.outputs ? "equal" : "differ", r.state ? "equal" : "differ", r.ring ? "equal" : "differ",
                    ok ? (c.expect_equal ? "matches the replay path" : "diverges, as the contract says it must") : "FAILED");
        if (!equal) std::printf("  [%zu floats, max |d| %.3g]", r.worst.n_differ, r.worst.max_abs);
        std::printf("\n");
    }

    const Result m = multi_sequence(seed);
    const bool multi_ok = m.outputs && m.state && m.ring;
    failures += !multi_ok;
    std::printf("%-44s %-7s %-7s %-7s  %s", "five passes, two sequences, shared buffers",
                m.outputs ? "equal" : "differ", m.state ? "equal" : "differ", m.ring ? "equal" : "differ",
                multi_ok ? "sequences do not interact" : "FAILED");
    if (!multi_ok) std::printf("  [%zu floats, max |d| %.3g]", m.worst.n_differ, m.worst.max_abs);
    std::printf("\n");

    std::printf("\n%s\n", failures ? "FAILURES above." : "Every legal schedule agrees with the replay path bit for bit; the illegal one diverges.");
    std::printf("Not covered here: the GPU, the unit decomposition across workgroups, the input prep and\n"
                "projections that feed these phases, and attention. Those belong to the full-model checks.\n");
    return failures ? 1 : 0;
}
