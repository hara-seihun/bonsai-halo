// Host exactness probe for the windowed attention unit list. No GPU.
//
//   make kernels/attn_window_check && kernels/attn_window_check
//
// `ph_attn` deals no score unit for a key chunk that a sliding window excludes, and the fold in
// `prep_chunk_r` starts above the same chunks. Both read `attn_chunk0`. The claim that makes that
// pair safe is arithmetic, not statistical, and the drafter cannot demonstrate it on the device:
// four of its matvecs drain a K-split with `atomicAdd`, so its own float state is not reproducible
// between two runs of one binary - the same arm differs from itself by up to 7.8e-3 in the block
// hidden state. So the exactness lives here, where it is decidable.
//
// Three claims, each checked against the engine's own expressions rather than a restatement:
//
//   1. A chunk below `attn_chunk0(pos, window)` is empty for a query at `pos`. The score body
//      computes lo = max(0, key_min - start), raised to qpos - window + 1 - start under a window,
//      and vis = count (noncausal) or min(count, qpos + 1 - start). Empty means lo >= vis, so the
//      max loop and the value loop both run zero iterations and the partial is (-inf, 0, 0).
//   2. A chunk at or above it is not empty, so the floor is exactly the first chunk the query can
//      see and the skip drops nothing a fold would read.
//   3. Folding a (-inf, 0, 0) partial is the identity on both accumulators, bit for bit: the
//      running max ignores -inf, the weight is exp(-inf - M) = 0, and fmaf(0, x, s) = s. This is
//      checked on the fold's own float32 expressions over random and adversarial partial sets,
//      including the fp16-representable value range the V cache holds.
//
// A group of rows shares one unit, so the producer's floor is the group's first row and the fold's
// floor is each row's own: claim 4 is that the first is never above the second.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "phases.hpp"

using halo::attn_chunk0;
using halo::ACHUNK;

namespace {

int failures = 0;
void fail(const char * what, long long a, long long b) {
    if (++failures < 20) fprintf(stderr, "FAIL %s: %lld vs %lld\n", what, a, b);
}

// The score body's visibility arithmetic, copied expression for expression from
// `attn_chunk_unit`'s softmax loop (kernels/phases.hpp). `count` is that unit's key count.
struct Visible { int lo, vis, count; };
Visible visible(int chunk, int qpos, int pos_last, int window, int key_min, int noncausal) {
    const int start = chunk * ACHUNK;
    const int count = std::min(ACHUNK, pos_last + 1 - start);
    int lo = std::max(0, key_min - start);
    if (window) lo = std::max(lo, qpos - window + 1 - start);
    const int vis = noncausal ? count : std::min(count, qpos + 1 - start);
    return { lo, vis, count };
}

// The fold in `prep_chunk_r`, one output element, over a chunk range. Same order, same fmaf, same
// reciprocal; `__expf` is the device's fast exponential and this uses `expf`, which is the only
// difference and the one the claim does not rest on - every implementation returns exactly 0 for
// -inf, which is the only argument an empty chunk ever produces.
struct Partial { float m, l, acc; };
void fold(const Partial * part, int cc0, int nchunks, float & den_out, float & v_out) {
    float M = -INFINITY;
    for (int cc = cc0; cc < nchunks; cc++) M = fmaxf(M, part[cc].m);
    float den = 0.0f, v = 0.0f;
    for (int cc = cc0; cc < nchunks; cc++) {
        const float w = expf(part[cc].m - M);
        den = fmaf(w, part[cc].l, den);
        v = fmaf(w, part[cc].acc, v);
    }
    den_out = den; v_out = v;
}

} // namespace

int main() {
    // ---- 1 and 2: the floor is exactly the first chunk a windowed query can see.
    long long empty_below = 0, live_above = 0;
    for (int window : { 0, 1, 2, 127, 128, 129, 2048, 4096, 32768 })
    for (int pos : { 0, 1, 127, 128, 129, 255, 2046, 2047, 2048, 2049, 3000, 4095, 4096, 8191,
                     11248, 16383, 32767 }) {
        const int c0 = attn_chunk0(pos, window);
        if (c0 < 0 || c0 > pos / ACHUNK) fail("floor out of range", c0, pos / ACHUNK);
        for (int chunk = 0; chunk <= pos / ACHUNK; chunk++) {
            // The drafter's shape: key_min 0, noncausal, the group's last row at `pos_last`.
            for (int pos_last : { pos, pos + 1, pos + 7 }) {
                const Visible v = visible(chunk, pos, pos_last, window, 0, 1);
                const bool empty = v.lo >= v.vis;
                if (chunk < c0) { if (!empty) fail("chunk below the floor is not empty", chunk, c0); else empty_below++; }
                else            { if (empty) fail("chunk at or above the floor is empty", chunk, c0); else live_above++; }
            }
            // The target model's shape: no window at all, so the floor is 0 and nothing is skipped.
            if (window == 0 && attn_chunk0(pos, 0) != 0) fail("windowless floor is not chunk 0", attn_chunk0(pos, 0), 0);
        }
    }

    // ---- 4: the producer's floor (the group's first row) is never above any row's own floor.
    for (int window : { 1, 128, 2048, 4096 })
    for (int pos0 = 0; pos0 < 40000; pos0 += 7)
    for (int nrows : { 1, 2, 4, 8 }) {
        const int gfloor = attn_chunk0(pos0, window);
        for (int i = 0; i < nrows; i++)
            if (gfloor > attn_chunk0(pos0 + i, window)) fail("group floor above a row's floor", gfloor, attn_chunk0(pos0 + i, window));
    }

    // ---- 3: folding empty partials is the identity, bit for bit.
    std::mt19937 rng(20260921);
    std::uniform_real_distribution<float> um(-40.f, 40.f), ul(0.f, 128.f), ua(-65504.f, 65504.f);
    long long cases = 0;
    for (int trial = 0; trial < 200000; trial++) {
        const int live = 1 + (int) (rng() % 32);      // chunks the window keeps
        const int dead = (int) (rng() % 96);          // chunks below the floor
        std::vector<Partial> full(dead + live);
        for (int i = 0; i < dead; i++) full[i] = { -INFINITY, 0.0f, 0.0f };
        for (int i = 0; i < live; i++) {
            Partial p { um(rng), ul(rng), ua(rng) };
            // An all-masked chunk inside the kept range is legal too: the last block row can sit
            // one chunk above the first, and a unit is dealt for the group.
            if ((rng() & 15) == 0) p = { -INFINITY, 0.0f, 0.0f };
            full[dead + i] = p;
        }
        // The kept range must hold at least one live chunk, which the schedule guarantees: a row
        // always sees its own position.
        full[dead + live - 1].m = um(rng); full[dead + live - 1].l = ul(rng); full[dead + live - 1].acc = ua(rng);
        float den_all, v_all, den_skip, v_skip;
        fold(full.data(), 0, dead + live, den_all, v_all);            // the schedule that dealt every chunk
        fold(full.data(), dead, dead + live, den_skip, v_skip);       // the schedule the window decides
        if (memcmp(&den_all, &den_skip, 4) || memcmp(&v_all, &v_skip, 4))
            fail("fold moved", (long long) den_all, (long long) den_skip);
        cases++;
    }

    // ---- the primitives the identity rests on, at the edges.
    for (float M : { -3.4e38f, -1.f, 0.f, 1e-30f, 1.f, 3.4e38f }) {
        if (fmaxf(M, -INFINITY) != M) fail("fmaxf(-inf) moved the max", 0, 0);
        if (expf(-INFINITY - M) != 0.0f) fail("exp(-inf - M) is not zero", 0, 0);
        for (float s : { -3.4e38f, -1.f, 0.f, 1.f, 3.4e38f }) {
            if (fmaf(0.0f, 0.0f, s) != s) fail("fmaf(0, 0, s) moved s", 0, 0);
            if (fmaf(0.0f, M, s) != s) fail("fmaf(0, x, s) moved s", 0, 0);
        }
    }

    printf("attn_window_check: %lld empty chunks below the floor, %lld live at or above, %lld folds\n",
           empty_below, live_above, cases);
    printf("attn_window_check: %s\n", failures ? "FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
