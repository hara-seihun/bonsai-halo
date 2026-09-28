// What the persistent kernel's scaffolding costs, measured with the engine's own scaffolding.
//
// A single-token step runs about 645 grid barriers and deals tens of thousands of units, and every
// phase map this lane has published prices those two things by subtraction: a phase that moves
// weight bytes is compared against those bytes at 242 GB/s and whatever is left over is attributed
// to a barrier, a unit boundary, a round quantisation or "latency". docs/single-stream.md leaves
// 3.3 ms of a 29.694 ms token in that residue and names blocks-per-wave-per-unit as the variable it
// tracks. This probe measures the three scaffolding costs directly instead.
//
// It includes kernels/phases.hpp and calls the real `grid_sync`, `next_unit` and `mv_rows_t`, so a
// number here is the cost of the code the engine runs, not of a replica that may have drifted.
//
//   mode 0   `phases` barriers, no units          -> nanoseconds per grid barrier
//   mode 1   units dealt, empty unit body         -> nanoseconds per unit deal (atomic + two syncs)
//   mode 2   units dealt, PW blocks each          -> GB/s at a chosen unit granularity
//
// Mode 2 is the decisive one and it is a sweep, not a point: hold the streamed bytes fixed and vary
// PW (blocks per wave per unit), which is exactly what differs between the engine's matvec phases
// (5 for the 40-block projections, 3 for the 48-block K-split pair, 8/9 for the 136-block down).
// If GB/s is flat across PW the unit boundary is free and the residue is somewhere else; if it
// rises with PW, the slope is the per-unit cost in nanoseconds and the intercept is the stream.
//
//   bench/coop_cost --mode 2 --pw 3,5,8,17,40 --bytes 2048 --grid 60
//
// Every arm runs in one process against one warmed clock, because two processes on this box differ
// by more than the effect being measured.
//
// `--tt` is how many activation rows the unit body carries, and it is the axis this probe was
// missing: the deployed drafted verify pass runs `mv_rows_t<PW, 8>`, where a block's 896 bytes
// serve eight rows of matrix work and 256 scalar activation loads instead of one row's 32. A
// number taken at one row prices the single-token step and nothing else.
//
// `--exact-units` deals the unit count asked for instead of rounding it down to whole grid rounds.
// Rounding is right when `pw` is the variable; it hides the round tail, which is the whole question
// for a phase whose 320 units land on a 100-workgroup grid.
//
// `--arm` is the prefetch axis and it reads this probe's own table a second way. At tt 8, pw 5 the
// stream is 141.9 GB/s over one grid round, 153.9 over three, 184.3 over eleven and 204.2 over
// seventy-eight - the same bytes, the same unit body, the same grid, and a 44% spread that belongs
// to how LONG the phase is rather than to how big its units are. The engine sits on the bad end of
// that curve: `mv_down`, `mv_ssm_out` and `mv_o` deal 320 units on a 100-workgroup grid, 3.2 rounds,
// and read 113 to 129 GB/s, while `mv_lm_head` deals 7760 in one phase and reads 203.3.
//
// Two places in that schedule expose a memory round trip that nothing covers, and each is an arm:
//
//   arm 1  THE PHASE START. After `grid_sync` all 800 waves ask for their first weight block into
//          an empty pipe. A workgroup already knows the unit it will take next - the loop opens at
//          `u = blockIdx.x` - so it can ask for those bytes BEFORE the barrier it is about to wait
//          at and arrive warm.
//   arm 2  THE UNIT BOUNDARY. A unit ends with the drain - two `__syncthreads`, the LDS reduction
//          and the output write - and only THEN deals the next unit and asks for its first block.
//          Dealing before the drain puts the drain's own work over that round trip. The deal is the
//          same atomic on the same counter; only when it is issued moves.
//   arm 4  THE ONE THE MEASUREMENT POINTED AT AFTER THE OTHER TWO WERE NULLS. 100 workgroups leave
//          the grid barrier together and every one of them is at the SAME OFFSET INSIDE ITS UNIT at
//          the same instant: unit bases are `nb * 896` apart, so 100 concurrent weight streams read
//          addresses that differ by a fixed stride and share their low bits. They de-phase only as
//          dynamic dealing lets them drift, which is why the rate climbs with the number of grid
//          rounds a phase gets. Rotating which wave takes which block range by `blockIdx.x` breaks
//          the lockstep at no cost in time: every wave still walks `PW` contiguous blocks in the
//          same order, and in the engine the drain reads the partials by BLOCK RANGE rather than by
//          wave, so the sum stays left to right and no output bit moves. Here it is a rate arm and
//          the probe does not check its values.
//   arm 8  WHAT MAKES EITHER OF THOSE POSSIBLE, and it is the finding this probe was built to
//          test. `__syncthreads()` is a seq_cst fence over every address space, and on gfx1151 the
//          memory legalizer compiles that to `s_waitcnt vmcnt(0) lgkmcnt(0); s_barrier`. A unit
//          boundary carries three of them and a phase boundary two, so NO VECTOR MEMORY REQUEST IN
//          THIS KERNEL CAN SURVIVE A BARRIER - a prefetch issued before one is waited on at it,
//          which is why the block cursor stops at the edge of a unit and why lookahead depth has
//          never bought anything across one. The drain and the unit deal publish LDS and read LDS;
//          their global writes are published by the phase barrier that follows. An LDS-scoped
//          fence around the raw barrier says exactly that and compiles to `s_waitcnt lgkmcnt(0);
//          s_barrier`, leaving the weight stream in flight.
//
// All of it is pure request scheduling: same bytes, same order, same arithmetic, same output. Arm 0
// keeps the loop the tree already had, instruction for instruction.
//
// `--stagger N` and `--trace-units` are the two diagnostics that say WHICH mechanism a short phase
// is losing to: a per-workgroup delay at the phase start tests whether 800 waves leaving one
// barrier together is a burst the memory system serves badly, and the trace times every unit of
// workgroup 0 so a ramp inside the phase can be seen instead of inferred.
#include "../kernels/phases.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

// COOP_GAP compiles the gap-phase body into the kernel. It is a build flag rather than a runtime
// one because the question it answers is what an EXTRA PHASE BODY costs the streamed phase that
// shares its kernel, which is the difference between this probe and the engine: `k_forward_rows`
// carries ten phase bodies and this probe carries one. Two binaries one define apart, the gap
// phase disabled at runtime in both, price that directly.
#ifndef COOP_GAP
#define COOP_GAP 0
#endif

using namespace halo;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e_)); exit(1);}}while(0)

namespace {

struct ProbeArgs {
    unsigned * bar; unsigned * work;
    const uint8_t * w; const int8_t * xq; const float * xs; const int * xsum;
    float * out;
    int phases, units, nb, mode;
    size_t span;            // units available in the streamed buffer; a phase walks a window of it
    // Where a phase's bytes live, which is the axis the engine differs from this probe on. `w` with
    // `regions == nullptr` is one allocation walked in ascending windows: phase p+1 starts exactly
    // where phase p stopped, and a launch is one sweep. `regions` is an array of `nregions`
    // SEPARATE hipMalloc allocations, each exactly one phase long, which is how `Engine::upload_halo`
    // stores a tensor. `stride_windows` scatters the window a phase takes inside one allocation, so
    // continuity can be broken without changing how the bytes were allocated.
    const uint8_t * const * regions; int nregions;
    unsigned stride_windows, nwindows;
    // What runs BETWEEN two streamed phases. The engine never runs two matvec phases back to back:
    // every one is preceded by a prep phase that reads a row of activations, normalises it and
    // writes forty kilobytes of quantised operand - 9.9 us against the matvec's 60 to 234. This
    // probe has always run one long stream against another, so a phase boundary here is a barrier
    // between two identical streams. `gap_units` inserts the engine's kind of neighbour: -1 none,
    // 0 a barrier with no work at all, N a phase of N units that each read 1 kB and write 256 B.
    int gap_units; const float * gap_in; float * gap_out;
    // The K split's stream geometry. `ph_matvec` with KS > 1 cuts a tile's blocks into KS parts and
    // deals them part-major: units 0..T-1 are part 0 of every tile, T..2T-1 are part 1. A unit
    // therefore reads PART blocks out of a row of KS*PART, so the grid's concurrent runs sit
    // KS*PART*896 bytes apart instead of PART*896. The three 5120-row projections of the deployed
    // verify pass run that way (KSV = KSFF = 2) and this probe has always read those same units
    // contiguously. `split` is KS: 0 or 1 keeps the contiguous walk.
    int split, total_tiles;
    // The last two things a probe unit does not do and an engine unit does. `drain` publishes the
    // eight waves' partial sums through LDS, reduces them in wave 0 in the engine's order and
    // writes the output row (1 store, or 2 atomicAdd, which is what KS > 1 pays); `rows` is passed
    // to the body instead of the compile-time TT, so the row guards survive as branches exactly as
    // they do in `ph_matvec`, where `nrows` is a runtime argument.
    int drain, rows; float * drain_out;
    // Diagnostics. `stagger` spins workgroup `blockIdx.x % 16` for that many sleep units at the top
    // of a phase, which is the burst hypothesis as a knob. `utrace` collects wall_clock64 at every
    // unit start of workgroup 0, so the inside of a phase is measured rather than modelled.
    int stagger;
    unsigned long long * utrace; int utrace_n;
};

// A workgroup barrier that orders LDS and nothing else. See the arm 8 note at the top of the file.
__device__ __forceinline__ void sync_lds() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
template <bool LDSBAR> __device__ __forceinline__ void unit_sync() {
    if constexpr (LDSBAR) sync_lds(); else __syncthreads();
}
// `next_unit` with the barrier the arm selects. The atomic, the counter and the LDS slot are the
// engine's; a workgroup still takes whatever unit it is handed.
template <bool LDSBAR> __device__ __forceinline__ int next_unit_arm(Ctx & c, int total) {
    if constexpr (!LDSBAR) return next_unit(c, total);
    unit_sync<LDSBAR>();
    if (threadIdx.x == 0) *c.s_unit = (int) (gridDim.x + atomicAdd(&c.work[c.phase], 1u));
    unit_sync<LDSBAR>();
    const int u = *c.s_unit;
    return u < total ? u : -1;
}

// The address map of one unit, factored out so a warm request and the phase body cannot drift
// apart: every arm below calls this and nothing else computes a weight address.
__device__ __forceinline__ const uint8_t * probe_run(const ProbeArgs & a, int p, int u, int wb) {
    const uint8_t * base = a.w;
    size_t uu;
    if (a.split > 1) {
        // Exactly ph_matvec's part-major address: tile = u % T, part = u / T, and a unit's run
        // starts at (tile * KS + part) * PART blocks.
        const int part = u / a.total_tiles, tile = u - part * a.total_tiles;
        uu = ((size_t) p * a.units + (size_t) tile * a.split + part) % a.span;
    } else if (a.regions) {
        base = a.regions[p % a.nregions];   // one allocation per phase, as the engine stores a tensor
        uu = (size_t) u;
    } else if (a.stride_windows) {
        const unsigned win = (unsigned) ((unsigned long long) p * a.stride_windows % a.nwindows);
        uu = ((size_t) win * a.units + u) % a.span;
    } else {
        uu = ((size_t) p * a.units + u) % a.span;
    }
    return base + (uu * a.nb + wb) * TILE_BLOCK_BYTES;
}

// A request, not a read. The value goes to a register the kernel does not look at until it exits,
// so the compiler's wait for it sits past the barrier (arm 1) or past the drain (arm 2) and the
// bytes travel while the workgroup is doing something else. One `global_load_dwordx4` a lane: the
// 512 bytes of the 896-byte block that `load_block` asks for first on the other side.
__device__ __forceinline__ uint4 probe_warm(const uint8_t * run, int lane) {
    return *(const uint4 *) (run + tile_off_qs_a(lane));
}

// A weight block as one wave lane holds it.
struct Blk { uint4 qa; uint2 qb; unsigned tail; };
__device__ __forceinline__ Blk blk_load(const uint8_t * run, int lane) {
    Blk b;
    b.qa = *(const uint4 *) (run + tile_off_qs_a(lane));
    b.qb = *(const uint2 *) (run + tile_off_qs_b(lane));
    b.tail = *(const unsigned *) (run + tile_off_tail(lane));
    return b;
}

// mv_rows_t's single-row body with a settable lookahead depth. AHEAD = 1 is what the engine ships:
// the wave asks for block b+1 while it works on block b, so the first block of every unit is
// exposed and a unit of PW blocks hides at most PW-1 round trips. AHEAD = 2 keeps two blocks in
// flight. The dot products and the fp32 chain are identical in both; only the request schedule
// moves, which is the whole question.
template <int PW, int AHEAD>
__device__ __forceinline__ float mv_probe(const uint8_t * __restrict__ run, int lane, cptr xq, cfptr xs, cptr xsum) {
    Blk q[AHEAD];
    #pragma unroll
    for (int i = 0; i < AHEAD; i++) q[i] = blk_load(run + min(i, PW - 1) * TILE_BLOCK_BYTES, lane);
    float y = 0.0f;
    #pragma unroll 1
    for (int b = 0; b < PW; b++) {
        Blk nxt = q[AHEAD - 1];
        if (b + AHEAD < PW) nxt = blk_load(run + (b + AHEAD) * TILE_BLOCK_BYTES, lane);
        unsigned tr[32];
        const unsigned dw[6] = { q[0].qa.x, q[0].qa.y, q[0].qa.z, q[0].qa.w, q[0].qb.x, q[0].qb.y };
        #pragma unroll
        for (int d = 0; d < 6; d++) {
            unsigned P0 = dw[d] & 0x00ff00ffu, P1 = (dw[d] >> 8) & 0x00ff00ffu;
            unsigned t0 = peel(P0), t1 = peel(P0), t2 = peel(P0), t3 = peel(P0), t4 = peel_last(P0);
            unsigned u0 = peel(P1), u1 = peel(P1), u2 = peel(P1), u3 = peel(P1), u4 = peel_last(P1);
            tr[4 * d] = t0 | (t1 << 8); tr[4 * d + 1] = t2 | (t3 << 8);
            tr[4 * d + 2] = u0 | (u1 << 8); tr[4 * d + 3] = u2 | (u3 << 8); tr[24 + d] = t4 | (u4 << 8);
        }
        unsigned P = (q[0].tail & 0xffu) | ((q[0].tail << 8) & 0xff0000u);
        unsigned h0 = peel(P), h1 = peel(P), h2 = peel(P), h3 = peel_last(P);
        tr[30] = h0 | (h1 << 8); tr[31] = h2 | (h3 << 8);
        int acc = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) acc = __builtin_amdgcn_sudot4(false, (int) tr[i], true, xq[b * 32 + i], acc, false);
        const float wscale = __half2float(__ushort_as_half((unsigned short) (q[0].tail >> 16)));
        y = fmaf((float) (xsum[b] - acc), wscale * xs[b], y);
        #pragma unroll
        for (int i = 0; i + 1 < AHEAD; i++) q[i] = q[i + 1];
        q[AHEAD - 1] = nxt;
    }
    return y;
}

// One phase of the probe deals `units` units and ends with the engine's grid barrier. The unit body
// is the engine's `mv_rows_t` over PW blocks, addressed exactly as `ph_matvec` addresses it: wave w
// takes blocks [w*PW, (w+1)*PW) of its unit's tile.
template <int PW, int TT = 1, int ARM = 0>
__global__ __launch_bounds__(NT) void probe(ProbeArgs a) {
    __shared__ float lds[LDS_FLOATS];
    __shared__ int s_unit;
    Ctx c{ a.bar, a.work, nullptr, 0u, 0u, false, lds, lds, &s_unit };
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    // The engine reads activations through the scalar cache; a pointer that stays in VGPRs would
    // measure a different machine.
    const int8_t * xq_l = a.xq; asm volatile("" : "+s"(xq_l));
    const float * xs_l = a.xs; asm volatile("" : "+s"(xs_l));
    const int * xsum_l = a.xsum; asm volatile("" : "+s"(xsum_l));
    float acc = 0.0f;
    // The warm requests live here and are read once, past the timed loop: the wait for one lands at
    // the next arm's request, which is a drain or a barrier away.
    uint4 warm{ 0u, 0u, 0u, 0u };
    constexpr bool LDSBAR = (ARM & 8) != 0;
    int tpos = 0;
    for (int p = 0; p < a.phases; p++) {
        if (a.stagger) for (int i = (int) (blockIdx.x & 15u) * a.stagger; i > 0; i--) __builtin_amdgcn_s_sleep(1);
        if (a.mode >= 1) {
            int u = blockIdx.x;
            while (u >= 0 && u < a.units) {
                c.dirty = true;
                if (a.utrace && blockIdx.x == 0 && threadIdx.x == 0 && tpos < a.utrace_n)
                    a.utrace[tpos++] = wall_clock64();
                int unext = -1;
                float y[TT];
                #pragma unroll
                for (int r = 0; r < TT; r++) y[r] = 0.0f;
                const int wb = ((ARM & 4) ? ((wave + (int) blockIdx.x) & (NW - 1)) : wave) * PW;
                if (a.mode >= 2) {
                    // Each phase reads a fresh window of the buffer, so a short phase is cold in the
                    // same way an engine phase is cold: the previous token's copy of these weights
                    // left the cache 30 ms ago.
                    const uint8_t * run = probe_run(a, p, u, wb);
                    if (a.mode == 2) {
                        mv_rows_t<PW, TT, false>(run, lane, (cptr) xq_l + wb * 32, a.nb * 32,
                                                 (cfptr) xs_l + wb, (cptr) xsum_l + wb, a.nb,
                                                 a.rows ? a.rows : TT, y);
                    } else if (a.mode == 3) {
                        acc += mv_probe<PW, 2>(run, lane, (cptr) xq_l + wb * 32, (cfptr) xs_l + wb, (cptr) xsum_l + wb);
                    } else {
                        acc += mv_probe<PW, 3>(run, lane, (cptr) xq_l + wb * 32, (cfptr) xs_l + wb, (cptr) xsum_l + wb);
                    }
                }
                // ARM 2: the deal moves ahead of the drain, so the next unit's first block is
                // travelling while this unit publishes its partials. The atomic is the same atomic
                // on the same counter; a workgroup still takes whatever unit it is handed.
                if constexpr (ARM & 2) {
                    unext = next_unit_arm<LDSBAR>(c, a.units);
                    if (a.mode >= 2 && unext >= 0 && unext < a.units)
                        warm = probe_warm(probe_run(a, p, unext, wb), lane);
                }
                if (a.mode == 2) {
                        if (a.drain) {
                            // ph_matvec's drain, instruction for instruction. drain 3 and 4 are the
                            // SPREAD form: every wave publishes (slot 0 of red[8][TT][32] is
                            // allocated today and never written), and wave w drains row w instead
                            // of wave 0 draining all eight. Same values summed in the same
                            // left-to-right order, so it is bit-identical by construction.
                            float * red = lds;
                            const int nrows = a.rows ? a.rows : TT;
                            if (a.drain >= 3) {
                                #pragma unroll
                                for (int r = 0; r < TT; r++) red[(wave * TT + r) * 32 + lane] = y[r];
                                unit_sync<LDSBAR>();
                                const int tile = a.total_tiles ? u % a.total_tiles : u;
                                #pragma unroll
                                for (int r = 0; r < TT; r++) {
                                    if (r % NW == wave && r < nrows) {
                                        float v = red[r * 32 + lane];
                                        #pragma unroll
                                        for (int w = 1; w < NW; w++) v += red[(w * TT + r) * 32 + lane];
                                        float * o = a.drain_out + (size_t) r * 5120 + (size_t) (tile % 160) * 32 + lane;
                                        if (a.drain > 3) atomicAdd(o, v); else *o = v;
                                    }
                                }
                                unit_sync<LDSBAR>();
                            } else {
                            if (wave > 0) {
                                #pragma unroll
                                for (int r = 0; r < TT; r++) red[(wave * TT + r) * 32 + lane] = y[r];
                            }
                            unit_sync<LDSBAR>();
                            if (wave == 0) {
                                const int tile = a.total_tiles ? u % a.total_tiles : u;
                                #pragma unroll
                                for (int r = 0; r < TT; r++) {
                                    if (r < nrows) {
                                        float v = y[r];
                                        #pragma unroll
                                        for (int w = 1; w < NW; w++) v += red[(w * TT + r) * 32 + lane];
                                        float * o = a.drain_out + (size_t) r * 5120 + (size_t) (tile % 160) * 32 + lane;
                                        if (a.drain > 1) atomicAdd(o, v); else *o = v;
                                    }
                                }
                            }
                            unit_sync<LDSBAR>();
                            }
                        } else {
                            #pragma unroll
                            for (int r = 0; r < TT; r++) acc += y[r];
                        }
                }
                u = (ARM & 2) ? unext : next_unit_arm<LDSBAR>(c, a.units);
            }
            if (a.utrace && blockIdx.x == 0 && threadIdx.x == 0 && tpos < a.utrace_n)
                a.utrace[tpos++] = wall_clock64() | (1ull << 63);   // phase end marker
        }
        // ARM 1: the first unit of the next phase is `blockIdx.x` by construction, so ask for its
        // weights before waiting at the barrier that separates the two phases.
        if constexpr (ARM & 1) {
            if (a.mode >= 2 && p + 1 < a.phases && (int) blockIdx.x < a.units)
                warm = probe_warm(probe_run(a, p + 1, (int) blockIdx.x, wave * PW), lane);
        }
        grid_sync(c);
        end_phase(c);
#if COOP_GAP
        if (a.gap_units >= 0) {
            for (int u = blockIdx.x; u >= 0 && u < a.gap_units; u = next_unit(c, a.gap_units)) {
                c.dirty = true;
                const float * src = a.gap_in + (size_t) (u & 31) * 256;
                float s = 0.0f;
                for (int i = threadIdx.x; i < 256; i += NT) s += src[i];
                a.gap_out[(size_t) u * 64 + (threadIdx.x & 63)] = s;
            }
            grid_sync(c);
            end_phase(c);
        }
#endif
    }
    // Keep the dots and the warm requests live without adding a store to the timed loop.
    if (acc == 12345.678f) a.out[blockIdx.x * NT + threadIdx.x] = acc + (float) (warm.x + warm.y + warm.z + warm.w);
}

using ProbeFn = void (*)(ProbeArgs);
// Arm 0 carries every block width this probe has ever been asked for. The prefetch arms carry the
// engine's own: 3 is the K-split pair, 5 the 40-block projections, 8 and 17 the 136-block down
// stage, 40 a whole tile in one unit. Each extra (pw, tt, arm) is another instantiation of a body
// that already compiles sixty of them, and a build that does not fit one tool call is a build
// nobody re-measures.
template <int TT, int ARM> ProbeFn probe_fn_tt(int pw) {
    if constexpr (ARM == 0) {
        switch (pw) {
            case 1: return probe<1, TT, 0>; case 2: return probe<2, TT, 0>; case 3: return probe<3, TT, 0>;
            case 4: return probe<4, TT, 0>; case 5: return probe<5, TT, 0>; case 6: return probe<6, TT, 0>;
            case 8: return probe<8, TT, 0>; case 9: return probe<9, TT, 0>; case 10: return probe<10, TT, 0>;
            case 17: return probe<17, TT, 0>; case 20: return probe<20, TT, 0>; case 34: return probe<34, TT, 0>;
            case 40: return probe<40, TT, 0>; case 68: return probe<68, TT, 0>; case 136: return probe<136, TT, 0>;
            default: return nullptr;
        }
    } else if constexpr (TT == 1 || TT == 8) {   // the prefetch and rotation arms, engine widths only
        switch (pw) {
            case 3: return probe<3, TT, ARM>; case 5: return probe<5, TT, ARM>; case 8: return probe<8, TT, ARM>;
            case 17: return probe<17, TT, ARM>; case 40: return probe<40, TT, ARM>;
            default: return nullptr;
        }
    } else return nullptr;
}
// 1 warms the next phase across the grid barrier, 2 deals the next unit before the drain and warms
// it there, 8 makes the unit-boundary barriers LDS-scoped so a request can survive them, and the
// sums are the compositions worth measuring.
template <int TT> ProbeFn probe_fn_arm(int pw, int arm) {
    switch (arm) {
        case 0: return probe_fn_tt<TT, 0>(pw);
        case 1: return probe_fn_tt<TT, 1>(pw);
        case 2: return probe_fn_tt<TT, 2>(pw);
        case 4: return probe_fn_tt<TT, 4>(pw);
        case 8: return probe_fn_tt<TT, 8>(pw);
        case 10: return probe_fn_tt<TT, 10>(pw);
        case 11: return probe_fn_tt<TT, 11>(pw);
        default: return nullptr;
    }
}
// The deployed row counts: one for a single-token step, eight for a drafted verify pass and the
// two intermediate widths a batched drafted step would hand it.
ProbeFn probe_fn(int pw, int tt, int arm) {
    switch (tt) {
        case 1: return probe_fn_arm<1>(pw, arm);
        case 2: return probe_fn_arm<2>(pw, arm);
        case 4: return probe_fn_arm<4>(pw, arm);
        case 8: return probe_fn_arm<8>(pw, arm);
        default: return nullptr;
    }
}

std::vector<int> parse_ints(const char * s) {
    std::vector<int> v; const char * p = s;
    while (*p) { v.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p == ',') p++; }
    return v;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char ** argv) {
    int mode = 2, grid = 60, phases = 64, rounds = 5, phase_units = 0, tt = 1, exact_units = 0;
    int regions = 0, scatter = 0, gap_units = -1, no_stream = 0, split = 0, drain = 0, runtime_rows = 0;
    int stagger = 0, trace_units = 0;
    size_t mib = 2048;                       // weight bytes the buffer holds
    std::vector<int> pws{ 3, 5, 8, 17, 40 };
    std::vector<int> arms{ 0 };              // prefetch arms, interleaved per shape in one process
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--mode") mode = atoi(next());   // 2 ships, 3 two blocks ahead, 4 three
        else if (a == "--grid") grid = atoi(next());
        else if (a == "--phases") phases = atoi(next());
        else if (a == "--rounds") rounds = atoi(next());
        else if (a == "--bytes") mib = (size_t) atoll(next());
        else if (a == "--pw") pws = parse_ints(next());
        // Units per phase. The engine's matvec phases are 320 to 1088 units - five to eighteen grid
        // rounds - and a probe that deals tens of thousands measures a machine in steady state that
        // no phase of the engine ever reaches.
        else if (a == "--phase-units") phase_units = atoi(next());
        else if (a == "--tt") tt = atoi(next());
        else if (a == "--exact-units") exact_units = 1;
        // Where the bytes live. `--regions N` gives every phase its own hipMalloc of exactly one
        // phase's weight bytes, which is how the engine stores a tensor; `--scatter` keeps the one
        // allocation and jumps the window instead. Both need `--phase-units`, because a phase's
        // length is what sets its region size.
        else if (a == "--regions") regions = atoi(next());
        else if (a == "--scatter") scatter = 1;
        else if (a == "--gap-units") { gap_units = atoi(next()); if (!COOP_GAP && gap_units >= 0) { printf("built without COOP_GAP=1\n"); return 1; } }
        else if (a == "--no-stream") no_stream = 1;   // gap phases only: the control that prices them
        // --split KS reads the units in the engine's part-major K-split geometry instead of
        // contiguously: same bytes, same unit count, same unit size, only the stream stride moves.
        else if (a == "--split") split = atoi(next());
        // The prefetch axis: 0 the deployed schedule, 1 warm the next phase's first unit before the
        // barrier, 2 deal the next unit before the drain and ask for its first block, 3 both.
        else if (a == "--arm") arms = parse_ints(next());
        // Diagnostics, both arm-independent: a per-workgroup delay at the phase start, and the
        // per-unit timeline of workgroup 0 for the first few phases.
        else if (a == "--stagger") stagger = atoi(next());
        else if (a == "--trace-units") trace_units = atoi(next());
        else if (a == "--drain") drain = atoi(next());          // 1 store, 2 atomicAdd
        else if (a == "--runtime-rows") runtime_rows = 1;
        else { printf("usage: coop_cost [--mode 0|1|2] [--grid 60] [--phases 64] [--rounds 5] [--bytes MiB] [--pw 3,5,8] [--tt 1|2|4|8] [--exact-units] [--regions N] [--scatter]\n"); return 1; }
    }

    hipDeviceProp_t prop{}; CK(hipGetDeviceProperties(&prop, 0));
    const size_t stream_bytes = mib << 20;
    uint8_t * w = nullptr; CK(hipMalloc(&w, stream_bytes));
    CK(hipMemset(w, 0x24, stream_bytes));    // a valid five-trit byte pattern; values do not matter
    // Activations: one row, wide enough for the largest block count any arm walks.
    const int max_blocks = *std::max_element(pws.begin(), pws.end()) * NW;
    int8_t * xq = nullptr; float * xs = nullptr; int * xsum = nullptr; float * out = nullptr;
    CK(hipMalloc(&xq, (size_t) max_blocks * 128 * tt));
    CK(hipMalloc(&xs, (size_t) max_blocks * 4 * tt));
    CK(hipMalloc(&xsum, (size_t) max_blocks * 4 * tt));
    CK(hipMalloc(&out, (size_t) grid * NT * 4));
    CK(hipMemset(xq, 1, (size_t) max_blocks * 128 * tt));
    CK(hipMemset(xs, 0, (size_t) max_blocks * 4 * tt));
    CK(hipMemset(xsum, 0, (size_t) max_blocks * 4 * tt));

    unsigned * bar = nullptr, * work = nullptr;
    CK(hipMalloc(&bar, 4));
    CK(hipMalloc(&work, (size_t) (2 * phases + 2) * 4));
    float * gap_in = nullptr, * gap_out = nullptr;
    CK(hipMalloc(&gap_in, 32 * 256 * 4));
    CK(hipMalloc(&gap_out, (size_t) std::max(gap_units, 1) * 64 * 4));
    CK(hipMemset(gap_in, 0, 32 * 256 * 4));
    float * drain_out = nullptr; CK(hipMalloc(&drain_out, (size_t) 8 * 5120 * 4)); CK(hipMemset(drain_out, 0, (size_t) 8 * 5120 * 4));
    unsigned long long * utrace = nullptr;
    const int utrace_n = trace_units > 0 ? trace_units : 0;
    if (utrace_n) { CK(hipMalloc(&utrace, (size_t) utrace_n * 8)); CK(hipMemset(utrace, 0, (size_t) utrace_n * 8)); }

    printf("gfx%s  grid %d  phases %d  rounds %d  stream %zu MiB  mode %d  tt %d%s%s%s%s%s\n",
           prop.gcnArchName, grid, phases, rounds, mib, mode, tt, exact_units ? "  exact-units" : "",
           regions > 1 ? "  separate-regions" : (scatter ? "  scattered-windows" : ""),
           split > 1 ? "  part-major-split" : "", drain ? (drain > 1 ? "  drain-atomic" : "  drain-store") : "",
           runtime_rows ? "  runtime-rows" : "");
    if (mode == 0) printf("%6s %12s %12s\n", "arm", "ms/round", "ns/barrier");
    else if (mode == 1) printf("%6s %10s %12s %12s\n", "units", "ms/round", "ns/unit", "ns/unit/wg");
    else printf("%6s %8s %5s %10s %12s %12s %10s\n", "pw", "units", "arm", "ms/round", "GB/s", "ns/unit", "ns/unit/wg");

    for (int pw : pws) {
        if (!probe_fn(pw, tt, arms[0])) { printf("no instantiation for pw %d tt %d\n", pw, tt); continue; }
        const int nb = pw * NW;                                  // blocks per tile, one unit per tile
        const size_t bytes_per_unit = (size_t) nb * TILE_BLOCK_BYTES;
        const size_t span = std::max<size_t>(1, stream_bytes / bytes_per_unit);
        int units = mode >= 2 ? (int) span : 4096 / pw;
        if (mode >= 2 && phase_units > 0) units = phase_units;
        if (mode == 0) units = 0;
        // Round the unit count *down* to a whole number of grid rounds: quantisation is not the
        // variable being swept, and a rounded-up count would read past the streamed buffer.
        // Under --exact-units the asked-for count stands, tail and all, and the window wraps.
        if (!exact_units) units = (units / grid) * grid;
        if (mode >= 2 && units == 0) { printf("pw %d needs more than %zu MiB\n", pw, mib); continue; }
        // The gap-only control still needs a legal unit count for the report line.

        // One allocation per phase, each exactly this phase's weight bytes. The engine's own
        // `mv_gate_up` reads two 19.5 MB allocations per launch and never the same ones twice in a
        // pass, so the count here only has to be large enough that a region is not still resident
        // when it comes round again.
        std::vector<uint8_t *> hregions;
        const uint8_t ** dregions = nullptr;
        if (regions > 1) {
            const size_t rbytes = (size_t) units * bytes_per_unit;
            for (int i = 0; i < regions; i++) {
                uint8_t * r = nullptr; CK(hipMalloc(&r, rbytes)); CK(hipMemset(r, 0x24, rbytes));
                hregions.push_back(r);
            }
            CK(hipMalloc(&dregions, sizeof(uint8_t *) * regions));
            CK(hipMemcpy(dregions, hregions.data(), sizeof(uint8_t *) * regions, hipMemcpyHostToDevice));
        }
        // A permutation of the windows of one allocation: stride coprime to the window count, so
        // every phase lands somewhere far from the last one and no window repeats until all have
        // been read. This changes nothing about how the bytes were allocated.
        unsigned nwindows = (unsigned) std::max<size_t>(1, span / (size_t) units), stride_windows = 0;
        if (scatter && nwindows > 2) {
            stride_windows = nwindows / 2 + 1;
            auto gcd = [](unsigned x, unsigned y) { while (y) { unsigned t = x % y; x = y; y = t; } return x; };
            while (gcd(stride_windows, nwindows) != 1) stride_windows++;
        }
        if (split > 1 && units % split) { printf("--split %d needs a unit count it divides\n", split); continue; }
        ProbeArgs a{ bar, work, w, xq, xs, xsum, out, phases, no_stream ? 0 : units, nb, mode, span,
                     dregions, regions, stride_windows, nwindows, gap_units, gap_in, gap_out,
                     split, split > 1 ? units / split : units, drain, runtime_rows ? tt : 0, drain_out,
                     stagger, utrace, utrace_n };

        for (int arm : arms) {
            ProbeFn fn = probe_fn(pw, tt, arm);
            if (!fn) { printf("no instantiation for pw %d tt %d arm %d\n", pw, tt, arm); continue; }
            double best = 1e30;
            for (int r = 0; r < rounds; r++) {
                CK(hipMemset(bar, 0, 4));
                CK(hipMemset(work, 0, (size_t) (2 * phases + 2) * 4));
                void * args[] = { (void *) &a };
                CK(hipDeviceSynchronize());
                const double t0 = now_ms();
                CK(hipLaunchCooperativeKernel((const void *) fn, dim3(grid), dim3(NT), args, 0, nullptr));
                CK(hipDeviceSynchronize());
                const double ms = now_ms() - t0;
                best = std::min(best, ms);
            }
            const double per_phase_ms = best / phases;
            const double gb = (double) (no_stream ? 0 : units) * bytes_per_unit / 1e9;
            const double rounds_of_units = units > 0 ? (double) ((units + grid - 1) / grid) : 0;
            if (mode == 0) printf("%6s %12.4f %12.1f\n", "-", best / phases, per_phase_ms * 1e6);
            else if (mode == 1)
                printf("%6d %10.4f %12.1f %12.1f\n", units, per_phase_ms,
                       per_phase_ms * 1e6 / units, per_phase_ms * 1e6 / rounds_of_units);
            else
                printf("%6d %8d %5d %10.4f %12.1f %12.1f %10.1f\n", pw, units, arm, per_phase_ms,
                       gb / (per_phase_ms / 1e3), per_phase_ms * 1e6 / units,
                       per_phase_ms * 1e6 / rounds_of_units);
            fflush(stdout);
            // The inside of a phase, from workgroup 0's own clock: unit starts and phase ends, in
            // microseconds at the device's 100 MHz reference. A ramp shows up as a long first unit;
            // a flat profile says the short-phase deficit is not a warm-up at all.
            if (utrace_n) {
                std::vector<unsigned long long> h((size_t) utrace_n);
                CK(hipMemcpy(h.data(), utrace, (size_t) utrace_n * 8, hipMemcpyDeviceToHost));
                const double tick_us = 1.0 / 100.0;   // wall_clock64 counts the 100 MHz constant clock
                printf("       trace pw %d arm %d, workgroup 0, us per unit (| = phase end):\n       ", pw, arm);
                int shown = 0;
                for (int i = 1; i < utrace_n && h[i] && shown < 64; i++) {
                    const unsigned long long prev = h[i - 1] & ~(1ull << 63), cur = h[i] & ~(1ull << 63);
                    printf("%.2f%s ", (double) (cur - prev) * tick_us, (h[i] >> 63) ? "|" : "");
                    if (++shown % 16 == 0) printf("\n       ");
                }
                printf("\n");
                CK(hipMemset(utrace, 0, (size_t) utrace_n * 8));
            }
        }
        for (uint8_t * r : hregions) CK(hipFree(r));
        if (dregions) CK(hipFree(dregions));
        if (mode <= 1) break;                 // modes 0 and 1 do not depend on pw
    }
    return 0;
}
