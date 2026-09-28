// Delivering the A4 projection's activation fragments through LDS instead of sixteen global loads.
//
// A workgroup of WV waves owns WV different weight row tiles and ONE token group, so every wave
// loads the same (slice, token) fragments out of the B image: 16 `global_load_b64` per wave per
// 128-block at two token tiles, 64 per workgroup, for the same 2 kB.
//
// The ablation ladder in ffn_batch.hip says what that costs and what it does not. Pinning the B
// address so every block reads the same 2 kB - perfect locality, identical request count - is an
// EXACT NULL (32.36/32.62 ms against 32.58/32.59 deployed). Collapsing the block's sixteen loads
// into one through common subexpressions - identical bytes on the weight stream, identical peel,
// identical matrix work, fifteen fewer requests - is -11.6% of the phase (28.77/28.86). And
// removing sixteen `ds_bpermute` per block is +2.4%, so the LDS pipe is free in this kernel while
// the vector memory pipe is charged by the instruction. docs/ffn-b-operand.md has the panel.
//
// So this stage buys the request cut without changing a value: the same 8-byte fragments reach the
// same lanes in the same order, through LDS instead of through L1. Bit-identical by construction.
//
// Two arms, because the shared one needs a barrier and the private one does not:
//
//   SHARED  one copy per workgroup. Thread `tid` stages EPT contiguous elements, the block's 2 kB
//           costs the workgroup ONE `global_load_b128` per thread, and every wave reads all of it.
//           One `s_barrier` per block, double-buffered so the barrier is the only synchronisation.
//   WAVE    one copy per wave. No barrier and no cross-wave ordering at all, at WV times the LDS
//           and WV times the staging requests - still 4 per wave-block against 16.
//
// The stage is one block deep and its global load for block b+1 is issued at the top of block b,
// which is the same cover the weight cursor gets (ffn_dense_stream.hpp).
#pragma once
#include <hip/hip_runtime.h>

namespace halo {
namespace ffnb {

enum BStageMode { BSTAGE_OFF = 0, BSTAGE_SHARED = 1, BSTAGE_WAVE = 2 };

// The fragments of one 128-block, in the order the block body reads them: element
// (slice s, token tile t, column col) at s * (TT * 16) + t * 16 + col. A read is one ds_read_b64
// whose sixteen lanes cover 128 contiguous bytes, so it is bank-conflict free.
template <int MODE, typename BF, int SLICES, int TT, int WV>
struct BStage {
    static constexpr int ELEMS = SLICES * TT * 16;             // 8-byte fragments per block
    static constexpr int COPIES = MODE == BSTAGE_WAVE ? WV : 1;
    static constexpr int THREADS = MODE == BSTAGE_WAVE ? 32 : WV * 32;
    static constexpr int EPT = ELEMS / THREADS;                // elements one thread stages
    // A wave's own copy needs no second buffer: the wave writes block b+1 after its own reads of
    // block b, in its own program order, and nothing else reads it. The shared copy does, because
    // another wave may still be reading when this one publishes. One buffer is 8 kB a workgroup
    // instead of 16, which is what keeps seven of them resident on a WGP's 128 kB.
    static constexpr int BUFS = MODE == BSTAGE_WAVE ? 1 : 2;
    static constexpr int BYTES = BUFS * COPIES * ELEMS * (int) sizeof(BF);
    static_assert(ELEMS % THREADS == 0, "a thread stages a whole number of fragments");
    static_assert(EPT * (int) sizeof(BF) <= 16 || EPT * (int) sizeof(BF) % 16 == 0,
                  "a thread's run of fragments is one or more 16-byte loads");

    BF * lds;                  // 2 * COPIES * ELEMS fragments
    const BF * src;            // B image, block 0 of this token group
    int npad;                  // slice stride in fragments
    int e0;                    // this thread's first element
    int s0;                    // its slice
    int j0;                    // its token inside the group
    BF pend[EPT];              // block b+1's fragments, in flight

    __device__ __forceinline__ BStage(BF * lds_, const BF * src_, int npad_, int tid, int wave)
        : lds(lds_ + (MODE == BSTAGE_WAVE ? (size_t) wave * BUFS * ELEMS : 0)), src(src_), npad(npad_) {
        const int t = MODE == BSTAGE_WAVE ? (tid & 31) : tid;
        e0 = t * EPT;
        s0 = e0 / (TT * 16);
        j0 = e0 - s0 * (TT * 16);
    }

    // Request block `blk`'s share of this thread's fragments. Wave-uniform except for the lane
    // offset, so each is a scalar base plus a 32-bit offset, exactly like the block body's loads.
    __device__ __forceinline__ void fetch(int blk) {
        const BF * p = src + (size_t) (blk * SLICES + s0) * npad + j0;
#pragma unroll
        for (int i = 0; i < EPT; ++i) pend[i] = p[i];
    }

    // Publish what `fetch` brought. The caller owns the barrier that makes it visible.
    __device__ __forceinline__ void commit(int blk) {
        BF * d = lds + (size_t) (BUFS == 1 ? 0 : blk & 1) * ELEMS + e0;
#pragma unroll
        for (int i = 0; i < EPT; ++i) d[i] = pend[i];
    }

    __device__ __forceinline__ void sync() const {
        if constexpr (MODE == BSTAGE_SHARED) __syncthreads();
    }

    // Fill block 0 and make it readable.
    __device__ __forceinline__ void prime() { fetch(0); commit(0); sync(); }

    __device__ __forceinline__ BF read(int blk, int slice, int t, int col) const {
        return lds[(size_t) (BUFS == 1 ? 0 : blk & 1) * ELEMS + slice * (TT * 16) + t * 16 + col];
    }
};

} // namespace ffnb
} // namespace halo
