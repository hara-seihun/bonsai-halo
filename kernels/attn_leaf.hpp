// The score unit's summation tree, written so that one lane can hold a whole key.
//
// A 256-dimension dot product in this engine is thirty-two eight-dimension leaves, each an
// eight-deep `fmaf` chain from `0.0f`. `warp_sum` spreads one leaf per lane and combines them with
// `xor 1`, `xor 2`, `row_ror 4`, `row_ror 8` and `permlanex16`, and only lane 0's result is stored.
// Writing that same combination inside ONE lane produces the same float, bit for bit, with no
// cross-lane traffic at all - which is why the reduction can be deleted rather than made cheaper.
//
// THE ORDER IS NOT THE INDEX ORDER, and assuming it was would have shipped a silent rounding
// change. `row_ror(N)` makes lane n read lane `((n % 16) - N) mod 16`, so within a row of sixteen
// lanes 0 accumulates its own quad, then the quad at -4 (quad 3), then lane 8's partial, which is
// quad 2 plus the quad at -4 from there (quad 1):
//
//     row total = (Q0 + Q3) + (Q2 + Q1),  Qk = (s(4k) + s(4k+1)) + (s(4k+2) + s(4k+3))
//
// and `permlanex16` adds row 1's identically shaped total. So the tree is balanced, its leaves and
// its quads are contiguous, and the four quads of a row are paired 0-3 and 2-1. `leaf_of_pos` is
// that relabeling: it turns a tree position into the leaf the hardware puts there. The rotation
// direction was read from the DPP definition and then measured against the real `warp_sum` by
// `kernels/attn_tree_check`, because a reduction's association is a hardware fact, not an
// algebraic one.
//
// `LeafTree<NL, NR>::run` evaluates `NL` leaves for `NR` query rows through a leaf functor that
// receives a compile-time tree position, so every address and every combination folds at compile
// time and the levels stay in registers. Nothing here reassociates: each combination is a plain
// add of two subtree values, never a multiply the compiler could contract into an `fma`.
#pragma once
#include <type_traits>

namespace halo {

// Tree position -> leaf index. Identity inside a quad, {0,3,2,1} across the quads of a row of 16.
__host__ __device__ constexpr int leaf_of_pos(int p) {
    return (p & ~15) | ((((4 - ((p >> 2) & 3)) & 3) << 2) | (p & 3));
}

template <int NL, int NR, int J0 = 0>
struct LeafTree {
    template <class LEAF>
    static __device__ __forceinline__ void run(LEAF & leaf, float * out) {
        float lo[NR], hi[NR];
        LeafTree<NL / 2, NR, J0>::run(leaf, lo);
        LeafTree<NL / 2, NR, J0 + NL / 2>::run(leaf, hi);
        #pragma unroll
        for (int i = 0; i < NR; i++) out[i] = lo[i] + hi[i];
    }
};

template <int NR, int J0>
struct LeafTree<1, NR, J0> {
    template <class LEAF>
    static __device__ __forceinline__ void run(LEAF & leaf, float * out) {
        leaf(std::integral_constant<int, leaf_of_pos(J0)>{}, out);
    }
};

} // namespace halo
