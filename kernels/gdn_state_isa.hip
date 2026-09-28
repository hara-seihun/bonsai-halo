// Census harness for the state codec's load and store loops. Not linked into the engine.
//
// `resident_state` is a 75-second translation unit and the only thing a coordinate change moves in
// it is these two loops, so they are instantiated here on their own with the format as a
// compile-time constant. Eight seconds an arm, no GPU lock, no model:
//
//   hipcc --offload-arch=gfx1151 --cuda-device-only -S -O3 -Isrc -Ikernels \
//         kernels/gdn_state_isa.hip -o /tmp/gdn.s
//   tools/gdn_state_isa.py /tmp/gdn.s
//
// The three arms are the accepted grouped coordinate (`i8`), the row-major one it replaced
// (`i8c`), and the row-major order with the full-exec scale store (`i8s`), which separates the
// store predicate from the request width. What to read: the per-wave request count and width on
// the state itself (four `global_load_b32` against one `global_load_b128`), and how many basic
// blocks the store loop is divided into, which is what a divergent scale store costs the R
// independent `warp_max` butterflies.
#include "device.hpp"
#include "gdn_state_codec.hpp"

using namespace halo;

// The base has to stay a global-address-space pointer derived from a kernel argument, exactly as
// `P.gdn_state + ((size_t) S.slot * 48 + gi) * GDN_STATE_FLOATS` is in the real kernel. Launder it
// through an `asm` constraint and the compiler flattens it to a generic pointer, every access
// becomes `flat_load_b32` with its own 64-bit vector address, and the load/store merger can no
// longer relate two offsets - which looks exactly like a coordinate that failed to widen.

template <int FMT, int SPLIT>
__global__ void __launch_bounds__(256) k_state_census(float * __restrict__ gdn_state, int h, float mul) {
    constexpr int R = 16 / SPLIT, RPU = 128 / SPLIT;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, part = blockIdx.x & (SPLIT - 1);
    float * region = gdn_state + (size_t) (blockIdx.x >> 4) * GDN_STATE_FLOATS;
    float * M = region + (size_t) h * 128 * 128;
    float m[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const GdnQuad q = gdn_state_load_row(FMT, M, region, h, part * RPU + wave + 8 * r, lane);
#pragma unroll
        for (int s = 0; s < 4; ++s) m[r][s] = q.v[s];
    }
    // Stands in for the token loop: one dependent multiply per element, so the stores cannot be
    // folded into the loads and the registers stay live across both loops.
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int s = 0; s < 4; ++s) m[r][s] *= mul;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        GdnQuad q;
#pragma unroll
        for (int s = 0; s < 4; ++s) q.v[s] = m[r][s];
        gdn_state_store_row(FMT, M, region, h, part * RPU + wave + 8 * r, lane, q);
    }
}

template __global__ void k_state_census<GDN_STATE_I8C, 4>(float *, int, float);
template __global__ void k_state_census<GDN_STATE_I8S, 4>(float *, int, float);
template __global__ void k_state_census<GDN_STATE_I8, 4>(float *, int, float);
template __global__ void k_state_census<GDN_STATE_I8C, 8>(float *, int, float);
template __global__ void k_state_census<GDN_STATE_I8, 8>(float *, int, float);
template __global__ void k_state_census<GDN_STATE_I8C, 16>(float *, int, float);
template __global__ void k_state_census<GDN_STATE_I8, 16>(float *, int, float);
template __global__ void k_state_census<GDN_STATE_F32, 4>(float *, int, float);
