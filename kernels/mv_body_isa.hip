// Census harness for the dot4 matvec body. Not linked into the engine; `tools/isa_loop_count.py`
// reads the assembly of one instantiation of `mv_rows_t` without compiling the whole persistent
// kernel, so an operand-map change can be priced in seconds.
//
//   hipcc --offload-arch=gfx1151 --cuda-device-only -S -O3 -Isrc -Ikernels kernels/mv_body_isa.hip -o /tmp/mv.s
//   tools/isa_loop_count.py /tmp/mv.s ILi5ELi8ELi4EE      # PW = 5, TT = 8, ACC = 4
//   tools/kernel_resources.py kernels/mv_body_isa.hip k_mv_census
//
// Eight seconds a sweep, against about forty for the persistent kernel and a GPU lock for a panel.
#include "device.hpp"

using namespace halo;

__device__ __forceinline__ cptr  xptr(const int8_t * p) { asm volatile("" : "+s"(p)); return (cptr) p; }
__device__ __forceinline__ cfptr sptr(const float * p)  { asm volatile("" : "+s"(p)); return (cfptr) p; }
__device__ __forceinline__ cptr  iptr(const int * p)    { asm volatile("" : "+s"(p)); return (cptr) p; }

template <int PW, int TT, int ACC = 1>
__global__ void __launch_bounds__(256) k_mv_census(const uint8_t * __restrict__ w, const int8_t * xq,
                                                  const float * xs, const int * xsum, float * out, int nrows) {
    const int lane = threadIdx.x & 31;
    float y[TT];
    #pragma unroll
    for (int r = 0; r < TT; r++) y[r] = 0.0f;
    const uint8_t * run = w + (size_t) blockIdx.x * PW * TILE_BLOCK_BYTES;
    mv_rows_t<PW, TT, false, ACC>(run, lane, xptr(xq), 40 * 32, sptr(xs), iptr(xsum), 40, nrows, y);
    #pragma unroll
    for (int r = 0; r < TT; r++) out[(size_t) blockIdx.x * TT + r] = y[r];
}

template __global__ void k_mv_census<5, 1, 1>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 1, 2>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 1, 4>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 1, 8>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 2, 1>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 2, 2>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 2, 4>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 2, 8>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 4, 1>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 4, 2>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 4, 4>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 4, 8>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 8, 1>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 8, 2>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 8, 4>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
template __global__ void k_mv_census<5, 8, 8>(const uint8_t *, const int8_t *, const float *, const int *, float *, int);
