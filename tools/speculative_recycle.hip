#include "speculative_recycle.h"
#include "../kernels/device.hpp"
#include <stdexcept>
#include <string>

namespace halo {
namespace {

constexpr int RECYCLE_THREADS = 256;
constexpr int RECYCLE_WAVES = RECYCLE_THREADS / 32;
static_assert(RECYCLE_THREADS == DF_RANK);

// This is the original selector's arithmetic and winner rule, with a corrected
// predecessor and a suffix origin. A target pass may overwrite its own scratch,
// but must not overwrite the drafter's topi/topv/hproj_out buffers.
__global__ void __launch_bounds__(RECYCLE_THREADS) recycle_suffix(DflashParams M, int begin_pos, int predecessor) {
    __shared__ int prev;
    __shared__ float part[RECYCLE_WAVES][DF_TOPK];
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    if (t == 0) prev = predecessor;
    __syncthreads();
    for (int pos = begin_pos; pos < DF_BLOCK; pos++) {
        const int pred = prev;
        const float pr = bf16_to_f32(M.w.pred_cb[(size_t) pred * DF_RANK + t]) * M.hproj_out[(size_t) pos * DF_RANK + t];
        float sc[DF_TOPK];
        #pragma unroll
        for (int k = 0; k < DF_TOPK; k++) {
            const int cand = M.topi[pos * DF_TOPK + k];
            float s = pr * bf16_to_f32(M.w.succ_cb[(size_t) cand * DF_RANK + t]);
            s = warp_sum(s);
            sc[k] = s;
        }
        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < DF_TOPK; k++) part[wave][k] = sc[k];
        }
        __syncthreads();
        if (t == 0) {
            float best = -INFINITY; int bk = 0;
            for (int k = 0; k < DF_TOPK; k++) {
                float s = M.topv[pos * DF_TOPK + k];
                for (int w = 0; w < RECYCLE_WAVES; w++) s += part[w][k];
                if (s > best) { best = s; bk = k; }
            }
            prev = M.topi[pos * DF_TOPK + bk];
            M.draft_out[pos - begin_pos] = prev;
        }
        __syncthreads();
    }
}

} // namespace

void launch_speculative_recycle(const DflashParams & block, int begin_pos, int predecessor, hipStream_t stream) {
    if (begin_pos < 1 || begin_pos >= DF_BLOCK || predecessor < 0 || predecessor >= VOCAB)
        throw std::invalid_argument("speculative recycle: invalid suffix position or predecessor token");
    if (!block.w.pred_cb || !block.w.succ_cb || !block.hproj_out || !block.topi || !block.topv || !block.draft_out)
        throw std::invalid_argument("speculative recycle: missing saved block buffers");
    hipLaunchKernelGGL(recycle_suffix, dim3(1), dim3(RECYCLE_THREADS), 0, stream, block, begin_pos, predecessor);
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) throw std::runtime_error(std::string("speculative recycle: ") + hipGetErrorString(error));
}

} // namespace halo
