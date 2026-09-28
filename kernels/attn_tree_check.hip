// Does one lane reproduce a 32-lane warp_sum, bit for bit?
//
// The score unit's new shape gives a whole key to one lane and deletes the cross-lane reduction.
// That is only free if the value it produces is the same float, and a reduction's association is a
// hardware fact: `row_ror` rotates, so lane 0 does not accumulate the quads of its row in index
// order. This probe answers it by running the engine's own `warp_sum` against `LeafTree` on the
// same random data, and by also running the tree WITHOUT the relabeling so the difference between
// the two is on the record rather than assumed.
//
//   make kernels/attn_tree_check && kernels/attn_tree_check [trials]
//
// It needs a GPU but no model, no lock and under a second.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include "halo_kernels.h"
#include "device.hpp"
#include "attn_leaf.hpp"

using namespace halo;

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("hip error %s at %d\n", hipGetErrorString(e_), __LINE__); return 2; } } while (0)

namespace {

constexpr int DIM = HD; // 256

__device__ __forceinline__ void leaf_dims(const float * q, const __half * k, int j, float * out) {
    const uint4 raw = *(const uint4 *) (k + (size_t) j * KLEAF);
    const __half2 * h2 = (const __half2 *) &raw;
    float kf[KLEAF];
    #pragma unroll
    for (int e = 0; e < KLEAF / 2; e++) { float2 f = __half22float2(h2[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
    float s = 0.0f;
    #pragma unroll
    for (int e = 0; e < KLEAF; e++) s = fmaf(q[j * KLEAF + e], kf[e], s);
    out[0] = s;
}

// One block of 32 threads per trial. Lane 0 writes three floats: the deployed cross-lane value,
// the in-lane tree with the hardware's quad order, and the in-lane tree in plain index order.
__global__ void __launch_bounds__(32) k_tree_check(const float * __restrict__ q,
                                                   const __half * __restrict__ k,
                                                   float * __restrict__ out, int trials) {
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x; t < trials; t += gridDim.x) {
        const float * qv = q + (size_t) t * DIM;
        const __half * kv = k + (size_t) t * DIM;

        float s[1];
        leaf_dims(qv, kv, lane, s);       // the deployed shape: lane l owns leaf l
        const float deployed = warp_sum(s[0]);

        float tree[1];
        auto leaf = [&](auto J, float * o) { leaf_dims(qv, kv, decltype(J)::value, o); };
        LeafTree<DIM / KLEAF, 1>::run(leaf, tree);

        float plain[1];
        // leaf_of_pos is an involution, so applying it again cancels the relabeling and leaves the
        // naive index-ordered tree - the one an algebraic argument would have produced.
        auto leaf_id = [&](auto J, float * o) { leaf_dims(qv, kv, leaf_of_pos(decltype(J)::value), o); };
        LeafTree<DIM / KLEAF, 1>::run(leaf_id, plain);

        if (lane == 0) { out[3 * t] = deployed; out[3 * t + 1] = tree[0]; out[3 * t + 2] = plain[0]; }
    }
}

} // namespace

int main(int argc, char ** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 20000;
    std::mt19937 rng(20260921);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q((size_t) trials * DIM);
    std::vector<__half> k((size_t) trials * DIM);
    for (size_t i = 0; i < q.size(); i++) {
        // The real operands: a query row scaled by 1/sqrt(256) and an fp16 key.
        q[i] = nd(rng) * 0.0625f;
        k[i] = __float2half(nd(rng));
    }
    float *dq = nullptr, *dout = nullptr; __half * dk = nullptr;
    CHECK(hipMalloc(&dq, q.size() * 4));
    CHECK(hipMalloc(&dk, k.size() * 2));
    CHECK(hipMalloc(&dout, (size_t) trials * 3 * 4));
    CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dk, k.data(), k.size() * 2, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_tree_check, dim3(trials < 4096 ? trials : 4096), dim3(32), 0, 0, dq, dk, dout, trials);
    CHECK(hipDeviceSynchronize());
    std::vector<float> out((size_t) trials * 3);
    CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));

    long tree_eq = 0, plain_eq = 0;
    double worst_tree = 0, worst_plain = 0;
    for (int t = 0; t < trials; t++) {
        const float a = out[3 * t], b = out[3 * t + 1], c = out[3 * t + 2];
        union { float f; uint32_t u; } ua{a}, ub{b}, uc{c};
        const uint32_t ai = ua.u, bi = ub.u, ci = uc.u;
        if (ai == bi) tree_eq++; else worst_tree = fmax(worst_tree, fabs((double) a - (double) b));
        if (ai == ci) plain_eq++; else worst_plain = fmax(worst_plain, fabs((double) a - (double) c));
    }
    printf("trials %d\n", trials);
    printf("  warp_sum == LeafTree (hardware quad order) : %ld / %d%s\n", tree_eq, trials,
           tree_eq == trials ? "   EXACT" : "");
    printf("  warp_sum == LeafTree (plain index order)   : %ld / %d\n", plain_eq, trials);
    printf("  largest absolute difference: relabelled %.3e, plain %.3e\n", worst_tree, worst_plain);
    (void) hipFree(dq); (void) hipFree(dk); (void) hipFree(dout);
    return tree_eq == trials ? 0 : 1;
}
