// Multi-row ternary matvec microbenchmark (the kernel used by the persistent forward).
#include "../kernels/halo_kernels.h"
#include "../kernels/device.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace halo;
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e)); exit(1);}}while(0)
template <int TT, int PW>
__global__ void __launch_bounds__(256) k(const uint8_t* w, const int8_t* xq_g, const float* xs_g, const int* xsum_g, float* out, int nb, int nrows) {
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int8_t* xq_l = xq_g; asm volatile("" : "+s"(xq_l)); cptr xq = (cptr) xq_l;
    const float* xs_l = xs_g; asm volatile("" : "+s"(xs_l)); cfptr xs = (cfptr) xs_l;
    const int* xsum_l = xsum_g; asm volatile("" : "+s"(xsum_l)); cptr xsum = (cptr) xsum_l;
    __shared__ float red[8][TT][32];
    const int wb = wave * PW;
    const uint8_t* run = w + ((size_t) blockIdx.x * nb + wb) * TILE_BLOCK_BYTES;
    float y[TT]; for (int r = 0; r < TT; r++) y[r] = 0.f;
    mv_rows_t<PW, TT, false>(run, lane, xq + wb * 32, nb * 32, xs + wb, xsum + wb, nb, nrows, y);
    if (wave > 0) for (int r = 0; r < TT; r++) red[wave][r][lane] = y[r];
    __syncthreads();
    if (wave == 0) for (int r = 0; r < TT; r++) { float v = y[r]; for (int ww = 1; ww < 8; ww++) v += red[ww][r][lane]; out[(size_t) r * gridDim.x * 32 + blockIdx.x * 32 + lane] = v; }
}
// variant: activations staged in LDS per wave, read with broadcast ds_read_b128
template <int TT, int PW>
__global__ void __launch_bounds__(256) kl(const uint8_t* w, const int8_t* xq_g, const float* xs_g, const int* xsum_g, float* out, int nb, int nrows) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    __shared__ __attribute__((aligned(16))) int xl[8][TT][PW * 32];
    __shared__ float red[8][TT][32];
    const int wb = wave * PW;
    for (int i = lane; i < TT * PW * 32; i += 32) { int r = i / (PW * 32), k = i % (PW * 32); xl[wave][r][k] = ((const int*) xq_g)[r * nb * 32 + wb * 32 + k]; }
    __syncthreads();
    const uint8_t* run = w + ((size_t) blockIdx.x * nb + wb) * TILE_BLOCK_BYTES;
    float y[TT]; for (int r = 0; r < TT; r++) y[r] = 0.f;
    uint4 qa = *(const uint4 *) (run + tile_off_qs_a(lane)); uint2 qb = *(const uint2 *) (run + tile_off_qs_b(lane)); unsigned tail = *(const unsigned *) (run + tile_off_tail(lane));
    #pragma unroll 1
    for (int b = 0; b < PW; b++) {
        uint4 nqa = qa; uint2 nqb = qb; unsigned ntail = tail;
        if (b + 1 < PW) { const uint8_t * nrun = run + (b + 1) * TILE_BLOCK_BYTES; nqa = *(const uint4 *) (nrun + tile_off_qs_a(lane)); nqb = *(const uint2 *) (nrun + tile_off_qs_b(lane)); ntail = *(const unsigned *) (nrun + tile_off_tail(lane)); }
        unsigned tr[32];
        const unsigned dw[6] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y };
        #pragma unroll
        for (int d = 0; d < 6; d++) {
            unsigned P0 = dw[d] & 0x00ff00ffu, P1 = (dw[d] >> 8) & 0x00ff00ffu;
            unsigned t0 = peel(P0), t1 = peel(P0), t2 = peel(P0), t3 = peel(P0), t4 = peel_last(P0);
            unsigned u0 = peel(P1), u1 = peel(P1), u2 = peel(P1), u3 = peel(P1), u4 = peel_last(P1);
            tr[4 * d] = t0 | (t1 << 8); tr[4 * d + 1] = t2 | (t3 << 8); tr[4 * d + 2] = u0 | (u1 << 8); tr[4 * d + 3] = u2 | (u3 << 8); tr[24 + d] = t4 | (u4 << 8);
        }
        { unsigned P = (tail & 0xffu) | ((tail << 8) & 0xff0000u); unsigned h0 = peel(P), h1 = peel(P), h2 = peel(P), h3 = peel_last(P); tr[30] = h0 | (h1 << 8); tr[31] = h2 | (h3 << 8); }
        const float wscale = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
        #pragma unroll
        for (int r = 0; r < TT; r++) {
            const int4* xb = (const int4*) &xl[wave][r][b * 32];
            int acc = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) { int4 v = xb[i]; acc = __builtin_amdgcn_sudot4(false, (int) tr[4*i], true, v.x, acc, false); acc = __builtin_amdgcn_sudot4(false, (int) tr[4*i+1], true, v.y, acc, false); acc = __builtin_amdgcn_sudot4(false, (int) tr[4*i+2], true, v.z, acc, false); acc = __builtin_amdgcn_sudot4(false, (int) tr[4*i+3], true, v.w, acc, false); }
            acc -= xsum_g[r * nb + wb + b];
            y[r] = fmaf((float) acc, wscale * xs_g[r * nb + wb + b], y[r]);
        }
        qa = nqa; qb = nqb; tail = ntail;
    }
    if (wave > 0) for (int r = 0; r < TT; r++) red[wave][r][lane] = y[r];
    __syncthreads();
    if (wave == 0) for (int r = 0; r < TT; r++) { float v = y[r]; for (int ww = 1; ww < 8; ww++) v += red[ww][r][lane]; out[(size_t) r * gridDim.x * 32 + blockIdx.x * 32 + lane] = v; }
}
int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 34816, K = argc > 2 ? atoi(argv[2]) : 5120; int nb = K / 128;
    size_t bytes = halo_tensor_bytes(N, K);
    std::vector<uint8_t> h(bytes); for (size_t i = 0; i < bytes; i++) h[i] = rand();
    int copies = (int) ((96u << 20) / bytes) + 1; std::vector<uint8_t*> ws(copies);
    for (int c = 0; c < copies; c++) { CK(hipMalloc(&ws[c], bytes)); CK(hipMemcpy(ws[c], h.data(), bytes, hipMemcpyHostToDevice)); }
    int8_t* xq; float* xs; int* xsum; float* out;
    CK(hipMalloc(&xq, 8 * K)); CK(hipMalloc(&xs, 8 * nb * 4)); CK(hipMalloc(&xsum, 8 * nb * 4)); CK(hipMalloc(&out, (size_t) 8 * N * 4));
    CK(hipMemset(xq, 1, 8 * K)); CK(hipMemset(xs, 0, 8 * nb * 4)); CK(hipMemset(xsum, 0, 8 * nb * 4));
    hipStream_t st; CK(hipStreamCreate(&st));
    auto time = [&](const char* nm, auto launch) {
        for (int i = 0; i < 3; i++) launch(ws[0]);
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1)); CK(hipEventRecord(e0, st));
        int reps = 40; for (int i = 0; i < reps; i++) launch(ws[i % copies]);
        CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1)); float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("N=%d K=%d %-8s %8.1f us %6.1f GB/s\n", N, K, nm, 1000 * ms / reps, bytes * (double) reps / ms / 1e6);
    };
    if (K == 5120) {
        time("TT=1", [&](uint8_t* w) { k<1, 5><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, 1); });
        time("TT=2", [&](uint8_t* w) { k<2, 5><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, 2); });
        time("TT=4", [&](uint8_t* w) { k<4, 5><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, 4); });
        time("TT=8", [&](uint8_t* w) { k<8, 5><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, 8); });
        time("TT=8/n4", [&](uint8_t* w) { k<8, 5><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, 4); });
        time("LDS TT=4", [&](uint8_t* w) { kl<4, 5><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, 4); });
        time("LDS TT=8", [&](uint8_t* w) { kl<8, 5><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, 8); });
    }
    return 0;
}
