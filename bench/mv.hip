// Matvec microbenchmark: GB/s of the HALO ternary kernel on synthetic tiles.
#include "../kernels/halo_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
using namespace halo;
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e)); exit(1);}}while(0)

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16384, K = argc > 2 ? atoi(argv[2]) : 5120;
    int reps = argc > 3 ? atoi(argv[3]) : 50;
    size_t bytes = halo_tensor_bytes(N, K);
    std::vector<uint8_t> h(bytes); for (size_t i = 0; i < bytes; i++) h[i] = rand();
    // rotate over enough copies to exceed the 32 MB MALL, so every call streams from DRAM like the real model
    int copies = (int) ((96u << 20) / bytes) + 1;
    std::vector<uint8_t*> ws(copies);
    for (int c = 0; c < copies; c++) { CK(hipMalloc(&ws[c], bytes)); CK(hipMemcpy(ws[c], h.data(), bytes, hipMemcpyHostToDevice)); }
    uint8_t* w = ws[0];
    int8_t* xq; float* xs; int* xsum; float* out;
    CK(hipMalloc(&xq, K)); CK(hipMalloc(&xs, K / 128 * 4)); CK(hipMalloc(&xsum, K / 128 * 4)); CK(hipMalloc(&out, N * 4));
    CK(hipMemset(xq, 1, K)); CK(hipMemset(xs, 0, K / 128 * 4)); CK(hipMemset(xsum, 0, K / 128 * 4));
    MvArgs a{}; a.nseg = 1; a.seg[0] = { w, out, N / 32, 0 }; a.xq = xq; a.xs = xs; a.xsum = xsum;
    hipStream_t st; CK(hipStreamCreate(&st));
    for (int i = 0; i < 3; i++) launch_matvec(a, K, st);
    CK(hipStreamSynchronize(st));
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0, st);
    for (int i = 0; i < reps; i++) { a.seg[0].w = ws[i % copies]; launch_matvec(a, K, st); }
    hipEventRecord(e1, st); CK(hipEventSynchronize(e1));
    float ms; hipEventElapsedTime(&ms, e0, e1);
    printf("N=%d K=%d: %.1f us/call, %.1f GB/s\n", N, K, 1000 * ms / reps, bytes * (double) reps / ms / 1e6);
    return 0;
}
