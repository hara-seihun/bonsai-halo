// GPU read-bandwidth probe: which allocation type gives the highest sustained read rate on gfx1151.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("%s failed: %s\n",#x,hipGetErrorString(e)); return -1;}}while(0)

typedef int v4i __attribute__((ext_vector_type(4)));
__global__ void rd(const int4* __restrict__ p, size_t n16, unsigned* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int acc = 0;
    const v4i* q = (const v4i*)p;
    #pragma unroll 4
    for (; i < n16; i += stride) { v4i v = __builtin_nontemporal_load(q + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678) out[0] = acc;
}
__global__ void rd_plain(const int4* __restrict__ p, size_t n16, unsigned* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int acc = 0;
    #pragma unroll 4
    for (; i < n16; i += stride) { int4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678) out[0] = acc;
}

static int run(const char* name, void* p, size_t bytes, unsigned* out, int blocks, int threads) {
    size_t n16 = bytes / 16;
    for (int nt = 0; nt < 2; nt++) {
        // warm
        if (nt) rd<<<blocks, threads>>>((int4*)p, n16, out); else rd_plain<<<blocks, threads>>>((int4*)p, n16, out);
        CK(hipDeviceSynchronize());
        double best = 1e9, sum = 0; int reps = 5;
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if (nt) rd<<<blocks, threads>>>((int4*)p, n16, out); else rd_plain<<<blocks, threads>>>((int4*)p, n16, out);
            CK(hipDeviceSynchronize());
            double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            best = ms < best ? ms : best; sum += ms;
        }
        printf("%-28s %s  %6.1f GB/s best  %6.1f GB/s mean  (%zu MB, %d x %d)\n", name, nt ? "nontemporal" : "plain      ",
               bytes / best / 1e6, bytes / (sum / reps) / 1e6, bytes >> 20, blocks, threads);
    }
    return 0;
}

int main(int argc, char** argv) {
    size_t mb = argc > 1 ? atoi(argv[1]) : 4096;
    size_t bytes = mb << 20;
    int blocks = argc > 2 ? atoi(argv[2]) : 40 * 8, threads = argc > 3 ? atoi(argv[3]) : 256;
    unsigned* out; CK(hipMalloc(&out, 4));
    std::vector<char> h(bytes, 1);

    { void* p = nullptr; hipError_t e = hipMalloc(&p, bytes);
      if (e == hipSuccess) { CK(hipMemcpy(p, h.data(), bytes, hipMemcpyHostToDevice)); run("hipMalloc", p, bytes, out, blocks, threads); hipFree(p); }
      else printf("hipMalloc %zu MB: %s\n", mb, hipGetErrorString(e)); }
    { void* p = nullptr; hipError_t e = hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached);
      if (e == hipSuccess) { CK(hipMemcpy(p, h.data(), bytes, hipMemcpyHostToDevice)); run("hipExtMalloc uncached", p, bytes, out, blocks, threads); hipFree(p); }
      else printf("hipExtMallocWithFlags uncached: %s\n", hipGetErrorString(e)); }
    { void* p = nullptr; hipError_t e = hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained);
      if (e == hipSuccess) { CK(hipMemcpy(p, h.data(), bytes, hipMemcpyHostToDevice)); run("hipExtMalloc finegrained", p, bytes, out, blocks, threads); hipFree(p); }
      else printf("hipExtMallocWithFlags finegrained: %s\n", hipGetErrorString(e)); }
    { void* p = nullptr; hipError_t e = hipMallocManaged(&p, bytes);
      if (e == hipSuccess) { memcpy(p, h.data(), bytes); run("hipMallocManaged", p, bytes, out, blocks, threads); hipFree(p); }
      else printf("hipMallocManaged: %s\n", hipGetErrorString(e)); }
    { void* p = nullptr; hipError_t e = hipHostMalloc(&p, bytes, hipHostMallocDefault);
      if (e == hipSuccess) { memcpy(p, h.data(), bytes); void* d; CK(hipHostGetDevicePointer(&d, p, 0)); run("hipHostMalloc default", d, bytes, out, blocks, threads); hipHostFree(p); }
      else printf("hipHostMalloc default: %s\n", hipGetErrorString(e)); }
    { void* p = nullptr; hipError_t e = hipHostMalloc(&p, bytes, hipHostMallocNonCoherent);
      if (e == hipSuccess) { memcpy(p, h.data(), bytes); void* d; CK(hipHostGetDevicePointer(&d, p, 0)); run("hipHostMalloc noncoherent", d, bytes, out, blocks, threads); hipHostFree(p); }
      else printf("hipHostMalloc noncoherent: %s\n", hipGetErrorString(e)); }
    { void* p = nullptr; hipError_t e = hipHostMalloc(&p, bytes, hipHostMallocCoherent);
      if (e == hipSuccess) { memcpy(p, h.data(), bytes); void* d; CK(hipHostGetDevicePointer(&d, p, 0)); run("hipHostMalloc coherent", d, bytes, out, blocks, threads); hipHostFree(p); }
      else printf("hipHostMalloc coherent: %s\n", hipGetErrorString(e)); }
    return 0;
}
