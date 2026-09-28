#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { auto e = (x); if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

__global__ void fill(uint32_t *p, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        p[i] = uint32_t(i * 2654435761u) ^ uint32_t(i >> 19);
}

__global__ void stream_known(const uint32_t *__restrict__ p, uint32_t *__restrict__ result, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (size_t i = tid; i < n; i += gridDim.x * blockDim.x) acc += p[i];
    result[tid] = acc;
}

int main() {
    constexpr size_t capacity = 256ull << 20;
    uint32_t *p = nullptr, *out = nullptr;
    CHECK(hipMalloc(&p, capacity));
    CHECK(hipMalloc(&out, (256 * 256) * sizeof(uint32_t)));
    hipLaunchKernelGGL(fill, dim3(256), dim3(256), 0, 0, p, capacity / 4);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    const size_t sizes[] = {16ull << 20, 16ull << 20, 16ull << 20,
                            256ull << 20, 256ull << 20, 16ull << 20};
    for (size_t i = 0; i < 6; ++i) {
        hipLaunchKernelGGL(stream_known, dim3(256), dim3(256), 0, 0, p, out, sizes[i] / 4);
        CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
        uint32_t check = 0;
        CHECK(hipMemcpy(&check, out, sizeof(check), hipMemcpyDeviceToHost));
        printf("arm=%zu bytes=%zu sum0=%u\n", i, sizes[i], check);
    }
    CHECK(hipFree(p)); CHECK(hipFree(out));
}
