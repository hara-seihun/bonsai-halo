#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { hipError_t status = (call); if (status != hipSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(status)); return 1; } } while (0)

// Each launch reads the declared contiguous initialized input once. The sizes
// straddle the gfx1151 last-level cache and include a 256 MiB streaming arm.
__global__ void fill_input(uint32_t *input, size_t words) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < words; i += gridDim.x * blockDim.x) {
        uint32_t v = static_cast<uint32_t>(i);
        v ^= v >> 16;
        v *= 0x7feb352dU;
        v ^= v >> 15;
        v *= 0x846ca68bU;
        input[i] = v ^ (v >> 16);
    }
}

__global__ void stream_once(const uint32_t *__restrict__ input,
                            uint32_t *__restrict__ output, size_t words) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t value = 0;
    for (size_t i = tid; i < words; i += gridDim.x * blockDim.x)
        value += input[i];
    output[tid] = value;
}

int main() {
    constexpr size_t max_bytes = 256ull << 20;
    constexpr size_t threads = 256 * 256;
    uint32_t *input = nullptr, *output = nullptr;
    CHECK(hipMalloc(&input, max_bytes));
    CHECK(hipMalloc(&output, threads * sizeof(uint32_t)));
    hipLaunchKernelGGL(fill_input, dim3(256), dim3(256), 0, 0,
                       input, max_bytes / sizeof(uint32_t));
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    constexpr size_t sizes[] = {16ull << 20, 64ull << 20, 128ull << 20, 256ull << 20};
    for (size_t bytes : sizes) {
        hipLaunchKernelGGL(stream_once, dim3(256), dim3(256), 0, 0,
                           input, output, bytes / sizeof(uint32_t));
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint32_t result = 0;
        CHECK(hipMemcpy(&result, output, sizeof(result), hipMemcpyDeviceToHost));
        printf("output0=%u input_bytes=%zu\n", result, bytes);
    }
    CHECK(hipFree(output));
    CHECK(hipFree(input));
}
