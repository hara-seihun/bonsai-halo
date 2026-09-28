#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

// One wave owns one 256-coordinate group. Each of its lanes reads one code
// from each of eight Q8_0 blocks. The two layouts have the same 272 bytes/group.
__global__ void fill(unsigned char *inter, unsigned char *codes, unsigned char *scales, int groups) {
    int g = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (g >= groups) return;
    for (int b = 0; b < 8; ++b) {
        unsigned char v = (unsigned char)((g * 17 + b * 53 + lane * 29) & 255);
        inter[g*272 + b*34 + 2 + lane] = v;
        codes[g*256 + b*32 + lane] = v;
        if (lane == b) {
            unsigned short h = (unsigned short)(0x3c00 + ((g * 13 + b * 3) & 63));
            inter[g*272 + b*34] = (unsigned char)h;
            inter[g*272 + b*34 + 1] = (unsigned char)(h >> 8);
            scales[g*16 + b*2] = (unsigned char)h;
            scales[g*16 + b*2 + 1] = (unsigned char)(h >> 8);
        }
    }
}

// A row is 32 consecutive wave groups = K8192, four waves per workgroup.
// Both arms execute the same ordered per-lane FMA and the same warp reduction.
// Input is a fixed nonzero signed activation. No model inference is claimed.
template<bool SPLIT>
__global__ void consume(const unsigned char *inter, const unsigned char *codes,
                        const unsigned char *scales, float *out, int rows) {
    int wave = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (wave >= rows) return;
    float sum = 0.0f;
    int a = (lane & 15) - 7;
    for (int k = 0; k < 32; ++k) {
        int g = wave * 32 + k;
        for (int b = 0; b < 8; ++b) {
            unsigned short scale_bits = 0;
            if (lane == b) {
                const unsigned char *s = SPLIT ? scales + g*16 + b*2 : inter + g*272 + b*34;
                scale_bits = (unsigned short)s[0] | ((unsigned short)s[1] << 8);
            }
            scale_bits = __shfl(scale_bits, b);
            union { unsigned short bits; __half value; } scale = {scale_bits};
            const unsigned char *q = SPLIT ? codes + g*256 + b*32 : inter + g*272 + b*34 + 2;
            int code = (int)(signed char)q[lane];
            sum = fmaf((float)(code * a), __half2float(scale.value), sum);
        }
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down(sum, offset);
    if (lane == 0) out[wave] = sum;
}

static double time_one(bool split, const unsigned char *inter, const unsigned char *codes,
                       const unsigned char *scales, float *out, int rows) {
    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start)); CHECK(hipEventCreate(&stop));
    CHECK(hipEventRecord(start));
    if (split) hipLaunchKernelGGL((consume<true>), dim3((rows+3)/4), dim3(128), 0, 0, inter, codes, scales, out, rows);
    else hipLaunchKernelGGL((consume<false>), dim3((rows+3)/4), dim3(128), 0, 0, inter, codes, scales, out, rows);
    CHECK(hipGetLastError()); CHECK(hipEventRecord(stop)); CHECK(hipEventSynchronize(stop));
    float ms; CHECK(hipEventElapsedTime(&ms, start, stop));
    CHECK(hipEventDestroy(start)); CHECK(hipEventDestroy(stop));
    return ms;
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 171520; // 1,492,910,080 Q8 bytes
    if (rows <= 0) return 2;
    size_t groups = (size_t) rows * 32;
    unsigned char *inter, *codes, *scales;
    float *out;
    CHECK(hipMalloc(&inter, groups*272)); CHECK(hipMalloc(&codes, groups*256));
    CHECK(hipMalloc(&scales, groups*16)); CHECK(hipMalloc(&out, rows*sizeof(float)));
    hipLaunchKernelGGL(fill, dim3((groups+127)/128), dim3(128), 0, 0, inter, codes, scales, (int) groups);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    std::vector<float> old(rows), now(rows);
    time_one(false, inter, codes, scales, out, rows);
    CHECK(hipMemcpy(old.data(), out, rows*sizeof(float), hipMemcpyDeviceToHost));
    time_one(true, inter, codes, scales, out, rows);
    CHECK(hipMemcpy(now.data(), out, rows*sizeof(float), hipMemcpyDeviceToHost));
    size_t mismatch = 0;
    for (int i=0; i<rows; ++i) mismatch += memcmp(&old[i], &now[i], sizeof(float)) != 0;
    printf("rows=%d groups=%zu image_bytes=%zu differing_f32=%zu\n", rows, groups, groups*272, mismatch);
    if (mismatch) return 3;
    for (int round=0; round<5; ++round) {
        if (round&1) {
            printf("round=%d split_ms=%.6f inter_ms=%.6f\n", round,
                   time_one(true,inter,codes,scales,out,rows), time_one(false,inter,codes,scales,out,rows));
        } else {
            printf("round=%d inter_ms=%.6f split_ms=%.6f\n", round,
                   time_one(false,inter,codes,scales,out,rows), time_one(true,inter,codes,scales,out,rows));
        }
    }
    CHECK(hipFree(inter)); CHECK(hipFree(codes)); CHECK(hipFree(scales)); CHECK(hipFree(out));
}
