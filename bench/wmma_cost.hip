// What a matrix instruction costs a SIMD32 on gfx1151, measured instead of assumed.
//
// Every block census in this repository converts instructions into time with a constant for
// `v_wmma_i32_16x16x16_*`, and those constants were read off a vendor table rather than this
// device. docs/ffn-decode-schedule.md already found one of them wrong by 2x - iu4 is 16 cycles per
// SIMD32, not 32, and "half of every A4 block census in this repository is priced against the wrong
// constant" - and nobody had checked iu8, which is what the deployed FFN slice, the deployed matvec
// and both sequence projections issue. A phase that is 74% of the default prompt pass deserves a
// measured constant.
//
// The measurement is throughput, not latency: each wave issues NIT independent matrix instructions
// over four accumulators that never feed each other, so nothing waits on a result and the number
// that comes out is the issue cost the SIMD pays. With W waves resident per SIMD32, elapsed time is
// W * NIT * cycles_per_instruction, so one launch at a known clock gives the constant directly.
//
//   tools/run-batch-compare --pin-clock --exec bench/wmma_cost
//   tools/run-batch-compare --pin-clock --exec bench/wmma_cost --clock 2900 --iters 4000
//
// The VALU arm is the harness control: `v_fmac_f32` must come out at one cycle per instruction per
// SIMD32, and if it does not, nothing else in the table means anything.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define UNROLL 16

__global__ void __launch_bounds__(256) k_iu8(int * out, int iters, int seed) {
    v4i a = {seed, seed + 1, seed + 2, seed + 3}, b = {seed + 4, seed + 5, seed + 6, seed + 7};
    v8i c0 = {0, 0, 0, 0, 0, 0, 0, 0}, c1 = c0, c2 = c0, c3 = c0;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < UNROLL / 4; u++) {
            c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c0, false);
            c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c1, false);
            c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c2, false);
            c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c3, false);
        }
    }
    if (out) out[threadIdx.x] = c0[0] + c1[1] + c2[2] + c3[3];
}

__global__ void __launch_bounds__(256) k_iu4(int * out, int iters, int seed) {
    v2i a = {seed, seed + 1}, b = {seed + 4, seed + 5};
    v8i c0 = {0, 0, 0, 0, 0, 0, 0, 0}, c1 = c0, c2 = c0, c3 = c0;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < UNROLL / 4; u++) {
            c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, false, b, c0, false);
            c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, false, b, c1, false);
            c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, false, b, c2, false);
            c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, false, b, c3, false);
        }
    }
    if (out) out[threadIdx.x] = c0[0] + c1[1] + c2[2] + c3[3];
}

__global__ void __launch_bounds__(256) k_f16(float * out, int iters, int seed) {
    v16h a, b;
#pragma unroll
    for (int i = 0; i < 16; i++) { a[i] = (_Float16) (seed + i); b[i] = (_Float16) (seed - i); }
    v8f c0 = {0, 0, 0, 0, 0, 0, 0, 0}, c1 = c0, c2 = c0, c3 = c0;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < UNROLL / 4; u++) {
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c1);
            c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c2);
            c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c3);
        }
    }
    if (out) out[threadIdx.x] = c0[0] + c1[1] + c2[2] + c3[3];
}

// The deployed eight-row matvec's arithmetic, for the same reason: `mv_rows_t` spends its block on
// `v_dot4_i32_iu8` and its census uses a constant too.
__global__ void __launch_bounds__(256) k_dot4(int * out, int iters, int seed) {
    int a = seed, b = seed + 7, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < UNROLL / 4; u++) {
            c0 = __builtin_amdgcn_sudot4(true, a, true, b, c0, false);
            c1 = __builtin_amdgcn_sudot4(true, a, true, b, c1, false);
            c2 = __builtin_amdgcn_sudot4(true, a, true, b, c2, false);
            c3 = __builtin_amdgcn_sudot4(true, a, true, b, c3, false);
        }
    }
    if (out) out[threadIdx.x] = c0 + c1 + c2 + c3;
}

// Harness control: a plain VALU instruction must read one cycle per instruction per SIMD32.
__global__ void __launch_bounds__(256) k_valu(float * out, int iters, int seed) {
    float a = (float) seed, b = a + 1.0f, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < UNROLL / 4; u++) {
            c0 = __builtin_fmaf(a, b, c0); c1 = __builtin_fmaf(a, b, c1);
            c2 = __builtin_fmaf(a, b, c2); c3 = __builtin_fmaf(a, b, c3);
        }
    }
    if (out) out[threadIdx.x] = c0 + c1 + c2 + c3;
}

static double run(const void * fn, int grid, int iters, void * out, double clock_ghz, int simds) {
    int seed = 1;
    void * args[] = {&out, &iters, &seed};
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    // warm
    hipLaunchKernel(fn, dim3(grid), dim3(256), args, 0, 0); hipDeviceSynchronize();
    hipEventRecord(a);
    hipLaunchKernel(fn, dim3(grid), dim3(256), args, 0, 0);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms = 0; hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    const double waves = (double) grid * 8.0;
    const double instrs = waves * (double) iters * (double) UNROLL;
    // cycles the whole machine spent, divided by the instructions it retired per SIMD
    return ms * 1e-3 * clock_ghz * 1e9 * simds / instrs;
}

int main(int argc, char ** argv) {
    int iters = 2000, grid = 0;
    double clock_ghz = 2.9;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--clock" && i + 1 < argc) clock_ghz = atof(argv[++i]) / 1000.0;
        else if (a == "--grid" && i + 1 < argc) grid = atoi(argv[++i]);
        else { fprintf(stderr, "usage: wmma_cost [--iters N] [--clock MHz] [--grid WORKGROUPS]\n"); return 2; }
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) { fprintf(stderr, "no device\n"); return 1; }
    const int wgp = prop.multiProcessorCount;
    const int simds = wgp * 4;            // four SIMD32 per WGP on gfx11
    if (!grid) grid = wgp;                // one 8-wave workgroup per WGP: two waves per SIMD32
    int * out = nullptr; hipMalloc(&out, 256 * sizeof(float));
    printf("gfx1151: %d WGPs, %d SIMD32, grid %d workgroups (%d waves per SIMD32), %d iters x %d unrolled, clock %.0f MHz\n",
           wgp, simds, grid, grid * 8 / simds, iters, UNROLL, clock_ghz * 1000);
    struct Arm { const char * name; const void * fn; double macs; } arms[] = {
        {"v_fmac_f32            ", (const void *) k_valu, 1},
        {"v_dot4_i32_iu8        ", (const void *) k_dot4, 4},
        {"v_wmma_i32_16x16x16_iu8", (const void *) k_iu8, 4096.0 / 32.0},
        {"v_wmma_i32_16x16x16_iu4", (const void *) k_iu4, 4096.0 / 32.0},
        {"v_wmma_f32_16x16x16_f16", (const void *) k_f16, 4096.0 / 32.0},
    };
    // The VALU arm calibrates the clock this run actually got: `v_fmac_f32` is one issue slot by
    // construction, so every other arm is reported in slots as well as in cycles at the nominal
    // clock. A census counts slots, so slots is the unit that converts a census into time; the
    // nominal-clock column is only right if the device really held that clock.
    double valu = run(arms[0].fn, grid, iters, out, clock_ghz, simds);
    printf("%-24s %14s %10s %16s %14s\n", "instruction", "cycles/SIMD32", "slots", "MAC/cycle/SIMD32", "device TOPS");
    for (auto & arm : arms) {
        double c = run(arm.fn, grid, iters, out, clock_ghz, simds);
        printf("%-24s %14.2f %10.2f %16.1f %14.1f\n", arm.name, c, c / valu, arm.macs / (c / valu),
               arm.macs / (c / valu) * simds * clock_ghz * 2.0);
    }
    printf("effective clock from the VALU arm: %.0f MHz\n", clock_ghz * 1000 / valu);
    hipFree(out);
    return 0;
}
