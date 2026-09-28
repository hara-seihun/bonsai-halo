// What a DEPENDENT matrix chain costs a SIMD32 on gfx1151, which is not what `bench/wmma_cost`
// measures.
//
// `wmma_cost` issues its matrix instructions over four accumulators that never feed each other, so
// it reports the issue cost with nothing waiting on a result: 21.91 slots per SIMD32 for
// `v_wmma_i32_16x16x16_iu8`. Every block census in this repository is converted with that number.
// THE DEPLOYED FFN SLICE DOES NOT ISSUE THAT WAY. `mvw_rows` walks eight K slices into TWO
// accumulators, `C1` and `C2`, and every matrix instruction reads the accumulator its predecessor
// two instructions earlier wrote, so a chain's next instruction can only issue once the previous
// one's result is available. If the result latency is longer than two issue costs, the wave stalls
// on the matrix pipe and no census can see it - which is the shape of the 30% of that phase that
// `docs/ffn-slice-activation.md` measures as neither its bytes nor its counted instructions.
//
// This walks the chain count with everything else fixed: same instruction, same operands, same
// unroll, same resident waves, only how many independent accumulators the wave rotates through.
// The answer is a constant the whole lane can spend, and it is the difference between "this block
// is 61% matrix" and "this block is 61% matrix ISSUE and pays a stall on top of it".
//
//   tools/run-batch-compare --pin-clock --exec bench/wmma_chain
//   tools/run-batch-compare --pin-clock --exec bench/wmma_chain --grid 60 --iters 4000
//
// `--grid 60` is the deployed FFN slice's cooperative grid: 256-thread workgroups, three per WGP,
// six waves per SIMD32. That matters here in a way it does not for a throughput measurement,
// because other waves are exactly what a dependent stall can hide behind.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

#define UNROLL 32

// CH independent accumulators, UNROLL matrix instructions per iteration, rotating through them in
// the order the FFN slice's block loop does. CH == 2 is `mvw_rows` exactly.
template <int CH>
__global__ void __launch_bounds__(256) k_chain(int * out, int iters, int seed) {
    v4i a = {seed, seed + 1, seed + 2, seed + 3}, b = {seed + 4, seed + 5, seed + 6, seed + 7};
    v8i c[CH];
#pragma unroll
    for (int i = 0; i < CH; i++) c[i] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++)
            c[u % CH] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, false, b, c[u % CH], false);
    }
    int s = 0;
#pragma unroll
    for (int i = 0; i < CH; i++) s += c[i][i & 7];
    if (out) out[threadIdx.x] = s;
}

// The same walk on the VALU, as the harness control: `v_fmac_f32` is one slot per instruction by
// construction, so its chain curve is the pure dependent-latency shape of an ordinary instruction
// and says whether this device hides a chain behind other waves at all.
template <int CH>
__global__ void __launch_bounds__(256) k_valu_chain(float * out, int iters, int seed) {
    float a = (float) seed, b = a + 1.0f, c[CH];
#pragma unroll
    for (int i = 0; i < CH; i++) c[i] = 0.0f;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) c[u % CH] = __builtin_fmaf(a, b, c[u % CH]);
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < CH; i++) s += c[i];
    if (out) out[threadIdx.x] = s;
}

static double run(const void * fn, int grid, int iters, void * out, double clock_ghz, int simds) {
    int seed = 1;
    void * args[] = {&out, &iters, &seed};
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernel(fn, dim3(grid), dim3(256), args, 0, 0); hipDeviceSynchronize();
    hipEventRecord(a);
    hipLaunchKernel(fn, dim3(grid), dim3(256), args, 0, 0);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms = 0; hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    const double waves = (double) grid * 8.0;
    const double instrs = waves * (double) iters * (double) UNROLL;
    return ms * 1e-3 * clock_ghz * 1e9 * simds / instrs;
}

int main(int argc, char ** argv) {
    int iters = 2000, grid = 60;
    double clock_ghz = 2.9;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--clock" && i + 1 < argc) clock_ghz = atof(argv[++i]) / 1000.0;
        else if (a == "--grid" && i + 1 < argc) grid = atoi(argv[++i]);
        else { fprintf(stderr, "usage: wmma_chain [--iters N] [--clock MHz] [--grid WORKGROUPS]\n"); return 2; }
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) { fprintf(stderr, "no device\n"); return 1; }
    // `multiProcessorCount` reports 20 on this part, which is its WGP count, not its 40 CUs: a WGP
    // is two CUs and each CU is two SIMD32, so the machine is 80 SIMD32. `bench/wmma_cost`,
    // `coop_grid` in phases.hpp and every phase model in docs/ use that number, and this probe
    // reported half of it, which halved every constant it printed - `iu8` read 17.14 where the
    // clock-free measurement in `bench/mvblock` reads 34.3. docs/ffn-matrix-rate.md carries both.
    const int simds = prop.multiProcessorCount * 4;   // four SIMD32 per reported WGP on gfx11
    int * out = nullptr; hipMalloc(&out, 256 * sizeof(int));
    const int grids[] = { grid, 20, 40, 80 };
    printf("gfx1151: %d WGPs, %d SIMD32, clock assumed %.0f MHz, unroll %d, iters %d\n",
           prop.multiProcessorCount, simds, clock_ghz * 1000.0, UNROLL, iters);
    printf("%8s %6s %7s  %12s %12s\n", "grid", "waves", "chains", "iu8 slots", "fmac slots");
    for (int gi = 0; gi < 4; gi++) {
        const int g = grids[gi];
        if (g <= 0) continue;
        const double wv = g * 8.0 / (double) simds;
        const double m1 = run((const void *) k_chain<1>, g, iters, out, clock_ghz, simds);
        const double m2 = run((const void *) k_chain<2>, g, iters, out, clock_ghz, simds);
        const double m4 = run((const void *) k_chain<4>, g, iters, out, clock_ghz, simds);
        const double m8 = run((const void *) k_chain<8>, g, iters, out, clock_ghz, simds);
        const double v1 = run((const void *) k_valu_chain<1>, g, iters, out, clock_ghz, simds);
        const double v2 = run((const void *) k_valu_chain<2>, g, iters, out, clock_ghz, simds);
        const double v4 = run((const void *) k_valu_chain<4>, g, iters, out, clock_ghz, simds);
        printf("%8d %6.1f %7s  %5.2f %5.2f %5.2f %5.2f  %5.2f %5.2f %5.2f\n",
               g, wv, "1/2/4/8", m1, m2, m4, m8, v1, v2, v4);
    }
    hipFree(out);
    return 0;
}
