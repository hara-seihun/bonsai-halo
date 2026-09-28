// What one `v_wmma_i32_16x16x16_iu8` costs a SIMD32 *in the shape the FFN slice issues it*.
//
// `bench/wmma_cost` measures 21.9 issue slots for that instruction with four independent
// accumulators and nothing else in the loop. The deployed FFN slice disagrees, and it disagrees the
// same way at three different widths. A `k_ffn_slice` launch is one layer and 65,280 wave-blocks,
// 816 per SIMD32; with the published per-launch times and the census of non-matrix slots:
//
//   16 rows  0.4225 ms/launch  1163 cycles a wave-block  16 wmma + 340 other  -> 51.4 cycles/wmma
//   32 rows  0.7133            1963                      32 wmma + 444        -> 47.5
//   64 rows  1.4289            3933                      64 wmma + 640        -> 51.5
//
// Three widths, one constant, and the census term does not move it. That constant is why every
// instruction cut this lane has made in that block converted at about a fifth of its slot count.
// The vendor peak for this part (512 int8 ops per clock per CU = 32.2 cycles per SIMD32) sits
// between the probe and the kernel, so at least one of the two is measuring something else.
//
// This probe takes the deployed shape apart one property at a time, with no memory traffic, no
// model and no host dependence:
//
//   chains      how many independent int32 accumulator chains a wave carries (1, 2, 4, 8).
//               The deployed block carries exactly TWO - C1 and C2 - and the `sched_barrier`
//               between column groups stops the scheduler from ever interleaving four.
//   hazard      whether each matrix instruction's A operand is written by a VALU op immediately
//               before it (the deployed peel and half-swap do exactly this) or hoisted out.
//   filler      independent VALU work interleaved with the matrix instructions, which is what the
//               block's other 444 slots are, to measure how much of it hides under the matrix pipe.
//   waves       waves per SIMD32, set by the grid. The deployed body runs six.
//
//   tools/run-batch-compare --pin-clock --exec bench/mvblock
//   tools/run-batch-compare --pin-clock --exec bench/mvblock --clock 2246 --waves 2,6,8
//
// Cycles are read inside the kernel and never converted through a clock. `SHADER_CYCLES` is a
// free-running shader-clock counter, so the cycles a SIMD32 spent on one wave's segment, divided by
// the matrix instructions every resident wave issued in that segment, is the per-SIMD cost of the
// instruction with no frequency assumption anywhere in it. The counter is 20 bits and wraps at
// about 1.05M cycles, so the segment is a bounded number of iterations rather than the whole run.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

#define WMMA(A, B, C) __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, (A), true, (B), (C), false)

// HW_REG_SHADER_CYCLES (id 29), offset 0, 20 bits: s_getreg_b32 simm16 = id | ((size-1) << 11).
__device__ __forceinline__ unsigned shader_cycles() {
    return __builtin_amdgcn_s_getreg((29) | (19 << 11));
}

struct Probe { unsigned long long ticks; unsigned cycles; unsigned seg; };

// Iterations the in-kernel counter spans. 64 x 16 matrix instructions x 8 waves per SIMD32 at ~30
// cycles is about 245k cycles, a quarter of the counter's range.
#define SEG 64

// One arm. CH independent accumulator chains, MI matrix instructions per iteration, FILL filler
// VALU operations per matrix instruction, HAZ = rewrite the A operand one instruction before use.
template <int CH, int MI, int FILL, bool HAZ, bool MAT = true>
__global__ void __launch_bounds__(256) k_arm(int * out, Probe * probe, int iters, int seed) {
    v4i a[CH], b;
    v8i c[CH];
#pragma unroll
    for (int i = 0; i < CH; i++) {
        a[i] = v4i{seed + i, seed + i + 1, seed + i + 2, seed + i + 3};
        c[i] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    }
    b = v4i{seed + 4, seed + 5, seed + 6, seed + 7};
    float f0 = (float) seed, f1 = f0 + 1.0f, f2 = f0 + 2.0f, f3 = f0 + 3.0f;
    unsigned long long t0 = 0, t1 = 0;
    unsigned s0 = 0, s1 = 0;
    for (int i = 0; i < iters; i++) {
        if (i == 1) { t0 = wall_clock64(); s0 = shader_cycles(); }
        if (i == 1 + SEG) { s1 = shader_cycles(); t1 = wall_clock64(); }
#pragma unroll
        for (int m = 0; m < MI; m++) {
            const int ch = m % CH;
            if (HAZ) {
                // The deployed block's A operand is produced by `v_perm_b32` (the radix-3 gather)
                // and `v_permlanex16_b32` (the half swap) a handful of slots before the matrix
                // instruction that reads it. Reproduce that dependence exactly.
                a[ch].x = __builtin_amdgcn_perm(a[ch].y, a[ch].x, 0x07030501u);
                a[ch].z = __builtin_amdgcn_permlanex16(a[ch].z, a[ch].z, 0x76543210u, 0xfedcba98u, false, false);
            }
            if (MAT) c[ch] = WMMA(a[ch], b, c[ch]);
#pragma unroll
            for (int k = 0; k < FILL; k++) {
                // Four independent float chains: the same kind of work the block's peel, drain and
                // addressing are, and none of it touches a matrix operand.
                float & f = (k % 4 == 0) ? f0 : (k % 4 == 1) ? f1 : (k % 4 == 2) ? f2 : f3;
                f = __builtin_fmaf(f, 1.0009765625f, 1.0f);
            }
        }
    }
    if (probe && blockIdx.x == 0 && threadIdx.x == 0) { probe->ticks = t1 - t0; probe->cycles = (s1 - s0) & 0xfffffu; probe->seg = SEG; }
    if (out) {
        int acc = 0;
#pragma unroll
        for (int i = 0; i < CH; i++) acc += c[i][i % 8];
        out[threadIdx.x] = acc + (int) (f0 + f1 + f2 + f3);
    }
}

struct Arm {
    const char * name;
    const void * fn;
    int mi;          // matrix instructions per iteration
    int fill;        // filler VALU per matrix instruction
    int chains;
};

static double ms_of(const void * fn, int grid, int iters, int * out, Probe * probe) {
    int seed = 1;
    void * args[] = {&out, &probe, &iters, &seed};
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernel(fn, dim3(grid), dim3(256), args, 0, 0); hipDeviceSynchronize();
    hipEventRecord(a);
    hipLaunchKernel(fn, dim3(grid), dim3(256), args, 0, 0);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms = 0; hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms;
}

int main(int argc, char ** argv) {
    int iters = 3000;
    std::vector<int> waves = {2, 6, 8};
    double nominal = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--clock" && i + 1 < argc) nominal = atof(argv[++i]);
        else if (a == "--waves" && i + 1 < argc) {
            waves.clear();
            char * s = strdup(argv[++i]);
            for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) waves.push_back(atoi(t));
            free(s);
        } else { fprintf(stderr, "usage: mvblock [--iters N] [--waves 2,6,8] [--clock MHz]\n"); return 2; }
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) { fprintf(stderr, "no device\n"); return 1; }
    const int wgp = prop.multiProcessorCount, simds = wgp * 4;

    // The deployed 32-row block: 32 matrix instructions in two chains and 444 other slots, which is
    // 13.9 filler operations per matrix instruction.
    static const Arm arms[] = {
        {"chains 1                ", (const void *) k_arm<1, 16, 0, false>, 16, 0, 1},
        {"chains 2                ", (const void *) k_arm<2, 16, 0, false>, 16, 0, 2},
        {"chains 4                ", (const void *) k_arm<4, 16, 0, false>, 16, 0, 4},
        {"chains 8                ", (const void *) k_arm<8, 16, 0, false>, 16, 0, 8},
        {"chains 2 + operand hazard", (const void *) k_arm<2, 16, 0, true>, 16, 0, 2},
        {"chains 4 + operand hazard", (const void *) k_arm<4, 16, 0, true>, 16, 0, 4},
        {"deployed shape: 2 chains, 14 filler", (const void *) k_arm<2, 16, 14, false>, 16, 14, 2},
        {"deployed shape + hazard ", (const void *) k_arm<2, 16, 14, true>, 16, 14, 2},
        {"four chains, 14 filler  ", (const void *) k_arm<4, 16, 14, false>, 16, 14, 4},
        {"filler only, no matrix  ", (const void *) k_arm<1, 16, 14, false, false>, 16, 14, 1},
    };
    int * out = nullptr; hipMalloc(&out, 256 * sizeof(int));
    Probe * probe = nullptr; hipHostMalloc(&probe, sizeof(Probe));
    printf("gfx1151: %d WGPs, %d SIMD32, %d iters, in-kernel segment %d iterations\n", wgp, simds, iters, SEG);
    printf("%-36s %7s %10s %12s %14s %14s %12s\n", "arm", "waves", "ms", "clock MHz",
           "cyc/wmma host", "cyc/wmma dev", "per filler");
    for (int w : waves) {
        const int grid = w * simds / 8;
        if (grid <= 0) continue;
        for (const Arm & arm : arms) {
            probe->ticks = 0; probe->cycles = 0;
            const double ms = ms_of(arm.fn, grid, iters, out, probe);
            // The kernel timed itself with both clocks over the same interval, so the delivered
            // shader frequency is a measurement rather than an assumption. The 20-bit counter wraps
            // at about 476 us; a wrapped sample reports a clock far from plausible and is shown.
            const double sec = probe->ticks / 100e6;
            const double mhz = sec > 0 ? probe->cycles / sec / 1e6 : 0;
            const double clk = nominal > 0 ? nominal : (mhz > 500 && mhz < 4000 ? mhz : 2246.0);
            const double waves_total = (double) grid * 8.0;
            const double mats = waves_total * iters * arm.mi;
            const double fills = waves_total * (double) iters * arm.mi * arm.fill;
            const double cycles = ms * 1e-3 * clk * 1e6 * simds;
            // The device number: shader cycles the SIMD32 advanced over one wave's segment,
            // divided by the matrix instructions its w resident waves issued in that segment.
            const double seg_mats = (double) SEG * arm.mi * w;
            const double dev = seg_mats > 0 && probe->cycles ? probe->cycles / seg_mats : 0.0;
            printf("%-36s %7d %10.3f %12.0f %14.2f %14.2f %12.3f\n", arm.name, w, ms, clk,
                   mats > 0 ? cycles / mats : 0.0, dev, fills > 0 ? cycles / fills : 0.0);
        }
    }
    hipFree(out); hipHostFree(probe);
    return 0;
}
