// What this device REALLY co-schedules, per register count, measured by counting blocks that are
// resident at the same time.
//
// Every grid in this engine comes from `coop_grid`, which asks
// `hipOccupancyMaxActiveBlocksPerMultiprocessor` and then clamps the answer with a granule-24
// model. Both are predictions. tools/occ_surface shows they disagree, and that the API's own ladder
// is not monotonic in registers - 120 VGPRs gets five blocks per WGP where 121 gets six and 112
// gets six. One of those two predictions decides the cooperative grid of the persistent kernel, and
// nothing in this repository has ever checked either against the hardware.
//
// This does, and it cannot hang: there is no grid sync. Each block announces itself, spins a fixed
// wall-clock interval, and leaves; the peak of the live count is how many blocks the device held at
// once. Oversubscribe the grid and the peak stops at the true co-residency.
//
//   make bench/occ_resident && bench/occ_resident [spin_us] [lds_bytes]
//
// Read it beside tools/occ_surface: the `api` column is what the runtime promised and `peak` is
// what the machine did. Where they differ, the grid this engine launches is wrong in that
// direction, and `coop_grid` is the one function that has to change.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef OCC_NT
#define OCC_NT 256
#endif

// gfx11 has no s_memrealtime; wall_clock64() is the portable constant-rate counter HIP exposes.
__device__ __forceinline__ unsigned long long realtime() {
    return wall_clock64();
}

// The register pressure is the point: N floats live across the spin, so the allocation the loader
// reports is the allocation the hardware schedules against.
extern __shared__ float dyn_lds[];

template <int N>
__global__ void __launch_bounds__(OCC_NT) k_res(const float * __restrict__ in, float * __restrict__ out,
                                                int * live, int * peak, unsigned long long ticks) {
    float v[N];
#pragma unroll
    for (int i = 0; i < N; i++) v[i] = in[threadIdx.x + i];
    // Touch the dynamic block so the allocation is real rather than elided.
    dyn_lds[threadIdx.x] = v[0];

    if (threadIdx.x == 0) {
        const int n = atomicAdd(live, 1) + 1;
        atomicMax(peak, n);
    }
    __syncthreads();

    const unsigned long long t0 = realtime();
    // Spin on the real-time counter, touching the live values so they cannot be rematerialised
    // after the wait. One iteration is a handful of VALU slots, so the loop is cheap and the block
    // stays resident for the whole interval.
    while (realtime() - t0 < ticks) {
#pragma unroll
        for (int i = 0; i < N; i++) v[i] = fmaf(v[i], 1.000001f, 1.0f);
    }

    __syncthreads();
    if (threadIdx.x == 0) atomicSub(live, 1);

    float s = dyn_lds[threadIdx.x];
#pragma unroll
    for (int i = 0; i < N; i++) s += v[i];
    if (s == 12345.678f) out[blockIdx.x * OCC_NT + threadIdx.x] = s;   // keep the array live
}

struct Arm { int regs; void (*fn)(const float *, float *, int *, int *, unsigned long long); };

static int wgp_count() {
    hipDeviceProp_t prop{};
    (void) hipGetDeviceProperties(&prop, 0);
    return prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
}

// The granule-24 model `coop_grid` already carries: VGPRs round to 24 out of a 1536-register file
// per SIMD32, at most sixteen waves live there, and a 256-thread workgroup is two of them.
static int model_blocks(int regs, int lds) {
    const int vg = (regs + 23) / 24 * 24;
    int waves = vg > 0 ? 1536 / vg : 16;
    if (waves > 16) waves = 16;
    int b = waves / (OCC_NT / 32 / 4);
    if (lds > 0) { const int by_lds = 65536 / lds; if (by_lds < b) b = by_lds; }
    return b;
}

template <int N>
static void run(int wgp, unsigned long long ticks, int lds, int * dlive, int * dpeak, const float * din, float * dout)
{
    const void * fn = (const void *) k_res<N>;
    hipFuncAttributes a{};
    (void) hipFuncGetAttributes(&a, fn);
    int api = 0;
    (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&api, fn, OCC_NT, (size_t) lds);

    // Oversubscribe by a wide margin so the peak is set by the device, not by the grid.
    const int grid = wgp * 12;
    int zero = 0, peak = 0;
    (void) hipMemcpy(dlive, &zero, sizeof(int), hipMemcpyHostToDevice);
    (void) hipMemcpy(dpeak, &zero, sizeof(int), hipMemcpyHostToDevice);
    hipLaunchKernelGGL((k_res<N>), dim3(grid), dim3(OCC_NT), (size_t) lds, 0, din, dout, dlive, dpeak, ticks);
    (void) hipDeviceSynchronize();
    (void) hipMemcpy(&peak, dpeak, sizeof(int), hipMemcpyDeviceToHost);

    const double per = (double) peak / wgp;
    const int mdl = model_blocks(a.numRegs, lds);
    printf("%5d %6d   %2d %5d %6.2f %7.1f   %-8s %s\n", a.numRegs, lds, api, mdl, per, per * wgp,
           per >= api + 0.9 ? "API LOW" : (per <= api - 0.9 ? "API HIGH" : "api ok"),
           (per >= mdl - 0.01 && per <= mdl + 0.01) ? "model ok" : "MODEL WRONG");
}

int main(int argc, char ** argv) {
    const double spin_us = argc > 1 ? atof(argv[1]) : 2000.0;
    hipDeviceProp_t prop{};
    (void) hipGetDeviceProperties(&prop, 0);
    const int wgp = wgp_count();
    // s_memrealtime runs at a fixed 100 MHz on this part; the probe reports the interval it used.
    const unsigned long long ticks = (unsigned long long) (spin_us * 100.0);
    printf("device %s: %d multiprocessors, workgroup %d threads, spin %.0f us (%llu ticks)\n\n",
           prop.name, wgp, OCC_NT, spin_us, ticks);
    printf("api = hipOccupancyMaxActiveBlocksPerMultiprocessor, model = coop_grid's granule-24 arithmetic,\n");
    printf("peak = blocks this device held at the same time.\n\n");
    printf(" VGPR    lds  api model   peak    grid   api?      model?\n");

    float * din = nullptr, * dout = nullptr;
    int * dlive = nullptr, * dpeak = nullptr;
    (void) hipMalloc(&din, 4096 * sizeof(float));
    (void) hipMalloc(&dout, (size_t) wgp * 12 * OCC_NT * sizeof(float));
    (void) hipMalloc(&dlive, sizeof(int));
    (void) hipMalloc(&dpeak, sizeof(int));
    (void) hipMemset(din, 0, 4096 * sizeof(float));

    for (int lds : { 0, 4132, 7204, 9508 }) {
        printf("-- LDS %d bytes per block\n", lds);
        run<64>(wgp, ticks, lds, dlive, dpeak, din, dout);
        run<88>(wgp, ticks, lds, dlive, dpeak, din, dout);
        run<92>(wgp, ticks, lds, dlive, dpeak, din, dout);    // 96 VGPR: the waves_per_eu(13) arm
        run<93>(wgp, ticks, lds, dlive, dpeak, din, dout);    // 97: first register of the next step
        run<103>(wgp, ticks, lds, dlive, dpeak, din, dout);
        run<111>(wgp, ticks, lds, dlive, dpeak, din, dout);
        run<116>(wgp, ticks, lds, dlive, dpeak, din, dout);   // 120 VGPR: THE DEPLOYED KERNEL
        run<117>(wgp, ticks, lds, dlive, dpeak, din, dout);   // 121: one register past it
        run<127>(wgp, ticks, lds, dlive, dpeak, din, dout);
        run<140>(wgp, ticks, lds, dlive, dpeak, din, dout);   // 144
        run<145>(wgp, ticks, lds, dlive, dpeak, din, dout);   // 149: the allocator's own choice
        run<175>(wgp, ticks, lds, dlive, dpeak, din, dout);
        printf("\n");
    }
    return 0;
}
