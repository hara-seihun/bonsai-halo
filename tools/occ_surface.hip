// The runtime's co-residency surface over (VGPRs, LDS bytes), without building a model kernel.
//
// `coop_grid` launches what `hipOccupancyMaxActiveBlocksPerMultiprocessor` says the device will
// hold, and that answer is a pure function of the loader's reported register count, the LDS a block
// asks for and the workgroup size. tools/rows_occ_probe reads it for the four or five budgets the
// persistent kernel happens to compile to, which costs a 60-second translation unit and samples the
// ladder wherever the allocator lands rather than where the steps are. This probe walks both axes
// directly: a synthetic body whose live register count is a template parameter, and the LDS axis
// swept through the dynamic-shared-memory argument of the same query, so one small compile prints
// the whole surface a kernel in this repository can land on.
//
//   make tools/occ_surface && tools/occ_surface
//
// Read it as: to buy another block per WGP, this is the register count and this is the LDS you have
// to be under. It launches nothing and times nothing, so it is not a panel and needs no GPU lock -
// the only device call is the property query the occupancy calculator needs.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#ifndef OCC_NT
#define OCC_NT 256
#endif

// A body that holds N floats live across a loop the compiler cannot unroll away, so the allocation
// the loader reports tracks N. The exact mapping does not matter: the probe reports what the loader
// says, not what was asked for.
template <int N>
__global__ void __launch_bounds__(OCC_NT) k_occ(const float * __restrict__ in, float * __restrict__ out, int n) {
    float v[N];
#pragma unroll
    for (int i = 0; i < N; i++) v[i] = in[threadIdx.x + i];
    for (int it = 0; it < n; it++) {
#pragma unroll
        for (int i = 0; i < N; i++) v[i] = fmaf(v[i], 1.000001f, (float) it);
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < N; i++) s += v[i];
    out[blockIdx.x * OCC_NT + threadIdx.x] = s;
}

struct Point { int regs; int scratch; int lds; const void * fn; };

static void row(const Point & p, const std::vector<int> & lds, int wgp, int waves_per_block) {
    printf("%5d %7d %5d ", p.regs, p.scratch, p.lds);
    for (int s : lds) {
        int per = 0;
        hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, p.fn, OCC_NT, (size_t) s);
        if (e != hipSuccess) { printf("   ?  "); continue; }
        printf("%3d   ", per);
    }
    // waves per SIMD32 at the widest LDS column that still holds, for the reader's convenience
    int per = 0;
    (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, p.fn, OCC_NT, 0);
    printf("| %2d waves/SIMD32, grid %d\n", per * waves_per_block / 4, per * wgp);
}

template <int... Ns>
static void collect(std::vector<Point> & pts) {
    const void * fns[] = { (const void *) k_occ<Ns>... };
    for (const void * f : fns) {
        hipFuncAttributes a{};
        if (hipFuncGetAttributes(&a, f) == hipSuccess)
            pts.push_back({ a.numRegs, (int) a.localSizeBytes, (int) a.sharedSizeBytes, f });
    }
}

int main(int argc, char ** argv) {
    hipDeviceProp_t prop{};
    (void) hipGetDeviceProperties(&prop, 0);
    const int wgp = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    const int waves_per_block = OCC_NT / 32;
    printf("device %s: %d multiprocessors, %zu B LDS per block, workgroup %d threads = %d waves\n\n",
           prop.name, wgp, (size_t) prop.sharedMemPerBlock, OCC_NT, waves_per_block);

    std::vector<int> lds;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) lds.push_back(atoi(argv[i]));
    } else {
        lds = { 0, 4132, 4288, 5120, 6144, 7204, 8192, 9508, 10240, 12288, 16384 };
    }

    std::vector<Point> pts;
    collect<8, 16, 24, 32, 40, 48, 56, 64,
            71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
            91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
            141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 160, 168, 176, 192, 208, 224, 240>(pts);

    printf("blocks per WGP the runtime will co-schedule, by reported VGPRs (rows) and LDS bytes per block (columns)\n\n");
    printf(" VGPR scratch  lds ");
    for (int s : lds) printf("%5d ", s);
    printf("\n");
    for (const Point & p : pts) row(p, lds, wgp, waves_per_block);
    return 0;
}
