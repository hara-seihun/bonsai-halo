// What grid the device really gives the persistent kernel, per register budget.
//
// tools/kernel_resources.py reads the compiler's occupancy estimate, which knows registers and LDS
// and nothing about what this runtime will co-schedule. `coop_grid` asks the runtime, and the two
// can disagree: a change that takes the kernel from 135 to 120 VGPRs crosses a compiler occupancy
// step and may still get the same number of workgroups. That difference is the whole value of a
// register change here, and it costs one compile and no model to see.
//
//   make tools/rows_occ_probe && tools/rows_occ_probe
//
// Prints, for each budget: the VGPRs the loader reports, the LDS per block, the blocks per WGP the
// occupancy API returns, what coop_grid does with that, and the resulting cooperative grid.
#define HALO_ROWS_PROBE 8
#define HALO_ROWS_PROBE_OCC 1
#include "../kernels/halo_rows.hip"

// The second budget of the same shape, and the two the narrow path uses, so one run covers every
// instantiation rows_grid_size takes a minimum over.
template __global__ void halo::k_forward_rows<8, 0, 0, 0, false, PK_OCC_WIDE>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<1, 0, 0, 0, false, 1>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 0, true, 1>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 0, true, PK_OCC_WIDE>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 0, false, 13>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 0, false, 14>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 0, true, 13>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 0, true, 14>(halo::FwdParams, int);

// The direct-commit bodies, because `rows_grid_for` takes a minimum with THEIR grid on any pass
// that commits its state, and nothing reported them. A budget that buys the replay body a block
// and leaves the commit body one short would hand a committed wide pass the old grid while every
// panel read the new one.
template __global__ void halo::k_forward_rows<8, 0, 0, 1, false, 1>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 1, false, PK_OCC_WIDE>(halo::FwdParams, int);
template __global__ void halo::k_forward_rows<8, 0, 0, 1, true, PK_OCC_WIDE>(halo::FwdParams, int);

#include <cstdio>

static void report(const char * name, const void * fn) {
    int per = 0;
    hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, halo::NT, 0);
    hipFuncAttributes attr{};
    hipError_t e2 = hipFuncGetAttributes(&attr, fn);
    hipDeviceProp_t prop{};
    (void) hipGetDeviceProperties(&prop, 0);
    const int vg = (attr.numRegs + 7) / 8 * 8;
    const int waves = per * halo::NW / 4;
    const int corrected = (vg > 0 && vg * waves > 1536 - 16) ? per - 1 : per;
    printf("%-28s numRegs %3d  lds %5zu  api %d blocks/WGP (%2d waves/SIMD32)  coop_grid %d blocks -> grid %d  %s%s\n",
           name, attr.numRegs, (size_t) attr.sharedSizeBytes, per, waves, corrected,
           corrected * prop.multiProcessorCount,
           e == hipSuccess ? "" : "[occupancy query failed] ",
           e2 == hipSuccess ? "" : "[attribute query failed]");
}

int main() {
    hipDeviceProp_t prop{};
    (void) hipGetDeviceProperties(&prop, 0);
    printf("device %s: %d multiprocessors, %zu B LDS per block, %d threads per block\n",
           prop.name, prop.multiProcessorCount, (size_t) prop.sharedMemPerBlock, prop.maxThreadsPerBlock);
    printf("workgroup %d threads = %d waves\n\n", halo::NT, halo::NW);
    report("fp32 TT=8 allocator", (const void *) halo::k_forward_rows<8, 0, 0, 0, false, 1>);
    report("fp32 TT=8 wide budget", (const void *) halo::k_forward_rows<8, 0, 0, 0, false, PK_OCC_WIDE>);
    report("fp32 TT=1 allocator", (const void *) halo::k_forward_rows<1, 0, 0, 0, false, 1>);
    report("packed TT=8 allocator", (const void *) halo::k_forward_rows<8, 0, 0, 0, true, 1>);
    report("packed TT=8 wide budget", (const void *) halo::k_forward_rows<8, 0, 0, 0, true, PK_OCC_WIDE>);
    report("fp32 TT=8 ask 13", (const void *) halo::k_forward_rows<8, 0, 0, 0, false, 13>);
    report("fp32 TT=8 ask 14", (const void *) halo::k_forward_rows<8, 0, 0, 0, false, 14>);
    report("packed TT=8 ask 13", (const void *) halo::k_forward_rows<8, 0, 0, 0, true, 13>);
    report("packed TT=8 ask 14", (const void *) halo::k_forward_rows<8, 0, 0, 0, true, 14>);
    printf("\n");
    report("fp32 TT=8 commit alloc", (const void *) halo::k_forward_rows<8, 0, 0, 1, false, 1>);
    report("fp32 TT=8 commit wide", (const void *) halo::k_forward_rows<8, 0, 0, 1, false, PK_OCC_WIDE>);
    report("packed TT=8 commit wide", (const void *) halo::k_forward_rows<8, 0, 0, 1, true, PK_OCC_WIDE>);
    return 0;
}
