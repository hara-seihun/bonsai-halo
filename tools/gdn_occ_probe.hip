// What the device gives the recurrent state kernel, per instantiation.
//
// `resident_state<SPLIT, GATE, DEFER>` is one template that serves two very different passes. A
// prefill pass walks many tokens, commits, and pays for the store path and its quantiser. A
// deferred generation step walks exactly one token, stores nothing, and returns. Register
// allocation is the maximum over both, so the cheap pass is charged for the expensive one, and
// `docs/decode-map.md` says that pass is memory bound: whatever costs it waves costs it bandwidth.
//
//   make tools/gdn_occ_probe && tools/gdn_occ_probe
//
// Prints, per instantiation: the VGPRs the loader reports, LDS per block, and the blocks per WGP
// the occupancy API returns. `tools/rows_occ_probe` established that this runtime's answer and the
// compiler's occupancy remark disagree, so ask the runtime.
#define HALO_ROWS_PROBE 1
#include "../kernels/halo_rows.hip"

template __global__ void halo::resident_state<1, halo::GATE_HOIST, false>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<1, halo::GATE_HOIST, true>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<2, halo::GATE_LOOP, false>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<2, halo::GATE_HOIST, false>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<2, halo::GATE_LOOP, true>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<2, halo::GATE_HOIST, true>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<4, halo::GATE_LOOP, false>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<4, halo::GATE_HOIST, false>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<4, halo::GATE_LOOP, true>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<4, halo::GATE_HOIST, true>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<8, halo::GATE_HOIST, true>(halo::GdnResidentParams, int);
template __global__ void halo::resident_state<16, halo::GATE_HOIST, true>(halo::GdnResidentParams, int);
template __global__ void halo::gdn_defer_flush_k<4>(float *, int, int);

// `launch_gdn_resident` is compiled in this translation unit and reaches the sequence quantiser's
// selector; nothing here calls it, so one definition satisfies the link.
namespace halo { int sequence_quant() { return 0; } }

#include <cstdio>

static void report(const char * name, const void * fn) {
    int per = 0;
    const hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, 256, 0);
    hipFuncAttributes attr{};
    const hipError_t e2 = hipFuncGetAttributes(&attr, fn);
    printf("%-34s numRegs %3d  spill %4zu  lds %5zu  api %d blocks/WGP (%2d waves/SIMD32)  %s%s\n",
           name, attr.numRegs, (size_t) attr.localSizeBytes, (size_t) attr.sharedSizeBytes,
           per, per * 8 / 4,
           e == hipSuccess ? "" : "[occupancy query failed] ",
           e2 == hipSuccess ? "" : "[attribute query failed]");
}

int main() {
    hipDeviceProp_t prop{};
    (void) hipGetDeviceProperties(&prop, 0);
    printf("device %s: %d multiprocessors, %zu B LDS per block\n\n",
           prop.name, prop.multiProcessorCount, (size_t) prop.sharedMemPerBlock);
    report("state<1,HOIST,commit>", (const void *) halo::resident_state<1, halo::GATE_HOIST, false>);
    report("state<1,HOIST,defer>",  (const void *) halo::resident_state<1, halo::GATE_HOIST, true>);
    report("state<2,LOOP,commit>",  (const void *) halo::resident_state<2, halo::GATE_LOOP, false>);
    report("state<2,HOIST,commit>", (const void *) halo::resident_state<2, halo::GATE_HOIST, false>);
    report("state<2,LOOP,defer>",   (const void *) halo::resident_state<2, halo::GATE_LOOP, true>);
    report("state<2,HOIST,defer>",  (const void *) halo::resident_state<2, halo::GATE_HOIST, true>);
    report("state<4,LOOP,commit>",  (const void *) halo::resident_state<4, halo::GATE_LOOP, false>);
    report("state<4,HOIST,commit>", (const void *) halo::resident_state<4, halo::GATE_HOIST, false>);
    report("state<4,LOOP,defer>",   (const void *) halo::resident_state<4, halo::GATE_LOOP, true>);
    report("state<4,HOIST,defer>",  (const void *) halo::resident_state<4, halo::GATE_HOIST, true>);
    report("state<8,HOIST,defer>",  (const void *) halo::resident_state<8, halo::GATE_HOIST, true>);
    report("state<16,HOIST,defer>", (const void *) halo::resident_state<16, halo::GATE_HOIST, true>);
    report("defer_flush<4>",        (const void *) halo::gdn_defer_flush_k<4>);
    return 0;
}
