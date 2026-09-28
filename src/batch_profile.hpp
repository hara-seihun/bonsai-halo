#pragma once
#include "engine.h"

namespace halo {
// Events locate launches on one stream. The kernel's existing barrier stamps
// divide each persistent slice without inserting synchronization on the host.
struct BatchProfiler {
    static constexpr int stamps = 8;
    struct Entry { hipEvent_t begin{}, end{}; BatchTrace trace; int phases = 0; };
    hipEvent_t origin{}, finish{};
    unsigned long long *device = nullptr;
    std::vector<Entry> entries;
    size_t used = 0;
    explicit BatchProfiler(int capacity) : entries((2 * NLAYER + 2) * ((capacity + 7) / 8) + 3 * NLAYER) {
        HIP_CHECK_H(hipEventCreate(&origin));
        HIP_CHECK_H(hipEventCreate(&finish));
        HIP_CHECK_H(hipMalloc(&device, entries.size() * stamps * sizeof(*device)));
        for (auto & e : entries) {
            HIP_CHECK_H(hipEventCreate(&e.begin)); HIP_CHECK_H(hipEventCreate(&e.end));
        }
    }
    ~BatchProfiler() {
        auto dispose = [](hipError_t status) {
            if (status != hipSuccess) { fprintf(stderr, "batch profile cleanup: %s\n", hipGetErrorString(status)); std::abort(); }
        };
        for (auto & e : entries) { dispose(hipEventDestroy(e.begin)); dispose(hipEventDestroy(e.end)); }
        dispose(hipEventDestroy(origin)); dispose(hipEventDestroy(finish)); dispose(hipFree(device));
    }
    void reset(hipStream_t stream) {
        used = 0;
        HIP_CHECK_H(hipMemsetAsync(device, 0, entries.size() * stamps * sizeof(*device), stream));
        HIP_CHECK_H(hipEventRecord(origin, stream));
    }
    unsigned long long *begin(const char *kind, int layer, int offset, int rows, int phases, hipStream_t stream) {
        if (used == entries.size()) throw std::runtime_error("batch profile capacity exceeded");
        auto & e = entries[used];
        e.trace = {kind, layer, offset, rows, 0, 0, {}}; e.phases = phases;
        HIP_CHECK_H(hipEventRecord(e.begin, stream));
        return device + used * stamps;
    }
    void end(hipStream_t stream) { HIP_CHECK_H(hipEventRecord(entries[used++].end, stream)); }
    void collect(Engine & engine) {
        HIP_CHECK_H(hipEventRecord(finish, engine.stream));
        HIP_CHECK_H(hipEventSynchronize(finish));
        std::vector<unsigned long long> host(used * stamps);
        HIP_CHECK_H(hipMemcpy(host.data(), device, host.size() * sizeof(*device), hipMemcpyDeviceToHost));
        float ms;
        HIP_CHECK_H(hipEventElapsedTime(&ms, origin, finish)); engine.batch_trace_span_ms = ms;
        engine.batch_trace.clear();
        for (size_t i = 0; i < used; ++i) {
            auto & e = entries[i];
            HIP_CHECK_H(hipEventElapsedTime(&ms, origin, e.begin)); e.trace.start_ms = ms;
            HIP_CHECK_H(hipEventElapsedTime(&ms, origin, e.end)); e.trace.end_ms = ms;
            for (int j = 0; j < e.phases; ++j) {
                auto a = host[i * stamps + j], b = host[i * stamps + j + 1];
                if (!a || b < a) throw std::runtime_error("missing or reversed kernel profile stamp");
                e.trace.phase_us.push_back(double(b - a) / 100.0); // s_memrealtime: 100 MHz on this target
            }
            engine.batch_trace.push_back(e.trace);
        }
    }
};
void Engine::set_batch_profile(bool enabled) {
    if (enabled && !batch_capacity) throw std::runtime_error("prepare_batch before profiling");
    if (enabled && !batch_profiler) batch_profiler = std::make_shared<BatchProfiler>(batch_capacity);
    batch_profile = enabled;
}
}
