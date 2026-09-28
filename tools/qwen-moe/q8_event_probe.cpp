// Low-overhead HIP event census of quantized Qwen decode launches. Does not edit
// the selected engine or its kernels. Only valid with graphs disabled: launches
// inside a replay are invisible to hipLaunchKernel interposition.
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using Launch = hipError_t (*)(const void *, dim3, dim3, void **, size_t, hipStream_t);
struct Entry {
    hipEvent_t start, stop;
    unsigned grid, block;
    std::string name;
};
static std::mutex mu;
static std::vector<Entry> entries;
static size_t graph_launches = 0, ordinary_launches = 0;
static bool registered = false;
static void require(hipError_t status) {
    if (status != hipSuccess) {
        fprintf(stderr, "q8-event: HIP error: %s\n", hipGetErrorString(status));
        abort();
    }
}

static void report() {
    std::lock_guard<std::mutex> lock(mu);
    const char *path = getenv("Q8_EVENT_OUTPUT");
    if (!path) return;
    FILE *file = fopen(path, "w");
    if (!file) { perror(path); return; }
    fprintf(file, "ordinal,name,grid,block,device_ms\n");
    for (size_t i = 0; i < entries.size(); ++i) {
        auto &e = entries[i];
        // This waits only after benchmark timing has finished; no per-launch
        // synchronization is inserted in the model's execution.
        hipError_t wait = hipEventSynchronize(e.stop);
        float ms = -1;
        hipError_t elapsed = wait == hipSuccess ? hipEventElapsedTime(&ms, e.start, e.stop) : wait;
        if (elapsed != hipSuccess) fprintf(stderr, "q8-event: event %zu failed: %s\n", i, hipGetErrorString(elapsed));
        fprintf(file, "%zu,%s,%u,%u,%.9g\n", i, e.name.c_str(), e.grid, e.block, ms);
        require(hipEventDestroy(e.start));
        require(hipEventDestroy(e.stop));
    }
    fclose(file);
    fprintf(stderr, "q8-event: %zu selected launches, %zu total launches, %zu graph replays; %s\n",
            entries.size(), ordinary_launches, graph_launches, path);
}

extern "C" hipError_t hipLaunchKernel(const void *function, dim3 grid, dim3 block,
                                       void **args, size_t shared, hipStream_t stream) {
    static auto original = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "hipLaunchKernel"));
    const char *name = hipKernelNameRefByPtr(function, stream);
    const bool selected = getenv("Q8_EVENT_OUTPUT") && name && strstr(name, "mul_mat_vec_q") &&
        (strstr(name, "ggml_type8E") || strstr(name, "ggml_type12E") ||
         strstr(name, "ggml_type13E") || strstr(name, "ggml_type14E"));
    hipEvent_t start{}, stop{};
    if (selected) {
        require(hipEventCreate(&start));
        require(hipEventCreate(&stop));
        require(hipEventRecord(start, stream));
    }
    auto result = original(function, grid, block, args, shared, stream);
    if (selected) {
        require(hipEventRecord(stop, stream));
        std::lock_guard<std::mutex> lock(mu);
        entries.push_back({start, stop, grid.x * grid.y * grid.z,
                           block.x * block.y * block.z, name});
    }
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!registered) { atexit(report); registered = true; }
        ++ordinary_launches;
    }
    return result;
}
using GraphLaunch = hipError_t (*)(hipGraphExec_t, hipStream_t);
extern "C" hipError_t hipGraphLaunch(hipGraphExec_t graph, hipStream_t stream) {
    static auto original = reinterpret_cast<GraphLaunch>(dlsym(RTLD_NEXT, "hipGraphLaunch"));
    { std::lock_guard<std::mutex> lock(mu); ++graph_launches; }
    return original(graph, stream);
}
