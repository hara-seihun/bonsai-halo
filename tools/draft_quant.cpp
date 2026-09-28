// Build a drafter weight cache beside its safetensors file, without a GPU.
//
// `Engine::load_dflash` builds a missing cache itself, but quantising 1.7 G parameters is minutes
// of CPU and the engine only reaches that code with the model resident and, in this lane, with the
// GPU measurement lock held. Run this first and a measured process starts against a warm cache.
//
//   tools/draft_quant ~/data/bonsai2/drafters/dflash2.safetensors [q8|q4|both] [threads]
#include "q8.h"
#include "halo_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace halo;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s DRAFTER.safetensors [q8|q4|both] [threads]\n", argv[0]); return 2; }
    const std::string path = argv[1];
    const std::string which = argc > 2 ? argv[2] : "q4";
    const int nthreads = argc > 3 ? atoi(argv[3]) : (int) std::thread::hardware_concurrency();

    Safetensors st;
    st.open(path);
    std::vector<std::string> names = { "fc.weight", "candidate_selector.hidden_projection.weight" };
    for (int l = 0; l < DF_LAYERS; l++) {
        const std::string pre = "layers." + std::to_string(l) + ".";
        for (const char * n : { "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight" })
            names.push_back(pre + n);
    }
    size_t params = 0;
    for (auto & n : names) { const StTensor & t = st.t(n); params += (size_t) t.shape[0] * t.shape[1]; }
    printf("%s: %zu weight tensors, %.3f G parameters\n", path.c_str(), names.size(), params / 1e9);
    printf("  q8 image %.3f GB, q4 image %.3f GB\n",
           (double) params * Q8_TILE_BLOCK_BYTES / (TILE_ROWS * BLOCK) / 1e9,
           (double) params * 2112 / (TILE_ROWS * BLOCK) / 1e9);

    for (int bits : { 8, 4 }) {
        if (which != "both" && which != (bits == 8 ? "q8" : "q4")) continue;
        Q8Cache cache;
        const bool built = cache.open_or_build(st, path, names, nthreads, bits);
        if (!built) printf("q%d cache already present and current\n", bits);
    }
    return 0;
}
