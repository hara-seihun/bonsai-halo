#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

struct Capture {
    std::string prefix;
    int tokens = 0;
    std::array<int, 40> seen{};
    bool failed = false;
};

static bool capture_topk(ggml_tensor * tensor, bool ask, void * user) {
    auto & c = *static_cast<Capture *>(user);
    int layer = -1;
    int end = 0;
    if (std::sscanf(tensor->name, "ffn_moe_topk-%d%n", &layer, &end) != 1 ||
        end != int(std::strlen(tensor->name)) || layer < 0 || layer >= 40) return true;
    if (ask) return true;
    if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 8 || tensor->ne[1] != c.tokens ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != sizeof(int32_t)) {
        std::fprintf(stderr, "unexpected %s shape=%lld,%lld stride=%zu,%zu\n", tensor->name,
            (long long) tensor->ne[0], (long long) tensor->ne[1], tensor->nb[0], tensor->nb[1]);
        c.failed = true;
        return false;
    }
    std::vector<int32_t> ids(size_t(c.tokens) * 8);
    // Top-k is a strided view of the router sort. Copy each row at its true stride.
    for (int row = 0; row < c.tokens; row++) {
        auto * device_row = reinterpret_cast<ggml_tensor *>(tensor);
        ggml_backend_tensor_get(device_row, ids.data() + size_t(row) * 8,
                                size_t(row) * tensor->nb[1], 8 * sizeof(int32_t));
    }
    const auto path = c.prefix + ".layer-" + std::to_string(layer) + ".i32";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(ids.data()), ids.size() * sizeof(int32_t));
    if (!out.good()) c.failed = true;
    c.seen[layer]++;
    return !c.failed;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: capture_all_routes MODEL PREFIX TEXT_FILE\n");
        return 2;
    }
    std::ifstream input(argv[3]);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof()) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    Capture cap{argv[2]};
    cp.cb_eval = capture_topk;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto * vocab = llama_model_get_vocab(model);
    int needed = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (needed < 1 || needed > 256) return 5;
    std::vector<llama_token> tokens(needed);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) != needed) return 6;
    cap.tokens = needed;
    std::ofstream tok(cap.prefix + ".tokens", std::ios::binary);
    tok.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(llama_token));
    tok.close();
    int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    llama_synchronize(ctx);
    for (int layer = 0; layer < 40; layer++) {
        if (cap.seen[layer] != 1) { std::fprintf(stderr, "layer %d seen %d\n", layer, cap.seen[layer]); cap.failed = true; }
    }
    std::printf("tokens=%d layers=40\n", needed);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc || cap.failed ? 7 : 0;
}
