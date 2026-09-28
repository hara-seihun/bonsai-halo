#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

struct Capture {
    bool enabled = false;
    bool failed = false;
    int row = 0;
    std::array<int, 40> seen{};
    std::array<std::vector<int32_t>, 40> ids;
};

static bool capture_topk(ggml_tensor * tensor, bool ask, void * user) {
    auto & c = *static_cast<Capture *>(user);
    int layer = -1, end = 0;
    if (std::sscanf(tensor->name, "ffn_moe_topk-%d%n", &layer, &end) != 1 ||
        end != int(std::strlen(tensor->name)) || layer < 0 || layer >= 40) return true;
    if (ask) return c.enabled;
    if (!c.enabled) return true;
    if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 8 || tensor->ne[1] != 1 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != sizeof(int32_t)) {
        std::fprintf(stderr, "unexpected topk %s shape=%lld,%lld stride=%zu\n", tensor->name,
                     (long long) tensor->ne[0], (long long) tensor->ne[1], tensor->nb[0]);
        c.failed = true;
        return false;
    }
    int32_t ids[8];
    ggml_backend_tensor_get(tensor, ids, 0, sizeof(ids));
    for (int id : ids) if (id < 0 || id >= 256) c.failed = true;
    c.ids[layer].insert(c.ids[layer].end(), ids, ids + 8);
    c.seen[layer]++;
    return !c.failed;
}

static int best(const float * values, int n) {
    return int(std::max_element(values, values + n) - values);
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: capture_generated_routes MODEL PREFIX TEXT_FILE STEPS PREFIX_TOKENS\n");
        return 2;
    }
    const int steps = std::stoi(argv[4]), prefix_tokens = std::stoi(argv[5]);
    if (steps < 2 || steps > 128 || prefix_tokens < 1 || prefix_tokens > 256) return 2;
    std::ifstream input(argv[3]);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof()) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 256; cp.n_ubatch = 256;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    Capture cap;
    cp.cb_eval = capture_topk; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto * vocab = llama_model_get_vocab(model);
    int needed = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (needed < prefix_tokens || needed > 256) return 5;
    std::vector<llama_token> tokens(needed);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) != needed) return 6;
    const int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> plain(steps), observed(steps);
    for (int arm = 0; arm < 2; arm++) {
        cap.enabled = false;
        if (arm) llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), prefix_tokens)) != 0) return 7;
        auto next = llama_token(best(llama_get_logits_ith(ctx, -1), nv));
        for (int row = 0; row < steps; row++) {
            cap.enabled = arm == 1; cap.row = row;
            if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) return 8;
            const float * values = llama_get_logits_ith(ctx, -1);
            next = llama_token(best(values, nv));
            if (arm) observed[row] = next;
            else {
                plain[row] = next;
            }
            if (cap.failed) return 9;
        }
    }
    for (int layer = 0; layer < 40; layer++) {
        if (cap.seen[layer] != steps || cap.ids[layer].size() != size_t(steps) * 8) return 10;
        std::ofstream out(std::string(argv[2]) + ".layer-" + std::to_string(layer) + ".i32", std::ios::binary);
        out.write(reinterpret_cast<const char *>(cap.ids[layer].data()), cap.ids[layer].size() * sizeof(int32_t));
        if (!out.good()) return 11;
    }
    std::ofstream out(std::string(argv[2]) + ".tokens", std::ios::binary);
    out.write(reinterpret_cast<const char *>(plain.data()), plain.size() * sizeof(llama_token));
    out.write(reinterpret_cast<const char *>(observed.data()), observed.size() * sizeof(llama_token));
    if (!out.good()) return 12;
    std::printf("prefix=%d steps=%d greedy_equal=%d/%d\n", prefix_tokens, steps,
                int(std::equal(plain.begin(), plain.end(), observed.begin())), steps);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
