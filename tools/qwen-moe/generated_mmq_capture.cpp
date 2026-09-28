// Capture real generated 32-sequence routes. Callback materialization changes graph fusion;
// this is a route geometry instrument, not a native throughput probe.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Capture {
    bool enabled = false;
    bool failed = false;
    std::array<int, 40> seen{};
    std::array<std::vector<int32_t>, 40> ids;
};

static bool topk(ggml_tensor * tensor, bool ask, void * user) {
    auto & cap = *static_cast<Capture *>(user);
    int layer = -1, end = 0;
    if (std::sscanf(tensor->name, "ffn_moe_topk-%d%n", &layer, &end) != 1 ||
        end != int(std::strlen(tensor->name)) || layer < 0 || layer >= 40) return true;
    if (ask) return cap.enabled;
    if (!cap.enabled) return true;
    if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 8 || tensor->ne[1] != 32 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != 4 || tensor->nb[1] < 32) {
        std::fprintf(stderr, "unexpected %s shape=%lld,%lld,%lld,%lld strides=%zu,%zu\n", tensor->name,
            (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
            (long long) tensor->ne[3], tensor->nb[0], tensor->nb[1]);
        cap.failed = true;
        return false;
    }
    int32_t ids[256];
    for (int row = 0; row < 32; ++row)
        ggml_backend_tensor_get(tensor, ids + row * 8, row * tensor->nb[1], 8 * sizeof(int32_t));
    for (int id : ids) if (id < 0 || id >= 256) cap.failed = true;
    cap.ids[layer].insert(cap.ids[layer].end(), ids, ids + 256);
    cap.seen[layer]++;
    return !cap.failed;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: generated_mmq_capture MODEL PREFIX STEPS OBSERVE(0|1)\n");
        return 2;
    }
    const int steps = std::atoi(argv[3]);
    if (steps < 2 || steps > 16) return 2;
    const bool observe = std::atoi(argv[4]) != 0;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 512; cp.n_ubatch = 256; cp.n_seq_max = 32;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    Capture cap;
    cp.cb_eval = topk; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string text = "The astronomer recorded brightness across the northern sky and compared measurements with previous observations. "
        "Write a function to compute the average and return it with an explanation of the observations and their uncertainties. "
        "On clear nights the stars appeared brighter near the horizon than the scientists expected.";
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (n < 40) return 5;
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n, true, true) != n) return 6;
    const int nv = llama_vocab_n_tokens(vocab);
    auto batch = llama_batch_init(256, 0, 1);
    std::vector<llama_token> next(32);
    std::ofstream decisions(std::string(argv[2]) + ".tokens", std::ios::binary);
    if (!decisions) return 7;
    for (int step = 0; step < steps; ++step) {
        const int rows = step == 0 ? 256 : 32;
        cap.enabled = observe && step > 0;
        batch.n_tokens = rows;
        for (int i = 0; i < 32; ++i) for (int p = 0; p < (step == 0 ? 8 : 1); ++p) {
            const int j = step == 0 ? i * 8 + p : i;
            batch.token[j] = step == 0 ? tokens[(i + p) % n] : next[i];
            batch.pos[j] = step == 0 ? p : 7 + step;
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = i;
            batch.logits[j] = step == 0 ? p == 7 : true;
        }
        if (llama_decode(ctx, batch) != 0) return 8;
        llama_synchronize(ctx);
        for (int i = 0; i < 32; ++i) {
            const float * values = llama_get_logits_ith(ctx, step == 0 ? i * 8 + 7 : i);
            if (!values) return 9;
            next[i] = llama_token(std::max_element(values, values + nv) - values);
        }
        decisions.write(reinterpret_cast<const char *>(next.data()), next.size() * sizeof(llama_token));
        if (cap.failed) return 10;
        std::printf("step=%d first=%d last=%d\n", step, next.front(), next.back());
        std::fflush(stdout);
    }
    for (int layer = 0; layer < 40 && observe; ++layer) {
        if (cap.seen[layer] != steps - 1 || cap.ids[layer].size() != size_t(steps - 1) * 256) return 11;
        std::ofstream out(std::string(argv[2]) + ".layer-" + std::to_string(layer) + ".i32", std::ios::binary);
        out.write(reinterpret_cast<const char *>(cap.ids[layer].data()), cap.ids[layer].size() * 4);
        if (!out.good()) return 12;
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return decisions.good() ? 0 : 13;
}
