// Compare matched row orders while extracting layer-zero input through the
// installed runtime's built-in output tensor, without an evaluation callback.
// Check complete heads and serialized state against the uninstrumented arms:
// marking an additional graph output may itself alter allocation/lifetimes.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

// Exported by the selected libllama, though not declared in its public header.
void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool enable);
float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid);

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: gdn_permutation_equiv MODEL OUTPUT_PREFIX normal|swap STEPS\n");
        return 2;
    }
    const std::string order = argv[3];
    if (order != "normal" && order != "swap") return 2;
    const int steps = std::atoi(argv[4]);
    if (steps != 3) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_seq_max = 32;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    llama_set_embeddings_layer_inp(ctx, 0, true);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string text = "The astronomer recorded brightness across the northern sky and compared measurements with previous observations. "
        "Write a function to compute the average and return it with an explanation of the observations and their uncertainties. "
        "On clear nights the stars appeared brighter near the horizon than the scientists expected.";
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (n < 40) return 5;
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n, true, true) != n) return 6;
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream logits(std::string(argv[2]) + ".f32", std::ios::binary);
    if (!logits) return 7;
    auto batch = llama_batch_init(256, 0, 1);
    std::vector<llama_token> next(32);
    for (int step = 0; step < steps; ++step) {
        batch.n_tokens = step == 0 ? 256 : 32;
        for (int i = 0; i < 32; ++i) {
            const int seq = order == "swap" && step == 2 && i < 2 ? 1 - i : i;
            for (int p = 0; p < (step == 0 ? 8 : 1); ++p) {
                const int j = step == 0 ? i * 8 + p : i;
                batch.token[j] = step == 0 ? tokens[(seq + p) % n] : next[seq];
                batch.pos[j] = step == 0 ? p : 7 + step;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = seq;
                batch.logits[j] = step == 0 ? p == 7 : true;
            }
        }
        const auto start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) return 8;
        llama_synchronize(ctx);
        const float * layer0 = llama_get_embeddings_layer_inp(ctx, 0);
        if (!layer0) return 13;
        std::ofstream input(std::string(argv[2]) + ".step" + std::to_string(step) + ".layer0.f32", std::ios::binary);
        if (!input.write(reinterpret_cast<const char *>(layer0), (size_t) batch.n_tokens * 2048 * sizeof(float))) return 14;
        const auto end = std::chrono::steady_clock::now();
        std::vector<float> heads(32 * (size_t) nv);
        for (int i = 0; i < 32; ++i) {
            const int seq = order == "swap" && step == 2 && i < 2 ? 1 - i : i;
            const float * values = llama_get_logits_ith(ctx, step == 0 ? i * 8 + 7 : i);
            if (!values) return 9;
            std::copy(values, values + nv, heads.begin() + seq * (size_t) nv);
            next[seq] = std::max_element(values, values + nv) - values;
            std::printf("{\"step\":%d,\"row\":%d,\"sequence\":%d,\"milliseconds\":%.6f,\"top\":%d}\n",
                step, i, seq, std::chrono::duration<double, std::milli>(end - start).count(), next[seq]);
        }
        logits.write(reinterpret_cast<const char *>(heads.data()), heads.size() * sizeof(float));
        if (!logits) return 10;
        if (step == 2) {
            std::vector<uint8_t> state(llama_state_get_size(ctx));
            const size_t written = llama_state_get_data(ctx, state.data(), state.size());
            if (!written || written > state.size()) return 11;
            std::ofstream out(std::string(argv[2]) + ".state", std::ios::binary);
            if (!out.write(reinterpret_cast<const char *>(state.data()), written)) return 12;
            std::printf("{\"state_bytes\":%zu}\n", written);
        }
        std::fflush(stdout);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
