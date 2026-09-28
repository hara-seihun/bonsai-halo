// Thirty-two independent one-token streams: same full model and one requested
// vocabulary row per step in both J policies; use distinct natural text tokens.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: short_j16_streams MODEL OUTPUT_PREFIX STEPS\n");
        return 2;
    }
    const int steps = std::atoi(argv[3]);
    if (steps < 2 || steps > 16) return 2;
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
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string text = "The astronomer recorded brightness across the northern sky and compared measurements with previous observations. "
        "Write a function to compute the average and return it with an explanation of the observations and their uncertainties. "
        "On clear nights the stars appeared brighter near the horizon than the scientists expected.";
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (n < 40) return 5;
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n, true, true) != n) return 6;
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream logits(std::string(argv[2]) + ".f32", std::ios::binary);
    if (!logits) return 7;
    auto batch = llama_batch_init(32, 0, 1);
    for (int step = 0; step < steps; ++step) {
        batch.n_tokens = 32;
        for (int i = 0; i < 32; ++i) {
            batch.token[i] = tokens[(i + 3*step) % n];
            batch.pos[i] = step;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = i;
            batch.logits[i] = (i == 31);
        }
        auto start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) return 8;
        llama_synchronize(ctx);
        auto end = std::chrono::steady_clock::now();
        auto values = llama_get_logits_ith(ctx, -1);
        if (!values) return 9;
        logits.write(reinterpret_cast<const char *>(values), nv * sizeof(float));
        const int top = std::max_element(values, values + nv) - values;
        std::printf("{\"step\":%d,\"milliseconds\":%.6f,\"top\":%d,\"vocabulary\":%d}\n",
            step, std::chrono::duration<double, std::milli>(end - start).count(), top, nv);
        std::fflush(stdout);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
