#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: short_width_probe MODEL OUTPUT_PREFIX REPEATS STEPS\n");
        return 2;
    }
    const int repeats = std::atoi(argv[3]);
    const int steps = std::atoi(argv[4]);
    if (repeats < 2 || repeats > 32 || steps < 1 || steps > 16) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt =
        "The astronomer recorded the brightness of each star on a clear night, then compared the measurements "
        "with previous observations. Write a Python function that computes their average and handles an empty list. ";
    const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (count < 32) return 5;
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, true) != count) return 6;
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream logits(std::string(argv[2]) + ".f32", std::ios::binary);
    if (!logits) return 7;
    for (int width : {16, 24, 32}) {
        for (int repeat = 0; repeat < repeats; ++repeat) {
            llama_memory_clear(llama_get_memory(ctx), true);
            auto start = std::chrono::steady_clock::now();
            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), width)) != 0) return 8;
            llama_synchronize(ctx);
            auto end = std::chrono::steady_clock::now();
            const float * values = llama_get_logits_ith(ctx, -1);
            if (!values) return 9;
            logits.write(reinterpret_cast<const char *>(values), nv * sizeof(float));
            int top = std::max_element(values, values + nv) - values;
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::printf("{\"width\":%d,\"repeat\":%d,\"milliseconds\":%.6f,\"top\":%d,\"vocabulary\":%d}\n",
                width, repeat, ms, top, nv);
            std::fflush(stdout);
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
