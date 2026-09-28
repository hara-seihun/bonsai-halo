#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: qwen-accept MODEL OUTPUT_PREFIX PROMPT_REPEATS STEPS [--short-prompt]\n");
        return 2;
    }
    const bool short_prompt = argc == 6 && std::string(argv[5]) == "--short-prompt";
    if (argc == 6 && !short_prompt) return 2;
    const int repeats = std::atoi(argv[3]);
    const int steps = std::atoi(argv[4]);
    if (repeats < 1 || repeats > 64 || steps < 1 || steps > 128) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    cp.n_ubatch = 256;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt;
    if (short_prompt) {
        for (int i = 0; i < repeats; ++i) prompt += "Stars appear at night. ";
        prompt += "Write a Python function that computes the mean of a nonempty list of measurements.\n";
    } else {
        for (int i = 0; i < repeats; ++i) {
            prompt += "A small observatory measures the stars each night. Its notebook records positions, times, and temperatures. "
                      "The researchers compare observations before drawing conclusions. ";
        }
        prompt += "Write a Python function that computes the mean of a nonempty list of measurements.\n";
    }
    const int required = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (required <= 0 || required > 2048) return 5;
    std::vector<llama_token> tokens(required);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) != required) return 6;
    const auto start = std::chrono::steady_clock::now();
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) return 7;
    llama_synchronize(ctx);
    const auto prefill_end = std::chrono::steady_clock::now();
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream logits(std::string(argv[2]) + ".f32", std::ios::binary);
    std::ofstream ids(std::string(argv[2]) + ".tokens");
    if (!logits || !ids) return 8;
    for (int step = 0; step < steps; ++step) {
        const float * values = llama_get_logits_ith(ctx, -1);
        if (!values) return 9;
        logits.write(reinterpret_cast<const char *>(values), nv * sizeof(float));
        llama_token next = std::max_element(values, values + nv) - values;
        ids << next << '\n';
        if (step + 1 < steps && llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) return 10;
    }
    const auto finish = std::chrono::steady_clock::now();
    std::printf("{\"prompt_tokens\":%d,\"steps\":%d,\"vocabulary\":%d,\"prompt_seconds\":%.6f,\"decode_and_dump_seconds\":%.6f}\n",
        required, steps, nv, std::chrono::duration<double>(prefill_end-start).count(),
        std::chrono::duration<double>(finish-prefill_end).count());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
