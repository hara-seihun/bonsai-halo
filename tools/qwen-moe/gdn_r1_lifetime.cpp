// Save the stable 32-row pre-swap state for exact old-history source lookup.
// Identical workload to gdn_swap_probe.cpp; only the serialization point moves.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: generated_stream_j16 MODEL OUTPUT_PREFIX STEPS\n");
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
        // The first eight positions seed distinct, natural-text prefixes. Later
        // positions consume the prior step's actual greedy target decisions.
        const int rows = step == 0 ? 256 : 32;
        batch.n_tokens = rows;
        for (int i = 0; i < 32; ++i) {
            for (int p = 0; p < (step == 0 ? 8 : 1); ++p) {
                const int j = (step == 0 ? i * 8 + p : i);
                batch.token[j] = step == 0 ? tokens[(i + p) % n] : next[i];
                batch.pos[j] = step == 0 ? p : 7 + step;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = step == 2 && i < 2 ? 1 - i : i;
                batch.logits[j] = (step == 0 ? p == 7 : true);
            }
        }
        const auto start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) return 8;
        llama_synchronize(ctx);
        const auto end = std::chrono::steady_clock::now();
        for (int i = 0; i < 32; ++i) {
            const auto * values = llama_get_logits_ith(ctx, step == 0 ? i * 8 + 7 : i);
            if (!values) return 9;
            logits.write(reinterpret_cast<const char *>(values), nv * sizeof(float));
            next[i] = std::max_element(values, values + nv) - values;
            std::printf("{\"step\":%d,\"stream\":%d,\"milliseconds\":%.6f,\"top\":%d,\"vocabulary\":%d}\n",
                step, i, std::chrono::duration<double, std::milli>(end - start).count(), next[i], nv);
        }
        // The full context serializer includes every recurrent cache row, not
        // only the state that happens to affect the next few greedy decisions.
        // This is deliberately outside the timed llama_decode interval.
        if (step > 0) {
            const size_t capacity = llama_state_get_size(ctx);
            std::vector<uint8_t> state(capacity);
            const size_t written = llama_state_get_data(ctx, state.data(), state.size());
            if (written == 0 || written > state.size()) return 10;
            uint64_t hash = 14695981039346656037ull;
            for (size_t j = 0; j < written; ++j) {
                hash = (hash ^ state[j]) * 1099511628211ull;
            }
            if (step == 1) {
                std::ofstream snapshot(std::string(argv[2]) + ".state", std::ios::binary);
                if (!snapshot.write(reinterpret_cast<const char *>(state.data()), written)) return 11;
            }
            std::printf("{\"state_step\":%d,\"state_bytes\":%zu,\"state_fnv64\":\"%016llx\"}\n",
                step, written, (unsigned long long) hash);
        }
        std::fflush(stdout);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
