// Matched complete-head generated streams after real, occupied per-sequence prefixes.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 7) {
        std::fprintf(stderr, "usage: generated_context MODEL PREFIX DEPTH STEPS [STATE_PREFIX save|load]\n");
        return 2;
    }
    const int depth = std::atoi(argv[3]), steps = std::atoi(argv[4]);
    if (depth < 8 || depth > 512 || depth % 8 || steps < 3 || steps > 12) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(8192, 32 * (depth + steps + 16));
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_seq_max = 32;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string text = "The astronomer recorded brightness across the northern sky and compared measurements with previous observations. Write a function to compute the average and return it with an explanation of the observations and their uncertainties. On clear nights the stars appeared brighter near the horizon than the scientists expected.";
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (n < 40) return 5;
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n, true, true) != n) return 6;
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream logits(std::string(argv[2]) + ".f32", std::ios::binary);
    std::ofstream seed_logits(std::string(argv[2]) + ".seed.f32", std::ios::binary);
    if (!logits || !seed_logits) return 7;
    auto batch = llama_batch_init(256, 0, 1);
    std::vector<llama_token> next(32);
    const bool load = argc == 7 && std::string(argv[6]) == "load";
    if (argc == 7 && !load && std::string(argv[6]) != "save") return 2;
    for (int base = 0; !load && base < depth; base += 8) {
        batch.n_tokens = 256;
        for (int i = 0; i < 32; ++i) for (int p = 0; p < 8; ++p) {
            const int j = i * 8 + p;
            batch.token[j] = tokens[(i + base + p) % n];
            batch.pos[j] = base + p;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = i;
            batch.logits[j] = base + 8 == depth && p == 7;
        }
        if (llama_decode(ctx, batch) != 0) return 8;
        if (base + 8 == depth) {
            llama_synchronize(ctx);
            for (int i = 0; i < 32; ++i) {
                const auto * v = llama_get_logits_ith(ctx, i * 8 + 7);
                if (!v) return 9;
                seed_logits.write(reinterpret_cast<const char *>(v), nv * sizeof(float));
                next[i] = std::max_element(v, v + nv) - v;
            }
        }
    }
    if (argc == 7) {
        const std::string prefix = argv[5];
        if (load) {
            std::ifstream state(prefix + ".state", std::ios::binary | std::ios::ate);
            std::ifstream ids(prefix + ".next", std::ios::binary);
            if (!state || !ids) return 12;
            const auto length = state.tellg();
            if (length <= 0) return 13;
            std::vector<uint8_t> bytes(static_cast<size_t>(length));
            state.seekg(0);
            if (!state.read(reinterpret_cast<char *>(bytes.data()), bytes.size()) ||
                !ids.read(reinterpret_cast<char *>(next.data()), next.size() * sizeof(llama_token)) ||
                llama_state_set_data(ctx, bytes.data(), bytes.size()) != bytes.size()) return 14;
        } else {
            const size_t length = llama_state_get_size(ctx);
            std::vector<uint8_t> bytes(length);
            const size_t written = llama_state_get_data(ctx, bytes.data(), length);
            std::ofstream state(prefix + ".state", std::ios::binary);
            std::ofstream ids(prefix + ".next", std::ios::binary);
            if (!written || written > length ||
                !state.write(reinterpret_cast<const char *>(bytes.data()), written) ||
                !ids.write(reinterpret_cast<const char *>(next.data()), next.size() * sizeof(llama_token))) return 15;
        }
    }
    for (int step = 0; step < steps; ++step) {
        batch.n_tokens = 32;
        for (int i = 0; i < 32; ++i) {
            batch.token[i] = next[i];
            batch.pos[i] = depth + step;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = i;
            batch.logits[i] = true;
        }
        auto start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, batch) != 0) return 10;
        llama_synchronize(ctx);
        auto end = std::chrono::steady_clock::now();
        for (int i = 0; i < 32; ++i) {
            const auto * v = llama_get_logits_ith(ctx, i);
            if (!v) return 11;
            logits.write(reinterpret_cast<const char *>(v), nv * sizeof(float));
            next[i] = std::max_element(v, v + nv) - v;
        }
        std::printf("{\"step\":%d,\"milliseconds\":%.6f,\"heads\":32,\"vocabulary\":%d}\n", step,
            std::chrono::duration<double, std::milli>(end - start).count(), nv);
        std::fflush(stdout);
    }
    if (load) {
        const size_t capacity = llama_state_get_size(ctx);
        std::vector<uint8_t> bytes(capacity);
        const size_t written = llama_state_get_data(ctx, bytes.data(), capacity);
        std::ofstream state(std::string(argv[2]) + ".endstate", std::ios::binary);
        if (!written || written > capacity ||
            !state.write(reinterpret_cast<const char *>(bytes.data()), written)) return 16;
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
