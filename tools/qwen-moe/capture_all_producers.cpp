#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

struct Capture {
    std::string prefix;
    int tokens = 0;
    std::array<std::array<int, 5>, 40> seen{};
    bool failed = false;
};

static bool observe(ggml_tensor * t, bool ask, void * user) {
    auto & c = *static_cast<Capture *>(user);
    constexpr const char * names[] = {"attn_post_norm", "ffn_moe_topk", "ffn_moe_weights_norm", "ffn_moe_down", "ffn_moe_swiglu"};
    int layer = -1, end = 0, kind = 0;
    for (; kind < 5; ++kind) {
        std::string pattern = std::string(names[kind]) + "-%d%n";
        if (std::sscanf(t->name, pattern.c_str(), &layer, &end) == 1 &&
            end == int(std::strlen(t->name)) && layer >= 0 && layer < 40) break;
        end = 0;
    }
    if (kind == 5 || ask) return true;
    const int width = kind == 0 || kind == 3 ? 2048 : kind == 4 ? 512 : 8;
    const int slots = kind == 3 || kind == 4 ? 8 : 1;
    if (t->type != (kind == 1 ? GGML_TYPE_I32 : GGML_TYPE_F32) ||
        t->ne[0] != width || t->ne[1] != (slots == 8 ? 8 : c.tokens) ||
        t->ne[2] != (slots == 8 ? c.tokens : 1) || t->ne[3] != 1 || t->nb[0] != 4) {
        std::fprintf(stderr, "unexpected %s: type=%d shape=%lld,%lld,%lld,%lld stride=%zu,%zu,%zu\n",
            t->name, t->type, (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3], t->nb[0], t->nb[1], t->nb[2]);
        c.failed = true;
        return false;
    }
    std::vector<char> bytes(size_t(width) * c.tokens * slots * 4);
    for (int row = 0; row < c.tokens; ++row)
        for (int slot = 0; slot < slots; ++slot)
            ggml_backend_tensor_get(t, bytes.data() + (size_t(row)*slots + slot)*width*4,
                                    slots == 8 ? size_t(row)*t->nb[2] + size_t(slot)*t->nb[1]
                                              : size_t(row)*t->nb[1], size_t(width)*4);
    const auto path = c.prefix + ".layer-" + std::to_string(layer) + "." + names[kind] + (kind == 1 ? ".i32" : ".f32");
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    if (!out.good() || ++c.seen[layer][kind] != 1) c.failed = true;
    return !c.failed;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: capture_all_producers MODEL PREFIX TEXT_FILE MAX_TOKENS (1..256)\n");
        return 2;
    }
    const int maximum = std::atoi(argv[4]);
    if (maximum < 1 || maximum > 256) return 2;
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
    cp.cb_eval = observe;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto * vocab = llama_model_get_vocab(model);
    const int needed = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (needed < 1) return 5;
    std::vector<llama_token> tokens(needed);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) != needed) return 6;
    if ((int)tokens.size() > maximum) tokens.resize(maximum);
    cap.tokens = tokens.size();
    std::ofstream tok(cap.prefix + ".tokens", std::ios::binary);
    tok.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(llama_token));
    tok.close();
    const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    llama_synchronize(ctx);
    for (int layer = 0; layer < 40; ++layer)
        for (int kind = 0; kind < 5; ++kind)
            if (cap.seen[layer][kind] != 1) {
                std::fprintf(stderr, "layer %d kind %d seen %d\n", layer, kind, cap.seen[layer][kind]);
                cap.failed = true;
            }
    std::printf("tokens=%d layers=40 nodes=5\n", cap.tokens);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc || cap.failed ? 7 : 0;
}
