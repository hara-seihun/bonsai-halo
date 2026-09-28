#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Capture {
    std::string prefix;
    int batch = 0;
    bool failed = false;
    std::array<int, 5> seen{};
};

static bool capture_tensor(ggml_tensor * t, bool ask, void * ptr) {
    auto & c = *static_cast<Capture *>(ptr);
    const char * names[] = {"attn_post_norm-0", "ffn_moe_topk-0", "ffn_moe_weights_norm-0",
                          "ffn_moe_swiglu-0", "ffn_moe_down-0"};
    int index = 0;
    for (; index < 5; ++index) if (std::strcmp(t->name, names[index]) == 0) break;
    if (index == 5) return true;
    if (ask) return true;
    if (t->type != (index == 1 ? GGML_TYPE_I32 : GGML_TYPE_F32) || t->nb[0] != 4 || t->ne[3] != 1) {
        std::fprintf(stderr, "unexpected tensor %s type=%d strides=%zu,%zu\n", t->name, t->type, t->nb[0], t->nb[1]);
        c.failed = true;
        return false;
    }
    auto path = c.prefix + "." + std::to_string(c.batch) + "." + names[index] + ".bin";
    std::vector<char> bytes(ggml_nelements(t) * sizeof(float));
    std::vector<char> storage(ggml_nbytes(t));
    ggml_backend_tensor_get(t, storage.data(), 0, storage.size());
    for (int64_t k = 0; k < t->ne[2]; ++k)
        for (int64_t j = 0; j < t->ne[1]; ++j)
            std::memcpy(bytes.data() + (k*t->ne[1] + j)*t->ne[0]*4,
                        storage.data() + k*t->nb[2] + j*t->nb[1], t->ne[0]*4);
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    if (!out.good()) { c.failed = true; return false; }
    std::fprintf(stdout, "%s shape=%lld,%lld,%lld,%lld bytes=%zu\n", path.c_str(),
                 (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], bytes.size());
    c.seen[index]++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: qwen-capture-routes MODEL PREFIX TEXT_FILE\n");
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
    Capture cap{argv[2]};
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = capture_tensor;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto * vocab = llama_model_get_vocab(model);
    int required = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (required < 1 || required > 256) return 5;
    std::vector<llama_token> tokens(required);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) != required) return 6;
    std::ofstream tok(cap.prefix + ".tokens", std::ios::binary);
    tok.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(llama_token));
    tok.close();
    const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    llama_synchronize(ctx);
    std::printf("tokens=%d callback_counts=%d,%d,%d,%d,%d\n", required,
                cap.seen[0], cap.seen[1], cap.seen[2], cap.seen[3], cap.seen[4]);
    for (int count : cap.seen) if (count != 1) cap.failed = true;
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc || cap.failed ? 7 : 0;
}
