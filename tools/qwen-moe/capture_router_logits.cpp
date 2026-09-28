#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Capture { std::string path; int seen = 0; bool failed = false; };

static bool capture(ggml_tensor * t, bool ask, void * arg) {
    auto & c = *static_cast<Capture *>(arg);
    if (std::strcmp(t->name, "ffn_moe_logits-0") != 0) return true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != 256 || t->nb[0] != 4 || t->ne[2] != 1 || t->ne[3] != 1) {
        std::fprintf(stderr, "unexpected router tensor shape or type\n"); c.failed = true; return false;
    }
    std::vector<char> storage(ggml_nbytes(t));
    ggml_backend_tensor_get(t, storage.data(), 0, storage.size());
    std::ofstream out(c.path, std::ios::binary);
    for (int64_t row = 0; row < t->ne[1]; ++row)
        out.write(storage.data() + row * t->nb[1], 256 * sizeof(float));
    c.failed |= !out.good();
    ++c.seen;
    std::printf("rows=%lld bytes=%lld\n", (long long)t->ne[1], (long long)t->ne[1]*1024);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: capture_router_logits MODEL TEXT_FILE OUTPUT\n"); return 2; }
    std::ifstream in(argv[2]);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!in.good() && !in.eof()) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    Capture cap{argv[3]};
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 256; cp.n_ubatch = 256;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = capture; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto * vocab = llama_model_get_vocab(model);
    const int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (n < 1 || n > 256) return 5;
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n, true, true) != n) return 6;
    const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), n));
    llama_synchronize(ctx);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return rc || cap.failed || cap.seen != 1 ? 7 : 0;
}
