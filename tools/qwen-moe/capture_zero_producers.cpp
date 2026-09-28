#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Capture {
    std::string prefix;
    int seen[4]{};
    bool failed = false;
};

static bool capture(ggml_tensor * t, bool ask, void * user) {
    auto & c = *static_cast<Capture *>(user);
    const char * names[] = {"ffn_moe_topk-0", "ffn_moe_gate-0", "ffn_moe_up-0", "ffn_moe_swiglu-0"};
    int i = 0;
    while (i < 4 && std::strcmp(t->name, names[i])) ++i;
    if (i == 4 || ask) return true;
    if (t->type != (i == 0 ? GGML_TYPE_I32 : GGML_TYPE_F32) || t->nb[0] != 4 || t->ne[3] != 1) {
        std::fprintf(stderr, "bad tensor %s type=%d stride=%zu\n", t->name, t->type, t->nb[0]);
        c.failed = true;
        return false;
    }
    std::vector<char> storage(ggml_nbytes(t)), flat(ggml_nelements(t)*4);
    ggml_backend_tensor_get(t, storage.data(), 0, storage.size());
    for (int64_t k = 0; k < t->ne[2]; ++k)
        for (int64_t j = 0; j < t->ne[1]; ++j)
            std::memcpy(flat.data() + (k*t->ne[1]+j)*t->ne[0]*4,
                        storage.data() + k*t->nb[2]+j*t->nb[1], t->ne[0]*4);
    std::ofstream out(c.prefix + "." + names[i] + ".bin", std::ios::binary);
    out.write(flat.data(), flat.size());
    if (!out.good()) c.failed = true;
    std::printf("%s %lld %lld %lld %zu\n", names[i], (long long)t->ne[0],
                (long long)t->ne[1], (long long)t->ne[2], flat.size());
    ++c.seen[i];
    return !c.failed;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: capture_zero_producers MODEL OUTPUT_PREFIX TEXT_FILE\n");
        return 2;
    }
    std::ifstream input(argv[3]);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof()) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    Capture cap{argv[2]};
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = capture;
    cp.cb_eval_user_data = &cap;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto * vocab = llama_model_get_vocab(model);
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (n < 1 || n > 256) return 5;
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n, true, true) != n) return 6;
    int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), n));
    llama_synchronize(ctx);
    std::printf("tokens=%d counts=%d,%d,%d,%d\n", n, cap.seen[0], cap.seen[1], cap.seen[2], cap.seen[3]);
    for (int v : cap.seen) if (v != 1) cap.failed = true;
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc || cap.failed ? 7 : 0;
}
