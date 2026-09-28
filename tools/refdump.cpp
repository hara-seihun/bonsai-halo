// Reference activation dump from llama.cpp (CPU backend) for a raw prompt, last token only.
// Usage: refdump MODEL OUTDIR "prompt"  -> OUTDIR/<name>.bin (f32) and OUTDIR/index.txt (name shape)
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <sys/stat.h>

struct Ctx { std::string dir; bool active = false; FILE * index; std::set<std::string> seen; };

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    Ctx * c = (Ctx *) ud;
    std::string name = t->name;
    if (ask) {
        return c->active;
    }
    if (!c->active) return true;
    if (t->type != GGML_TYPE_F32) return true;
    // keep only 2D-ish activations of modest size
    size_t n = ggml_nelements(t);
    if (n > 4 * 1024 * 1024) return true;
    if (!ggml_is_contiguous(t)) return true;
    if (c->seen.count(name)) return true;
    c->seen.insert(name);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n * 4);
    std::string fn = c->dir + "/" + name + ".bin";
    for (auto & ch : fn) if (ch == '/' && &ch != &fn[0] && fn.compare(0, c->dir.size(), c->dir) == 0 && (&ch - &fn[0]) > (long) c->dir.size()) ch = '_';
    FILE * f = fopen(fn.c_str(), "wb"); if (f) { fwrite(buf.data(), 4, n, f); fclose(f); }
    fprintf(c->index, "%s %lld %lld %lld %lld\n", name.c_str(), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: refdump MODEL OUTDIR PROMPT [ngl]\n"); return 1; }
    std::string model_path = argv[1], dir = argv[2], prompt = argv[3];
    int ngl = argc > 4 ? atoi(argv[4]) : 0;
    mkdir(dir.c_str(), 0700);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), false, true);
    toks.resize(n);
    fprintf(stderr, "tokens:"); for (auto t : toks) fprintf(stderr, " %d", t); fprintf(stderr, "\n");

    Ctx c; c.dir = dir; c.index = fopen((dir + "/index.txt").c_str(), "w");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 1; cp.n_ubatch = 1; cp.n_seq_max = 1;
    cp.cb_eval = cb; cp.cb_eval_user_data = &c;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    for (int i = 0; i < n; i++) {
        c.active = (i == n - 1);
        llama_batch b = llama_batch_get_one(&toks[i], 1);
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    // logits
    const float * lg = llama_get_logits_ith(ctx, -1);
    int nv = llama_vocab_n_tokens(vocab);
    FILE * f = fopen((dir + "/logits.bin").c_str(), "wb"); fwrite(lg, 4, nv, f); fclose(f);
    int best = 0; for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    char buf[64]; int pn = llama_token_to_piece(vocab, best, buf, sizeof buf, 0, true);
    fprintf(stderr, "argmax %d '%.*s'\n", best, pn, buf);
    fclose(c.index);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
