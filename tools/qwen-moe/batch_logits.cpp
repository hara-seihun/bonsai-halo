// Replay one target-generated continuation at several verification widths.
// This diagnoses target arithmetic independently of any draft or sampler.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

static void require(bool yes, const char * what) {
    if (!yes) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}
static int top(const float * p, int n) { return int(std::max_element(p, p + n) - p); }
static void compare(const std::vector<float> & reference, const float * actual, int n, int width, int offset, int row) {
    int differing = 0;
    float maximum = 0;
    for (int j = 0; j < n; ++j) {
        uint32_t a, b;
        std::memcpy(&a, &reference[j], 4);
        std::memcpy(&b, actual + j, 4);
        differing += a != b;
        maximum = std::max(maximum, std::abs(reference[j] - actual[j]));
    }
    const int winner = top(reference.data(), n);
    const int other = top(actual, n);
    float runner = -INFINITY;
    int runner_id = -1;
    for (int j = 0; j < n; ++j) if (j != winner && reference[j] > runner) {
        runner = reference[j]; runner_id = j;
    }
    std::printf("{\"width\":%d,\"offset\":%d,\"row\":%d,\"bit_differences\":%d,\"max_abs\":%.9g,\"plain_top\":%d,\"batch_top\":%d,\"plain_margin\":%.9g,\"batch_at_plain_top\":%.9g,\"batch_at_plain_runner\":%.9g}\n",
        width, offset, row, differing, maximum, winner, other, reference[winner] - runner, actual[winner],
        actual[runner_id]);
}
struct LayerProbe {
    int stage = 0;
    int width = 0;
    int row = 0;
    std::map<std::string, std::vector<float>> plain;
    std::map<std::string, std::vector<float>> batch;
};
static bool layer_callback(ggml_tensor * t, bool ask, void * userdata) {
    auto & p = *static_cast<LayerProbe *>(userdata);
    if (std::getenv("QWEN_PROBE_ALIAS")) {
        if (ask && p.stage && (std::strncmp(t->name, "ffn_moe_", 8) == 0) &&
            (std::strstr(t->name, "-0") != nullptr) && t->data) {
            std::fprintf(stderr, "router_alloc stage=%d width=%d row=%d name=%s op=%d shape=%lld,%lld bytes=%zu data=%p buffer=%p\n",
                p.stage, p.width, p.row, t->name, int(t->op), (long long)t->ne[0], (long long)t->ne[1],
                ggml_nbytes(t), t->data, (void *)t->buffer);
        }
        return false;
    }
    if (std::getenv("QWEN_PROBE_NO_CUT") || !p.stage || t->type != GGML_TYPE_F32 || t->ne[0] > 2048 || t->ne[1] < 1 || t->ne[1] > 8) return false;
    const std::string name(t->name);
    const char * route = std::getenv("QWEN_PROBE_ROUTER");
    const char * layer = std::getenv("QWEN_PROBE_LAYER");
    const bool router_selected = route && name.rfind(std::string(route) + "-", 0) == 0 &&
        (!layer || name == std::string(route) + "-" + layer);
    const bool selected = router_selected || (!std::getenv("QWEN_PROBE_ROUTER_ONLY") && (name.rfind("ffn_moe_weighted-", 0) == 0 ||
        name.rfind("ffn_moe_gate_up-", 0) == 0 ||
        name.rfind("ffn_moe_swiglu-", 0) == 0 ||
        name.rfind("ffn_moe_down-", 0) == 0 ||
        name.rfind("ffn_moe_out-", 0) == 0 ||
        name.rfind("ffn_shexp-", 0) == 0 ||
        name.rfind("ffn_shexp_gated-", 0) == 0 ||
        name.rfind("attn_norm-", 0) == 0 ||
        name.rfind("attn_residual-", 0) == 0 ||
        name.rfind("attn_post_norm-", 0) == 0 ||
        name.rfind("ffn_out-", 0) == 0 ||
        name.rfind("post_moe-", 0) == 0 ||
        name.rfind("l_out-", 0) == 0));
    if (!selected || ask) return selected;
    if (std::getenv("QWEN_PROBE_CUT_ONLY")) return true;
    auto & outputs = p.stage == 1 ? p.plain : p.batch;
    auto & row = outputs[name];
    row.resize(t->ne[0] * t->ne[1]);
    ggml_backend_tensor_get(t, row.data(), 0, row.size() * sizeof(float));
    return true;
}
static void print_layers(const LayerProbe & p) {
    for (const auto & [name, a] : p.plain) {
        auto it = p.batch.find(name);
        require(it != p.batch.end(), "missing batched layer tensor");
        const auto & b = it->second;
        int bits = 0;
        float max_abs = 0;
        for (size_t j = 0; j < a.size(); ++j) {
            uint32_t x, y;
            std::memcpy(&x, &a[j], 4); std::memcpy(&y, &b[j], 4);
            bits += x != y;
            max_abs = std::max(max_abs, std::abs(a[j] - b[j]));
        }
        std::printf("{\"tensor\":\"%s\",\"elements\":%zu,\"bit_differences\":%d,\"max_abs\":%.9g}\n", name.c_str(), a.size(), bits, max_abs);
    }
}
int main(int argc, char ** argv) {
    const bool layer_probe = argc == 6 && std::strcmp(argv[5], "--layer-probe") == 0;
    const bool causal_probe = argc == 6 && std::strcmp(argv[5], "--causal-probe") == 0;
    require(argc == 5 || layer_probe || causal_probe,
        "usage: batch-logits MODEL PROMPT STEPS OUTPUT_PREFIX [--layer-probe|--causal-probe]");
    LayerProbe probe;
    int steps = std::atoi(argv[3]);
    require(steps >= 4 && steps <= 128, "steps must be 4..128");
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    require(model != nullptr, "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (layer_probe || std::getenv("QWEN_PROBE_ALIAS")) { cp.cb_eval = layer_callback; cp.cb_eval_user_data = &probe; }
    llama_context * ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "context init failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::string prompt = argv[2];
    int needed = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    require(needed > 0 && needed < 512, "prompt token length invalid");
    std::vector<llama_token> prefix(needed);
    require(llama_tokenize(vocab, prompt.data(), prompt.size(), prefix.data(), needed, true, true) == needed, "tokenization failed");
    std::vector<llama_token> generated(steps);
    std::vector<std::vector<float>> gold(steps + 1, std::vector<float>(nv));
    require(llama_decode(ctx, llama_batch_get_one(prefix.data(), needed)) == 0, "plain prefill failed");
    for (int i = 0; i < steps; ++i) {
        if (layer_probe && i == 0) probe.stage = 1;
        auto p = llama_get_logits_ith(ctx, -1);
        require(p != nullptr, "plain logits unavailable");
        std::copy(p, p + nv, gold[i].begin());
        generated[i] = top(p, nv);
        if (std::getenv("QWEN_PROBE_ALIAS")) { probe.stage = 1; probe.width = 1; probe.row = i + 1; }
        require(llama_decode(ctx, llama_batch_get_one(&generated[i], 1)) == 0, "plain decode failed");
        probe.stage = 0;
        if (layer_probe && i == 1) break;
    }
    if (layer_probe) {
        llama_memory_clear(llama_get_memory(ctx), true);
        require(llama_decode(ctx, llama_batch_get_one(prefix.data(), needed)) == 0, "probe prefill failed");
        auto batch = llama_batch_init(2, 0, 1);
        batch.n_tokens = 2;
        for (int k = 0; k < 2; ++k) {
            batch.token[k] = generated[k]; batch.pos[k] = needed + k;
            batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0; batch.logits[k] = 1;
        }
        probe.stage = 2;
        require(llama_decode(ctx, batch) == 0, "probe batch failed");
        probe.stage = 0;
        compare(gold[1], llama_get_logits_ith(ctx, 0), nv, 2, 0, 1);
        if (!std::getenv("QWEN_PROBE_CUT_ONLY") && !std::getenv("QWEN_PROBE_NO_CUT")) print_layers(probe);
        llama_batch_free(batch);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }
    std::copy(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + nv, gold[steps].begin());
    std::ofstream ids(std::string(argv[4]) + ".tokens");
    for (auto id : generated) ids << id << '\n';
    if (causal_probe) {
        require(steps >= 16, "causal probe needs 16 plain steps");
        const int anchor = 11;
        std::vector<float> first_head;
        for (int arm = 0; arm < 5; ++arm) {
            llama_memory_clear(llama_get_memory(ctx), true);
            require(llama_decode(ctx, llama_batch_get_one(prefix.data(), needed)) == 0, "causal prefill failed");
            compare(gold[0], llama_get_logits_ith(ctx, -1), nv, 1, arm, 0);
            for (int i = 0; i < anchor; ++i) {
                auto batch = llama_batch_init(1, 0, 1);
                batch.n_tokens = 1;
                batch.token[0] = generated[i]; batch.pos[0] = needed + i;
                batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
                require(llama_decode(ctx, batch) == 0, "causal anchor decode failed");
                compare(gold[i + 1], llama_get_logits_ith(ctx, 0), nv, 1, arm, i + 1);
                llama_batch_free(batch);
            }
            auto batch = llama_batch_init(4, 0, 1);
            batch.n_tokens = 4;
            for (int k = 0; k < 4; ++k) {
                llama_token token = generated[anchor + k];
                if (k && arm == 1) token = generated[anchor];
                if (k && arm == 2) token = generated[anchor + 4 - k];
                if (k && arm == 3) token = generated[anchor - 1];
                batch.token[k] = token;
                batch.pos[k] = needed + anchor + k;
                batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0;
                batch.logits[k] = arm == 4 ? (k == 0) : 1;
            }
            require(llama_decode(ctx, batch) == 0, "causal multirow decode failed");
            auto p = llama_get_logits_ith(ctx, 0);
            require(p != nullptr, "causal row 12 logits unavailable");
            compare(gold[12], p, nv, 4, arm, 12);
            if (arm == 0) first_head.assign(p, p + nv);
            else compare(first_head, p, nv, 4, arm, -12);
            std::ofstream head(std::string(argv[4]) + ".arm" + std::to_string(arm) + ".f32", std::ios::binary);
            head.write(reinterpret_cast<const char *>(p), nv * sizeof(float));
            require(head.good(), "head write failed");
            llama_batch_free(batch);
        }
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }
    const bool anchored = std::getenv("QWEN_SERIAL_ANCHOR") != nullptr;
    const std::vector<std::pair<int,int>> configurations = anchored ?
        std::vector<std::pair<int,int>>{{1, 0}, {4, 0}} :
        std::vector<std::pair<int,int>>{{1, 0}, {2, 0}, {2, 1}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {8, 0}};
    for (auto configuration : configurations) {
        const int width = configuration.first, offset = configuration.second;
        llama_memory_clear(llama_get_memory(ctx), true);
        require(llama_decode(ctx, llama_batch_get_one(prefix.data(), needed)) == 0, "replay prefill failed");
        compare(gold[0], llama_get_logits_ith(ctx, -1), nv, width, offset, 0);
        int anchor = anchored && width == 4 ? 11 : 0;
        for (int i = 0; i < anchor; ++i) {
            auto batch = llama_batch_init(1, 0, 1);
            batch.n_tokens = 1;
            batch.token[0] = generated[i]; batch.pos[0] = needed + i;
            batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
            require(llama_decode(ctx, batch) == 0, "anchor serial decode failed");
            compare(gold[i + 1], llama_get_logits_ith(ctx, 0), nv, width, offset, i + 1);
            llama_batch_free(batch);
        }
        for (int i = anchor; i < steps;) {
            const int count = std::min(i == 0 && offset ? offset : width, steps - i);
            auto batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (int k = 0; k < batch.n_tokens; ++k) {
                batch.token[k] = generated[i + k];
                batch.pos[k] = needed + i + k;
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k] = 1;
            }
            if (std::getenv("QWEN_PROBE_ALIAS")) { probe.stage = 2; probe.width = width; probe.row = i + 1; }
            require(llama_decode(ctx, batch) == 0, "replay decode failed");
            probe.stage = 0;
            for (int k = 0; k < batch.n_tokens; ++k) {
                auto p = llama_get_logits_ith(ctx, k);
                require(p != nullptr, "replay logits unavailable");
                compare(gold[i + k + 1], p, nv, width, offset, i + k + 1);
            }
            llama_batch_free(batch);
            i += count;
        }
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
}
