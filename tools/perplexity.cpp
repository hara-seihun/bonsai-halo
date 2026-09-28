// Teacher-forced WikiText perplexity and paired full-distribution fidelity on Halo's real route.
// Match llama-perplexity -c 512: independent 512-token windows, BOS at every window's
// first position when the vocabulary requests one, score logits at positions 256..510
// against tokens 257..511 (255 predictions per window).
#include "engine.h"
#include "tokenizer.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace halo;

int main(int argc, char ** argv) {
    try {
        std::string model = "../../data/bonsai2/PTQ1_0.gguf", corpus, output, reference;
        int chunks = 0, start_chunk = 0, mode = 0, rows = 8, threads = 8;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("missing value for " + a); return argv[i]; };
            if (a == "--model") model = next();
            else if (a == "--ppl") corpus = next();
            else if (a == "--chunks") chunks = std::stoi(next());
            else if (a == "--start-chunk") start_chunk = std::stoi(next());
            else if (a == "--route") {
                const auto route = next();
                if (route == "deployed") { mode = 0; rows = 8; }
                else if (route == "wide-deployed") { mode = 20; rows = 128; }
                else if (route == "wide-a4") { mode = 19; rows = 128; }
                else throw std::runtime_error("--route wants deployed, wide-deployed or wide-a4");
            } else if (a == "--logits-out") output = next();
            else if (a == "--reference") reference = next();
            else if (a == "--threads") threads = std::stoi(next());
            else throw std::runtime_error("usage: tools/perplexity --ppl wiki.test.raw [--chunks N] [--route deployed|wide-deployed|wide-a4] [--reference exact.f32] [--logits-out exact.f32] [--model FILE]");
        }
        if (corpus.empty() || chunks < 0 || start_chunk < 0 || (!output.empty() && output == reference))
            throw std::runtime_error("--ppl required; --chunks must be >=0; reference and output must differ");
        std::ifstream input(corpus, std::ios::binary);
        if (!input) throw std::runtime_error("cannot read corpus " + corpus);
        std::string text(std::istreambuf_iterator<char>{input}, {});
        Tokenizer tk; tk.load(model);
        auto tokens = tk.encode(text, false);
        const auto * vocab = static_cast<const llama_vocab *>(tk.vocab);
        if (llama_vocab_get_add_bos(vocab)) tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
        const int available = int(tokens.size() / 512);
        if (available < 2) throw std::runtime_error("llama-perplexity requires at least two full windows");
        if (start_chunk >= available || (chunks && start_chunk + chunks > available))
            throw std::runtime_error("requested chunks exceed the complete corpus windows");
        if (!chunks) chunks = available - start_chunk;
        const size_t chunk_bytes = (size_t) 255 * VOCAB * sizeof(float);
        std::ofstream out;
        if (!output.empty()) {
            if (start_chunk) {
                std::ifstream existing(output, std::ios::binary | std::ios::ate);
                if (!existing || (size_t) existing.tellg() != (size_t) start_chunk * chunk_bytes)
                    throw std::runtime_error("reference output size does not match --start-chunk");
            }
            out.open(output, std::ios::binary | (start_chunk ? std::ios::app : std::ios::trunc));
            if (!out) throw std::runtime_error("cannot open logits output");
        }
        std::ifstream ref;
        if (!reference.empty()) {
            ref.open(reference, std::ios::binary);
            if (!ref) throw std::runtime_error("cannot open reference logits");
            ref.seekg((size_t) start_chunk * chunk_bytes);
        }
        Engine e;
        e.load(model, threads, 1, 512);
        e.batch_mode = mode;
        if (mode) { e.prepare_batch(rows, mode == 20 ? Engine::BATCH_WORKSPACE_ONLY : 1u << 8); e.prepare_sequence(); }
        fprintf(stderr, "PPL route=%d state=%s seq_quant=%s chunks=%d tokens=%zu scored=%d\n",
                mode, getenv("HALO_GDN_STATE") ? getenv("HALO_GDN_STATE") : "f32",
                getenv("HALO_SEQ_QUANT") ? getenv("HALO_SEQ_QUANT") : "default", chunks, tokens.size(), chunks * 255);
        double nll = 0, kl = 0; long agree = 0, scored = 0;
        std::vector<float> baseline(VOCAB);
        for (int c = start_chunk; c < start_chunk + chunks; ++c) {
            Engine::Seq s; e.reset_seq(s);
            std::vector<Engine::Seq *> seq{&s};
            for (int pos = 0; pos < 512; pos += rows) {
                const int count = std::min(rows, 512 - pos);
                std::vector<std::vector<int>> feed(1);
                feed[0].assign(tokens.begin() + c * 512 + pos, tokens.begin() + c * 512 + pos + count);
                if (pos == 0 && llama_vocab_get_add_bos(vocab)) feed[0][0] = llama_vocab_bos(vocab);
                std::vector<float> logits;
                // Don't run the vocabulary projection on the first half. Position 255 is
                // also unscored: llama-perplexity begins with the logit at position 256.
                const bool score = pos + count > 256;
                e.forward_batch(seq, feed, score, nullptr, score ? &logits : nullptr);
                if (!score) continue;
                for (int r = std::max(0, 256 - pos); r < count && pos + r < 511; ++r) {
                    const float * p = logits.data() + (size_t) r * VOCAB;
                    const int target = tokens[c * 512 + pos + r + 1];
                    const float mx = *std::max_element(p, p + VOCAB);
                    double z = 0; int top = 0;
                    for (int j = 0; j < VOCAB; ++j) { z += std::exp(double(p[j] - mx)); if (p[j] > p[top]) top = j; }
                    const double logz = double(mx) + std::log(z);
                    nll += logz - p[target];
                    if (out.is_open()) out.write(reinterpret_cast<const char *>(p), (size_t) VOCAB * sizeof(float));
                    if (ref.is_open()) {
                        ref.read(reinterpret_cast<char *>(baseline.data()), (size_t) VOCAB * sizeof(float));
                        if (!ref) throw std::runtime_error("reference logits ended before the scored token stream");
                        const float bm = *std::max_element(baseline.begin(), baseline.end());
                        double bz = 0, weighted = 0; int btop = 0;
                        for (int j = 0; j < VOCAB; ++j) {
                            const double w = std::exp(double(baseline[j] - bm));
                            bz += w; weighted += w * (double(baseline[j]) - p[j]);
                            if (baseline[j] > baseline[btop]) btop = j;
                        }
                        kl += weighted / bz + logz - (double(bm) + std::log(bz));
                        agree += btop == top;
                    }
                    ++scored;
                }
            }
            fprintf(stderr, "chunk %d/%d: segment ppl %.6f\n", c + 1, available, std::exp(nll / scored));
        }
        if (ref.is_open() && start_chunk + chunks == available && ref.peek() != std::char_traits<char>::eof()) throw std::runtime_error("reference contains more scored rows than the complete corpus");
        printf("{\"start_chunk\":%d,\"chunks\":%d,\"scored\":%ld,\"route\":%d,\"nll\":%.9f,\"perplexity\":%.9f,\"kl_exact_to_route\":%.9f,\"top1_agreement\":%.9f}\n",
               start_chunk, chunks, scored, mode, nll / scored, std::exp(nll / scored), ref.is_open() ? kl / scored : 0.0, ref.is_open() ? double(agree) / scored : 1.0);
    } catch (const std::exception & ex) { fprintf(stderr, "perplexity: %s\n", ex.what()); return 1; }
}
