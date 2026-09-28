#include "tokenizer.h"
#include "llama.h"
#include <stdexcept>

namespace halo {

static void quiet_log(ggml_log_level, const char *, void *) {}

void Tokenizer::load(const std::string & path) {
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params p = llama_model_default_params();
    p.vocab_only = true;
    model = llama_model_load_from_file(path.c_str(), p);
    if (!model) throw std::runtime_error("tokenizer: cannot load vocab from " + path);
    vocab = llama_model_get_vocab((const llama_model *) model);
}

Tokenizer::~Tokenizer() { if (model) llama_model_free((llama_model *) model); }

std::vector<int> Tokenizer::encode(const std::string & text, bool parse_special) const {
    std::vector<int> out(text.size() + 16);
    int n = llama_tokenize((const llama_vocab *) vocab, text.c_str(), (int) text.size(), out.data(), (int) out.size(), false, parse_special);
    if (n < 0) { out.resize(-n); n = llama_tokenize((const llama_vocab *) vocab, text.c_str(), (int) text.size(), out.data(), (int) out.size(), false, parse_special); }
    out.resize(n);
    return out;
}

std::string Tokenizer::piece(int token) const {
    char buf[256];
    int n = llama_token_to_piece((const llama_vocab *) vocab, token, buf, sizeof buf, 0, true);
    if (n < 0) return "";
    return std::string(buf, n);
}

bool Tokenizer::is_eog(int token) const { return llama_vocab_is_eog((const llama_vocab *) vocab, token); }

std::string Tokenizer::chat_prompt(const std::string & user, bool thinking) {
    std::string s = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    s += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return s;
}

}
