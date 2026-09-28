// Tokenizer boundary: Qwen byte-level BPE via libllama loaded vocab-only. No inference code from llama.cpp runs.
#pragma once
#include <string>
#include <vector>

namespace halo {
struct Tokenizer {
    void * model = nullptr; const void * vocab = nullptr;
    void load(const std::string & gguf_path);
    ~Tokenizer();
    std::vector<int> encode(const std::string & text, bool parse_special) const;
    std::string piece(int token) const;
    bool is_eog(int token) const;
    // chat template for a single user turn (thinking on/off)
    static std::string chat_prompt(const std::string & user, bool thinking);
};
}
