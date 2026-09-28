// Qwen3.5 chat template (the model's tokenizer.chat_template) rendered from OpenAI-style messages,
// and the parser for the model's <tool_call> output format.
#pragma once
#include <string>
#include <vector>
#include "../vendor/nlohmann/json.hpp"

namespace halo {

using json = nlohmann::json;

struct ChatOptions {
    bool thinking = true;              // enable_thinking
    std::string reasoning_effort = "xhigh"; // xhigh | medium | low (thinking only)
};

// Render messages (+ optional tools array) with add_generation_prompt = true. When sys_end is given
// it receives the rendered length of the leading system block (tools and system message), which is a
// stable prefix across conversations that share them.
std::string render_chat(const json & messages, const json * tools, const ChatOptions & opt, size_t * sys_end = nullptr);

struct ToolCall { std::string name; json arguments; };
struct ParsedOutput { std::string reasoning; std::string content; std::vector<ToolCall> tool_calls; };

// Split a complete assistant generation into reasoning (inside <think>), content and tool calls.
// `thinking_open` tells whether the prompt ended inside an open <think> block.
ParsedOutput parse_output(const std::string & text, bool thinking_open);

// Incremental splitter for streaming: feed text, receive reasoning/content deltas; tool call blocks
// are withheld from content and returned complete from finish().
struct StreamSplitter {
    bool in_reasoning; std::string pending; std::string tool_buffer; bool in_tool = false;
    std::vector<ToolCall> tool_calls;
    explicit StreamSplitter(bool thinking_open) : in_reasoning(thinking_open) {}
    // returns { reasoning_delta, content_delta }
    std::pair<std::string, std::string> feed(const std::string & piece);
    std::pair<std::string, std::string> finish();
};

} // namespace halo
