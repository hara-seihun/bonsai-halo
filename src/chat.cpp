#include "chat.h"
#include <stdexcept>

namespace halo {

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

static std::string content_text(const json & content) {
    if (content.is_null()) return "";
    if (content.is_string()) return content.get<std::string>();
    if (content.is_array()) {
        std::string out;
        for (const auto & item : content) {
            if (item.contains("text")) out += item["text"].get<std::string>();
            else if (item.value("type", "") == "image_url" || item.contains("image") || item.contains("image_url")) throw std::runtime_error("images are not supported by this engine");
            else throw std::runtime_error("unexpected content item");
        }
        return out;
    }
    throw std::runtime_error("unexpected content type");
}

static std::string args_value(const json & v) { return v.is_string() ? v.get<std::string>() : v.dump(); }

std::string render_chat(const json & messages, const json * tools, const ChatOptions & opt, size_t * sys_end) {
    if (!messages.is_array() || messages.empty()) throw std::runtime_error("no messages provided");
    std::string reasoning_instructions;
    if (opt.thinking) {
        if (opt.reasoning_effort == "xhigh") reasoning_instructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        else if (opt.reasoning_effort == "low") reasoning_instructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
        else if (opt.reasoning_effort != "medium") throw std::runtime_error("unexpected reasoning effort " + opt.reasoning_effort);
    }
    std::string out;
    const bool has_tools = tools && tools->is_array() && !tools->empty();
    auto role_of = [](const json & m) { std::string r = m.value("role", ""); return r == "developer" ? std::string("system") : r; };
    const bool first_system = role_of(messages[0]) == "system";
    if (has_tools) {
        out += "<|im_start|>system\n";
        if (!reasoning_instructions.empty()) out += reasoning_instructions + "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto & tool : *tools) out += "\n" + tool.dump();
        out += "\n</tools>";
        out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (first_system) { const std::string c = trim(content_text(messages[0]["content"])); if (!c.empty()) out += "\n\n" + c; }
        out += "<|im_end|>\n";
    } else if (first_system) {
        const std::string c = trim(content_text(messages[0]["content"]));
        if (!c.empty()) out += "<|im_start|>system\n" + (reasoning_instructions.empty() ? "" : reasoning_instructions + "\n\n") + c + "<|im_end|>\n";
        else if (!reasoning_instructions.empty()) out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
    } else if (!reasoning_instructions.empty()) {
        out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
    }
    if (sys_end) *sys_end = out.size();
    // last real user query (tool responses wrapped as user messages do not count)
    int last_query = -1;
    for (int i = (int) messages.size() - 1; i >= 0; i--) {
        if (messages[i].value("role", "") != "user") continue;
        const std::string c = trim(content_text(messages[i]["content"]));
        if (!(c.rfind("<tool_response>", 0) == 0 && c.size() >= 16 && c.compare(c.size() - 16, 16, "</tool_response>") == 0)) { last_query = i; break; }
    }
    if (last_query < 0) throw std::runtime_error("no user query found in messages");
    const int n = (int) messages.size();
    for (int i = 0; i < n; i++) {
        const json & m = messages[i];
        const std::string role = role_of(m);
        const std::string content = trim(content_text(m.contains("content") ? m["content"] : json()));
        if (role == "system") {
            if (i != 0) throw std::runtime_error("system message must be at the beginning");
        } else if (role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning;
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) reasoning = trim(m["reasoning_content"].get<std::string>());
            else if (m.contains("reasoning") && m["reasoning"].is_string()) reasoning = trim(m["reasoning"].get<std::string>());
            // preserve_thinking defaults to true in this template: every assistant turn keeps its block
            out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                bool first = true;
                for (const auto & tc0 : m["tool_calls"]) {
                    const json & tc = tc0.contains("function") ? tc0["function"] : tc0;
                    const std::string name = tc.value("name", "");
                    if (first) out += content.empty() ? "<tool_call>\n<function=" + name + ">\n" : "\n\n<tool_call>\n<function=" + name + ">\n";
                    else out += "\n<tool_call>\n<function=" + name + ">\n";
                    first = false;
                    json args;
                    if (tc.contains("arguments")) {
                        const json & a = tc["arguments"];
                        if (a.is_string()) { const std::string as = a.get<std::string>(); if (!as.empty()) { try { args = json::parse(as); } catch (...) { args = json::object(); } } }
                        else args = a;
                    }
                    if (args.is_object()) {
                        for (auto it = args.begin(); it != args.end(); ++it) out += "<parameter=" + it.key() + ">\n" + args_value(it.value()) + "\n</parameter>\n";
                    }
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            const bool prev_tool = i > 0 && messages[i - 1].value("role", "") == "tool";
            if (!prev_tool) out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            const bool next_tool = i + 1 < n && messages[i + 1].value("role", "") == "tool";
            if (!next_tool) out += "<|im_end|>\n";
        } else {
            throw std::runtime_error("unexpected message role: " + role);
        }
    }
    out += "<|im_start|>assistant\n";
    out += opt.thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return out;
}

// ---- output parsing

static json parse_param_value(const std::string & raw) {
    // the template renders non-string arguments through tojson; recover those, keep the rest as text
    const std::string v = trim(raw);
    if (v.empty()) return raw;
    const char c = v[0];
    if (c == '{' || c == '[' || c == '"' || v == "true" || v == "false" || v == "null" || c == '-' || (c >= '0' && c <= '9')) {
        try { return json::parse(v); } catch (...) {}
    }
    return raw;
}

static void parse_tool_block(const std::string & block, std::vector<ToolCall> & out) {
    // block: text between <tool_call> and </tool_call>
    size_t f = block.find("<function=");
    if (f == std::string::npos) return;
    size_t fe = block.find('>', f);
    if (fe == std::string::npos) return;
    ToolCall tc; tc.name = block.substr(f + 10, fe - f - 10); tc.arguments = json::object();
    size_t p = fe + 1;
    while (true) {
        size_t ps = block.find("<parameter=", p);
        if (ps == std::string::npos) break;
        size_t pe = block.find('>', ps);
        if (pe == std::string::npos) break;
        const std::string key = block.substr(ps + 11, pe - ps - 11);
        size_t vs = pe + 1;
        if (vs < block.size() && block[vs] == '\n') vs++;
        size_t ve = block.find("</parameter>", vs);
        if (ve == std::string::npos) ve = block.size();
        std::string value = block.substr(vs, ve - vs);
        if (!value.empty() && value.back() == '\n') value.pop_back();
        tc.arguments[key] = parse_param_value(value);
        p = ve;
    }
    out.push_back(tc);
}

static void split_tool_calls(const std::string & text, std::string & content, std::vector<ToolCall> & calls) {
    size_t p = 0;
    while (true) {
        size_t s = text.find("<tool_call>", p);
        if (s == std::string::npos) { content += text.substr(p); break; }
        content += text.substr(p, s - p);
        size_t e = text.find("</tool_call>", s);
        const std::string block = e == std::string::npos ? text.substr(s + 11) : text.substr(s + 11, e - s - 11);
        parse_tool_block(block, calls);
        if (e == std::string::npos) break;
        p = e + 12;
    }
    content = trim(content);
}

ParsedOutput parse_output(const std::string & text, bool thinking_open) {
    ParsedOutput o;
    std::string rest = text;
    if (thinking_open) {
        size_t e = text.find("</think>");
        if (e == std::string::npos) { o.reasoning = trim(text); return o; }
        o.reasoning = trim(text.substr(0, e));
        rest = text.substr(e + 8);
    }
    split_tool_calls(rest, o.content, o.tool_calls);
    return o;
}

// ---- streaming

static const char * THINK_END = "</think>";
static const char * TC_OPEN = "<tool_call>";
static const char * TC_CLOSE = "</tool_call>";

// longest suffix of s that is a prefix of tag
static size_t partial_tag(const std::string & s, const char * tag) {
    const std::string t = tag;
    for (size_t k = std::min(s.size(), t.size() - 1); k > 0; k--) if (s.compare(s.size() - k, k, t, 0, k) == 0) return k;
    return 0;
}

std::pair<std::string, std::string> StreamSplitter::feed(const std::string & piece) {
    std::string r, c;
    pending += piece;
    while (true) {
        if (in_reasoning) {
            size_t e = pending.find(THINK_END);
            if (e != std::string::npos) { r += pending.substr(0, e); pending = pending.substr(e + 8); in_reasoning = false; continue; }
            size_t keep = partial_tag(pending, THINK_END);
            r += pending.substr(0, pending.size() - keep); pending = pending.substr(pending.size() - keep);
            break;
        }
        if (in_tool) {
            size_t e = pending.find(TC_CLOSE);
            if (e != std::string::npos) { parse_tool_block(pending.substr(0, e), tool_calls); pending = pending.substr(e + 12); in_tool = false; continue; }
            break; // keep buffering the tool block
        }
        size_t s = pending.find(TC_OPEN);
        if (s != std::string::npos) { c += pending.substr(0, s); pending = pending.substr(s + 11); in_tool = true; continue; }
        size_t keep = partial_tag(pending, TC_OPEN);
        c += pending.substr(0, pending.size() - keep); pending = pending.substr(pending.size() - keep);
        break;
    }
    return { r, c };
}

std::pair<std::string, std::string> StreamSplitter::finish() {
    std::string r, c;
    if (in_reasoning) { r = pending; pending.clear(); return { r, c }; }
    if (in_tool) { parse_tool_block(pending, tool_calls); pending.clear(); return { r, c }; }
    c = pending; pending.clear();
    return { r, c };
}

} // namespace halo
