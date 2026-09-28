// OpenAI-compatible HTTP server: GET /v1/models, POST /v1/chat/completions (streaming and not),
// GET /health. Concurrent requests decode together through `ServeBatch`, which owns the engine on
// its own thread; the engine keeps a prefix snapshot between requests.
#define CPPHTTPLIB_NO_EXCEPTIONS 0
#include "../vendor/cpp-httplib/httplib.h"
#include "engine.h"
#include "serve_batch.h"
#include "tokenizer.h"
#include "chat.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>

namespace halo {

using json = nlohmann::json;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Holds back an incomplete trailing UTF-8 sequence so every emitted chunk is valid UTF-8.
struct Utf8Buffer {
    std::string held;
    std::string push(const std::string & piece) {
        held += piece;
        size_t cut = held.size();
        for (size_t back = 1; back <= 3 && back <= held.size(); back++) {
            const unsigned char c = held[held.size() - back];
            if ((c & 0xC0) == 0x80) continue;              // continuation byte, keep looking for the lead
            const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            if (need > back) cut = held.size() - back;    // lead byte without all its continuations
            break;
        }
        std::string out = held.substr(0, cut); held = held.substr(cut);
        return out;
    }
    std::string flush() { std::string out = held; held.clear(); return out; }
};

struct ServerConfig { std::string host = "127.0.0.1"; int port = 8471; std::string model_id = "bonsai-2-27b"; int default_max_tokens = 4096; int max_active = 1; };
int run_server(Engine & e, Tokenizer & tk, const ServerConfig & cfg);

static std::string thinking_effort(const json & req, bool & enabled) {
    enabled = true;
    std::string effort = "xhigh";
    if (req.contains("reasoning_effort") && req["reasoning_effort"].is_string()) {
        const std::string e = req["reasoning_effort"].get<std::string>();
        if (e == "none" || e == "off" || e == "minimal") enabled = false;
        else if (e == "low") effort = "low";
        else if (e == "medium") effort = "medium";
        else effort = "xhigh";
    } else if (req.contains("reasoning_effort") && req["reasoning_effort"].is_null()) {
        enabled = false;
    }
    if (req.contains("enable_thinking") && req["enable_thinking"].is_boolean()) enabled = req["enable_thinking"].get<bool>();
    if (req.contains("chat_template_kwargs") && req["chat_template_kwargs"].is_object()) {
        const json & kw = req["chat_template_kwargs"];
        if (kw.contains("enable_thinking") && kw["enable_thinking"].is_boolean()) enabled = kw["enable_thinking"].get<bool>();
        if (kw.contains("reasoning_effort") && kw["reasoning_effort"].is_string()) effort = kw["reasoning_effort"].get<std::string>();
    }
    return effort;
}

static json tool_call_json(const ToolCall & tc, int index, const std::string & id) {
    return json{ { "index", index }, { "id", id }, { "type", "function" }, { "function", { { "name", tc.name }, { "arguments", tc.arguments.dump() } } } };
}

int run_server(Engine & e, Tokenizer & tk, const ServerConfig & cfg) {
    httplib::Server svr;
    ServeBatch sched(e, cfg.max_active);
    std::atomic<long> request_counter{0};
    // A handler blocks until its request has generated, so the HTTP pool has to be able to hold
    // every request the scheduler can decode at once, plus room for health checks. The default pool
    // is the host's thread count, which would quietly serialize the scheduler on a smaller machine.
    { const size_t pool = (size_t) std::max(16, sched.max_active() * 2);
      svr.new_task_queue = [pool] { return new httplib::ThreadPool(pool); }; }

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{ { "status", "ok" }, { "active", sched.active() }, { "max_active", sched.max_active() } }.dump(), "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json body = { { "object", "list" }, { "data", json::array({ json{ { "id", cfg.model_id }, { "object", "model" }, { "owned_by", "bonsai-halo" }, { "context_window", e.context } } }) } };
        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        if (getenv("HALO_LOG_REQUESTS")) { FILE * lf = fopen(getenv("HALO_LOG_REQUESTS"), "a"); if (lf) { fwrite(req.body.data(), 1, req.body.size(), lf); fputs("\n", lf); fclose(lf); } }
        try { body = json::parse(req.body); } catch (const std::exception & ex) { res.status = 400; res.set_content(json{ { "error", { { "message", std::string("invalid JSON: ") + ex.what() } } } }.dump(), "application/json"); return; }
        const bool stream = body.value("stream", false);
        bool thinking = true;
        ChatOptions opt; opt.reasoning_effort = thinking_effort(body, thinking); opt.thinking = thinking;
        std::string prompt;
        std::vector<int> toks;
        size_t sys_tokens = 0;
        try {
            const json * tools = body.contains("tools") && body["tools"].is_array() && !body["tools"].empty() ? &body["tools"] : nullptr;
            size_t sys_end = 0;
            prompt = render_chat(body.at("messages"), tools, opt, &sys_end);
            toks = tk.encode(prompt, true);
            // the system block ends at a special token, so its tokens are a prefix of the prompt's
            if (sys_end > 0 && sys_end < prompt.size()) sys_tokens = tk.encode(prompt.substr(0, sys_end), true).size();
        } catch (const std::exception & ex) { res.status = 400; res.set_content(json{ { "error", { { "message", ex.what() } } } }.dump(), "application/json"); return; }
        Engine::GenParams gp;
        gp.max_tokens = body.contains("max_tokens") && body["max_tokens"].is_number() ? body["max_tokens"].get<int>() : body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number() ? body["max_completion_tokens"].get<int>() : cfg.default_max_tokens;
        gp.temp = body.value("temperature", 0.0f);
        gp.top_p = body.value("top_p", 0.95f);
        gp.top_k = body.value("top_k", 20);
        if (body.contains("seed") && body["seed"].is_number()) gp.seed = body["seed"].get<unsigned>();
        if (sys_tokens > 0) gp.snapshot_points.push_back(sys_tokens);
        const int room = e.context - DF_BLOCK - 2 - (int) toks.size();
        if (room < 16) { res.status = 400; res.set_content(json{ { "error", { { "message", "context window exceeded: the prompt has " + std::to_string(toks.size()) + " tokens of " + std::to_string(e.context) }, { "type", "context_length_exceeded" } } } }.dump(), "application/json"); return; }
        if (gp.max_tokens > room) gp.max_tokens = room;
        const long id = ++request_counter;
        const std::string completion_id = "chatcmpl-halo-" + std::to_string(id);
        const long created = (long) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        fprintf(stderr, "[%ld] prompt %zu tokens, max %d, thinking %s/%s, stream %d\n", id, toks.size(), gp.max_tokens, thinking ? "on" : "off", opt.reasoning_effort.c_str(), (int) stream);

        auto run = [&, toks, gp, thinking, completion_id, created, id](const std::function<bool(const json &)> & emit, json & final_message, std::string & finish_reason, Engine::SpecStats & st, int & completion_tokens) mutable {
            StreamSplitter split(thinking);
            Utf8Buffer utf8;
            std::atomic<bool> aborted{false};
            std::string reasoning_all, content_all;
            finish_reason = "length";
            bool content_started = false;
            auto deliver = [&](std::pair<std::string, std::string> d) {
                if (!content_started) {
                    // the template puts a blank line after </think>; do not stream it as content
                    size_t k = d.second.find_first_not_of(" \n\r\t");
                    d.second = k == std::string::npos ? "" : d.second.substr(k);
                    if (!d.second.empty()) content_started = true;
                }
                if (d.first.empty() && d.second.empty()) return true;
                reasoning_all += d.first; content_all += d.second;
                json delta = json::object();
                if (!d.first.empty()) delta["reasoning_content"] = d.first;
                if (!d.second.empty()) delta["content"] = d.second;
                return emit(json{ { "id", completion_id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", cfg.model_id }, { "choices", json::array({ json{ { "index", 0 }, { "delta", delta }, { "finish_reason", nullptr } } }) } });
            };
            Engine::GenParams p = gp;
            p.aborted = [&]() { return aborted.load(); };
            sched.run(toks, p, [&](int tok) {
                if (tk.is_eog(tok)) { finish_reason = "stop"; return false; }
                completion_tokens++;
                const std::string piece = utf8.push(tk.piece(tok));
                if (piece.empty()) return true;
                if (!deliver(split.feed(piece))) { aborted = true; return false; }
                return true;
            }, st);
            deliver(split.feed(utf8.flush()));
            deliver(split.finish());
            json tool_calls = json::array();
            for (size_t i = 0; i < split.tool_calls.size(); i++) tool_calls.push_back(tool_call_json(split.tool_calls[i], (int) i, "call_" + std::to_string(id) + "_" + std::to_string(i)));
            if (!tool_calls.empty()) {
                finish_reason = "tool_calls";
                emit(json{ { "id", completion_id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", cfg.model_id }, { "choices", json::array({ json{ { "index", 0 }, { "delta", { { "tool_calls", tool_calls } } }, { "finish_reason", nullptr } } }) } });
            }
            final_message = json{ { "role", "assistant" }, { "content", content_all.empty() ? json() : json(content_all) } };
            if (!reasoning_all.empty()) final_message["reasoning_content"] = reasoning_all;
            if (!tool_calls.empty()) final_message["tool_calls"] = tool_calls;
        };
        auto usage_json = [](const Engine::SpecStats & st, int completion_tokens, double seconds) {
            return json{ { "prompt_tokens", st.prompt_tokens }, { "completion_tokens", completion_tokens }, { "total_tokens", st.prompt_tokens + completion_tokens },
                         { "prompt_tokens_details", { { "cached_tokens", st.cached_tokens } } },
                         { "halo", { { "prefill_seconds", st.t_prefill }, { "seconds", seconds }, { "tokens_per_second", seconds > 0 ? completion_tokens / seconds : 0.0 }, { "speculative_steps", st.steps }, { "drafted", st.drafted }, { "accepted", st.accepted } } } };
        };

        if (stream) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            // the provider runs after this handler returned: everything it uses is captured by value
            res.set_chunked_content_provider("text/event-stream", [&e, &tk, &cfg, run, usage_json, completion_id, created, id](size_t, httplib::DataSink & sink) mutable {
                json final_message; std::string finish_reason; Engine::SpecStats st; int completion_tokens = 0;
                const double t0 = now();
                bool alive = true;
                auto emit = [&](const json & chunk) { if (!alive) return false; const std::string s = "data: " + chunk.dump() + "\n\n"; alive = sink.write(s.data(), s.size()); return alive; };
                try { run(emit, final_message, finish_reason, st, completion_tokens); }
                catch (const std::exception & ex) { fprintf(stderr, "[%ld] error: %s\n", id, ex.what()); emit(json{ { "error", { { "message", ex.what() } } } }); sink.done(); return true; }
                const double seconds = now() - t0;
                emit(json{ { "id", completion_id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", cfg.model_id }, { "choices", json::array({ json{ { "index", 0 }, { "delta", json::object() }, { "finish_reason", finish_reason } } }) }, { "usage", usage_json(st, completion_tokens, seconds) } });
                const std::string done = "data: [DONE]\n\n"; if (alive) sink.write(done.data(), done.size());
                sink.done();
                fprintf(stderr, "[%ld] %s: %d tokens in %.2f s (%.1f tok/s), prefill %.2f s (%ld cached), %ld spec steps\n", id, finish_reason.c_str(), completion_tokens, seconds, seconds > 0 ? completion_tokens / seconds : 0.0, st.t_prefill, st.cached_tokens, st.steps);
                return true;
            });
            return;
        }
        json final_message; std::string finish_reason; Engine::SpecStats st; int completion_tokens = 0;
        const double t0 = now();
        try { run([](const json &) { return true; }, final_message, finish_reason, st, completion_tokens); }
        catch (const std::exception & ex) { fprintf(stderr, "[%ld] error: %s\n", id, ex.what()); res.status = 500; res.set_content(json{ { "error", { { "message", ex.what() } } } }.dump(), "application/json"); return; }
        const double seconds = now() - t0;
        json out = { { "id", completion_id }, { "object", "chat.completion" }, { "created", created }, { "model", cfg.model_id },
                     { "choices", json::array({ json{ { "index", 0 }, { "message", final_message }, { "finish_reason", finish_reason } } }) }, { "usage", usage_json(st, completion_tokens, seconds) } };
        res.set_content(out.dump(), "application/json");
        fprintf(stderr, "[%ld] %s: %d tokens in %.2f s (%.1f tok/s), prefill %.2f s (%ld cached)\n", id, finish_reason.c_str(), completion_tokens, seconds, seconds > 0 ? completion_tokens / seconds : 0.0, st.t_prefill, st.cached_tokens);
    });

    fprintf(stderr, "bonsai-halo serving %s on http://%s:%d/v1, %d request%s decode together\n",
            cfg.model_id.c_str(), cfg.host.c_str(), cfg.port, sched.max_active(), sched.max_active() == 1 ? "" : "s");
    if (!svr.listen(cfg.host, cfg.port)) { fprintf(stderr, "cannot listen on %s:%d\n", cfg.host.c_str(), cfg.port); return 1; }
    return 0;
}

} // namespace halo
