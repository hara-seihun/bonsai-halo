#include "engine.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <string>
#include <thread>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include "../vendor/nlohmann/json.hpp"


using namespace halo;

namespace halo { struct ServerConfig { std::string host = "127.0.0.1"; int port = 8471; std::string model_id = "bonsai-2-27b"; int default_max_tokens = 4096; int max_active = 1; }; int run_server(Engine & e, Tokenizer & tk, const ServerConfig & cfg); }

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<int> token_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open token IDs: " + path);
    auto ids = nlohmann::json::parse(in).get<std::vector<int>>();
    if (ids.empty()) throw std::runtime_error("empty token IDs: " + path);
    return ids;
}

int main(int argc, char ** argv) {
    std::string model = "PTQ1_0.gguf";
    std::string prompt = "Explain why the sky is blue in two sentences.";
    bool raw = false, thinking = false, bench = false, nograph = false;
    std::string dump_dir, logits_out, readback, mtp_path, dflash_path, prompts_file, prompt_ids, batch_ids_dir; bool v1 = false; int n_draft = 3, batch = 0;
    bool serve = false, max_profile = false, plain = false; ServerConfig sc;
    int context_capacity = MAXCTX, ffn_mode = 0;
    int n_gen = 128, nthreads = std::thread::hardware_concurrency();
    float temp = 0.0f, top_p = 0.95f; int top_k = 20; unsigned seed = 42; int prefill_rows = 8;
    // -1 until the routes are known: the wide-deployed default is resolved after parsing, because
    // it depends on which other paths the run asked for. `--prefill-ffn off` pins the eight-row
    // route explicitly.
    int prefill_ffn = -1;
    std::vector<int> prefill_sweep; int sweep_rounds = 1;
    // Drafter weight coordinate(s), DFlash2 only. A drafted run's output tokens do not depend on
    // it, so the arms of a sweep produce the same text and differ only in steps, acceptance and
    // time. Q4 is the default; `--draft-weights q8` is the control (docs/drafter-q4.md).
    std::vector<int> draft_arms{ 1 };
    // Workgroup counts for the persistent kernel, walked as a case axis of the drafted bench.
    // The verify pass of a drafted step is the only wide shape the product path launches, and its
    // grid is set by occupancy rather than by any choice: two processes cannot order a few percent
    // on this box, so the arms share a process, a loaded model, a clock and a thermal state.
    // -1 leaves the grid the build's own occupancy allows. `HALO_BENCH_GRID` is the same axis for
    // plain decode, per token; this one is per run because a drafted step is not one token.
    std::vector<int> grid_arms{ -1 };
    // Register budget for the persistent kernel, walked as a case axis so both arms run under one
    // clock: -1 leaves it as configured, 0 lets the allocator choose, 1 asks for twelve waves.
    // Crossed with --grid-rows this gives the three cells that separate the change: budget 0 is
    // today, budget 1 pinned to today's grid is the spill alone, budget 1 on its own grid is the
    // whole change. A fourth cell, budget 0 asked for a grid its occupancy cannot hold, silently
    // returns today's grid and is a free in-process null.
    std::vector<int> occ_arms{ -1 };
    // Workgroups for the drafter's cooperative kernel, as an in-process arm list. 0 keeps the
    // build's own occupancy grid and G pins G. A drafted bench re-prefills per arm, so one process
    // reports per-step draft ms, verify ms, acceptance and the greedy digest for every arm on one
    // clock - and the digest is an exact control, because nothing here changes an operand. Give the
    // list as a palindrome (40,60,80,80,60,40) and every arm is measured early and late, which is
    // what separates a real difference from the warm-up drift of the first arm in a process.
    std::vector<int> draft_grids{ 0 };
    // Whether the drafter's 2048-key window decides its attention unit list as well as its mask,
    // as an in-process arm list: 1 skips the chunks the window excludes, 0 deals every chunk from
    // position 0 and masks it to zero. Bit-identical arms - same drafts, same accepted count, same
    // greedy digest - so the difference this walks is time alone. Palindromes work here too
    // (1,0,0,1). docs/drafter-window-units.md.
    std::vector<int> deal_arms{ -1 };
    // Rows one wide prompt pass carries. Every weight-bearing kernel re-streams its whole image
    // once per pass, so this trades device residency (the head's logit buffer and the activation
    // workspace, both linear in rows) against how many times a prompt pays that stream.
    int pass_rows = PASS_ROWS_DEFAULT;
    // Prompt ingestion asks the head for the one logit row it reads. `--head-rows all` puts the head
    // back on every row of the pass, which is the control arm for that selection.
    bool head_rows_all = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (i == 1 && a == "serve") { serve = true; continue; }
        if (serve && a == "--port") { sc.port = atoi(argv[++i]); continue; }
        if (serve && a == "--host") { sc.host = argv[++i]; continue; }
        if (serve && a == "--model-id") { sc.model_id = argv[++i]; continue; }
        if (serve && a == "--max-tokens") { sc.default_max_tokens = atoi(argv[++i]); continue; }
        // How many requests decode together. Each one needs its own sequence state slot, which is
        // its own K/V cache: about 90 KB per token of context, so 2.95 GB per slot at the full
        // 32768 and 738 MB at 8192. `--slots 1` is the serialized server this replaced, and it is
        // the control arm for every concurrency measurement.
        if (serve && a == "--slots") { sc.max_active = atoi(argv[++i]); continue; }
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "-p" || a == "--prompt") prompt = next();
        else if (a == "-n") n_gen = atoi(next());
        else if (a == "--raw") raw = true;
        else if (a == "--think") thinking = true;
        else if (a == "--bench") bench = true;
        else if (a == "--profile") { std::string v = next(); if (v != "max") throw std::runtime_error("--profile wants max"); max_profile = true; }
        else if (a == "--plain") plain = true;
        else if (a == "--no-graph") nograph = true;
        else if (a == "-t") nthreads = atoi(next());
        else if (a == "--temp") temp = atof(next());
        else if (a == "--top-k") top_k = atoi(next());
        else if (a == "--top-p") top_p = atof(next());
        else if (a == "--seed") seed = atoi(next());
        else if (a == "--prefill-rows") prefill_rows = atoi(next());
        else if (a == "--pass-rows") pass_rows = atoi(next());
        else if (a == "--head-rows") { std::string v = next(); if (v != "tail" && v != "all") { fprintf(stderr, "--head-rows wants tail or all\n"); return 1; } head_rows_all = v == "all"; }
        else if (a == "--prefill-ffn") {
            const std::string mode = next();
            if (mode == "off" || mode == "deployed") prefill_ffn = 0;
            else if (mode == "wide-deployed") prefill_ffn = 20;
            else if (mode == "wide-commit") prefill_ffn = 18;
            else if (mode == "wide-commit-a4") prefill_ffn = 19;
            else { fprintf(stderr, "unknown prefill route: %s\n", mode.c_str()); return 1; }
        }
        // Two prompt-ingestion routes measured in two processes cannot be compared on this box:
        // its clock ramps over seconds and back-to-back processes have moved the same workload by
        // 8%. --prefill-sweep walks the routes inside one process against one loaded model, one
        // warmed device and one clock, resetting the sequence between arms.
        else if (a == "--prefill-sweep") {
            std::string list = next();
            for (size_t p = 0; p <= list.size(); ) {
                const size_t q = std::min(list.find(',', p), list.size());
                const std::string one = list.substr(p, q - p);
                if (one == "off" || one == "deployed") prefill_sweep.push_back(0);
                else if (one == "wide-deployed") prefill_sweep.push_back(20);
                else if (one == "wide-commit") prefill_sweep.push_back(18);
                else if (one == "wide-commit-a4") prefill_sweep.push_back(19);
                else { fprintf(stderr, "unknown prefill route: %s\n", one.c_str()); return 1; }
                p = q + 1;
            }
        }
        else if (a == "--sweep-rounds") sweep_rounds = atoi(next());
        else if (a == "--grid-rows") {
            grid_arms.clear();
            const std::string v = next();
            for (size_t i = 0; i < v.size();) { size_t j = v.find(',', i); if (j == std::string::npos) j = v.size(); grid_arms.push_back(atoi(v.substr(i, j - i).c_str())); i = j + 1; }
            if (grid_arms.empty()) throw std::runtime_error("--grid-rows needs at least one workgroup count");
        }
        else if (a == "--attn-deal") {
            deal_arms.clear();
            std::string v = next();
            for (size_t i = 0; i < v.size();) { size_t j = v.find(',', i); if (j == std::string::npos) j = v.size(); deal_arms.push_back(atoi(v.substr(i, j - i).c_str())); i = j + 1; }
            if (deal_arms.empty()) throw std::runtime_error("--attn-deal needs at least one arm");
        }
        else if (a == "--draft-grid") {
            draft_grids.clear();
            const std::string v = next();
            for (size_t i = 0; i < v.size();) { size_t j = v.find(',', i); if (j == std::string::npos) j = v.size(); draft_grids.push_back(atoi(v.substr(i, j - i).c_str())); i = j + 1; }
            if (draft_grids.empty()) throw std::runtime_error("--draft-grid needs at least one workgroup count");
        }
        else if (a == "--draft-weights") {
            draft_arms.clear();
            std::string s = next(), one;
            for (size_t i = 0; i <= s.size(); i++) {
                if (i == s.size() || s[i] == ',') {
                    if (one == "q8") draft_arms.push_back(0);
                    else if (one == "q4") draft_arms.push_back(1);
                    else throw std::runtime_error("--draft-weights takes q8, q4 or a comma list of them");
                    one.clear();
                } else one += s[i];
            }
            if (draft_arms.empty()) throw std::runtime_error("--draft-weights needs at least one coordinate");
        }
        else if (a == "--mtp") mtp_path = next();
        else if (a == "--dflash") dflash_path = next();
        else if (a == "--batch") batch = atoi(next());
        else if (a == "--context") context_capacity = atoi(next());
        else if (a == "--ffn") {
            const std::string mode = next();
            if (mode == "deployed") ffn_mode = 0;
            else if (mode == "a8") ffn_mode = 1;
            else if (mode == "a4") ffn_mode = 2;
            else if (mode == "scaled-a8") ffn_mode = 3;
            else if (mode == "sliced") ffn_mode = 4;
            else if (mode == "auto") ffn_mode = 5;
            else if (mode == "map-a8") ffn_mode = 6;
            else if (mode == "map-scaled-a8") ffn_mode = 7;
            else if (mode == "map-a4") ffn_mode = 8;
            else if (mode == "auto-map-a8") ffn_mode = 9;
            else if (mode == "auto-map-scaled-a8") ffn_mode = 10;
            else if (mode == "auto-map-a4") ffn_mode = 11;
            else if (mode == "single-map") ffn_mode = 12;
            else if (mode == "single-state") ffn_mode = 13;
            else if (mode == "single-retile") ffn_mode = 14;
            else if (mode == "single-grid") ffn_mode = 15;
            else if (mode == "commit-scaled") ffn_mode = 16;
            else if (mode == "wide-sequence") ffn_mode = 17;
            else if (mode == "wide-commit") ffn_mode = 18;
            else if (mode == "wide-commit-a4") ffn_mode = 19;
            else if (mode == "wide-deployed") ffn_mode = 20;
            else { fprintf(stderr, "unknown FFN mode: %s\n", mode.c_str()); return 1; }
        }
        else if (a == "--prompts") prompts_file = next();
        else if (a == "--prompt-ids") prompt_ids = next();
        else if (a == "--batch-ids-dir") batch_ids_dir = next();
        else if (a == "--pk-occ") {
            occ_arms.clear();
            std::string v = next();
            for (size_t i = 0, j; i <= v.size(); i = j + 1) {
                j = v.find(',', i); if (j == std::string::npos) j = v.size();
                const std::string one = v.substr(i, j - i);
                if (one == "0") occ_arms.push_back(0);
                else if (one == "1") occ_arms.push_back(1);
                else if (one == "2") occ_arms.push_back(2);
                else if (!one.empty()) throw std::runtime_error("--pk-occ takes 0 (allocator's choice), 1 (twelve waves) and/or 2 (thirteen)");
            }
            if (occ_arms.empty()) throw std::runtime_error("--pk-occ needs at least one budget");
        }
        else if (a == "--draft-n") n_draft = atoi(next());
        else if (a == "--dump") { dump_dir = next(); nograph = true; }
        else if (a == "--v1") v1 = true;
        else if (a == "--logits") logits_out = next();
        else if (a == "--readback") readback = next();
        else { fprintf(stderr, "usage: %s [-m model.gguf] [-p prompt] [-n tokens] [--raw] [--think] [--temp T] [--top-k K] [--top-p P] [--seed S] [--dflash FILE [--draft-weights q8|q4|q8,q4] [--grid-rows N,N] [--pk-occ 0|1|0,1]] [--mtp FILE] [--batch N] [--context N] [--ffn auto|deployed|a8|a4|scaled-a8|sliced|map-a8|map-scaled-a8|map-a4|auto-map-a8|auto-map-scaled-a8|auto-map-a4|single-map|single-state|single-retile|single-grid|commit-scaled|wide-sequence|wide-commit|wide-commit-a4|wide-deployed] [--pass-rows N] [--head-rows tail|all] [--draft-grid G[,G...]] [--attn-deal 0|1[,...]] [--profile max [--plain]] [--bench]\n       %s serve [--port 8471] [--host 127.0.0.1] [--model-id ID] [--max-tokens N] [--slots N] [--context N] [-m model.gguf] [--dflash FILE]\n", argv[0], argv[0]); return 1; }
    }
    try {
        // The max preset chooses a different measured route for independent streams and for
        // one client's exact-output speculation. The target remains the same GGUF in both.
        if (max_profile) {
            if (batch > 0) {
                if (!ffn_mode) ffn_mode = 19;
                if (context_capacity == MAXCTX) context_capacity = 256;
                if (!getenv("HALO_GDN_STATE")) gdn_state_boot_format(GDN_STATE_I8);
                if (!getenv("HALO_GDN_DEFER")) gdn_defer_set_depth(4);
            } else if (!plain && mtp_path.empty() && dflash_path.empty())
                dflash_path = "../../data/bonsai2/drafters/dflash2.safetensors";
        }
        if (!batch_ids_dir.empty() && batch <= 0) throw std::runtime_error("--batch-ids-dir needs --batch");
        if (plain && (!dflash_path.empty() || !mtp_path.empty()))
            throw std::runtime_error("--plain excludes a drafter");
        if (ffn_mode && (batch <= 0 || serve || v1 || !mtp_path.empty() || !dflash_path.empty()))
            throw std::runtime_error("--ffn requires plain --batch inference, without serving or drafters");
        // DFlash2 rides the wide route: its features are captured for every row of a wide pass.
        // The MTP head still needs a per-row final hidden that nothing writes there.
        const bool prefill_ffn_asked = prefill_ffn >= 0;
        if (prefill_ffn_asked && prefill_ffn && (batch > 0 || v1 || !mtp_path.empty()))
            throw std::runtime_error("--prefill-ffn covers plain, DFlash2 and served generation, without --batch, --v1 or --mtp");
        // The engine's fastest EXACT prompt ingestion is the wide schedule over the deployed FFN
        // (mode 20): same weights, same arithmetic, same output bits as the eight-row route, 2.9x
        // the rate on a served prompt. It is the default for every path that can take it. The
        // paths left on the eight-row route are the ones that cannot: `--batch` drives
        // `batch_mode` itself, `--v1` is the per-op reference path, `--mtp` has no per-row final
        // hidden on a wide pass, `--ffn` is a whole-run route selection, `--prefill-sweep` sets
        // the route per arm, and `--dump` writes per-op reference activations.
        if (!prefill_ffn_asked)
            prefill_ffn = (batch > 0 || v1 || !mtp_path.empty() || ffn_mode || !prefill_sweep.empty() || !dump_dir.empty()) ? 0 : 20;
        // The wide route needs 32 rows before `forward_batch` selects it at all, and the ceiling is
        // what the pass buffers are built for.
        if (pass_rows < 32 || pass_rows > PASSMAX)
            throw std::runtime_error("--pass-rows must be 32.." + std::to_string(PASSMAX));
        if (serve && (sc.max_active < 1 || sc.max_active > MAXSLOTS))
            throw std::runtime_error("--slots must be 1.." + std::to_string(MAXSLOTS));
        Engine e; e.use_graph = !nograph; e.fused = !v1 && dump_dir.empty(); e.head_rows_all = head_rows_all;
        e.load(model, nthreads, batch > 0 ? batch : serve ? sc.max_active : 1, context_capacity);
        // One budget on the command line pins it for the whole run; two are walked as a case axis
        // in the drafted bench below, where the arms share a clock.
        if (occ_arms.size() == 1 && occ_arms[0] >= 0) e.set_pk_occ(occ_arms[0]);
        e.batch_mode = ffn_mode;
        if (ffn_mode) {
            const int module_mode = ffn_mode >= 16 && ffn_mode != 20 ? (ffn_mode == 19 ? 8 : 7) : ffn_mode == 5 ? 1 : ffn_mode >= 9 ? ffn_mode - 3 : ffn_mode;
            // Mode 20 runs the deployed FFN under the wide schedule, so it wants the shared
            // workspace and none of the 4.28 GB weight images.
            e.prepare_batch(pass_rows, ffn_mode == 4 || ffn_mode == 20 || (ffn_mode >= 12 && ffn_mode <= 15) ? Engine::BATCH_WORKSPACE_ONLY : 1u << module_mode);
            if (ffn_mode >= 17) e.prepare_sequence();
        }
        if (!prefill_sweep.empty()) {
            // One process holds every route in the sweep, so it prepares the union of their
            // modules once. Modes 18 and 19 together exceed this host's allocation budget, which
            // is why a sweep carries at most one weight image.
            unsigned mask = 0; bool wide = false;
            for (int r : prefill_sweep) {
                if (!r) continue;
                wide = true;
                if (r != 20) mask |= 1u << (r == 19 ? 8 : 7);
            }
            if (mask & (mask - 1)) throw std::runtime_error("one weight-image route per sweep: 18 and 19 do not fit together");
            const size_t before = e.device_bytes;
            if (wide) { e.prepare_batch(pass_rows, mask ? mask : Engine::BATCH_WORKSPACE_ONLY); e.prepare_sequence(); }
            e.prefill_batch_mode = prefill_sweep.front();
            fprintf(stderr, "prefill sweep: %zu routes x %d rounds, +%.2f GB on device\n",
                    prefill_sweep.size(), sweep_rounds, (e.device_bytes - before) / 1e9);
        }
        // A single value is a plain setting for the whole process, so --grid-rows N works on any
        // run; two or more make it a case axis of the drafted bench below.
        if (grid_arms.size() == 1 && grid_arms[0] >= 0) e.set_grid_rows(grid_arms[0]);
        if (occ_arms.size() == 1 && occ_arms[0] >= 0) e.set_pk_occ(occ_arms[0]);
        if (prefill_ffn && prefill_sweep.empty()) {
            // Prompt ingestion only. Mode 20 keeps the deployed FFN and needs no weight image.
            const size_t before = e.device_bytes;
            e.prepare_batch(pass_rows, prefill_ffn == 20 ? Engine::BATCH_WORKSPACE_ONLY
                                                         : 1u << (prefill_ffn == 19 ? 8 : 7));
            e.prepare_sequence();
            e.prefill_batch_mode = prefill_ffn;
            fprintf(stderr, "wide prompt ingestion: mode %d%s, %d rows per pass, +%.2f GB on device\n",
                    prefill_ffn, prefill_ffn_asked ? "" : " (default)", e.prefill_width(),
                    (e.device_bytes - before) / 1e9);
        }
        Tokenizer tk; tk.load(model);
        unsigned draft_want = 0;
        for (int arm : draft_arms) draft_want |= arm ? Engine::DRAFT_Q4 : Engine::DRAFT_Q8;
        if (serve) {
            if (!dflash_path.empty()) e.load_dflash(dflash_path, nthreads, draft_want);
            if (sc.max_active > 1)
                fprintf(stderr, "concurrent serving: up to %d requests decode together, %d tokens of context each\n",
                        sc.max_active, e.context);
            return run_server(e, tk, sc);
        }
        if (batch > 0) {
            // N independent sequences; each step is one pass with one row per sequence
            std::vector<std::string> prompts(batch, raw ? prompt : Tokenizer::chat_prompt(prompt, thinking));
            if (!prompts_file.empty()) {
                std::ifstream f(prompts_file); std::string line; int i = 0;
                if (!f) throw std::runtime_error("cannot open prompts file: " + prompts_file);
                while (i < batch && std::getline(f, line))
                    if (!line.empty() && line[0] != '#') prompts[i++] = raw ? line : Tokenizer::chat_prompt(line, thinking);
                if (i < batch) throw std::runtime_error("prompts file has fewer usable lines than the batch size");
            }
            std::vector<Engine::Seq> seqs(batch);
            std::vector<std::vector<int>> outs(batch);
            std::vector<int> nexts(batch, -1);
            std::vector<bool> done(batch, false);
            double t0 = now();
            size_t prompt_tokens = 0;
            for (int b = 0; b < batch; b++) {
                seqs[b].slot = b; e.reset_seq(seqs[b]);
                std::vector<int> t = batch_ids_dir.empty() ? (prompt_ids.empty() ? tk.encode(prompts[b], true) : token_file(prompt_ids))
                    : token_file(batch_ids_dir + "/" + (b < 10 ? "00" : b < 100 ? "0" : "") + std::to_string(b) + ".ids.json");
                prompt_tokens += t.size();
                std::vector<Engine::Seq *> sv { &seqs[b] };
                std::vector<int> am;
                const int width = ffn_mode ? 128 : RMAX;
                for (size_t i = 0; i < t.size(); i += width) {
                    size_t k = std::min((size_t) width, t.size() - i);
                    std::vector<std::vector<int>> tv { std::vector<int>(t.begin() + i, t.begin() + i + k) };
                    bool last = i + k == t.size();
                    e.forward_batch(sv, tv, last, last ? &am : nullptr, nullptr, Engine::LogitRows::Tail);
                    if (last) nexts[b] = am.back();
                }
            }
            double t1 = now();
            fprintf(stderr, "prefill: %d sequences, %zu tokens in %.2f s (%.1f tok/s)\n", batch, prompt_tokens, t1 - t0, prompt_tokens / (t1 - t0));
            long produced = 0; int steps = 0;
            for (int step = 0; step < n_gen; step++) {
                std::vector<Engine::Seq *> sv; std::vector<std::vector<int>> tv; std::vector<int> idx;
                for (int b = 0; b < batch; b++) {
                    if (done[b]) continue;
                    if (tk.is_eog(nexts[b])) { done[b] = true; continue; }
                    outs[b].push_back(nexts[b]); produced++;
                    sv.push_back(&seqs[b]); tv.push_back({ nexts[b] }); idx.push_back(b);
                }
                if (sv.empty()) break;
                std::vector<int> am;
                e.forward_batch(sv, tv, true, &am);
                for (size_t i = 0; i < idx.size(); i++) nexts[idx[i]] = am[i];
                steps++;
            }
            double t2 = now();
            if (!bench) for (int b = 0; b < batch; b++) { printf("--- seq %d ---\n", b); for (int t : outs[b]) fputs(tk.piece(t).c_str(), stdout); printf("\n"); }
            fprintf(stderr, "batch %d: %ld tokens in %d steps, %.2f s: %.1f tok/s aggregate, %.1f ms/step\n", batch, produced, steps, t2 - t1, produced / (t2 - t1), 1000.0 * (t2 - t1) / steps);
            return 0;
        }
        std::string text = raw ? prompt : Tokenizer::chat_prompt(prompt, thinking);
        std::vector<int> toks = prompt_ids.empty() ? tk.encode(text, true) : token_file(prompt_ids);
        fprintf(stderr, "prompt: %zu tokens\n", toks.size());

        if (!mtp_path.empty() || !dflash_path.empty()) {
            const bool use_df = !dflash_path.empty();
            if (use_df) e.load_dflash(dflash_path, nthreads, draft_want); else e.load_mtp(mtp_path, nthreads);
            // Acceptance is workload dependent and a drafter change is judged on it, so a drafted
            // bench can carry a prompt set: --prompts FILE, one per line, walked inside one process
            // against one loaded model. Without it the single -p prompt is the only case.
            std::vector<std::vector<int>> cases{ toks };
            if (!prompts_file.empty()) {
                cases.clear();
                std::ifstream f(prompts_file); std::string line;
                if (!f) throw std::runtime_error("cannot open prompts file: " + prompts_file);
                while (std::getline(f, line))
                    if (!line.empty() && line[0] != '#') cases.push_back(tk.encode(raw ? line : Tokenizer::chat_prompt(line, thinking), true));
                if (cases.empty()) throw std::runtime_error("prompts file has no usable lines");
                fprintf(stderr, "prompt set: %zu cases\n", cases.size());
            }
            const std::vector<int> routes = prefill_sweep.empty()
                ? std::vector<int>{ e.prefill_batch_mode } : prefill_sweep;
            const std::vector<int> arms = use_df ? draft_arms : std::vector<int>{ 0 };
            for (int round = 0; round < std::max(1, sweep_rounds); round++)
            for (size_t ci = 0; ci < cases.size(); ci++)
            for (int route : routes)
            for (int arm : arms)
            for (int occ : occ_arms)
            for (int dg : draft_grids)
            for (int deal : deal_arms)
            for (int grid : grid_arms) {
                const std::vector<int> & toks = cases[ci];
                if (routes.size() > 1 || sweep_rounds > 1 || arms.size() > 1 || cases.size() > 1
                    || occ_arms.size() > 1 || grid_arms.size() > 1 || draft_grids.size() > 1
                    || deal_arms.size() > 1) { e.reset(); e.prefill_batch_mode = route; }
                if (deal >= 0) e.set_attn_deal(deal);
                if (occ >= 0) e.set_pk_occ(occ);
                e.set_grid_rows(grid);
                if (draft_grids.size() > 1 || draft_grids[0] != 0) set_dflash_grid(dg);
                if (use_df) e.df.use_q4 = arm != 0;
                // Each case is its own drafter timeline. Without this the second arm of
                // `--draft-weights q8,q4` reports the average of both, which reads as a real
                // number and is not one.
                e.reset_draft_profile();
                Engine::SpecStats st;
                double t0 = now();
                int produced = 0; std::string out;
                double tg0 = 0;
                // The greedy token stream of a drafted run is what a lossless drafter must not
                // change. One 64-bit digest per arm says so without diffing text across processes.
                unsigned long long digest = 1469598103934665603ull;
                auto on_token = [&](int tok) {
                    if (tg0 == 0) tg0 = now();
                    if (tk.is_eog(tok)) return false;
                    for (int b = 0; b < 4; b++) { digest ^= (unsigned) (tok >> (8 * b)) & 0xffu; digest *= 1099511628211ull; }
                    std::string p = tk.piece(tok); out += p;
                    if (!bench) { fputs(p.c_str(), stdout); fflush(stdout); }
                    produced++;
                    return true;
                };
                if (use_df) e.generate_dflash(toks, n_gen, on_token, st); else e.generate_mtp(toks, n_gen, n_draft, on_token, st);
                double t1 = now();
                if (!bench) printf("\n");
                fprintf(stderr, "prefill+first draft: %.2f s\n", tg0 - t0);
                fprintf(stderr, "%s: weights %s pk-occ %d grid %d dgrid %d deal %d route %d round %d case %zu, prompt %zu tokens in %.3f s: %.1f tok/s; %d tokens in %.2f s: %.2f tok/s; %ld steps, %ld drafted, %ld accepted (%.1f%%), %.2f tokens/step; per step: draft %.3f ms, verify %.3f ms; digest %llu\n",
                        use_df ? "dflash2" : "mtp", arm ? "q4" : "q8", occ, grid, dg, e.attn_deal(), route, round, ci, toks.size(), tg0 - t0, toks.size() / (tg0 - t0),
                        produced, t1 - tg0, produced / (t1 - tg0), st.steps, st.drafted, st.accepted, 100.0 * st.accepted / std::max(1L, st.drafted), (double) (st.accepted + st.steps) / std::max(1L, st.steps),
                        1000.0 * st.t_draft / std::max(1L, st.steps), 1000.0 * st.t_verify / std::max(1L, st.steps), digest);
                // One `prof` buffer serves both cooperative kernels, so a run reads the drafter's
                // segments or the verify pass's phases, never both: the second writer would be
                // printed under the first one's names.
                if (getenv("HALO_PROFILE_DRAFT")) e.print_draft_profile();
                else if (getenv("HALO_PROFILE")) e.print_profile();
            }
            return 0;
        }
        std::mt19937 rng(seed);
        std::vector<float> lg;
        auto pick = [&](int argmax) { if (temp <= 0.0f) return argmax; e.get_logits(lg); return sample_from_logits(lg, temp, top_k, top_p, rng); };
        double t0 = now();
        int next = -1, last_row = 0;
        if (e.prefill_width() > RMAX) {
            e.prefill(e.seq0, toks, 0, next, nullptr);
            last_row = e.fwd.nrows - 1;
            e.n_pos = (int) toks.size();
        } else if (e.fused && prefill_rows > 1) {
            std::vector<Engine::Seq *> sv { &e.seq0 };
            for (size_t i = 0; i < toks.size(); i += prefill_rows) {
                size_t k = std::min((size_t) prefill_rows, toks.size() - i);
                std::vector<std::vector<int>> tv { std::vector<int>(toks.begin() + i, toks.begin() + i + k) };
                std::vector<int> am;
                bool last = i + k == toks.size();
                e.forward(sv, tv, last, last ? &am : nullptr);
                if (last) { next = am.back(); last_row = (int) k - 1; }
            }
            e.n_pos = (int) toks.size();
        } else {
            for (size_t i = 0; i < toks.size(); i++) {
                if (i + 1 == toks.size() && !dump_dir.empty()) e.dump_dir = dump_dir;
                next = e.step(toks[i], i + 1 == toks.size());
            }
        }
        e.dump_dir.clear();
        next = pick(next);
        double t1 = now();
        if (!readback.empty()) {
            auto wr = [&](const char * nm, const float * d, size_t n) { auto v = e.read(d, n); FILE * f = fopen((readback + "/" + nm + ".bin").c_str(), "wb"); fwrite(v.data(), 4, n, f); fclose(f); };
            wr("x", e.x, RMAX * D); wr("xn", e.xn, D); wr("big", e.big, RMAX * QKV_OUT); wr("zbuf", e.zbuf, RMAX * VDIM); wr("obuf", e.obuf, RMAX * VDIM); wr("ab", e.ab, RMAX * 2 * HV); wr("gu", e.gu, 2 * FMAX * FF);
            wr("blk", e.blk_cache, 2 * 48 * RMAX * BLK_TOKEN_FLOATS); wr("ninv", e.ninv, RMAX);
            wr("qrot", e.qrot, RMAX * ATTN_OUT);
            { auto v = e.read((const float *) e.xq, RMAX * FF / 4); FILE * f = fopen((readback + "/xq.bin").c_str(), "wb"); fwrite(v.data(), 4, v.size(), f); fclose(f); }
            wr("xs", e.xs, NB_FF); wr("dbg", e.amax_val, 256 + 1280 + 80); wr("xsum", (const float *) e.xsum, NB_FF);
            wr("qfull", e.qfull, RMAX * Q_OUT); wr("kbuf", e.kbuf, RMAX * KV_OUT); wr("vbuf", e.vbuf, RMAX * KV_OUT); wr("y", e.y, VDIM);
        }
        if (!logits_out.empty()) { std::vector<float> lg; e.get_logits(lg, last_row); FILE * f = fopen(logits_out.c_str(), "wb"); fwrite(lg.data(), 4, lg.size(), f); fclose(f); }
        fprintf(stderr, "prefill: %zu tokens in %.2f s (%.1f tok/s)\n", toks.size(), t1 - t0, toks.size() / (t1 - t0));

        int produced = 0;
        unsigned long long digest = 1469598103934665603ull;
        double tg0 = now();
        std::string out;
        // HALO_BENCH_GRID=60,48,... alternates the persistent kernel's workgroup count token by
        // token inside this process, so the arms share an executable, a clock and a thermal state
        // and their samples are paired. A pair of processes cannot resolve a few percent on this
        // box; see orchestration/HANDOFF.md. HALO_PROFILE adds one phase map per arm.
        std::vector<int> arms;
        const char * ab_env = getenv("HALO_BENCH_GRID");
        if (ab_env) {
            for (const char * p = ab_env; *p;) { arms.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; }
        }
        const int ab = arms.size() >= 2;
        std::vector<std::vector<double>> tok_ms(arms.size());
        while (produced < n_gen) {
            if (tk.is_eog(next)) break;
            std::string p = tk.piece(next);
            out += p;
            if (!bench) { fputs(p.c_str(), stdout); fflush(stdout); }
            produced++;
            for (int b = 0; b < 4; b++) { digest ^= (unsigned) (next >> (8 * b)) & 0xffu; digest *= 1099511628211ull; }
            const int slot = ab ? (produced - 1) % (int) arms.size() : -1;
            if (slot >= 0) e.set_grid_rows(arms[slot]);
            const double ts = now();
            next = pick(e.step(next, true));
            if (slot >= 0) tok_ms[slot].push_back(1000.0 * (now() - ts));
            // The phase map of one token of each arm, taken from consecutive tokens of one process.
            if (slot >= 0 && getenv("HALO_PROFILE") && produced > n_gen - (int) arms.size() - 1) {
                fprintf(stderr, "---- phase map, grid %d\n", arms[slot]);
                e.print_profile();
            }
        }
        if (ab) {
            for (size_t a = 0; a < arms.size(); a++) {
                auto & v = tok_ms[a];
                if (v.empty()) continue;
                std::vector<double> s = v;
                std::sort(s.begin(), s.end());
                const double med = s[s.size() / 2];
                fprintf(stderr, "grid %d: %zu tokens, median %.3f ms (%.2f tok/s), min %.3f, mean %.3f\n",
                        arms[a], v.size(), med, 1000.0 / med, s.front(),
                        std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size());
            }
            for (size_t a = 1; a < arms.size(); a++) {
                const size_t np = std::min(tok_ms[0].size(), tok_ms[a].size());
                std::vector<double> d(np);
                size_t win = 0;
                for (size_t i = 0; i < np; i++) { d[i] = tok_ms[0][i] - tok_ms[a][i]; if (d[i] > 0) win++; }
                std::sort(d.begin(), d.end());
                fprintf(stderr, "paired %d minus %d: median %+.3f ms (%+.2f%%), %zu of %zu tokens favour %d\n",
                        arms[0], arms[a], np ? d[np / 2] : 0.0,
                        np ? 100.0 * d[np / 2] / tok_ms[0][np / 2] : 0.0, win, np, arms[a]);
            }
        }
        double tg1 = now();
        if (!bench) printf("\n");
        if (getenv("HALO_PROFILE")) e.print_profile();
        fprintf(stderr, "generated %d tokens in %.2f s: %.2f tok/s (%.2f ms/token); digest %llu\n", produced, tg1 - tg0,
                produced / (tg1 - tg0), produced ? 1000.0 * (tg1 - tg0) / produced : 0.0, digest);
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
