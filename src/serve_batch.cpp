// The scheduler behind the HTTP server: one engine thread, many requests.
//
// Every engine call in this file is one an ordinary `Engine::generate` makes, in the same order,
// with the same arguments. The only thing that differs is how many sequences a decode pass carries,
// and a pass's rows do not interact: `Engine::forward` gives row i the token, state slot and
// position of its own sequence, the recurrent state and the K/V cache are indexed by slot, and the
// weights are read-only. That is why a batched step needs no numerical argument of its own.
#include "serve_batch.h"
#include "batch_route.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace halo {

static double serve_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ServeBatch::Impl {
    Engine & e;
    const int max_active;

    struct Req {
        std::vector<int> prompt;
        Engine::GenParams p;
        // consumer-visible state, guarded by `m`
        std::deque<int> out;
        bool done = false;       // the scheduler is finished with this request
        bool cancel = false;     // the consumer asked to stop
        std::string error;
        Engine::SpecStats st;
        // scheduler-owned
        Engine::Seq seq;
        bool started = false;    // prompt ingested
        // The drafter's context holds every token of this sequence: its prompt was ingested with
        // features and every shared step fed its row in (`batched_step`). Such a request drafts
        // whenever it is alone, however much company it had before.
        bool drafter_current = false;
        int next = -1;
        int produced = 0;
        std::vector<int> drafts;
        std::mt19937 rng;
        // Ingestion, resumable. A prompt is admitted over several calls so the requests that are
        // already generating are not stopped for its duration; `ingested` is how much of it the
        // sequence holds and `stops` the snapshot points still ahead of that.
        size_t ingested = 0;
        bool ingest_open = false;
        std::vector<size_t> stops;
    };

    mutable std::mutex m;
    std::condition_variable wake;      // the engine thread waits here for work
    std::condition_variable out_cv;    // consumers wait here for their tokens
    std::deque<Req *> queued;
    std::vector<Req *> active;
    std::vector<int> free_slots;
    bool stopping = false;
    std::thread thread;

    Impl(Engine & engine, int slots) : e(engine), max_active(slots) {
        for (int i = e.nslots - 1; i >= 0; i--) free_slots.push_back(i);
        // Which route a shared step will take is a property of the process, and a panel that
        // prices two of them has to be able to see which one it started.
        const int route = decode_route();
        if (route)
            fprintf(stderr, "served decode: mode %d above %d rows, wide floor %d rows, up to %d rows per pass\n",
                    route, route_min_rows(), halo::batch_wide_min(), std::min(e.batch_capacity, PASSMAX));
        else
            fprintf(stderr, "served decode: eight-row passes\n");
        thread = std::thread([this] { loop(); });
    }

    ~Impl() {
        { std::lock_guard<std::mutex> lk(m); stopping = true; }
        wake.notify_all();
        if (thread.joinable()) thread.join();
    }

    // ---- consumer side ----

    void submit(Req * r) {
        { std::lock_guard<std::mutex> lk(m); queued.push_back(r); }
        wake.notify_one();
    }

    void wait_for(Req * r, const std::function<bool(int)> & on_token) {
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            out_cv.wait(lk, [&] { return !r->out.empty() || r->done; });
            while (!r->out.empty()) {
                const int tok = r->out.front();
                r->out.pop_front();
                lk.unlock();
                const bool go = on_token(tok);
                lk.lock();
                if (!go) {
                    r->cancel = true;
                    wake.notify_one();
                    out_cv.wait(lk, [&] { return r->done; });
                    return;
                }
            }
            if (r->done) return;
        }
    }

    // ---- engine side ----

    // Publishes a token and reports whether the request should keep going.
    bool emit(Req * r, int tok) {
        std::lock_guard<std::mutex> lk(m);
        r->out.push_back(tok);
        out_cv.notify_all();
        return !r->cancel;
    }

    bool cancelled(Req * r) {
        std::lock_guard<std::mutex> lk(m);
        return r->cancel;
    }

    // The request's slot goes back to the pool and the consumer is released. Nothing may touch the
    // request after this: `run` returns as soon as `done` is set and the request lives on its
    // caller's stack.
    void finish(Req * r, const std::string & error = {}) {
        std::lock_guard<std::mutex> lk(m);
        active.erase(std::remove(active.begin(), active.end(), r), active.end());
        free_slots.push_back(r->seq.slot);
        if (!error.empty()) r->error = error;
        r->done = true;
        out_cv.notify_all();
    }

    void admit() {
        std::lock_guard<std::mutex> lk(m);
        // A client that disconnected while queued leaves whether or not a slot is free; otherwise it
        // would wait behind requests it is no longer interested in.
        for (auto it = queued.begin(); it != queued.end();) {
            if ((*it)->cancel) { (*it)->done = true; it = queued.erase(it); }
            else ++it;
        }
        out_cv.notify_all();
        while (!queued.empty() && (int) active.size() < max_active && !free_slots.empty()) {
            Req * r = queued.front();
            queued.pop_front();
            r->seq = Engine::Seq{};
            r->seq.slot = free_slots.back();
            free_slots.pop_back();
            active.push_back(r);
        }
    }

    bool stop_now(Req * r) { return r->produced >= r->p.max_tokens || cancelled(r) || (r->p.aborted && r->p.aborted()); }

    int sample_row(Req * r, int row) {
        if (r->p.temp <= 0.0f) throw std::runtime_error("sampled row requested for a greedy request");
        std::vector<float> lg;
        e.get_logits(lg, row);
        return sample_from_logits(lg, r->p.temp, r->p.top_k, r->p.top_p, r->rng);
    }

    // Which route a step of several requests takes.
    //
    // A batched step used to call `Engine::forward`, whose pass is `RMAX` rows, so N requests
    // became `ceil(N/8)` complete forwards and every eight rows re-read the whole weight set. Mode
    // 4 keeps that persistent kernel for prep, the projections and attention and takes only the
    // two phases that are not bound by it out of the slice loop: the deployed FFN body reads no
    // row metadata and no recurrent state, so one launch serves up to `FMAX` rows, and the
    // vocabulary head runs once over the batch. Same weights, same K order, same per-block int32
    // accumulation, same FP32 fold - which rows share a launch is a schedule, and the acceptance
    // is that the served completions do not move.
    //
    // The wide schedule (mode 20) is the default now, and what it costs is measured rather than
    // deferred. `forward_batch` gives a pass the wide route only at `batch_wide_min()` rows or more
    // (32 unless a process moves it), so mode 20 is two different things to a served step:
    //
    //   9..31 rows   the same sliced route this served on before, plus the direct state commit.
    //                BIT-IDENTICAL: 7,946,240 full-vocabulary logits of a 32-row decode pass at
    //                `HALO_WIDE_MIN=33` agree to the bit between mode 4 and mode 20, and mode 4
    //                reproduces itself across processes on the same dump. It is worth about +6%.
    //   >=32 rows    the wide schedule: wide prep, wide sequence projections, resident GDN state.
    //                +36..55% aggregate at 32 streams, and a DIFFERENT NUMERICAL MAP - teacher-
    //                forced NLL +0.0064 +- 0.0072 nats over 512 paired predictions against a
    //                zero-floor control, at most +0.021 at two sigma, where the A4 FFN map this
    //                engine already serves prompts with costs +0.0615 on the same instrument.
    //                Top-1 agreement with the document is 0.4199 against 0.4121.
    //
    // docs/serve-decode-wide.md has both panels. `HALO_SERVE_DECODE_MODE=4` restores the
    // predecessor exactly, and an explicit `decode_batch_mode` keeps the contract it always had.
    static int decode_route_default() {
        static const int m = [] {
            const char * v = getenv("HALO_SERVE_DECODE_MODE");
            return v ? atoi(v) : 20;
        }();
        return m;
    }
    int decode_route() const {
        if (!e.batch_capacity) return 0;
        if (e.decode_batch_mode) {                       // an explicit choice keeps its exact contract
            const int m = e.decode_batch_mode;
            return m >= 17 && !e.batch_sequence ? 0 : m;
        }
        const int m = decode_route_default();
        // A process that built no sequence modules cannot take the wide route at any width; it gets
        // the sliced one it had before this default existed rather than the one-row persistent pass.
        if (m >= 17 && !e.batch_sequence) return 4;
        return m;
    }
    // Below this a step keeps the persistent kernel it has always used. Nine rows is the first
    // width that would need a second eight-row pass, so one to eight rows are untouched code.
    static int route_min_rows() {
        static const int n = [] {
            const char * v = getenv("HALO_SERVE_ROUTE_MIN");
            const int k = v ? atoi(v) : RMAX + 1;
            return k >= 1 ? k : RMAX + 1;
        }();
        return n;
    }

    // How many prompt tokens one admission chunk carries while other requests are generating.
    //
    // A chunk boundary is a boundary `Engine::prefill` already has: it takes the prompt it has
    // ingested so far as `from` and walks `prefill_width()` rows a pass from there, which is
    // exactly what the snapshot points in `prefill_begin` have always done. Rounding the budget
    // down to whole passes keeps the pass composition of an unchunked ingestion, so what a chunk
    // adds is the state commit at its last pass and nothing else.
    size_t chunk_tokens() const {
        static const long want = [] {
            const char * v = getenv("HALO_SERVE_PREFILL_CHUNK");
            return v ? atol(v) : 0;
        }();
        if (want < 0) return 0;   // the server this replaced: one prompt stops every other request
        const size_t w = (size_t) std::max(1, e.prefill_width());
        // One wide pass by default. The floor matters for a server whose ingestion did not get the
        // wide route - there a pass is RMAX tokens and costs a whole weight stream, so a decode
        // step every pass would be half the work of ingesting.
        const size_t n = want > 0 ? (size_t) want : std::max(w, (size_t) 256);
        return std::max(w, n / w * w);
    }

    // The prompt-prefix cache and the snapshot points of `Engine::generate`, against this request's
    // own slot. Everything here happens once, before the first token of the prompt is ingested.
    void prefill_begin(Req * r) {
        Engine::Seq & s = r->seq;
        size_t from = 0;
        const int area = e.snapshot_find(r->prompt);
        if (area >= 0) { e.snapshot_restore(s, area); from = e.snaps[area].tokens.size(); }
        else e.reset_seq(s);
        r->st.prompt_tokens = (long) r->prompt.size();
        r->st.cached_tokens = (long) from;
        std::vector<size_t> points;
        for (size_t pt : r->p.snapshot_points) {
            const size_t at = pt / RMAX * RMAX;
            if (at > from && at < r->prompt.size()) points.push_back(at);
        }
        {
            size_t best_lcp = 0;
            for (const Engine::Snapshot & sn : e.snaps) {
                if (!sn.valid) continue;
                size_t k = 0;
                const size_t n = std::min(sn.tokens.size(), r->prompt.size());
                while (k < n && sn.tokens[k] == r->prompt[k]) k++;
                if (k < sn.tokens.size() && k > best_lcp) best_lcp = k;
            }
            const size_t at = best_lcp / RMAX * RMAX;
            if (best_lcp >= 256 && at > from && at < r->prompt.size()) points.push_back(at);
        }
        { const size_t at = r->prompt.size() > RMAX ? (r->prompt.size() - 1) / RMAX * RMAX : 0; if (at > from) points.push_back(at); }
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        r->stops = points;
        r->ingested = from;
        r->ingest_open = true;
    }

    // Ingest up to `budget` more prompt tokens (0 means all of them) and report whether the prompt
    // is now in the sequence. A request ingests alone - two prompts are never interleaved - so the
    // prefix cache keeps the property it was built on: the second request of a shared preamble
    // still looks for its snapshot after the first request saved it.
    bool prefill_step(Req * r, size_t budget) {
        if (!r->ingest_open) prefill_begin(r);
        const double t0 = serve_now();
        Engine::Seq & s = r->seq;
        const size_t end = r->prompt.size();
        const bool spec = r->drafter_current;
        size_t spent = 0;
        int next = -1;
        while (r->ingested < end) {
            size_t stop = r->stops.empty() ? end : r->stops.front();
            const size_t room = budget ? budget - spent : end - r->ingested;
            if (r->ingested + room < stop) stop = r->ingested + room;
            const bool save = !r->stops.empty() && stop == r->stops.front();
            const bool whole = stop == end;
            std::vector<int> part(r->prompt.begin(), r->prompt.begin() + stop);
            int last = -1;
            e.prefill(s, part, r->ingested, last, whole && spec ? &r->drafts : nullptr);
            if (save) { e.snapshot_save(s, part); r->stops.erase(r->stops.begin()); }
            spent += stop - r->ingested;
            r->ingested = stop;
            if (whole) next = last;
            if (budget && spent >= budget) break;
        }
        r->st.t_prefill += serve_now() - t0;
        if (r->ingested < end) return false;
        r->started = true;
        r->next = r->p.temp <= 0.0f ? next : sample_row(r, e.fwd.nrows - 1);
        r->produced = 1;
        const bool go = emit(r, r->next);
        if (!go || r->produced >= r->p.max_tokens) finish(r);
        return true;
    }

    // The speculative step `Engine::generate` runs when a drafter is loaded and decoding is greedy.
    void drafted_step(Req * r) {
        std::vector<Engine::Seq *> sv { &r->seq };
        std::vector<int> block { r->next };
        block.insert(block.end(), r->drafts.begin(), r->drafts.end());
        std::vector<std::vector<int>> tv { block };
        std::vector<int> am;
        const double t0 = serve_now();
        e.fwd.hcap = e.hcap;
        e.forward(sv, tv, true, &am);
        e.fwd.hcap = nullptr;
        r->st.t_verify += serve_now() - t0;
        int a = 0;
        while (a < (int) r->drafts.size() && am[a] == r->drafts[a]) a++;
        r->st.steps++;
        r->st.drafted += (long) r->drafts.size();
        r->st.accepted += a;
        bool go = true;
        for (int i = 0; i < a && go; i++) { go = emit(r, r->drafts[i]); r->produced++; if (stop_now(r)) go = false; }
        if (go) { go = emit(r, am[a]); r->produced++; if (stop_now(r)) go = false; }
        const int start_old = r->seq.len - (int) block.size();
        Engine::rollback(r->seq, a + 1);
        if (!go) { finish(r); return; }
        r->next = am[a];
        const double td0 = serve_now();
        e.dflash_step(r->seq, a + 1, start_old, r->next, r->seq.len, &r->drafts);
        r->st.t_draft += serve_now() - td0;
    }

    // One token for every request in `rs`. A pass carries as many rows as the route allows:
    // `MAXSEQ` on the persistent kernel, the prepared batch capacity on the wide one.
    void batched_step(const std::vector<Req *> & rs) {
        const int route = decode_route();
        // The pass width belongs to the route that will carry it: the persistent kernel takes
        // MAXSEQ sequences and RMAX rows, the wide one takes the prepared capacity. Deciding the
        // width before the route is what makes a 16-row step ask `Engine::forward` for sixteen
        // sequences and be told it cannot have them.
        const bool wide = route && (int) rs.size() >= route_min_rows();
        const size_t width = wide ? (size_t) std::min(e.batch_capacity, PASSMAX) : (size_t) MAXSEQ;
        for (size_t i = 0; i < rs.size(); i += width) {
            const size_t n = std::min(width, rs.size() - i);
            std::vector<Engine::Seq *> sv;
            std::vector<std::vector<int>> tv;
            bool sampled = false;
            for (size_t j = 0; j < n; j++) {
                sv.push_back(&rs[i + j]->seq);
                tv.push_back({ rs[i + j]->next });
                sampled = sampled || rs[i + j]->p.temp > 0.0f;
            }
            std::vector<int> am;
            std::vector<float> lg;
            if (wide) {
                // One D2H copy of the rows a sampler will read, instead of one synchronised
                // `get_logits` per sampled row: a wide pass leaves its logits in the head module,
                // and `publish_last_logits` only ever moves the tail row.
                const int saved = e.batch_mode;
                e.batch_mode = route;
                try { e.forward_batch(sv, tv, true, &am, sampled ? &lg : nullptr); }
                catch (...) { e.batch_mode = saved; throw; }
                e.batch_mode = saved;
            } else {
                // Capture this pass's drafter features and feed each sequence its row, so a request
                // that shares steps keeps a current drafter and speculates again once it is alone.
                // Rows are in sequence order, one per sequence. The target's map is unchanged:
                // capture copies residuals the pass already computes.
                bool feed = false;
                for (size_t j = 0; j < n; j++) feed = feed || rs[i + j]->drafter_current;
                if (feed) e.fwd.hcap = e.hcap;
                try { e.forward(sv, tv, true, &am); }
                catch (...) { e.fwd.hcap = nullptr; throw; }
                e.fwd.hcap = nullptr;
                if (feed) e.dflash_ingest(sv);
            }
            for (size_t j = 0; j < n; j++) {
                Req * r = rs[i + j];
                // A wide pass captures nothing this route feeds; its drafts are stale either way.
                if (wide) r->drafter_current = false;
                r->drafts.clear();
                if (r->p.temp <= 0.0f) r->next = am[j];
                else if (wide) {
                    std::vector<float> row(lg.begin() + (size_t) j * VOCAB, lg.begin() + (size_t) (j + 1) * VOCAB);
                    r->next = sample_from_logits(row, r->p.temp, r->p.top_k, r->p.top_p, r->rng);
                } else r->next = sample_row(r, (int) j);
                r->produced++;
                r->st.steps++;
                const bool go = emit(r, r->next);
                if (!go || stop_now(r)) finish(r);
            }
        }
    }

    void loop() {
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            wake.wait(lk, [&] { return stopping || !queued.empty() || !active.empty(); });
            if (stopping && active.empty()) {
                for (Req * r : queued) { r->done = true; }
                queued.clear();
                out_cv.notify_all();
                return;
            }
            std::vector<Req *> batch;
            lk.unlock();
            admit();
            { std::lock_guard<std::mutex> g(m); batch = active; }
            std::string err;
            try {
                // A cancelled request leaves before it can cost a pass.
                for (Req * r : batch) if (cancelled(r)) finish(r);
                { std::lock_guard<std::mutex> g(m); batch = active; }
                Req * pre = nullptr;
                for (Req * r : batch) if (!r->started) { pre = r; break; }
                if (pre) {
                    // Prompt ingestion always feeds the drafter, so any greedy request may draft once
                    // it is alone, whether or not it arrived into company.
                    if (!pre->ingest_open)
                        pre->drafter_current = pre->p.speculative && pre->p.temp <= 0.0f && e.df.loaded;
                    // The requests that are already generating take a step between this prompt's
                    // chunks. A prompt used to be one call, so a 2000-token ingestion emitted
                    // nothing for every other request for its whole duration - and the cost of
                    // that is not only the wait: a request held out of the batch finishes later,
                    // which is how a served step loses the rows that make it worth its weight
                    // stream. With nothing else to run this is one call and the code the server
                    // has always run.
                    std::vector<Req *> gen;
                    for (Req * r : batch) if (r->started) gen.push_back(r);
                    const bool done = prefill_step(pre, gen.empty() ? 0 : chunk_tokens());
                    if (!done && !gen.empty()) batched_step(gen);
                } else if (batch.size() == 1 && batch[0]->drafter_current) {
                    drafted_step(batch[0]);
                } else if (!batch.empty()) {
                    batched_step(batch);
                }
            } catch (const std::exception & ex) {
                err = ex.what();
            }
            if (!err.empty()) {
                std::vector<Req *> victims;
                { std::lock_guard<std::mutex> g(m); victims = active; }
                for (Req * r : victims) finish(r, err);
            }
            lk.lock();
        }
    }
};

ServeBatch::ServeBatch(Engine & e, int max_active) {
    const int cap = std::max(1, std::min(max_active, e.nslots));
    impl.reset(new Impl(e, cap));
}

ServeBatch::~ServeBatch() = default;

int ServeBatch::max_active() const { return impl->max_active; }

int ServeBatch::active() const {
    std::lock_guard<std::mutex> lk(impl->m);
    return (int) impl->active.size() + (int) impl->queued.size();
}

void ServeBatch::run(const std::vector<int> & prompt, const Engine::GenParams & params,
                     const std::function<bool(int)> & on_token, Engine::SpecStats & stats) {
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if ((int) prompt.size() + params.max_tokens + DF_BLOCK >= impl->e.context)
        throw std::runtime_error("prompt and max_tokens exceed the context of " + std::to_string(impl->e.context));
    Impl::Req r;
    r.prompt = prompt;
    r.p = params;
    r.rng.seed(params.seed);
    impl->submit(&r);
    impl->wait_for(&r, on_token);
    stats = r.st;
    std::string error;
    { std::lock_guard<std::mutex> lk(impl->m); error = r.error; }
    if (!error.empty()) throw std::runtime_error(error);
}

} // namespace halo
