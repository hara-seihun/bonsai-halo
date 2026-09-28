// Mixed-mode sequence state acceptance, on the real model.
//
// Direct commit (FwdParams::commit_state, batch modes 16, 18, 19) moves where the GDN state and
// conv ring become durable: to the end of the pass that produced them, instead of the start of the
// next pass that replays them. The arithmetic does not change, so a sequence that switches between
// a replaying mode and a committing mode part way through must end up with exactly the state it
// would have had if it had never switched.
//
// This tool asserts that on the loaded 64-layer model, pass by pass, by running two copies of the
// same token stream in one engine:
//
//   reference   every pass in the replaying mode (10, or 17 for the wide projections)
//   mixed       the same passes, some of them in the committing twin of that mode (16, or 18)
//
// After every pass it compares the two copies on full-vocabulary logits, per-row argmax, the whole
// GDN state of each sequence slot and the whole conv ring. Differences are counted and reported.
// Nothing here assumes the two sides agree; "bitwise equal" is a measured count of differing
// floats, and non-finite values are counted on both sides rather than presumed absent.
//
// The two copies do not hold their state in the same place at every moment, and the tool must not
// pretend they do. A durable image covers tokens [0, len - keep): a replaying pass leaves its own
// rows in blk_cache for its successor to commit, while a committing pass has already stored them
// and sets keep to 0. Straight after a committing pass the mixed copy is durable at a later prefix
// than the reference, by exactly that pass's rows, and raw gdn_state and conv_ring differ for a
// correct implementation. The logical states agree; the durable images have not yet met.
//
// So logits and argmax are required equal after every pass, being a function of the logical state,
// which never lags. Raw state and ring are compared and reported after every pass for every slot,
// including one that sat the pass out, but equality is only required where both copies of that
// slot are durable at the same prefix. Each scenario then ends with one identical committing pass
// on both copies, which flushes both deferred prefixes, and there the raw state and ring must
// match. That pass is appended after the trajectory, never inserted into it, so the history under
// test keeps its original commit points. A raw difference at an unaligned prefix is reported with
// both prefixes and the gap between them, and is never relabelled as a pass.
//
// Only modes with identical arithmetic are paired. 16 against 10 and 18 against 17 differ solely
// in commit_state. Mode 19 is wide, committing and A4, and has no uncommitted wide A4 twin to pair
// with, so it is not a state comparison and is not run here; comparing it against 11 would measure
// the wide projection map, which is tools/batch_compare's job.
//
// The pass shapes are ragged and cross the dispatch boundaries in Engine::forward_batch: at most 4
// rows runs the deployed whole-pass kernel, 5 to 31 the sliced schedule, 32 and above the batched
// FFN and (modes 17 and up) the wide sequence projections. The default plan visits 1, 3, 8, 17 and
// 40 rows, with two sequences whose rows share and straddle the 8-row slices.
//
// It also exercises the rollback boundary main added alongside commit: a committed pass must
// refuse to have tokens rejected, and must refuse without touching the Seq; an uncommitted pass
// after a committed one must still roll back correctly.
//
// What this does not cover: timing, the wide projection's numerics against the deployed map, the
// drafters, and anything about a mode's speed. It answers one question, whether mixing commit into
// a sequence's history changes that sequence's state or output.
//
// Build: see tools/direct-commit/Makefile (or the root Makefile's tools target).
// Run:   tools/sequence_state_check --model PATH --out run.json

#include "engine.h"
#include "ffn_batch.h"
#include "sequence_batch.h"
#include "tokenizer.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace halo;
using json = nlohmann::json;
namespace fs = std::filesystem;

// Recurrent layers per sequence slot: the stride engine.cpp uses for gdn_state and conv_ring.
constexpr int GDN_LAYERS = NLAYER - NLAYER / 4;

// ------------------------------------------------------------------ float comparison

// A monotone integer key for a float, so "how far apart" is meaningful for values that differ in
// the last bits. Equal values share a key; ordering matches float ordering.
static long long ordered_key(float f) {
    int32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    return bits >= 0 ? (long long) bits : -2147483648LL - (long long) bits;
}

static json jnum(double v) {
    if (std::isfinite(v)) return json(v);
    return json(std::isnan(v) ? "nan" : v > 0 ? "inf" : "-inf");
}

struct Diff {
    size_t count = 0;            // floats compared
    size_t differing = 0;        // floats whose bits differ
    size_t nonfinite_ref = 0, nonfinite_mix = 0;
    double max_abs = 0.0;        // largest |reference - mixed| over pairs where both are finite
    long long max_ulp = 0;
    long long first_index = -1;
    std::vector<size_t> groups_differing;   // e.g. logit rows, or GDN layer indices
    size_t groups_differing_total = 0;
    bool equal() const { return differing == 0 && nonfinite_ref == 0 && nonfinite_mix == 0; }

    json to_json(size_t group_size = 0) const {
        json j{ { "compared", count }, { "differing", differing },
                { "nonfinite_reference", nonfinite_ref }, { "nonfinite_mixed", nonfinite_mix },
                { "max_abs_diff", jnum(max_abs) }, { "max_ulp", max_ulp },
                { "bitwise_equal", differing == 0 } };
        if (first_index >= 0) {
            j["first_difference"] = group_size ? json{ { "flat", first_index },
                                                       { "group", (size_t) first_index / group_size },
                                                       { "offset", (size_t) first_index % group_size } }
                                               : json{ { "flat", first_index } };
        }
        if (groups_differing_total) {
            j["groups_differing"] = groups_differing_total;
            j["groups_differing_first"] = groups_differing;
        }
        return j;
    }
};

// group_size splits the buffers into logical rows (logits) or layers (state, ring) so a failure
// says where it is instead of only how big it is.
static Diff compare_floats(const std::vector<float> & ref, const std::vector<float> & mix,
                           size_t group_size = 0, size_t list_groups = 8) {
    Diff d;
    if (ref.size() != mix.size())
        throw std::runtime_error("comparing buffers of different length: " + std::to_string(ref.size()) +
                                 " against " + std::to_string(mix.size()));
    d.count = ref.size();
    long long last_group = -1;
    for (size_t i = 0; i < ref.size(); i++) {
        const float a = ref[i], b = mix[i];
        const bool fa = std::isfinite(a), fb = std::isfinite(b);
        if (!fa) d.nonfinite_ref++;
        if (!fb) d.nonfinite_mix++;
        uint32_t ba, bb;
        std::memcpy(&ba, &a, 4); std::memcpy(&bb, &b, 4);
        if (ba == bb) continue;
        d.differing++;
        if (d.first_index < 0) d.first_index = (long long) i;
        if (fa && fb) {
            d.max_abs = std::fmax(d.max_abs, std::fabs((double) a - (double) b));
            d.max_ulp = std::max(d.max_ulp, std::llabs(ordered_key(a) - ordered_key(b)));
        } else {
            d.max_abs = std::numeric_limits<double>::infinity();
        }
        if (group_size) {
            const long long g = (long long) (i / group_size);
            if (g != last_group) {
                last_group = g;
                d.groups_differing_total++;
                if (d.groups_differing.size() < list_groups) d.groups_differing.push_back((size_t) g);
            }
        }
    }
    return d;
}

// ------------------------------------------------------------------ configuration

struct Cfg {
    std::string model = "../../data/bonsai2/PTQ1_0.gguf";
    std::string doc = "PLAN.md";
    std::string out = "../../data/bonsai2/sequence-state-check/run.json";
    std::vector<std::pair<int, int>> shapes{ { 1, 0 }, { 2, 1 }, { 5, 3 }, { 9, 8 }, { 24, 16 }, { 0, 1 }, { 3, 5 } };
    // Which passes the mixed run commits. The default visits every transition: uncommitted first,
    // into commit, commit to commit, back out, and in again across the wide pass.
    std::vector<bool> commit{ false, true, true, false, true, false, true };
    std::vector<int> references{ 10, 17 };   // 10 pairs with 16, 17 pairs with 18
    int context = 512;
    int nthreads = (int) std::thread::hardware_concurrency();
    bool compare_state = true;               // the 151 MB per-slot GDN state read
    bool run_rollback = true;
};

static int commit_twin(int reference_mode) {
    switch (reference_mode) {
        case 10: return 16;   // auto scaled A8, committing
        case 17: return 18;   // wide sequence projections, committing
        default: return -1;
    }
}

// The FFN weight image a mode reaches at 32 rows and above, mirroring the dispatch in
// Engine::forward_batch. Modes below that threshold run the engine's own FFN and need no image.
static int mode_image(int m) {
    switch (m) {
        case 9:  return 6;
        case 10: case 16: case 17: case 18: return 7;
        case 11: case 19: return 8;
        default: return -1;
    }
}

// Which schedule a pass of this many rows actually takes, for the report. Same conditions as
// Engine::forward_batch: it is the row count, not the mode, that picks the schedule.
static std::string schedule_of(int mode, int rows) {
    const bool sequence_mode = mode >= 16;
    if (rows <= 4) return "deployed";
    if (rows < 32) return "sliced";
    if (sequence_mode && mode >= 17) return "wide-sequence";
    return "batched-ffn";
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream s; s << f.rdbuf();
    return s.str();
}

// ------------------------------------------------------------------ driver

struct Checker {
    Engine & e;
    const Cfg & cfg;
    std::vector<int> doc;
    size_t failures = 0;
    size_t deferred_observed = 0;   // raw differences at prefixes that had not met yet: expected, not faults

    // Two copies of the same two sequences: the reference pair and the mixed pair, in their own
    // slots of one engine, reset before each scenario.
    Engine::Seq ref_a, ref_b, mix_a, mix_b;

    Checker(Engine & en, const Cfg & c) : e(en), cfg(c) {
        ref_a.slot = 0; ref_b.slot = 1; mix_a.slot = 2; mix_b.slot = 3;
    }

    void reset_all() {
        for (Engine::Seq * s : { &ref_a, &ref_b, &mix_a, &mix_b }) e.reset_seq(*s);
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
    }

    // Sequence A reads the document from the front, B from the middle, so the two sequences carry
    // genuinely different tokens and a slot mix-up cannot pass unnoticed.
    std::vector<int> tokens(char which, int from, int n) const {
        const size_t base = which == 'a' ? 0 : doc.size() / 2;
        if (base + from + n > doc.size())
            throw std::runtime_error("document has too few tokens for the pass plan");
        return std::vector<int>(doc.begin() + base + from, doc.begin() + base + from + n);
    }

    static json seq_json(const Engine::Seq & s) {
        return json{ { "slot", s.slot }, { "len", s.len }, { "keep", s.keep },
                     { "last_n", s.last_n }, { "parity", s.parity }, { "rollbackable", s.rollbackable },
                     { "durable_prefix", s.len - s.keep } };
    }

    // Tokens whose effect is already in gdn_state and conv_ring.
    static int durable_prefix(const Engine::Seq & s) { return s.len - s.keep; }

    struct SlotCheck { json j; bool asserted = false, ok = true, deferred_difference = false; };

    // One slot's durable image: compared and reported always, required to match only where the two
    // copies have stored the same tokens.
    SlotCheck compare_slot(const char * name, Engine::Seq & rs, Engine::Seq & ms,
                           bool allow_parity_difference = false) {
        SlotCheck c;
        const int rp = durable_prefix(rs), mp = durable_prefix(ms);
        c.asserted = rp == mp;
        // keep and rollbackable differ by design after a committing pass; len and last_n never may,
        // and parity only where a rejected pass legitimately puts one copy a flip ahead.
        const bool book_ok = rs.len == ms.len && rs.last_n == ms.last_n &&
                             (allow_parity_difference || rs.parity == ms.parity);
        const Diff dr = compare_floats(conv_ring(rs), conv_ring(ms), GDN_RING_FLOATS);
        bool raw_equal = dr.equal();
        json js{ { "sequence", name },
                 { "seq_reference", seq_json(rs) }, { "seq_mixed", seq_json(ms) },
                 { "durable_prefix", json{ { "reference", rp }, { "mixed", mp },
                                           { "aligned", c.asserted }, { "mixed_ahead_by", mp - rp } } },
                 { "bookkeeping_consistent", book_ok },
                 { "conv_ring", dr.to_json(GDN_RING_FLOATS) } };
        if (allow_parity_difference) js["parity_difference_allowed"] = true;
        if (cfg.compare_state) {
            const size_t region = gdn_region_floats();
            const Diff ds = compare_floats(gdn_state(rs), gdn_state(ms), region);
            js["gdn_state"] = ds.to_json(region);
            raw_equal = raw_equal && ds.equal();
        }
        js["raw_state_bitwise_equal"] = raw_equal;
        js["state_asserted"] = c.asserted;
        if (!c.asserted && raw_equal) {
            // The mixed copy says it has stored more tokens than the reference, yet its durable
            // image is identical. Worth seeing: that is what a commit_state that stored nothing
            // would look like here, and the later passes would then be the ones to fail.
            js["unaligned_but_identical"] = true;
        }
        js["state_note"] = c.asserted
            ? "both copies have stored the same " + std::to_string(rp) + " tokens, so their durable images must match"
            : "the mixed copy has stored " + std::to_string(mp - rp) +
              " tokens the reference still holds in blk_cache; a difference here is expected, and the "
              "synchronising pass at the end of the scenario is where it has to vanish";
        c.ok = book_ok && (!c.asserted || raw_equal);
        c.deferred_difference = !c.asserted && !raw_equal;
        deferred_observed += c.deferred_difference;
        js["ok"] = c.ok;
        c.j = std::move(js);
        return c;
    }

    std::vector<float> gdn_state(const Engine::Seq & s) {
        // A slot's whole durable image, at whatever stride this process allocated its regions.
        return e.read(e.gdn_state + (size_t) s.slot * GDN_LAYERS * gdn_region_floats(),
                      (size_t) GDN_LAYERS * gdn_region_floats());
    }
    std::vector<float> conv_ring(const Engine::Seq & s) {
        return e.read(e.conv_ring + (size_t) s.slot * GDN_LAYERS * GDN_RING_FLOATS,
                      (size_t) GDN_LAYERS * GDN_RING_FLOATS);
    }

    // One pass over whichever of the two sequences has rows this time.
    struct PassOut { std::vector<float> logits; std::vector<int> argmax; };
    PassOut run_pass(int mode, Engine::Seq & a, Engine::Seq & b,
                     const std::vector<int> & ta, const std::vector<int> & tb) {
        std::vector<Engine::Seq *> seqs;
        std::vector<std::vector<int>> toks;
        if (!ta.empty()) { seqs.push_back(&a); toks.push_back(ta); }
        if (!tb.empty()) { seqs.push_back(&b); toks.push_back(tb); }
        if (seqs.empty()) throw std::runtime_error("a pass with no rows");
        PassOut out;
        e.batch_mode = mode;
        e.forward_batch(seqs, toks, true, &out.argmax, &out.logits);
        return out;
    }

    // ---- scenario: the same token stream, one copy switching modes -------------------------
    json scenario(int reference_mode) {
        const int commit_mode = commit_twin(reference_mode);
        json sc{ { "name", "modes " + std::to_string(reference_mode) + " and " + std::to_string(commit_mode) +
                           " mixed, against " + std::to_string(reference_mode) + " throughout" },
                 { "reference_mode", reference_mode }, { "commit_mode", commit_mode } };
        reset_all();

        int at_a = 0, at_b = 0;
        bool all_equal = true;
        json passes = json::array();
        for (size_t p = 0; p < cfg.shapes.size(); p++) {
            const int na = cfg.shapes[p].first, nb = cfg.shapes[p].second;
            const bool commit = p < cfg.commit.size() ? cfg.commit[p] : (p % 2 == 1);
            const int mixed_mode = commit ? commit_mode : reference_mode;
            const std::vector<int> ta = tokens('a', at_a, na), tb = tokens('b', at_b, nb);
            at_a += na; at_b += nb;

            const PassOut r = run_pass(reference_mode, ref_a, ref_b, ta, tb);
            const PassOut m = run_pass(mixed_mode, mix_a, mix_b, ta, tb);

            json jp{ { "index", p },
                     { "rows", json{ { "a", na }, { "b", nb }, { "total", na + nb } } },
                     { "mixed_mode", mixed_mode }, { "commits", commit },
                     { "schedule", json{ { "reference", schedule_of(reference_mode, na + nb) },
                                         { "mixed", schedule_of(mixed_mode, na + nb) } } } };

            const Diff dl = compare_floats(r.logits, m.logits, VOCAB);
            jp["logits"] = dl.to_json(VOCAB);
            size_t argmax_same = 0;
            for (size_t i = 0; i < r.argmax.size() && i < m.argmax.size(); i++) argmax_same += r.argmax[i] == m.argmax[i];
            jp["argmax"] = json{ { "rows", r.argmax.size() }, { "matching", argmax_same } };
            bool pass_equal = dl.equal() && argmax_same == r.argmax.size();

            json per_seq = json::array();
            int deferred = 0, asserted = 0;
            for (auto & pair : slot_pairs()) {
                const SlotCheck c = compare_slot(pair.first, *pair.second.first, *pair.second.second);
                asserted += c.asserted;
                deferred += c.deferred_difference;
                pass_equal = pass_equal && c.ok;
                per_seq.push_back(c.j);
            }
            jp["sequences"] = std::move(per_seq);
            jp["slots_state_asserted"] = asserted;
            jp["slots_state_deferred_and_differing"] = deferred;
            jp["equal"] = pass_equal;
            all_equal = all_equal && pass_equal;
            if (!pass_equal) failures++;
            passes.push_back(std::move(jp));
            std::fprintf(stderr, "  pass %zu  %2d+%2d rows  mode %2d %-13s %s  (state asserted on %d of 2 slots)\n",
                         p, na, nb, mixed_mode, schedule_of(mixed_mode, na + nb).c_str(),
                         pass_equal ? "equal" : "DIFFERS", asserted);
        }
        sc["passes"] = std::move(passes);

        // Both copies now take one identical committing pass. It flushes whatever each still held
        // in blk_cache, so both slots become durable at len, and a sequence that mixed commit into
        // its history must now hold exactly the state of one that never did. This is where the
        // deferred differences above have to disappear.
        json sync = json{ { "what", "one identical committing pass on both copies, appended after the "
                                    "trajectory to bring both durable prefixes to len" },
                          { "mode", commit_mode }, { "rows", json{ { "a", 2 }, { "b", 2 } } } };
        const PassOut rs = run_pass(commit_mode, ref_a, ref_b, tokens('a', at_a, 2), tokens('b', at_b, 2));
        const PassOut ms = run_pass(commit_mode, mix_a, mix_b, tokens('a', at_a, 2), tokens('b', at_b, 2));
        const Diff dsl = compare_floats(rs.logits, ms.logits, VOCAB);
        sync["logits"] = dsl.to_json(VOCAB);
        bool sync_ok = dsl.equal();
        json sync_seqs = json::array();
        for (auto & pair : slot_pairs()) {
            const SlotCheck c = compare_slot(pair.first, *pair.second.first, *pair.second.second);
            // After a committing pass on both sides there is nothing left deferred; if this is not
            // asserted the tool has lost track of the contract and should be believed, not patched.
            sync_ok = sync_ok && c.ok && c.asserted;
            sync_seqs.push_back(c.j);
        }
        sync["sequences"] = std::move(sync_seqs);
        sync["ok"] = sync_ok;
        if (!sync_ok) failures++;
        all_equal = all_equal && sync_ok;
        sc["synchronisation"] = std::move(sync);
        sc["equal"] = all_equal;
        std::fprintf(stderr, "  synchronising commit pass  %s\n", sync_ok ? "state and ring equal" : "DIFFERS");
        return sc;
    }

    std::vector<std::pair<const char *, std::pair<Engine::Seq *, Engine::Seq *>>> slot_pairs() {
        return { { "a", { &ref_a, &mix_a } }, { "b", { &ref_b, &mix_b } } };
    }

    // ---- rollback boundary ------------------------------------------------------------------

    // A committed pass must refuse a rejection, and must refuse before it touches the Seq.
    json rollback_rejection(int reference_mode) {
        const int commit_mode = commit_twin(reference_mode);
        reset_all();
        json out{ { "name", "a committed pass refuses rejection without mutating the Seq" },
                  { "commit_mode", commit_mode } };
        run_pass(commit_mode, mix_a, mix_b, tokens('a', 0, 6), tokens('b', 0, 2));

        auto attempt = [&](const char * what, Engine::Seq & s, int keep, bool expect_throw) {
            const Engine::Seq before = s;
            std::string message;
            bool threw = false;
            try { Engine::rollback(s, keep); } catch (const std::exception & ex) { threw = true; message = ex.what(); }
            const bool unchanged = before.len == s.len && before.keep == s.keep && before.last_n == s.last_n &&
                                   before.parity == s.parity && before.slot == s.slot && before.rollbackable == s.rollbackable;
            const bool ok = threw == expect_throw && (!expect_throw || unchanged);
            if (!ok) failures++;
            return json{ { "attempt", what }, { "keep", keep }, { "expected_rejection", expect_throw },
                         { "rejected", threw }, { "message", message },
                         { "seq_before", seq_json(before) }, { "seq_after", seq_json(s) },
                         { "seq_unchanged", unchanged }, { "ok", ok } };
        };

        json checks = json::array();
        checks.push_back(attempt("reject one token of a committed pass", mix_a, mix_a.last_n - 1, true));
        checks.push_back(attempt("reject every token of a committed pass", mix_a, 0, true));
        // Accepting the whole pass is not a rejection; it stays legal and leaves the committed
        // sequence with nothing to replay.
        {
            const int n = mix_a.last_n;
            json j = attempt("accept the whole committed pass", mix_a, n, false);
            j["keep_after"] = mix_a.keep;
            j["ok"] = j["ok"].get<bool>() && mix_a.keep == 0 && mix_a.last_n == n;
            if (!j["ok"].get<bool>()) failures++;
            checks.push_back(std::move(j));
        }
        // An uncommitted pass takes rejections, but still refuses a prefix outside its rows.
        run_pass(reference_mode, mix_a, mix_b, tokens('a', 6, 4), tokens('b', 2, 2));
        checks.push_back(attempt("prefix longer than the pass", mix_a, mix_a.last_n + 1, true));
        checks.push_back(attempt("negative prefix", mix_a, -1, true));
        out["checks"] = std::move(checks);
        out["ok"] = std::all_of(out["checks"].begin(), out["checks"].end(),
                                [](const json & j) { return j["ok"].get<bool>(); });
        return out;
    }

    // An uncommitted pass that follows a committed one must still roll back to exactly the state
    // of a sequence that was only ever fed the accepted tokens.
    json rollback_after_commit(int reference_mode, bool reject_everything) {
        const int commit_mode = commit_twin(reference_mode);
        const int k1 = 6, accept = reject_everything ? 0 : 4, reject = 3, tail = 5;
        reset_all();
        json out{ { "name", reject_everything ? "reject every token of an uncommitted pass after a committed one"
                                              : "reject part of an uncommitted pass after a committed one" },
                  { "commit_mode", commit_mode }, { "reference_mode", reference_mode },
                  { "committed_rows", k1 }, { "accepted", accept }, { "rejected", reject }, { "tail_rows", tail } };

        // The rejected tokens must differ from the accepted ones, or the check proves nothing.
        const std::vector<int> accepted = tokens('a', k1, accept);
        const std::vector<int> rejected = tokens('a', doc.size() / 4, reject);
        const std::vector<int> tail_toks = tokens('a', k1 + accept, tail);
        if (reject && accept && std::equal(rejected.begin(), rejected.end(), accepted.begin(),
                                           accepted.begin() + std::min<size_t>(accepted.size(), rejected.size())))
            throw std::runtime_error("the rejected tokens equal the accepted ones; the rollback check would be vacuous");
        out["rejected_tokens_differ"] = true;

        // The pass that gets rolled back carries sequence A alone. When every token of it is
        // rejected the reference has no pass to match it with, and letting B ride along would
        // leave B a token further on in one copy than the other, which is a divergence this check
        // did not intend to make and would then have reported as a failure.
        const std::vector<int> empty_b;

        // Reference: the committed pass, then only the accepted tokens, then the tail.
        run_pass(commit_mode, ref_a, ref_b, tokens('a', 0, k1), tokens('b', 0, 2));
        if (accept) run_pass(reference_mode, ref_a, ref_b, accepted, empty_b);
        const PassOut r = run_pass(reference_mode, ref_a, ref_b, tail_toks, tokens('b', 2, 1));

        // Mixed: the same committed pass, then the accepted tokens plus rejected ones, rolled back.
        run_pass(commit_mode, mix_a, mix_b, tokens('a', 0, k1), tokens('b', 0, 2));
        std::vector<int> attempted = accepted;
        attempted.insert(attempted.end(), rejected.begin(), rejected.end());
        run_pass(reference_mode, mix_a, mix_b, attempted, empty_b);
        Engine::rollback(mix_a, accept);
        // Rejecting the whole pass leaves parity flipped once more than the reference. With nothing
        // to replay, parity selects the half of blk_cache that nothing reads, so the two copies
        // must still agree; that is exactly what this case measures, so the difference is recorded
        // rather than assumed.
        const bool parity_by_design = !accept;
        if (parity_by_design) out["parity_differs_by_design"] = mix_a.parity != ref_a.parity;
        const PassOut m = run_pass(reference_mode, mix_a, mix_b, tail_toks, tokens('b', 2, 1));

        const Diff dl = compare_floats(r.logits, m.logits, VOCAB);
        out["logits"] = dl.to_json(VOCAB);
        bool ok = dl.equal();
        json seqs = json::array();
        for (auto & pair : slot_pairs()) {
            const bool is_a = pair.first[0] == 'a';
            const SlotCheck c = compare_slot(pair.first, *pair.second.first, *pair.second.second,
                                             parity_by_design && is_a);
            ok = ok && c.ok;
            seqs.push_back(c.j);
        }
        out["sequences"] = std::move(seqs);

        // The tail is still deferred on both copies, so flush it the same way the scenarios do and
        // require the durable images to match once nothing is outstanding.
        const PassOut rs = run_pass(commit_mode, ref_a, ref_b, tokens('a', k1 + accept + tail, 2), tokens('b', 3, 2));
        const PassOut ms = run_pass(commit_mode, mix_a, mix_b, tokens('a', k1 + accept + tail, 2), tokens('b', 3, 2));
        json sync{ { "what", "identical committing pass on both copies, so the tail stops being deferred" },
                   { "mode", commit_mode } };
        const Diff dsl = compare_floats(rs.logits, ms.logits, VOCAB);
        sync["logits"] = dsl.to_json(VOCAB);
        bool sync_ok = dsl.equal();
        json sync_seqs = json::array();
        for (auto & pair : slot_pairs()) {
            const bool is_a = pair.first[0] == 'a';
            const SlotCheck c = compare_slot(pair.first, *pair.second.first, *pair.second.second,
                                             parity_by_design && is_a);
            sync_ok = sync_ok && c.ok && c.asserted;
            sync_seqs.push_back(c.j);
        }
        sync["sequences"] = std::move(sync_seqs);
        sync["ok"] = sync_ok;
        out["synchronisation"] = std::move(sync);

        ok = ok && sync_ok;
        out["ok"] = ok;
        if (!ok) failures++;
        return out;
    }
};

// ------------------------------------------------------------------ main

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(std::stoi(s.substr(i, j - i)));
        i = j + 1;
    }
    return v;
}

static std::vector<std::pair<int, int>> parse_shapes(const std::string & s) {
    std::vector<std::pair<int, int>> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string item = s.substr(i, j - i);
        const size_t colon = item.find(':');
        if (colon == std::string::npos) throw std::runtime_error("--shapes wants A:B pairs, got " + item);
        v.emplace_back(std::stoi(item.substr(0, colon)), std::stoi(item.substr(colon + 1)));
        i = j + 1;
    }
    return v;
}

int main(int argc, char ** argv) {
    Cfg cfg;
    bool shapes_given = false, pattern_given = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") cfg.model = next();
        else if (a == "--doc") cfg.doc = next();
        else if (a == "--out") cfg.out = next();
        else if (a == "--context") cfg.context = std::stoi(next());
        else if (a == "--threads") cfg.nthreads = std::stoi(next());
        else if (a == "--references") cfg.references = parse_ints(next());
        else if (a == "--shapes") { cfg.shapes = parse_shapes(next()); shapes_given = true; }
        else if (a == "--commit-pattern") {
            const std::string p = next();
            cfg.commit.assign(p.size(), false);
            for (size_t k = 0; k < p.size(); k++) cfg.commit[k] = p[k] == '1';
            pattern_given = true;
        }
        else if (a == "--no-state") cfg.compare_state = false;
        else if (a == "--no-rollback") cfg.run_rollback = false;
        else if (a == "-h" || a == "--help") {
            std::printf("sequence_state_check [--model GGUF] [--doc FILE] [--out JSON] [--context N]\n"
                        "                     [--references 10,17] [--shapes A:B,A:B,...] [--commit-pattern 0110101]\n"
                        "                     [--threads N] [--no-state] [--no-rollback]\n\n"
                        "Compares a sequence that switches between a replaying mode and its committing twin\n"
                        "against the same sequence that never switches, on logits, GDN state and conv ring.\n");
            return 0;
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (shapes_given && !pattern_given) {
        cfg.commit.assign(cfg.shapes.size(), false);
        for (size_t k = 1; k < cfg.commit.size(); k += 2) cfg.commit[k] = true;
    }
    for (int r : cfg.references)
        if (commit_twin(r) < 0) { std::fprintf(stderr, "no committing twin for mode %d; use 10 or 17\n", r); return 2; }

    try {
        if(!getenv("BONSAI_BENCH_SERVER_MASKED"))
            throw std::runtime_error("use tools/run-batch-compare --state-check");
        unsigned mask = 0;
        for (int r : cfg.references) {
            for (int m : { r, commit_twin(r) }) {
                const int img = mode_image(m);
                if (img >= 0) mask |= ffn_batch_mode_bit(img);
            }
        }
        const bool wants_wide = std::any_of(cfg.references.begin(), cfg.references.end(),
                                            [](int r) { return r >= 17 || commit_twin(r) >= 17; });

        std::fprintf(stderr, "loading %s\n", cfg.model.c_str());
        Engine e;
        e.load(cfg.model, cfg.nthreads, 4, cfg.context);   // four slots: the reference pair and the mixed pair
        e.prepare_batch(128, mask ? mask : Engine::BATCH_WORKSPACE_ONLY);
        if (wants_wide) e.prepare_sequence();

        Tokenizer tk;
        tk.load(cfg.model);
        Checker ck(e, cfg);
        ck.doc = tk.encode(read_file(cfg.doc), false);

        int need_a = 0, need_b = 0;
        for (auto & s : cfg.shapes) { need_a += s.first; need_b += s.second; }
        need_a = std::max(need_a, 32);   // the rollback scenarios feed their own short passes
        if ((int) ck.doc.size() / 2 < std::max(need_a, need_b) + 16)   // the synchronising passes and the rollback checks feed a few more
            throw std::runtime_error(cfg.doc + " tokenizes to " + std::to_string(ck.doc.size()) +
                                     " tokens, too few for the pass plan");

        json run;
        run["tool"] = "sequence_state_check";
        run["gdn_resident"] = sequence_resident_enabled(e.batch_sequence);
        run["gdn_state_split"] = getenv("HALO_GDN_SPLIT")?getenv("HALO_GDN_SPLIT"):"4";
        run["sequence_layout"] = sequence_direct_layout(e.batch_sequence)?"direct":"staged";
        run["sequence_operand"] = getenv("HALO_SEQUENCE_OPERAND")?getenv("HALO_SEQUENCE_OPERAND"):"int8";
        run["resident_server_masked"] = true;
        auto command=[](const char *text) {
            FILE *p=popen(text,"r");if(!p)throw std::runtime_error("cannot collect state-check provenance");
            std::string value;char buf[4096];while(fgets(buf,sizeof(buf),p))value+=buf;
            if(pclose(p))throw std::runtime_error("state-check provenance command failed");
            return value;
        };
        run["git_revision"]=command("git rev-parse HEAD");
        run["source_sha256_rollup"]=command("git ls-files --cached --others --exclude-standard -z -- src kernels Makefile tools/sequence_state_check.cpp tools/direct-commit/Makefile | xargs -0 sha256sum | sha256sum");
        run["executable_sha256"]=command("sha256sum tools/direct-commit/sequence_state_check");
        {
            char buf[64]; const time_t t = time(nullptr);
            strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", localtime(&t));
            run["generated"] = buf;
        }
        json shapes = json::array();
        for (size_t i = 0; i < cfg.shapes.size(); i++)
            shapes.push_back(json{ { "a", cfg.shapes[i].first }, { "b", cfg.shapes[i].second },
                                   { "total", cfg.shapes[i].first + cfg.shapes[i].second },
                                   { "commits", i < cfg.commit.size() ? cfg.commit[i] : (i % 2 == 1) } });
        run["config"] = json{ { "model", cfg.model }, { "document", cfg.doc }, { "context", cfg.context },
                              { "references", cfg.references }, { "passes", shapes },
                              { "gdn_state_compared", cfg.compare_state },
                              { "document_tokens", ck.doc.size() } };
        run["scenarios"] = json::array();
        for (int r : cfg.references) {
            std::fprintf(stderr, "scenario: %d with %d mixed in\n", r, commit_twin(r));
            run["scenarios"].push_back(ck.scenario(r));
        }
        if (cfg.run_rollback) {
            const int r = cfg.references.front();
            std::fprintf(stderr, "rollback boundary on mode %d\n", r);
            run["rollback"] = json::array();
            run["rollback"].push_back(ck.rollback_rejection(r));
            run["rollback"].push_back(ck.rollback_after_commit(r, false));
            run["rollback"].push_back(ck.rollback_after_commit(r, true));
        }
        run["failures"] = ck.failures;
        run["deferred_state_differences_reported"] = ck.deferred_observed;
        run["how_to_read"] =
            "Logits and argmax are required equal after every pass. Raw gdn_state and conv_ring are "
            "compared after every pass for every slot, but required equal only where both copies are "
            "durable at the same prefix (len - keep); straight after a committing pass the mixed copy "
            "is ahead by that pass's rows and a difference there is the contract working. Each "
            "scenario ends with one identical committing pass on both copies, and the raw state and "
            "ring must match there.";
        run["verdict"] = ck.failures ? "differences found; see the failing entries"
                                     : "mixing commit into a sequence changed neither its output at any "
                                       "pass nor its state once both copies had committed";

        const fs::path out = cfg.out;
        if (out.has_parent_path()) fs::create_directories(out.parent_path());
        std::ofstream(out) << run.dump(1) << "\n";
        std::fprintf(stderr, "\n%s\n%s\n", run["verdict"].get<std::string>().c_str(), out.c_str());
        return ck.failures ? 1 : 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "sequence_state_check failed: %s\n", ex.what());
        return 2;
    }
}
