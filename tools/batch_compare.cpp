// Full-model batch comparison driver.
//
// Runs the real engine (all 64 layers, real weights, real tokenizer) over three workloads and
// compares every batch execution mode against the deployed path:
//
//   mode 0  deployed        (forward_batch slices the pass into the original 8-row forward)
//   mode 1  integer A8
//   mode 2  integer A4
//   mode 3  scaled A8
//   mode 4  sliced deployed control: the layer-wise batched schedule with the original FFN, which
//           separates a scheduling or replay bug from an FFN numerics bug
//   mode 5  auto: the engine picks per pass by row count (<=4 deployed, 5..31 sliced, >=32 A8)
//   mode 6  optimized IU8/A8      the newly adopted FFN module, direct
//   mode 7  optimized scaled A8   likewise
//   mode 8  optimized A4          likewise
//   mode 9  auto -> mode 6 at >=32 rows (<=4 deployed, 5..31 sliced)
//   mode 10 auto -> mode 7 at >=32 rows
//   mode 11 auto -> mode 8 at >=32 rows
//   mode 12 single-map: the original per-pass scheduling with the optimized kernel at one token,
//           the single-token path, whose matched reference is mode 0
//
// Modes 6, 7 and 8 are intended to be bit-identical to 1, 3 and 2 respectively. This driver treats
// that as a question, not a premise: it dumps both sides and tools/batch_compare.py counts the
// elements that actually differ, for whichever pairs the operator names with --reference-pairs.
// Nothing here assumes a pairing on its own, because at fewer than 32 rows an automatic mode runs
// the deployed or sliced path and a family pairing would compare the wrong thing.
//
//   prefill    one real document, 32 and 128 rows per pass, one sequence
//   decode     8 and 32 independent streams with distinct real prompts, one row per stream per
//              step, fixed greedy steps, aggregate tokens/s
//   quality    full-vocabulary logits on fixed teacher-forced tokens, prefill-shaped (rows from one
//              sequence) and decode-shaped (one row per stream); dumped for tools/batch_compare.py
//   multistep  32 sequences carried over several greedy steps, token matrix compared against
//              mode 0: catches state, replay and parity errors a single pass cannot show
//
// Nothing here times an FFN surrogate: every measured region is Engine::forward_batch on the loaded
// model. Mode order is reshuffled every round inside one process, sequence state is reset before
// every timed region, and every individual wall time is kept in the JSON result so the analysis can
// show spread instead of a single averaged number.
//
// Build: tools/batch-compare/build.sh   (see tools/batch-compare/README.md)

#include "engine.h"
#include "ffn_batch.h"
#include "sequence_batch.h"
#include "head_batch.h"
#include "prep_batch.h"
#include "tokenizer.h"
#include "halo_env.hpp"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <vector>

using namespace halo;
using json = nlohmann::json;
namespace fs = std::filesystem;

static std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) out += c == '\'' ? "'\\''" : std::string(1, c);
    return out + "'";
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- telemetry

// amdgpu clocks/temperature/power/busy straight from sysfs, sampled by a host thread. -1 means the
// file was unreadable. These are host snapshots, not per-kernel counters, and the GPU is not proven
// idle for other work; the analysis prints them as context, never as a measurement of the kernel.
struct Telemetry {
    struct Sample { double t; long long gfx_hz = -1, temp_mc = -1, power_uw = -1, busy_pct = -1; };
    fs::path device, hwmon;
    int interval_ms = 5;
    std::atomic<bool> stop{false};
    std::thread worker;
    std::vector<Sample> samples;

    static long long number(const fs::path & p) { std::ifstream f(p); long long v = -1; f >> v; return v; }
    static std::string text(const fs::path & p) { std::ifstream f(p); std::ostringstream s; s << f.rdbuf(); return s.str(); }

    // first DRM card whose hwmon reports the amdgpu driver
    bool discover() {
        std::error_code ec;
        for (const auto & card : fs::directory_iterator("/sys/class/drm", ec)) {
            const std::string n = card.path().filename().string();
            if (n.rfind("card", 0) != 0 || n.find('-') != std::string::npos) continue;
            fs::path dev = card.path() / "device";
            std::error_code e2;
            for (const auto & h : fs::directory_iterator(dev / "hwmon", e2)) {
                if (text(h.path() / "name").find("amdgpu") != std::string::npos) { device = dev; hwmon = h.path(); return true; }
            }
        }
        return false;
    }
    void start() {
        if (interval_ms <= 0 || hwmon.empty()) return;
        stop.store(false); samples.clear();
        worker = std::thread([this] {
            while (!stop.load()) {
                Sample s{}; s.t = now();
                s.gfx_hz = number(hwmon / "freq1_input");
                s.temp_mc = number(hwmon / "temp1_input");
                s.power_uw = number(hwmon / "power1_input");
                s.busy_pct = number(device / "gpu_busy_percent");
                samples.push_back(s);
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            }
        });
    }
    void stop_now() { stop.store(true); if (worker.joinable()) worker.join(); }
    // Without this the sampler thread is still joinable when an exception leaves a run, and
    // `~thread` calls `std::terminate` while the stack unwinds - so the panel aborts with a core
    // dump and the engine's own message, which is the thing worth reading, is never printed.
    ~Telemetry() { try { stop_now(); } catch (...) {} }
    json summary(bool raw) {
        stop_now();
        json j;
        j["device_sysfs"] = device.string();
        j["samples_taken"] = (int) samples.size();
        if (samples.empty()) { j["available"] = false; return j; }
        j["available"] = true;
        auto stat = [&](long long Sample::*f, const char * name) {
            long long lo = -1, hi = -1; double sum = 0; int n = 0;
            for (auto & s : samples) { long long v = s.*f; if (v < 0) continue; if (n == 0 || v < lo) lo = v; if (n == 0 || v > hi) hi = v; sum += (double) v; n++; }
            if (!n) { j[name] = nullptr; return; }
            j[name] = json{ { "min", lo }, { "max", hi }, { "mean", sum / n }, { "n", n } };
        };
        stat(&Sample::gfx_hz, "gfx_hz");
        stat(&Sample::temp_mc, "temperature_mc");
        stat(&Sample::power_uw, "power_uw");
        stat(&Sample::busy_pct, "busy_percent");
        if (raw) { json arr = json::array(); for (auto & s : samples) arr.push_back(json{ { "t", s.t }, { "gfx_hz", s.gfx_hz }, { "temperature_mc", s.temp_mc }, { "power_uw", s.power_uw }, { "busy_percent", s.busy_pct } }); j["raw"] = arr; }
        return j;
    }
};

// ---------------------------------------------------------------- host diagnostics

// A timed region can stall on the host while the GPU sits idle. Clocks and busy percent show that
// symptom and cannot name the cause: an idle, cooling GPU looks the same whether the process was
// waiting on memory reclaim, faulting pages back from disk, preempted by another job, or blocked on
// I/O. These counters separate those. PSI says how long anything on the machine was stalled on each
// resource, rusage says whether this process was running, preempted or faulting, and MemAvailable
// says whether the machine was short of memory while it happened.
struct HostDiag {
    struct Snap {
        bool ok = false;
        double utime = 0, stime = 0;
        long majflt = 0, minflt = 0, nvcsw = 0, nivcsw = 0;
        long long psi[3][2];               // [cpu, memory, io] x [some, full], microseconds, -1 unknown
        long long mem_available_kb = -1, swap_free_kb = -1, swap_total_kb = -1;
        Snap() { for (auto & r : psi) r[0] = r[1] = -1; }
    };

    // "some" is any task stalled, "full" is every runnable task stalled. /proc/pressure/cpu has no
    // full line on most kernels, and the whole interface is absent without CONFIG_PSI.
    static void read_psi(const char * path, long long out[2]) {
        out[0] = out[1] = -1;
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            const int idx = line.rfind("some", 0) == 0 ? 0 : line.rfind("full", 0) == 0 ? 1 : -1;
            const size_t p = line.find("total=");
            if (idx >= 0 && p != std::string::npos) out[idx] = atoll(line.c_str() + p + 6);
        }
    }

    static long long meminfo_kb(const std::string & text, const char * key) {
        const size_t p = text.find(key);
        if (p == std::string::npos) return -1;
        const size_t c = text.find(':', p);
        return c == std::string::npos ? -1 : atoll(text.c_str() + c + 1);
    }

    static Snap take() {
        Snap s;
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            s.ok = true;
            s.utime = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6;
            s.stime = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
            s.majflt = ru.ru_majflt; s.minflt = ru.ru_minflt;
            s.nvcsw = ru.ru_nvcsw;   s.nivcsw = ru.ru_nivcsw;
        }
        read_psi("/proc/pressure/cpu", s.psi[0]);
        read_psi("/proc/pressure/memory", s.psi[1]);
        read_psi("/proc/pressure/io", s.psi[2]);
        const std::string mi = Telemetry::text("/proc/meminfo");
        s.mem_available_kb = meminfo_kb(mi, "MemAvailable");
        s.swap_free_kb = meminfo_kb(mi, "SwapFree");
        s.swap_total_kb = meminfo_kb(mi, "SwapTotal");
        return s;
    }

    static json delta(const Snap & a, const Snap & b, double wall) {
        json j;
        j["available"] = a.ok && b.ok;
        if (!a.ok || !b.ok) return j;
        const double cpu = (b.utime - a.utime) + (b.stime - a.stime);
        j["cpu_user_s"] = b.utime - a.utime;
        j["cpu_system_s"] = b.stime - a.stime;
        j["cpu_total_s"] = cpu;
        // One busy thread over the region is 1.0. Far below that on a long region means the process
        // was blocked rather than working, which is the first thing to know about a stall.
        j["cpu_busy_fraction_of_wall"] = wall > 0 ? cpu / wall : 0.0;
        j["major_faults"] = (long long) (b.majflt - a.majflt);
        j["minor_faults"] = (long long) (b.minflt - a.minflt);
        j["voluntary_context_switches"] = (long long) (b.nvcsw - a.nvcsw);
        j["involuntary_context_switches"] = (long long) (b.nivcsw - a.nivcsw);
        static const char * res[3] = { "cpu", "memory", "io" };
        static const char * kind[2] = { "some", "full" };
        json psi = json::object();
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < 2; k++)
                psi[std::string(res[r]) + "_" + kind[k] + "_stall_s"] =
                    (a.psi[r][k] < 0 || b.psi[r][k] < 0) ? json(nullptr)
                                                         : json((b.psi[r][k] - a.psi[r][k]) / 1e6);
        j["psi"] = psi;
        auto pair = [](long long x, long long y) {
            return x < 0 || y < 0 ? json(nullptr) : json{ { "before", x }, { "after", y }, { "delta", y - x } };
        };
        j["mem_available_kb"] = pair(a.mem_available_kb, b.mem_available_kb);
        j["swap_free_kb"] = pair(a.swap_free_kb, b.swap_free_kb);
        j["swap_total_kb"] = b.swap_total_kb < 0 ? json(nullptr) : json(b.swap_total_kb);
        return j;
    }
};

// Optional external telemetry (e.g. a Kelana probe): run a command, capture stdout verbatim.
static std::string run_capture(const std::string & cmd) {
    if (cmd.empty()) return "";
    std::string out; FILE * p = popen(cmd.c_str(), "r");
    if (!p) return "";
    char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
    pclose(p);
    return out;
}

// ---------------------------------------------------------------- timed region

// Wall clock plus, optionally, HIP event time on the engine stream. The stream is synchronised
// before the clock stops, so a pass that never copies argmax back is still fully accounted.
struct Region {
    Engine & e; bool use_events;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    double t0 = 0, wall = 0; float event_ms = -1;
    HostDiag::Snap h0, h1;
    Region(Engine & en, bool events) : e(en), use_events(events) {
        if (use_events) { HIP_CHECK_H(hipEventCreate(&ev0)); HIP_CHECK_H(hipEventCreate(&ev1)); }
    }
    void begin() {
        h0 = HostDiag::take();                 // before the clock: these reads are not measured work
        if (use_events) HIP_CHECK_H(hipEventRecord(ev0, e.stream));
        t0 = now();
    }
    void end() {
        if (use_events) HIP_CHECK_H(hipEventRecord(ev1, e.stream));
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
        wall = now() - t0;
        h1 = HostDiag::take();
        if (use_events) { HIP_CHECK_H(hipEventSynchronize(ev1)); HIP_CHECK_H(hipEventElapsedTime(&event_ms, ev0, ev1)); }
    }
    json host() const { return HostDiag::delta(h0, h1, wall); }
    ~Region() { if (ev0) (void) hipEventDestroy(ev0); if (ev1) (void) hipEventDestroy(ev1); }
};

// ---------------------------------------------------------------- inputs

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream s; s << f.rdbuf(); return s.str();
}

static std::vector<std::string> read_lines(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<std::string> out; std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty() && line[0] != '#') out.push_back(line);
    }
    return out;
}

static std::vector<int> parse_ints(const std::string & csv) {
    std::vector<int> out; std::string cur;
    std::istringstream s(csv);
    while (std::getline(s, cur, ',')) if (!cur.empty()) out.push_back(atoi(cur.c_str()));
    return out;
}

// ---------------------------------------------------------------- workloads

enum PrefillLogits { PL_NONE = 0, PL_LAST = 1, PL_ALL = 2 };

struct Cfg {
    std::string model = "../../data/bonsai2/PTQ1_0.gguf";
    std::string doc = "PLAN.md";
    std::string prompts = "tools/batch-compare/prompts.txt";
    std::string out_dir = "../../data/bonsai2/batch-comparison";
    std::string tag;
    std::string telemetry_cmd;
    std::string reference_pairs;               // recorded for the analysis; never interpreted here
    unsigned a4_images = 0;                    // FFN_BATCH_A4_WIDE_ONLY / _DENSE_ONLY, 0 keeps both
    std::vector<int> modes{ 0, 4, 1, 2, 3 };
    // A4 row-tile ownerships to measure, interleaved with the modes inside one process. Empty
    // means "whatever the build selects", one entry pins it, several compare them without the
    // clock drift a pair of processes carries. -1 is the configured default.
    std::vector<int> ffn_scheds{ -1 };
    std::vector<int> gdn_pregates{ -1 };
    std::vector<int> gdn_splits{ -1 };   // state-unit width: rows per head group, docs/gdn-unit-width.md
    // State-unit lane ownerships, interleaved the same way: -1 leaves whatever the build selects,
    // 0 pins the deployed rows-per-lane units and 1 pins one lane per 32 columns of one row. Both
    // arms produce the same bits; docs/gdn-lane-layout.md.
    std::vector<int> gdn_colss{ -1 };
    // A4 gate/up operand liveness orders to measure, interleaved the same way. -1 is the
    // configured default, 0 the slice-outer order, 1, 2 and 4 token tiles per accumulator group.
    std::vector<int> ffn_orders{ -1 };
    // Sequence-projection row-tile ownerships, the same idea on the other large phase: -1 leaves
    // whatever the build selects, 0 keeps a wave on its own row tile, 1 and 2 give one row tile to
    // two or four waves of a workgroup.
    std::vector<int> seq_scheds{ -1 };
    // The stored order of the sequence weight image's (row tile, block) runs: -1 leaves the build's
    // choice, 0 tile-major, 1 block-major. Permuted in place between arms; bit-identical axis.
    std::vector<int> seq_images{ -1 };
    // A4 weight images to measure, per stage, interleaved the same way. -1 leaves the build's
    // choice; 1 dense, 2 pair, 3 dense gate/up + pair down, 4 the reverse. Bit-identical axis.
    std::vector<int> ffn_images{ -1 };
    std::vector<int> ffn_runs{ -1 };      // stored weight-image run order; see ffn_batch_set_run_order
    // How many of the down projection's two weight matrices one wave owns on the dense A4 arm:
    // -1 leaves the build's rule, 1 splits them across two waves, 2 keeps the pair. Bit-identical.
    std::vector<int> ffn_dn_mats{ -1 };
    // The recurrent state's storage coordinate, walked like a mode: every workload here resets and
    // re-prefills its sequences, so each arm writes and reads its own coordinate.
    std::vector<int> gdn_states{ -1 };
    // How many generation steps share one state write-back. Only a packed coordinate has room for
    // the pending updates, so an arm's effective depth is reported beside the requested one.
    std::vector<int> gdn_defers{ -1 };
    // Activation precision of the sequence INPUT projection: -1 leaves the build's choice, 0 the
    // deployed eight-bit operand, 1 the four-bit map on the eight-bit instruction, 2 the four-bit
    // instruction. 1 and 2 must agree bit for bit; 0 against either is a numerical change with its
    // own quality evidence. Only modes that take the wide sequence route read this.
    std::vector<int> seq_quants{ -1 };
    std::vector<int> prefill_rows{ 32, 128 };
    std::vector<int> streams{ 8, 32 };
    std::vector<int> quality_rows{ 8, 32 };
    std::vector<std::string> quality_shapes{ "prefill", "decode" };
    int context = 512;
    int slots = 32;
    int prefill_tokens = 384;
    int rounds = 3;
    int gen_steps = 8;
    int multistep_streams = 32;
    int multistep_steps = 4;
    int quality_ctx = 64;      // teacher-forced context fed before the logits pass
    // Horizon: the same teacher-forced question asked over a whole generated-length sequence.
    // A packed recurrent state rounds once per token per layer in the generation shape and once
    // per *pass* in prefill, so a 64-row teacher-forced window carries at most 64 roundings and
    // cannot see rounding that accumulates. These run `horizon_tokens` single-token steps per
    // stream and score the distribution every `horizon_every` of them.
    int horizon_tokens = 512;
    int horizon_ctx = 128;
    int horizon_every = 8;
    int horizon_streams = 32;
    std::string horizon_doc;   // empty keeps --doc; the horizon needs far more than --prefill-tokens
    // State dump: the fp32 recurrent state itself, for the format search that has to happen before
    // a coordinate is proposed. Layers are gdn indices 0..47.
    std::vector<int> state_dump_layers{ 0, 12, 24, 36, 47 };
    int state_dump_ctx = 384;
    int warmup_steps = 3;
    unsigned seed = 1234;
    int nthreads = (int) std::thread::hardware_concurrency();
    int telemetry_ms = 5;
    bool telemetry_raw = false;
    bool hip_events = false;
    bool dump_logits = true;
    PrefillLogits prefill_logits = PL_LAST;
    bool do_prefill = true, do_decode = true, do_quality = true, do_multistep = true;
    bool do_horizon = false, do_state_dump = false;
    // Hash every logit of the document at each --prefill-rows width and compare the widths.
    bool prefill_identity = false;
    // The head's row selection, walked as a case axis: 1 is the tail row alone, which is what
    // prompt ingestion reads, and 0 is every row of the pass, which is what the head did before the
    // selection existed and what every published prefill number before it measured. Two processes
    // cannot resolve this - prefill is clock-elastic and a decode control is not, so neither arm
    // normalises the other - and `both` interleaves them inside one warmed process.
    std::vector<int> head_rows{ 1 };
};

static const int MODE_MAX = 20;

static const char * mode_name(int m) {
    switch (m) {
        case 0: return "deployed";
        case 1: return "integer-a8";
        case 2: return "integer-a4";
        case 3: return "scaled-a8";
        case 4: return "sliced-deployed";
        case 5: return "auto-a8";
        case 6: return "optimized-iu8-a8";
        case 7: return "optimized-scaled-a8";
        case 8: return "optimized-a4";
        case 9: return "auto-optimized-a8";
        case 10: return "auto-optimized-scaled-a8";
        case 11: return "auto-optimized-a4";
        case 12: return "single-map";
        case 13: return "single-state";
        case 14: return "single-retile";
        case 15: return "single-grid";
        case 16: return "commit-scaled";
        case 17: return "wide-sequence";
        case 18: return "wide-commit";
        case 19: return "wide-commit-a4";
        case 20: return "wide-deployed";
        default: return "unknown";
    }
}

// The engine's own --ffn spelling for the same mode, so a report can be turned back into a command.
static const char * mode_engine_flag(int m) {
    switch (m) {
        case 0: return "deployed";
        case 1: return "a8";
        case 2: return "a4";
        case 3: return "scaled-a8";
        case 4: return "sliced";
        case 5: return "auto";
        case 6: return "map-a8";
        case 7: return "map-scaled-a8";
        case 8: return "map-a4";
        case 9: return "auto-map-a8";
        case 10: return "auto-map-scaled-a8";
        case 11: return "auto-map-a4";
        case 12: return "single-map";
        case 13: return "single-state";
        case 14: return "single-retile";
        case 15: return "single-grid";
        case 16: return "commit-scaled";
        case 17: return "wide-sequence";
        case 18: return "wide-commit";
        case 19: return "wide-commit-a4";
        case 20: return "wide-deployed";
        default: return "?";
    }
}

// Row-count dispatch for the automatic modes, mirroring src/batch.cpp. Recorded in run.json so the
// analysis can say which reference a pairing is only valid against, and at which row counts.
// -1 means the mode is not automatic: it runs the same path at every row count.
static int auto_wide_mode(int m) {
    switch (m) {
        case 5:  return 1;
        case 9:  return 6;
        case 10: return 7;
        case 11: return 8;
        case 16: case 17: case 18: return 7;
        case 19: return 8;
        case 20: return 4;   // wide schedule, deployed FFN: no weight image, no new numerical map
        default: return -1;
    }
}

// Which module mode a measured mode actually enters, or -1 for the paths that never reach the
// batched FFN module. Modes 0 and 4 run the engine's own FFN, and mode 12 keeps the original
// scheduling with an optimized kernel inside it, so none of them needs a repacked weight image. An
// automatic mode needs whatever it dispatches to at 32 rows and above, because that is the only
// place it leaves the deployed path.
static int mode_module_image(int m) {
    const int wide = auto_wide_mode(m);
    const int needs = wide < 0 ? m : wide;
    return needs == 0 || needs == 4 || needs >= 12 ? -1 : needs;
}

// Contract with Engine::prepare_batch: build the shared workspace and no weight image at all. A zero
// mask still means every mode, which for a run of modes 0 and 12 would repack 16.58 GB of
// representations that nothing in the run reads.
static constexpr unsigned BATCH_WORKSPACE_ONLY = 1u << 30;

// Bit per module mode, for prepare_batch's image selection. Zero means no measured mode enters the
// module; the engine reads that as FFN_BATCH_ALL, so a run of modes 0 and 4 alone would pay for
// every weight image. prepare_batch is still called, because mode 4 needs the shared workspace.
static unsigned modes_mask(const std::vector<int> & modes) {
    unsigned mask = 0;
    for (int m : modes) {
        const int img = mode_module_image(m);
        if (img >= 0) mask |= ffn_batch_mode_bit(img);
    }
    return mask;
}

static const char * image_name(int image) {
    switch (image) {
        case FFN_IMAGE_SCALES: return "SCALES";
        case FFN_IMAGE_SLICE2: return "SLICE2";
        case FFN_IMAGE_LANE2:  return "LANE2";
        case FFN_IMAGE_PAIR:   return "PAIR";
        case FFN_IMAGE_DENSE5: return "DENSE5";
        default:               return "?";
    }
}

struct Driver {
    Engine & e; Tokenizer & tk; const Cfg & cfg;
    std::vector<int> doc_tokens;                 // real document, raw tokenization
    std::vector<int> doc_full;                   // the same document before --prefill-tokens truncates it
    std::vector<int> horizon_doc;                // the long document the horizon workload walks
    std::vector<std::vector<int>> prompt_tokens; // distinct chat-formatted prompts
    std::vector<Engine::Seq> seqs;

    Driver(Engine & en, Tokenizer & t, const Cfg & c) : e(en), tk(t), cfg(c) {}

    // `required` slots must be servable or the run is wrong; `desired` is how many distinct prompts
    // the widest optional workload would like. Quality groups that exceed what the prompts file or
    // the slot count can serve are skipped and recorded, rather than forcing a huge slot capacity.
    void prepare_inputs(int required, int desired) {
        doc_tokens = tk.encode(read_file(cfg.doc), false);
        doc_full = doc_tokens;
        horizon_doc = cfg.horizon_doc.empty() ? doc_full : tk.encode(read_file(cfg.horizon_doc), false);
        if ((int) doc_tokens.size() < cfg.prefill_tokens)
            throw std::runtime_error("document " + cfg.doc + " tokenizes to " + std::to_string(doc_tokens.size()) +
                                     " tokens, fewer than --prefill-tokens " + std::to_string(cfg.prefill_tokens));
        doc_tokens.resize(cfg.prefill_tokens);

        std::vector<std::string> lines = read_lines(cfg.prompts);
        if ((int) lines.size() < required)
            throw std::runtime_error("prompts file has " + std::to_string(lines.size()) + " usable lines, fewer than the " +
                                     std::to_string(required) + " distinct prompts the decode and multistep streams need");
        // distinct prompts only: repeating one prompt across rows would not be an independent workload
        const int take = std::min<int>(std::max({ required, desired, 1 }), (int) lines.size());
        std::vector<std::string> uniq(lines.begin(), lines.begin() + take);
        std::sort(lines.begin(), lines.end());
        if (std::adjacent_find(lines.begin(), lines.end()) != lines.end())
            throw std::runtime_error("prompts file contains duplicate lines");
        for (auto & l : uniq) prompt_tokens.push_back(tk.encode(Tokenizer::chat_prompt(l, false), true));

        seqs.assign(std::min<int>(std::max(take, 1), cfg.slots), Engine::Seq{});
        for (int i = 0; i < (int) seqs.size(); i++) seqs[i].slot = i;
    }

    // rows a decode-shaped workload can actually be given: one slot and one distinct prompt each
    int decode_row_limit() const { return std::min<int>(prompt_tokens.size(), seqs.size()); }

    // ---- prefill: one sequence, one real document, rows_per_pass rows per pass
    json prefill(int mode, int rows_per_pass, Telemetry & tel) {
        e.batch_mode = mode;
        Engine::Seq & s = seqs[0];
        e.reset_seq(s);
        HIP_CHECK_H(hipStreamSynchronize(e.stream));

        const int n = (int) doc_tokens.size();
        int logit_passes = 0;
        std::vector<int> am;
        std::vector<double> pass_wall;
        Region r(e, cfg.hip_events);
        tel.start();
        r.begin();
        for (int i = 0; i < n; i += rows_per_pass) {
            const int k = std::min(rows_per_pass, n - i);
            const bool last = i + k == n;
            const bool logits = cfg.prefill_logits == PL_ALL || (cfg.prefill_logits == PL_LAST && last);
            std::vector<Engine::Seq *> sv{ &s };
            std::vector<std::vector<int>> tv{ std::vector<int>(doc_tokens.begin() + i, doc_tokens.begin() + i + k) };
            double p0 = now();
            e.forward_batch(sv, tv, logits, logits ? &am : nullptr, nullptr, Engine::LogitRows::Tail);
            pass_wall.push_back(now() - p0);
            if (logits) logit_passes++;
        }
        r.end();
        json j;
        // The tail row of the last pass, read straight out of the head's own buffer. It is the row
        // ingestion consumes, and it is the row a selection has to reproduce exactly: the same
        // 248,320 logits under `--head-rows tail` and under `--head-rows all`. Untimed, and the
        // hash travels between processes where a timing does not.
        if (logit_passes && head_batch_capacity(e.batch_head) > 0) {
            const int last_rows = n - (n - 1) / rows_per_pass * rows_per_pass;
            if (last_rows > RMAX && last_rows <= head_batch_capacity(e.batch_head)) {
                std::vector<float> tail(VOCAB);
                HIP_CHECK_H(hipMemcpy(tail.data(), head_logits(e.batch_head) + (size_t) (last_rows - 1) * VOCAB,
                    (size_t) VOCAB * sizeof(float), hipMemcpyDeviceToHost));
                uint64_t h = 14695981039346656037ull;
                const auto * b = reinterpret_cast<const unsigned char *>(tail.data());
                for (size_t i = 0; i < tail.size() * sizeof(float); ++i) h = (h ^ b[i]) * 1099511628211ull;
                j["tail_logits_fnv64"] = std::to_string(h);
                j["tail_logits_rows"] = last_rows;
            }
        }
        j["case"] = "prefill";
        j["mode"] = mode; j["mode_name"] = mode_name(mode);
        j["rows_per_pass"] = rows_per_pass;
        j["tokens"] = n;
        j["passes"] = (int) pass_wall.size();
        j["logits_passes"] = logit_passes;
        j["logits_policy"] = cfg.prefill_logits == PL_ALL ? "all-passes" : cfg.prefill_logits == PL_LAST ? "last-pass-only" : "none";
        j["lm_head_cost_included"] = logit_passes > 0;
        j["wall_s"] = r.wall;
        j["tokens_per_s"] = n / r.wall;
        j["pass_wall_s"] = pass_wall;               // every individual pass, not just the total
        if (cfg.hip_events) j["hip_event_ms"] = r.event_ms;
        j["last_argmax"] = am.empty() ? -1 : am.back();
        j["telemetry"] = tel.summary(cfg.telemetry_raw);
        j["host"] = r.host();
        return j;
    }

    // Feed a sequence's prompt with no logits, return the argmax of the final row.
    int prefill_stream(Engine::Seq & s, const std::vector<int> & toks, int rows_per_pass) {
        std::vector<int> am;
        for (size_t i = 0; i < toks.size(); i += rows_per_pass) {
            const int k = (int) std::min((size_t) rows_per_pass, toks.size() - i);
            const bool last = i + k == toks.size();
            std::vector<Engine::Seq *> sv{ &s };
            std::vector<std::vector<int>> tv{ std::vector<int>(toks.begin() + i, toks.begin() + i + k) };
            e.forward_batch(sv, tv, last, last ? &am : nullptr, nullptr, Engine::LogitRows::Tail);
        }
        return am.empty() ? -1 : am.back();
    }

    // ---- decode: nstream independent sequences, one row each per step, fixed greedy steps
    json decode(int mode, int nstream, Telemetry & tel, bool capture_text) {
        if (nstream > e.nslots) throw std::runtime_error("streams exceed engine slots");
        e.batch_mode = mode;

        // state setup and prompt prefill sit outside the timed region
        double setup0 = now();
        std::vector<int> next(nstream, -1);
        long prompt_toks = 0;
        for (int b = 0; b < nstream; b++) {
            e.reset_seq(seqs[b]);
            next[b] = prefill_stream(seqs[b], prompt_tokens[b], std::min(128, (int) prompt_tokens[b].size()));
            prompt_toks += (long) prompt_tokens[b].size();
        }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
        double setup = now() - setup0;

        std::vector<std::vector<int>> out(nstream);
        std::vector<double> step_wall;
        int eos_hits = 0;
        std::vector<Engine::Seq *> sv(nstream);
        for (int b = 0; b < nstream; b++) sv[b] = &seqs[b];

        Region r(e, cfg.hip_events);
        tel.start();
        r.begin();
        for (int step = 0; step < cfg.gen_steps; step++) {
            std::vector<std::vector<int>> tv(nstream);
            for (int b = 0; b < nstream; b++) { tv[b] = { next[b] }; out[b].push_back(next[b]); }
            std::vector<int> am;
            double s0 = now();
            e.forward_batch(sv, tv, true, &am, nullptr);
            step_wall.push_back(now() - s0);
            for (int b = 0; b < nstream; b++) next[b] = am[b];
        }
        r.end();
        for (int b = 0; b < nstream; b++) for (int t : out[b]) if (tk.is_eog(t)) eos_hits++;

        const long produced = (long) nstream * cfg.gen_steps;
        json j;
        j["case"] = "decode";
        j["mode"] = mode; j["mode_name"] = mode_name(mode);
        j["streams"] = nstream;
        j["steps"] = cfg.gen_steps;
        j["tokens_emitted"] = produced;            // every one of these is a token the model produced
        j["eos_tokens_emitted"] = eos_hits;
        j["eos_policy"] = "fixed step count; EOS is emitted and fed back, streams are not stopped";
        j["wall_s"] = r.wall;
        j["aggregate_tokens_per_s"] = produced / r.wall;
        j["per_stream_tokens_per_s"] = cfg.gen_steps / r.wall;
        j["step_wall_s"] = step_wall;
        j["setup_s"] = setup;                      // reset + prompt prefill, excluded from the timing
        j["prompt_tokens"] = prompt_toks;
        if (cfg.hip_events) j["hip_event_ms"] = r.event_ms;
        j["telemetry"] = tel.summary(cfg.telemetry_raw);
        j["host"] = r.host();
        if (capture_text) {
            json texts = json::array();
            for (int b = 0; b < nstream; b++) {
                std::string s;
                for (int t : out[b]) s += tk.piece(t);
                texts.push_back(json{ { "stream", b }, { "tokens", out[b] }, { "text", s } });
            }
            j["streams_out"] = texts;
        }
        return j;
    }

    // ---- multistep: nstream sequences carried over several greedy steps from their real prompts.
    // Returns the token matrix; the analysis compares it to mode 0 step by step. A mode whose
    // single pass matches but whose third step diverges has a state, replay or parity bug.
    json multistep(int mode, int nstream, int steps) {
        e.batch_mode = mode;
        std::vector<Engine::Seq *> sv(nstream);
        std::vector<int> next(nstream, -1);
        for (int b = 0; b < nstream; b++) {
            e.reset_seq(seqs[b]);
            next[b] = prefill_stream(seqs[b], prompt_tokens[b], std::min(128, (int) prompt_tokens[b].size()));
            sv[b] = &seqs[b];
        }
        std::vector<std::vector<int>> out(nstream);
        for (int s = 0; s < steps; s++) {
            std::vector<std::vector<int>> tv(nstream);
            for (int b = 0; b < nstream; b++) { tv[b] = { next[b] }; out[b].push_back(next[b]); }
            std::vector<int> am;
            e.forward_batch(sv, tv, true, &am, nullptr);
            for (int b = 0; b < nstream; b++) next[b] = am[b];
        }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
        json j;
        j["case"] = "multistep";
        j["mode"] = mode; j["mode_name"] = mode_name(mode);
        j["gdn_state"] = gdn_state_format_name(gdn_state_format());
        j["seq_quant"] = sequence_quant_name(sequence_quant());
        j["streams"] = nstream;
        j["steps"] = steps;
        j["seq_len_after"] = seqs[0].len;
        json t = json::array(), txt = json::array();
        for (int b = 0; b < nstream; b++) {
            t.push_back(out[b]);
            std::string s;
            for (int tok : out[b]) s += tk.piece(tok);
            txt.push_back(s);
        }
        j["tokens"] = t;          // [stream][step], greedy, prompts identical across modes
        j["text"] = txt;
        return j;
    }

    // ---- width identity: the same document ingested at two pass widths must produce the same
    // logit bits for the same token. Nothing in a pass is supposed to depend on how many rows share
    // it: the quantisers are per row, the FFN arms are published bit-identical across widths, and a
    // row's attention reads the same keys whether the earlier rows arrived in this pass or the one
    // before it. This runs outside every timed region and asks for logits on every pass, so the
    // digest covers every token of the document rather than the last row of the last pass.
    json prefill_identity(int mode, int rows_per_pass) {
        e.batch_mode = mode;
        Engine::Seq & s = seqs[0];
        e.reset_seq(s);
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
        const int n = (int) doc_tokens.size();
        uint64_t hash = 14695981039346656037ull;
        size_t values = 0, nonfinite = 0;
        std::vector<float> lg;
        std::vector<int> am, argmaxes;
        for (int i = 0; i < n; i += rows_per_pass) {
            const int k = std::min(rows_per_pass, n - i);
            std::vector<Engine::Seq *> sv{ &s };
            std::vector<std::vector<int>> tv{ std::vector<int>(doc_tokens.begin() + i, doc_tokens.begin() + i + k) };
            e.forward_batch(sv, tv, true, &am, &lg);
            const auto * bytes = reinterpret_cast<const unsigned char *>(lg.data());
            for (size_t b = 0; b < lg.size() * sizeof(float); ++b) hash = (hash ^ bytes[b]) * 1099511628211ull;
            for (float v : lg) if (!std::isfinite(v)) nonfinite++;
            values += lg.size();
            argmaxes.insert(argmaxes.end(), am.begin(), am.end());
        }
        json j;
        j["case"] = "prefill-identity";
        j["mode"] = mode; j["mode_name"] = mode_name(mode);
        j["rows_per_pass"] = rows_per_pass;
        j["tokens"] = n;
        j["passes"] = (n + rows_per_pass - 1) / rows_per_pass;
        j["logit_values"] = values;
        j["nonfinite"] = nonfinite;
        j["logits_fnv64"] = std::to_string(hash);
        j["argmax"] = argmaxes;
        return j;
    }

    // ---- quality: full-vocabulary logits on fixed teacher-forced tokens
    // prefill-shaped: rows from one sequence in a single pass. decode-shaped: one row per stream.
    json quality(int mode, const std::string & shape, int rows, const fs::path & dir) {
        e.batch_mode = mode;
        std::vector<float> lg;
        std::vector<int> am;

        if (shape == "prefill") {
            Engine::Seq & s = seqs[0];
            e.reset_seq(s);
            const int ctx = cfg.quality_ctx;
            if (ctx + rows > (int) doc_tokens.size()) throw std::runtime_error("quality: document too short");
            for (int i = 0; i < ctx; i += 128) {
                const int k = std::min(128, ctx - i);
                std::vector<Engine::Seq *> sv{ &s };
                std::vector<std::vector<int>> tv{ std::vector<int>(doc_tokens.begin() + i, doc_tokens.begin() + i + k) };
                e.forward_batch(sv, tv, false, nullptr, nullptr);
            }
            std::vector<Engine::Seq *> sv{ &s };
            std::vector<std::vector<int>> tv{ std::vector<int>(doc_tokens.begin() + ctx, doc_tokens.begin() + ctx + rows) };
            e.forward_batch(sv, tv, true, &am, &lg);
        } else {
            std::vector<Engine::Seq *> sv(rows);
            std::vector<std::vector<int>> tv(rows);
            for (int b = 0; b < rows; b++) {
                e.reset_seq(seqs[b]);
                std::vector<int> t = prompt_tokens[b];
                const int fed = (int) t.size() - 1;      // last token becomes the timed-shape row
                for (int i = 0; i < fed; i += 128) {
                    const int k = std::min(128, fed - i);
                    std::vector<Engine::Seq *> one{ &seqs[b] };
                    std::vector<std::vector<int>> tt{ std::vector<int>(t.begin() + i, t.begin() + i + k) };
                    e.forward_batch(one, tt, false, nullptr, nullptr);
                }
                sv[b] = &seqs[b];
                tv[b] = { t.back() };
            }
            e.forward_batch(sv, tv, true, &am, &lg);
        }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));

        if ((int) lg.size() != rows * VOCAB)
            throw std::runtime_error("quality: forward_batch returned " + std::to_string(lg.size()) +
                                     " logits, expected " + std::to_string((size_t) rows * VOCAB));
        // A NaN reproduces its own bit pattern perfectly. Counting non-finite values here keeps a
        // pair comparison from reading two identically broken dumps as a successful reimplementation.
        size_t nonfinite = 0, nonfinite_rows = 0;
        for (int r = 0; r < rows; r++) {
            size_t bad = 0;
            for (int v = 0; v < VOCAB; v++) if (!std::isfinite(lg[(size_t) r * VOCAB + v])) bad++;
            nonfinite += bad;
            if (bad) nonfinite_rows++;
        }
        if (nonfinite)
            fprintf(stderr, "warning: mode %d (%s) %s rows %d produced %zu non-finite logits across %zu rows\n",
                    mode, mode_name(mode), shape.c_str(), rows, nonfinite, nonfinite_rows);

        json j;
        j["case"] = "quality";
        j["shape"] = shape;
        j["mode"] = mode; j["mode_name"] = mode_name(mode);
        j["gdn_state"] = gdn_state_format_name(gdn_state_format());
        j["gdn_defer"] = gdn_defer_effective();
        j["seq_quant"] = sequence_quant_name(sequence_quant());
        j["rows"] = rows;
        j["vocab"] = VOCAB;
        j["argmax"] = am;
        j["nonfinite_logits"] = (double) nonfinite;
        j["nonfinite_rows"] = (double) nonfinite_rows;
        if (cfg.dump_logits) {
            // The state coordinate is part of the arm's identity, so two arms never share a dump.
            const std::string fn = "logits-" + shape + "-r" + std::to_string(rows) + "-mode" + std::to_string(mode) +
                                   (gdn_state_format() == GDN_STATE_F32 ? "" : std::string("-state") + gdn_state_format_name(gdn_state_format())) +
                                   (gdn_defer_effective() != 1 ? "-defer" + std::to_string(gdn_defer_effective()) : "") +
                                   (sequence_quant() == SEQ_QUANT_A8 ? "" : std::string("-seq") + sequence_quant_name(sequence_quant())) + ".f32";
            FILE * f = fopen((dir / fn).c_str(), "wb");
            if (!f) throw std::runtime_error("cannot write logits dump");
            fwrite(lg.data(), 4, lg.size(), f);
            fclose(f);
            j["logits_file"] = fn;
            j["logits_dtype"] = "float32";
            j["logits_layout"] = "row-major [rows][vocab]";
        }
        return j;
    }

    // ---- horizon: one teacher-forced document per stream, walked one token per step in the
    // generation shape, with the distribution scored every `every` steps.
    //
    // Why this workload exists. A packed recurrent state is re-rounded every time the state crosses
    // memory, which in a generation step is once per token per layer and in a 128-row prefill pass
    // is once per 128 tokens per layer. The teacher-forced quality window measures at most 64
    // roundings, so it cannot separate a coordinate whose error accumulates from one whose error
    // does not: int16 scored +0.0024 nats and int8 +0.0035 there, both inside their own error bars,
    // with sixteen-fold different rounding between them. This runs 512 roundings per layer and
    // reports the score as a function of how many have happened.
    //
    // Teacher forcing is the point. Every arm is fed exactly the same tokens, so the comparison is
    // paired per (stream, step) against a fixed target and no arm's own output is privileged.
    // Scoring an arm's *free-running* continuation under the fp32 engine cannot rank coordinates:
    // the fp32 arm's greedy continuation is by construction the reference's own argmax path, so it
    // takes the minimum achievable NLL whatever its quality.
    json horizon(int mode, int nstream, int ctx, int steps, int every) {
        e.batch_mode = mode;
        const int need = ctx + steps + 1;            // last step's target is one past its input
        const int have = (int) horizon_doc.size();
        if (have < need) throw std::runtime_error("horizon: document has " + std::to_string(have) +
            " tokens, needs --horizon-ctx + --horizon-tokens + 1 = " + std::to_string(need));
        // Distinct windows while the document allows it, then evenly spaced overlapping ones. The
        // overlap is recorded rather than hidden: it costs independence between streams, not the
        // pairing between arms, which is what ranks them.
        const int stride = nstream > 1 ? std::min(need, (have - need) / (nstream - 1)) : need;
        std::vector<std::vector<int>> win(nstream);
        for (int b = 0; b < nstream; b++)
            win[b].assign(horizon_doc.begin() + (size_t) b * stride, horizon_doc.begin() + (size_t) b * stride + need);

        std::vector<Engine::Seq *> sv(nstream);
        for (int b = 0; b < nstream; b++) {
            e.reset_seq(seqs[b]);
            for (int i = 0; i < ctx; i += 128) {
                const int k = std::min(128, ctx - i);
                std::vector<Engine::Seq *> one{ &seqs[b] };
                std::vector<std::vector<int>> tt{ std::vector<int>(win[b].begin() + i, win[b].begin() + i + k) };
                e.forward_batch(one, tt, false, nullptr, nullptr);
            }
            sv[b] = &seqs[b];
        }

        std::vector<float> lg;
        std::vector<int> am;
        json scored = json::array();
        double nll_sum = 0.0; long scored_n = 0, agree_n = 0;
        for (int t = 0; t < steps; t++) {
            std::vector<std::vector<int>> tv(nstream);
            for (int b = 0; b < nstream; b++) tv[b] = { win[b][ctx + t] };
            const bool score = ((t + 1) % every) == 0 || t + 1 == steps;
            e.forward_batch(sv, tv, score, score ? &am : nullptr, score ? &lg : nullptr);
            if (!score) continue;
            HIP_CHECK_H(hipStreamSynchronize(e.stream));
            json row_nll = json::array(), row_arg = json::array(), row_tgt = json::array();
            for (int b = 0; b < nstream; b++) {
                const float * p = lg.data() + (size_t) b * VOCAB;
                const int tgt = win[b][ctx + t + 1];
                float mxf = p[0];
                for (int v = 1; v < VOCAB; v++) mxf = p[v] > mxf ? p[v] : mxf;
                const double mx = mxf;
                // `expf` on the shifted float, accumulated in double. The double `exp` this used to
                // call is 6x the cost and it made the HOST the panel's bound: scoring one step of 32
                // streams took about as long as eleven generation steps of the same batch, so a
                // sample large enough to resolve an activation quantiser did not fit a bounded
                // measurement call. The shift keeps every argument in [-inf, 0], so the relative
                // error is ~1e-7 against differences of ~1e-3, and every arm is scored by the same
                // code on the same tokens - the paired difference, which is what ranks arms, cannot
                // see it at all.
                double sum = 0.0;
                for (int v = 0; v < VOCAB; v++) sum += (double) expf(p[v] - mxf);
                const double nll = -((double) p[tgt] - mx - std::log(sum));
                row_nll.push_back(nll); row_arg.push_back(am[b]); row_tgt.push_back(tgt);
                nll_sum += nll; scored_n++;
                if (am[b] == tgt) agree_n++;
            }
            scored.push_back(json{ { "step", t }, { "nll", row_nll }, { "argmax", row_arg }, { "target", row_tgt } });
        }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));

        json j;
        j["case"] = "horizon";
        j["mode"] = mode; j["mode_name"] = mode_name(mode);
        j["gdn_state"] = gdn_state_format_name(gdn_state_format());
        j["gdn_defer"] = gdn_defer_effective();
        j["seq_quant"] = sequence_quant_name(sequence_quant());
        j["streams"] = nstream; j["ctx"] = ctx; j["steps"] = steps; j["score_every"] = every;
        j["window_stride"] = stride;
        j["distinct_windows"] = stride >= need;
        j["document"] = cfg.horizon_doc.empty() ? cfg.doc : cfg.horizon_doc;
        j["document_tokens"] = have;
        j["scored_predictions"] = (double) scored_n;
        j["nll_mean"] = scored_n ? nll_sum / (double) scored_n : 0.0;
        j["teacher_top1"] = scored_n ? (double) agree_n / (double) scored_n : 0.0;
        j["scored"] = scored;
        // The tokens each arm was fed, so a later pairing can prove the arms saw the same input.
        json wins = json::array();
        for (int b = 0; b < nstream; b++) wins.push_back(json{ { "stream", b }, { "offset", b * stride },
            { "first", win[b].front() }, { "last", win[b].back() } });
        j["windows"] = wins;
        return j;
    }

    // ---- state dump: the fp32 gated-delta state of one warmed sequence.
    //
    // Every packed coordinate this engine can hold is a choice of block and a choice of code, and
    // both are decided by what the state's own values look like. Nothing had measured that. One
    // dump answers the whole family offline: a row's crest factor says what a per-row scale costs,
    // the spread between a row's four 32-column groups says what a finer block would buy back, and
    // the exact readout error of any candidate can be evaluated against the real values rather than
    // against an assumed distribution.
    json state_dump(int mode, const fs::path & dir) {
        if (gdn_state_format() != GDN_STATE_F32)
            throw std::runtime_error("state dump reads the fp32 coordinate; run it without --gdn-state");
        e.batch_mode = mode;
        Engine::Seq & s = seqs[0];
        e.reset_seq(s);
        const int ctx = std::min(cfg.state_dump_ctx, (int) doc_full.size());
        for (int i = 0; i < ctx; i += 128) {
            const int k = std::min(128, ctx - i);
            std::vector<Engine::Seq *> sv{ &s };
            std::vector<std::vector<int>> tv{ std::vector<int>(doc_full.begin() + i, doc_full.begin() + i + k) };
            e.forward_batch(sv, tv, false, nullptr, nullptr);
        }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));

        json files = json::array();
        // A dump reads the durable image, so the engine has to be settled: a deferring process
        // keeps the last steps' rank-1 terms beside the state rather than in it. No-op at depth 1.
        gdn_defer_flush_pending(e.gdn_state, s.slot, e.stream);
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
        std::vector<float> buf((size_t) GDN_STATE_FLOATS);   // the dump is fp32-only, checked above
        for (int gi : cfg.state_dump_layers) {
            if (gi < 0 || gi >= 48) throw std::runtime_error("state dump layer outside 0..47");
            HIP_CHECK_H(hipMemcpy(buf.data(), e.gdn_state + ((size_t) s.slot * 48 + gi) * gdn_region_floats(),
                                  (size_t) GDN_STATE_FLOATS * sizeof(float), hipMemcpyDeviceToHost));
            const std::string fn = "state-gdn" + std::to_string(gi) + ".f32";
            FILE * f = fopen((dir / fn).c_str(), "wb");
            if (!f) throw std::runtime_error("cannot write state dump");
            fwrite(buf.data(), 4, buf.size(), f);
            fclose(f);
            size_t nonzero = 0, nonfinite = 0;
            for (float v : buf) { if (v != 0.0f) nonzero++; if (!std::isfinite(v)) nonfinite++; }
            files.push_back(json{ { "gdn_layer", gi }, { "file", fn }, { "nonzero", (double) nonzero },
                                  { "nonfinite", (double) nonfinite } });
        }
        json j;
        j["case"] = "state-dump";
        j["mode"] = mode; j["mode_name"] = mode_name(mode);
        j["context_tokens"] = ctx;
        j["heads"] = HV; j["rows"] = SS; j["cols"] = SS;
        j["layout"] = "row-major [head][row][col], fp32; a wave owns row j and lane l holds cols l+32s";
        j["files"] = files;
        return j;
    }
};

// ---------------------------------------------------------------- main

int main(int argc, char ** argv) {
    Cfg cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "-m" || a == "--model") cfg.model = next();
        else if (a == "--doc") cfg.doc = next();
        else if (a == "--prompts") cfg.prompts = next();
        else if (a == "--out") cfg.out_dir = next();
        else if (a == "--tag") cfg.tag = next();
        else if (a == "--modes") cfg.modes = parse_ints(next());
        else if (a == "--ffn-sched") cfg.ffn_scheds = parse_ints(next());
        else if (a == "--gdn-pregate") cfg.gdn_pregates = parse_ints(next());
        else if (a == "--gdn-split") cfg.gdn_splits = parse_ints(next());
        else if (a == "--gdn-cols") cfg.gdn_colss = parse_ints(next());
        else if (a == "--ffn-order") cfg.ffn_orders = parse_ints(next());
        else if (a == "--seq-sched") cfg.seq_scheds = parse_ints(next());
        else if (a == "--seq-image") cfg.seq_images = parse_ints(next());
        else if (a == "--ffn-image") cfg.ffn_images = parse_ints(next());
        else if (a == "--ffn-run") cfg.ffn_runs = parse_ints(next());
        else if (a == "--ffn-dn-mats") cfg.ffn_dn_mats = parse_ints(next());
        else if (a == "--gdn-state") cfg.gdn_states = parse_ints(next());
        else if (a == "--gdn-defer") cfg.gdn_defers = parse_ints(next());
        else if (a == "--seq-quant") {
            cfg.seq_quants.clear();
            std::string v = next(), tok;
            std::stringstream ss(v);
            while (std::getline(ss, tok, ',')) {
                if (tok == "a8" || tok == "8" || tok == "0") cfg.seq_quants.push_back(SEQ_QUANT_A8);
                else if (tok == "a4e" || tok == "1") cfg.seq_quants.push_back(SEQ_QUANT_A4E);
                else if (tok == "a4" || tok == "4" || tok == "2") cfg.seq_quants.push_back(SEQ_QUANT_A4);
                else if (tok == "a4e-out" || tok == "3") cfg.seq_quants.push_back(SEQ_QUANT_A4E_OUT);
                else if (tok == "a4-out") cfg.seq_quants.push_back(SEQ_QUANT_A4_OUT);
                else if (tok == "a4e-both" || tok == "5") cfg.seq_quants.push_back(SEQ_QUANT_A4E_BOTH);
                else if (tok == "a4-both" || tok == "6") cfg.seq_quants.push_back(SEQ_QUANT_A4_BOTH);
                else { fprintf(stderr, "--seq-quant wants a8, a4e, a4, a4e-out, a4-out, a4e-both or a4-both\n"); return 1; }
            }
            if (cfg.seq_quants.empty()) { fprintf(stderr, "--seq-quant wants a8, a4e, a4, a4e-out, a4-out, a4e-both or a4-both\n"); return 1; }
        }
        else if (a == "--prefill-rows") cfg.prefill_rows = parse_ints(next());
        else if (a == "--prefill-identity") cfg.prefill_identity = true;
        else if (a == "--head-rows") {
            std::string v = next();
            if (v == "tail") cfg.head_rows = { 1 };
            else if (v == "all") cfg.head_rows = { 0 };
            else if (v == "both") cfg.head_rows = { 1, 0 };
            else { fprintf(stderr, "--head-rows wants tail, all or both\n"); return 1; }
        }
        else if (a == "--streams") cfg.streams = parse_ints(next());
        else if (a == "--context") cfg.context = atoi(next().c_str());
        else if (a == "--slots") cfg.slots = atoi(next().c_str());
        else if (a == "--prefill-tokens") cfg.prefill_tokens = atoi(next().c_str());
        else if (a == "--rounds") cfg.rounds = atoi(next().c_str());
        else if (a == "--gen-steps" || a == "--steps") cfg.gen_steps = atoi(next().c_str());
        else if (a == "--quality-rows") cfg.quality_rows = parse_ints(next());
        else if (a == "--quality-shapes") {
            std::string v = next();
            cfg.quality_shapes.clear();
            if (v.find("prefill") != std::string::npos) cfg.quality_shapes.push_back("prefill");
            if (v.find("decode") != std::string::npos) cfg.quality_shapes.push_back("decode");
            if (cfg.quality_shapes.empty()) { fprintf(stderr, "--quality-shapes wants prefill, decode or both\n"); return 1; }
        }
        else if (a == "--reference-pairs") cfg.reference_pairs = next();
        else if (a == "--a4-images") {
            std::string v = next();
            if (v == "both") cfg.a4_images = 0;
            else if (v == "wide") cfg.a4_images = FFN_BATCH_A4_WIDE_ONLY;
            else if (v == "dense") cfg.a4_images = FFN_BATCH_A4_DENSE_ONLY;
            else { fprintf(stderr, "--a4-images wants both, wide or dense\n"); return 1; }
        }
        else if (a == "--quality-ctx") cfg.quality_ctx = atoi(next().c_str());
        else if (a == "--multistep-streams") cfg.multistep_streams = atoi(next().c_str());
        else if (a == "--multistep-steps") cfg.multistep_steps = atoi(next().c_str());
        else if (a == "--warmup-steps") cfg.warmup_steps = atoi(next().c_str());
        else if (a == "--seed") cfg.seed = (unsigned) atoi(next().c_str());
        else if (a == "-t" || a == "--threads") cfg.nthreads = atoi(next().c_str());
        else if (a == "--telemetry-ms") cfg.telemetry_ms = atoi(next().c_str());
        else if (a == "--telemetry-raw") cfg.telemetry_raw = true;
        else if (a == "--telemetry-cmd") cfg.telemetry_cmd = next();
        else if (a == "--hip-events") cfg.hip_events = true;
        else if (a == "--no-logits-dump") cfg.dump_logits = false;
        else if (a == "--prefill-logits") { std::string v = next(); cfg.prefill_logits = v == "all" ? PL_ALL : v == "none" ? PL_NONE : PL_LAST; }
        else if (a == "--horizon-tokens") cfg.horizon_tokens = atoi(next().c_str());
        else if (a == "--horizon-ctx") cfg.horizon_ctx = atoi(next().c_str());
        else if (a == "--horizon-every") cfg.horizon_every = atoi(next().c_str());
        else if (a == "--horizon-streams") cfg.horizon_streams = atoi(next().c_str());
        else if (a == "--horizon-doc") cfg.horizon_doc = next();
        else if (a == "--state-dump-layers") cfg.state_dump_layers = parse_ints(next());
        else if (a == "--state-dump-ctx") cfg.state_dump_ctx = atoi(next().c_str());
        else if (a == "--only") {
            std::string v = next();
            cfg.do_prefill = v.find("prefill") != std::string::npos;
            cfg.do_decode = v.find("decode") != std::string::npos;
            cfg.do_quality = v.find("quality") != std::string::npos;
            cfg.do_multistep = v.find("multistep") != std::string::npos;
            cfg.do_horizon = v.find("horizon") != std::string::npos;
            cfg.do_state_dump = v.find("state-dump") != std::string::npos;
        } else {
            fprintf(stderr,
                    "usage: batch_compare [-m model.gguf] [--doc FILE] [--prompts FILE] [--out DIR] [--tag NAME]\n"
                    "  [--modes 0,4,1,2,3,5] [--prefill-rows 32,128] [--prefill-identity] [--streams 8,32] [--context 512] [--slots 32]\n"
                    "    modes: 0 deployed, 1 integer A8, 2 integer A4, 3 scaled A8, 4 sliced deployed control,\n"
                    "           5 auto A8, 6 optimized IU8/A8, 7 optimized scaled A8, 8 optimized A4,\n"
                    "           9/10/11 automatic routes to 6/7/8 (<=4 deployed, 5..31 sliced, >=32 the module)\n"
                    "           12 single-map, 13 single-state, 14 single-retile; 15 single-grid control; 16 commit-scaled, 17 wide-sequence, 18 wide-commit, 19 wide-commit-a4\n"
                    "  [--prefill-tokens 384] [--rounds 3] [--gen-steps 8] [--quality-ctx 64]\n"
                    "  [--quality-rows 8,32] [--quality-shapes prefill,decode] [--reference-pairs 6:1,7:3,8:2]\n"
                    "  [--gdn-state 0,2] [--gdn-defer 0,1,4]\n"
                    "  [--a4-images both|wide|dense]  A4 holds two weight images and picks per call; either\n"
                    "      restriction keeps one and uses it at every row count, trading throughput for bytes\n"
                    "  [--multistep-streams 32] [--multistep-steps 4] [--seq-quant a8,a4e,a4] [--seq-image 0,1]\n"
                    "  [--prefill-logits last|all|none] [--head-rows tail|all|both] [--warmup-steps 3] [--seed 1234] [-t threads]\n"
                    "  --head-rows chooses how many rows of a prompt pass the vocabulary head produces:\n"
                    "      tail is one row, which is what ingestion reads, all is every row of the pass,\n"
                    "      and both walks them as a case axis inside one process.\n"
                    "  [--telemetry-ms 5] [--telemetry-raw] [--telemetry-cmd CMD] [--hip-events] [--no-logits-dump]\n"
                    "  [--only prefill,decode,quality,multistep,horizon,state-dump]\n"
                    "  [--horizon-tokens 512] [--horizon-ctx 128] [--horizon-every 8] [--horizon-streams 32]\n"
                    "  [--horizon-doc FILE]  teacher-forced single-token steps per stream, scored every Nth\n"
                    "      step. This is the shape a packed recurrent state has to survive: it rounds the\n"
                    "      state once per token per layer, where a 128-row prefill pass rounds it once per 128.\n"
                    "  [--state-dump-layers 0,12,24,36,47] [--state-dump-ctx 384]  fp32 state of a warmed\n"
                    "      sequence, for an offline search over storage coordinates\n"
                    "\n"
                    "--reference-pairs is recorded in run.json for tools/batch_compare.py; this driver never\n"
                    "interprets it. A decode-shaped quality group needs one slot and one distinct prompt per row;\n"
                    "groups past that are skipped and recorded, so --quality-shapes prefill measures 128 rows\n"
                    "without raising --slots to 128.\n");
            return 1;
        }
    }

    try {
        // A debug env var that truncates the model would let the baseline time a partial forward
        // while an optimized mode runs all 64 layers. Refuse rather than publish that number.
        for (const char * v : { "HALO_STOP", "HALO_LAYERS" }) {
            const char * s = getenv(v);
            if (s && atoi(s) != 0)
                throw std::runtime_error(std::string(v) + "=" + s + " truncates the forward pass; unset it before benchmarking");
        }
        for (int m : cfg.modes)
            if (m < 0 || m > MODE_MAX)
                throw std::runtime_error("mode " + std::to_string(m) + " is outside the engine's 0.." +
                                         std::to_string(MODE_MAX) + " batch modes");
        const bool timing_workload = cfg.rounds > 0 && (cfg.do_prefill || cfg.do_decode);
        const bool quality_decode_shape = cfg.do_quality &&
            std::find(cfg.quality_shapes.begin(), cfg.quality_shapes.end(), "decode") != cfg.quality_shapes.end();

        const int decode_streams = cfg.do_decode && !cfg.streams.empty() ? *std::max_element(cfg.streams.begin(), cfg.streams.end()) : 0;
        const int max_prefill_rows = cfg.do_prefill && !cfg.prefill_rows.empty() ? *std::max_element(cfg.prefill_rows.begin(), cfg.prefill_rows.end()) : 0;
        const int max_quality_rows = cfg.do_quality && !cfg.quality_rows.empty() ? *std::max_element(cfg.quality_rows.begin(), cfg.quality_rows.end()) : 0;
        // Slots a workload that must not be dropped needs at once. Decode-shaped quality is the one
        // optional consumer: it asks for as many slots as it can get and gives up the rest.
        const int required_streams = std::max({ decode_streams, cfg.do_multistep ? cfg.multistep_streams : 0,
                                                cfg.do_horizon ? cfg.horizon_streams : 0 });
        if (required_streams > cfg.slots)
            throw std::runtime_error("--streams/--multistep-streams need " + std::to_string(required_streams) +
                                     " slots, more than --slots " + std::to_string(cfg.slots));
        const int desired_streams = std::max(required_streams,
                                             quality_decode_shape ? std::min(max_quality_rows, cfg.slots) : 0);
        // A document longer than the context overflows the sequence inside forward_batch, and the
        // failure the caller sees is a glibc double free during unwinding rather than a message.
        // Say it here, where the two numbers are side by side.
        if (cfg.do_prefill && cfg.prefill_tokens > cfg.context)
            throw std::runtime_error("--prefill-tokens " + std::to_string(cfg.prefill_tokens) +
                                     " does not fit --context " + std::to_string(cfg.context) +
                                     "; raise --context or shorten the document");
        const int max_rows = std::max({ desired_streams, max_prefill_rows, max_quality_rows, 8 });
        if (max_rows > PASSMAX)
            throw std::runtime_error("a pass carries at most " + std::to_string(PASSMAX) + " rows");
        // Capacity is at least the full 128 rows whatever shapes are measured: prompt prefill inside
        // the decode and quality setup uses wider passes than any measured shape, and prepare_batch
        // runs once, outside every timed region. A wider measured shape raises it, and the extra
        // residency is the head's logit buffer (VOCAB floats per row) plus the activation workspace.
        const int capacity = std::max(128, max_rows);

        const std::string tag = cfg.tag.empty() ? [] {
            char buf[32]; time_t t = time(nullptr); strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", localtime(&t)); return std::string(buf);
        }() : cfg.tag;
        fs::path dir = fs::path(cfg.out_dir) / tag;
        fs::create_directories(dir);

        // Only the images the measured modes actually enter are built. Weight images are the bulk of
        // the module's residency, so a run that measures three modes should not pay for six.
        unsigned mask = modes_mask(cfg.modes);
        if (cfg.a4_images) {
            const bool wants_a4 = mask & (ffn_batch_mode_bit(2) | ffn_batch_mode_bit(8));
            if (!wants_a4) throw std::runtime_error("--a4-images only applies when a measured mode reaches A4");
            mask |= cfg.a4_images;                 // options ride in the same word as the mode bits
        }

        // No measured mode reads a weight image, so ask for the workspace alone rather than letting
        // a zero mask mean all of them.
        const unsigned prepare_mask = mask ? mask : BATCH_WORKSPACE_ONLY;

        // Every coordinate this panel will select has to be reserved before the engine allocates:
        // the regions it is about to size are the ones each arm will address. A case list that
        // names only packed coordinates also starts the process in one, so the fp32 default does
        // not reserve three times the memory for an arm this panel never runs.
        if (!cfg.gdn_states.empty() &&
            std::none_of(cfg.gdn_states.begin(), cfg.gdn_states.end(), [](int g) { return g < 0; }))
            gdn_state_boot_format(cfg.gdn_states.front());
        for (int gs : cfg.gdn_states) if (gs >= 0) gdn_state_reserve_format(gs);
        // The write-back depth is a reservation too at an exact coordinate: the pending triples
        // need room in a region whose values already fill it. See docs/gdn-defer-exact.md.
        for (int gd : cfg.gdn_defers) if (gd >= 0) gdn_defer_reserve_depth(gd);

        Engine e;
        e.head_rows_all = cfg.head_rows.size() == 1 && cfg.head_rows[0] == 0;
        e.load(cfg.model, cfg.nthreads, cfg.slots, cfg.context);
        const size_t bytes_after_load = e.device_bytes;
        e.prepare_batch(capacity, prepare_mask);   // allocation and any per-mode setup, outside every timed region
        if (std::any_of(cfg.modes.begin(), cfg.modes.end(), [](int m) { return m >= 17; })) e.prepare_sequence();
        const size_t bytes_after_prepare = e.device_bytes;
        // Workspace-only may leave no FFN state at all; every accessor below tolerates that.
        const size_t module_bytes = e.batch_ffn ? ffn_batch_bytes(e.batch_ffn) : 0;
        Tokenizer tk; tk.load(cfg.model);

        Driver drv(e, tk, cfg);
        drv.prepare_inputs(required_streams, desired_streams);
        const int max_streams = drv.decode_row_limit();

        Telemetry tel; tel.interval_ms = cfg.telemetry_ms; bool tel_ok = tel.discover();

        json run;
        run["schema"] = "bonsai-batch-compare/1";
        run["tag"] = tag;
        run["model"] = cfg.model;
        run["document"] = cfg.doc;
        run["prompts_file"] = cfg.prompts;
        run["context"] = cfg.context;
        run["slots"] = cfg.slots;
        run["max_rows"] = max_rows;
        run["batch_capacity"] = capacity;
        // run["rounds"] below is the measurement array. With no timed workload selected there are no
        // rounds to run, whatever --rounds said, so an empty round is never recorded as if it ran.
        run["rounds_requested"] = timing_workload ? cfg.rounds : 0;
        run["gen_steps"] = cfg.gen_steps;
        run["quality_rows"] = cfg.quality_rows;
        run["quality_shapes"] = cfg.quality_shapes;
        run["multistep"] = json{ { "streams", cfg.multistep_streams }, { "steps", cfg.multistep_steps } };
        run["prefill_tokens"] = cfg.prefill_tokens;
        run["modes"] = cfg.modes;
        run["ffn_scheds"] = cfg.ffn_scheds;
        run["gdn_pregates"] = cfg.gdn_pregates;
        run["gdn_splits"] = cfg.gdn_splits;
        run["gdn_colss"] = cfg.gdn_colss;
        run["ffn_orders"] = cfg.ffn_orders;
        run["seq_scheds"] = cfg.seq_scheds;
        run["seq_images"] = cfg.seq_images;
        run["ffn_images"] = cfg.ffn_images;
        run["ffn_runs"] = cfg.ffn_runs;
        run["ffn_dn_mats"] = cfg.ffn_dn_mats;
        run["gdn_states"] = cfg.gdn_states;
        run["gdn_state"] = gdn_state_format_name(gdn_state_format());
        run["gdn_defers"] = cfg.gdn_defers;
        run["gdn_defer"] = gdn_defer_depth();
        run["gdn_defer_effective"] = gdn_defer_effective();
        run["seq_quants"] = cfg.seq_quants;
        run["seq_quant"] = sequence_quant_name(sequence_quant());
        run["mode_names"] = [&] { json m = json::object(); for (int x : cfg.modes) m[std::to_string(x)] = mode_name(x); return m; }();
        run["mode_engine_flags"] = [&] { json m = json::object(); for (int x : cfg.modes) m[std::to_string(x)] = mode_engine_flag(x); return m; }();
        // Row-count dispatch of the automatic modes, mirroring src/batch.cpp. The analysis needs it
        // to know that pairing an automatic mode with a numerical family only holds at >=32 rows.
        run["sequence_routes"] = [&] {
            json routes=json::object();
            for(int m:cfg.modes) routes[std::to_string(m)]={
                {"wide_projection_min_rows",m>=17?json(32):json(nullptr)},
                {"commit_state",m==16||m==18||m==19}};
            return routes;
        }();
        // Every HALO_* variable this process saw, so a pairing can tell which arm it is holding
        // even for a route flag that did not exist when the analysis was written.
        run["halo_env"] = halo_env_snapshot();
        run["gdn_resident"] = sequence_resident_enabled(e.batch_sequence);
        run["wide_head"] = head_batch_capacity(e.batch_head) > 0;
        run["wide_head_tile_width"] = getenv("HALO_HEAD_TT") ? json(getenv("HALO_HEAD_TT")) : json("auto");
        run["head_rows"] = cfg.head_rows;
        run["gdn_state_split"] = getenv("HALO_GDN_SPLIT")?getenv("HALO_GDN_SPLIT"):"4";
        run["sequence_layout"] = sequence_direct_layout(e.batch_sequence) ? "direct" : "staged";
        run["sequence_operand"] = getenv("HALO_SEQUENCE_OPERAND") ? getenv("HALO_SEQUENCE_OPERAND") : "int8";
        run["sequence_tile_width"] = getenv("HALO_SEQUENCE_TT") ? json(getenv("HALO_SEQUENCE_TT")) : json("auto");
        run["auto_dispatch"] = [&] {
            json m = json::object();
            for (int x : cfg.modes) {
                const int wide = auto_wide_mode(x);
                if (wide < 0) continue;
                m[std::to_string(x)] = json{ { "rows_le_4", 0 }, { "rows_5_to_31", 4 }, { "rows_ge_32", wide },
                                             { "mirrors", "src/batch.cpp" } };
            }
            return m;
        }();
        // What the operator asked the analysis to compare; this driver does not interpret it.
        run["reference_pairs"] = cfg.reference_pairs.empty() ? json(nullptr) : json(cfg.reference_pairs);
        run["seed"] = cfg.seed;
        run["git_revision"] = run_capture("git rev-parse HEAD");
        run["git_diff_sha256"] = run_capture("git diff HEAD -- src kernels tools/batch_compare.cpp tools/run-batch-compare | sha256sum");
        run["executable_sha256"] = run_capture("sha256sum " + shell_quote(fs::read_symlink("/proc/self/exe").string()));
        run["input_sha256"] = run_capture("sha256sum " + shell_quote(cfg.doc) + " " + shell_quote(cfg.prompts));
        // Every engine, kernel and driver source that goes into this executable, listed by git rather
        // than by hand: a new header must not be able to slip out of the provenance record. Untracked
        // files in those paths are included too, so a not-yet-committed kernel still gets hashed.
        {
            const std::string list = "git ls-files -z --cached --others --exclude-standard -- "
                                     "src kernels vendor Makefile tools/batch_compare.cpp tools/batch_compare.py tools/halo_env.hpp tools/run-batch-compare";
            run["source_sha256"] = run_capture(list + " | xargs -0 sha256sum");
            run["source_sha256_rollup"] = run_capture(list + " | xargs -0 sha256sum | sha256sum");
        }
        run["hip_events"] = cfg.hip_events;
        // Resident device memory. The module's repacked weight images are additional representations
        // of all 64 layers held alongside the deployed weights, so they are a real deployment cost.
        // Budget is what the selected images should take; resident is what the engine actually holds.
        {
            // Several modes share one image, so pricing per mode would double count. The module
            // prices each image instead, and reports which of them this state actually holds.
            json images = json::array();
            double resident_images = 0;
            for (int i = 0; i < FFN_IMAGE_COUNT; i++) {
                const double resident = e.batch_ffn ? (double) ffn_batch_resident_image_bytes(e.batch_ffn, i) : 0.0;
                resident_images += resident;
                images.push_back(json{ { "image", i }, { "name", image_name(i) },
                                       { "bytes_all_layers", (double) ffn_batch_image_bytes(i) },
                                       { "resident_bytes", resident }, { "resident", resident > 0 } });
            }
            const double budget = mask ? (double) ffn_batch_weight_bytes(mask) : 0.0;
            const double budget_all = (double) ffn_batch_weight_bytes(FFN_BATCH_ALL);
            json served = json::object();
            for (int m : cfg.modes) {
                const int img = mode_module_image(m);
                served[std::to_string(m)] = img < 0 || !e.batch_ffn ? json(nullptr)
                                                                    : json(ffn_batch_supports(e.batch_ffn, img));
            }
            run["device_memory"] = json{
                { "after_load_bytes", (double) bytes_after_load },
                { "after_prepare_batch_bytes", (double) bytes_after_prepare },
                { "ffn_batch_module_bytes", (double) module_bytes },
                { "ffn_batch_weight_budget_bytes", budget },
                { "ffn_batch_weight_bytes_all_modes", budget_all },
                { "resident_image_bytes_total", resident_images },
                { "images", images },
                { "modes_mask", (unsigned long long) prepare_mask },
                { "modes_mask_hex", [&] { char b[16]; snprintf(b, sizeof b, "0x%x", prepare_mask); return std::string(b); }() },
                { "workspace_only", mask == 0 },
                { "ffn_state_created", e.batch_ffn != nullptr },
                { "wide_sequence_prepared", e.batch_sequence != nullptr },
                { "sequence_commit_modes", json::array({16,18,19}) },
                { "a4_image_option", cfg.a4_images == FFN_BATCH_A4_WIDE_ONLY ? "wide-only"
                                     : cfg.a4_images == FFN_BATCH_A4_DENSE_ONLY ? "dense-only" : "both" },
                { "module_modes_served", served },
                { "batch_rows_prepared", capacity },
                { "note", mask == 0
                      ? "no measured mode reads a weight image, so prepare_batch was asked for the workspace alone"
                      : "engine device_bytes before and after prepare_batch; the module total is the prepared weight images plus the shared activation workspace" },
            };
            fprintf(stderr, "device memory: %.2f GB model, %.2f GB after prepare_batch (+%.2f GB batched FFN module, "
                            "mask 0x%x, weight budget %.2f GB of %.2f GB for every mode)\n",
                    bytes_after_load / 1e9, bytes_after_prepare / 1e9, module_bytes / 1e9,
                    prepare_mask, budget / 1e9, budget_all / 1e9);
            if (e.batch_ffn)
                for (int i = 0; i < FFN_IMAGE_COUNT; i++)
                    if (ffn_batch_resident_image_bytes(e.batch_ffn, i))
                        fprintf(stderr, "  image %-6s %.2f GB\n", image_name(i),
                                ffn_batch_resident_image_bytes(e.batch_ffn, i) / 1e9);
            if (mask == 0)
                fprintf(stderr, "no measured mode reads a weight image; asked prepare_batch for the workspace alone "
                                "(0x%x) instead of the %.2f GB every image would cost\n", prepare_mask, budget_all / 1e9);
            // A mode whose image was never built throws at its first pass, deep into a long run.
            // Refuse now, while nothing has been measured and nothing has been written.
            for (int m : cfg.modes) {
                const int img = mode_module_image(m);
                if (img >= 0 && !ffn_batch_supports(e.batch_ffn, img))
                    throw std::runtime_error("mode " + std::to_string(m) + " (" + mode_name(m) + ") needs module mode " +
                                             std::to_string(img) + ", which prepare_batch did not build");
            }
        }
        // provenance: how the machine was set up for this run
        {
            json env = json::object();
            for (const char * v : { "BONSAI_BENCH_SERVER_PAUSED", "BONSAI_BENCH_SERVER_MASKED", "HALO_PROFILE", "HALO_PROFILE_ROWS", "HALO_SPEC_DEBUG", "HALO_SEQUENCE_OPERAND", "HALO_SEQUENCE_TT", "HALO_SEQUENCE_LAYOUT", "HALO_GDN_RESIDENT", "HALO_GDN_SPLIT", "HALO_GDN_COMPARE", "HIP_VISIBLE_DEVICES" }) {
                const char * s = getenv(v);
                env[v] = s ? json(std::string(s)) : json(nullptr);
            }
            env["HALO_STOP"] = nullptr; env["HALO_LAYERS"] = nullptr;  // refused above when nonzero
            run["environment"] = env;
            const char * paused = getenv("BONSAI_BENCH_SERVER_PAUSED");
            run["resident_server_paused"] = paused && atoi(paused) != 0;
            run["full_model_layers"] = NLAYER;
        }
        run["telemetry_available"] = tel_ok;
        run["started"] = [] { char b[32]; time_t t = time(nullptr); strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", localtime(&t)); return std::string(b); }();
        if (!cfg.telemetry_cmd.empty()) run["telemetry_cmd_before"] = run_capture(cfg.telemetry_cmd);

        // ---- warmup: every mode runs the decode shape once, discarded. Clocks settle and any
        // first-touch allocation inside a mode happens before the first timed round. A run with no
        // timed workload measures numerics only, where a clock ramp changes nothing, so it is skipped.
        const int warm_streams = std::max(1, max_streams);
        double warm0 = now();
        for (size_t wi = 0; timing_workload && wi < cfg.modes.size(); wi++) {
            const int mode = cfg.modes[wi];
            e.batch_mode = mode;
            for (int b = 0; b < warm_streams; b++) e.reset_seq(drv.seqs[b]);
            std::vector<Engine::Seq *> sv(warm_streams);
            std::vector<std::vector<int>> tv(warm_streams);
            for (int b = 0; b < warm_streams; b++) { sv[b] = &drv.seqs[b]; tv[b] = { drv.prompt_tokens[b][0] }; }
            std::vector<int> am;
            const double mode_warm_start = now();
            for (int s = 0; s < cfg.warmup_steps || now() - mode_warm_start < 2.0; s++) {
                if (drv.seqs[0].len + 1 >= e.context)
                    for (int b = 0; b < warm_streams; b++) e.reset_seq(drv.seqs[b]);
                e.forward_batch(sv, tv, true, &am, nullptr);
                for (int b = 0; b < warm_streams; b++) tv[b] = { am[b] };
            }
        }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
        run["warmup_min_s_per_mode"] = timing_workload ? json(2.0) : json(nullptr);
        run["warmup_s"] = now() - warm0;
        run["warmup_skipped"] = !timing_workload;
        if (timing_workload) fprintf(stderr, "warmup done in %.2f s\n", run["warmup_s"].get<double>());
        else fprintf(stderr, "no timed workload selected; skipping the clock ramp\n");

        // ---- timed rounds, mode order reshuffled each round inside this one process
        std::mt19937 rng(cfg.seed);
        json rounds = json::array();
        for (int r = 0; r < (timing_workload ? cfg.rounds : 0); r++) {
            std::vector<std::tuple<int,int,int,int,int,int,int,int,int,int,int,int,int,int,int>> order;
            for (int m : cfg.modes) for (int sc : cfg.ffn_scheds) for (int pg : cfg.gdn_pregates) for (int gc : cfg.gdn_colss) for (int od : cfg.ffn_orders) for (int sq : cfg.seq_scheds) for (int im : cfg.ffn_images) for (int dm : cfg.ffn_dn_mats) for (int gs : cfg.gdn_states) for (int gd : cfg.gdn_defers) for (int hr : cfg.head_rows) for (int qz : cfg.seq_quants) for (int si : cfg.seq_images) for (int fr : cfg.ffn_runs) for (int gsp : cfg.gdn_splits) order.emplace_back(m, sc, pg, gc, od, sq, im, dm, gs, gd, hr, qz, si, fr, gsp);
            std::shuffle(order.begin(), order.end(), rng);
            json jr; jr["round"] = r; jr["measurements"] = json::array();
            jr["mode_order"] = json::array();
            for (auto [m, sc, pg, gc, od, sq, im, dm, gs, gd, hr, qz, si, fr, gsp] : order) jr["mode_order"].push_back(json{ { "mode", m }, { "ffn_sched", sc }, { "gdn_pregate", pg }, { "gdn_cols", gc }, { "ffn_order", od }, { "seq_sched", sq }, { "ffn_image", im }, { "ffn_dn_mats", dm }, { "gdn_state", gs }, { "gdn_defer", gd }, { "head_rows", hr ? "tail" : "all" }, { "seq_quant", qz }, { "seq_image", si }, { "ffn_run", fr }, { "gdn_split", gsp } });
            for (auto [mode, sched, pregate, gcols, ord, seq_sched, image, dn_mats, gdn_state, gdn_defer, head_rows, seq_quant_arm, seq_image_arm, ffn_run_arm, gdn_split_arm] : order) {
                // A unit width carries no state and no numerical map, so it alternates between
                // arms of one process with nothing to synchronise. -1 is set too, because it has to
                // clear the previous arm's pin and let the shape rule choose. docs/gdn-unit-width.md.
                gdn_resident_set_split(gdn_split_arm);
                e.head_rows_all = head_rows == 0;
                // Rewrites every resident weight image from the source weights, which is where an
                // arm's cold start belongs: between arms, not inside the first timed pass of one.
                if (ffn_run_arm >= 0) ffn_batch_set_run_order(e.batch_ffn, ffn_run_arm, e.stream);
                if (seq_image_arm >= 0 && e.batch_sequence) sequence_batch_set_image(e.batch_sequence, seq_image_arm, e.stream);
                if (sched >= 0) ffn_batch_set_a4_sched(e.batch_ffn, sched);
                if (image >= 0) ffn_batch_set_a4_image(e.batch_ffn, image);
                if (dn_mats >= 0) ffn_batch_set_dn_mats(e.batch_ffn, dn_mats);
                if (pregate >= 0) gdn_resident_set_pregate(pregate);
                gdn_resident_set_cols(gcols);   // -1 clears a previous arm's pin, like the unit width
                if (ord >= 0) ffn_batch_set_a4_order(e.batch_ffn, ord);
                if (seq_sched >= 0 && e.batch_sequence) sequence_batch_set_sched(e.batch_sequence, seq_sched);
                if (gdn_state >= 0) gdn_state_set_format(gdn_state);
                if (gdn_defer >= 0) gdn_defer_set_depth(gdn_defer);
                if (seq_quant_arm >= 0) sequence_set_quant(seq_quant_arm);
                // The stored weight order follows the precision, and it is 1.3 GB of permutation:
                // pay it here, between arms, not inside the first timed pass of one.
                if (e.batch_sequence) {
                    sequence_batch_sync_quant(e.batch_sequence, e.stream);
                    HIP_CHECK_H(hipStreamSynchronize(e.stream));
                }
                const char * state_now = gdn_state_format_name(gdn_state_format());
                const char * quant_now = sequence_quant_name(sequence_quant());
                const int defer_now = gdn_defer_effective();
                const int sched_now = ffn_batch_a4_sched(e.batch_ffn), pregate_now = gdn_resident_pregate();
                const int gsplit_now = gdn_resident_split(), cols_now = gdn_resident_cols();
                const int ord_now = ffn_batch_a4_order(e.batch_ffn);
                const int seq_now = e.batch_sequence ? sequence_batch_sched(e.batch_sequence) : -1;
                const int seqimg_now = e.batch_sequence ? sequence_batch_image(e.batch_sequence) : -1;
                const int image_now = ffn_batch_a4_image(e.batch_ffn);
                const int run_now = ffn_batch_run_order(e.batch_ffn);
                const int dnmats_now = ffn_batch_dn_mats(e.batch_ffn);
                const char * head_now = e.head_rows_all ? "all" : "tail";
                if (cfg.do_prefill) for (int rows : cfg.prefill_rows) {
                    json m = drv.prefill(mode, rows, tel);
                    // Read the coordinate back after the pass, not before it: a route that cannot
                    // carry the process default drops a stage while the pass runs, and a panel
                    // that recorded what it asked for would name an arm that did not execute.
                    const char * quant_ran = sequence_quant_name(sequence_quant());
                    m["round"] = r; m["ffn_sched"] = sched_now; m["gdn_pregate"] = pregate_now; m["ffn_order"] = ord_now; m["seq_sched"] = seq_now; m["seq_image"] = seqimg_now; m["ffn_image"] = image_now; m["ffn_dn_mats"] = dnmats_now; m["gdn_state"] = state_now; m["head_rows"] = head_now; m["seq_quant"] = quant_ran; m["seq_quant_asked"] = quant_now; m["gdn_defer"] = defer_now; m["ffn_run"] = run_now; m["gdn_split"] = gsplit_now; m["gdn_cols"] = cols_now;
                    fprintf(stderr, "round %d  prefill mode %2d (%-24s) rows %3d sched %d pregate %d split %2d order %d seq %d seqimg %d image %d run %d dnmats %d state %s defer %d head %s quant %-3s: %6.2f s  %7.1f tok/s\n",
                            r, mode, mode_name(mode), rows, sched_now, pregate_now, gsplit_now, ord_now, seq_now, seqimg_now, image_now, run_now, dnmats_now, state_now, defer_now, head_now, quant_ran, m["wall_s"].get<double>(), m["tokens_per_s"].get<double>());
                    jr["measurements"].push_back(m);
                }
                if (cfg.do_decode) for (int st : cfg.streams) {
                    json m = drv.decode(mode, st, tel, r == 0);
                    const char * quant_ran = sequence_quant_name(sequence_quant());
                    m["round"] = r; m["ffn_sched"] = sched_now; m["gdn_pregate"] = pregate_now; m["ffn_order"] = ord_now; m["seq_sched"] = seq_now; m["seq_image"] = seqimg_now; m["ffn_image"] = image_now; m["ffn_dn_mats"] = dnmats_now; m["gdn_state"] = state_now; m["head_rows"] = head_now; m["seq_quant"] = quant_ran; m["seq_quant_asked"] = quant_now; m["gdn_defer"] = defer_now; m["ffn_run"] = run_now; m["gdn_split"] = gsplit_now; m["gdn_cols"] = cols_now;
                    fprintf(stderr, "round %d  decode  mode %2d (%-24s) streams %3d sched %d pregate %d split %2d order %d seq %d seqimg %d image %d run %d dnmats %d state %s defer %d head %s quant %-3s: %6.2f s  %7.1f tok/s aggregate (%.2f per stream)\n",
                            r, mode, mode_name(mode), st, sched_now, pregate_now, gsplit_now, ord_now, seq_now, seqimg_now, image_now, run_now, dnmats_now, state_now, defer_now, head_now, quant_ran, m["wall_s"].get<double>(),
                            m["aggregate_tokens_per_s"].get<double>(), m["per_stream_tokens_per_s"].get<double>());
                    jr["measurements"].push_back(m);
                }
            }
            rounds.push_back(jr);
            // keep partial results on disk: a crash in a later round must not lose the earlier ones
            run["rounds"] = rounds;
            std::ofstream(dir / "run.json") << run.dump(1) << "\n";
        }
        run["rounds"] = rounds;

        // ---- multistep state check: identical prompts, several greedy steps, 32 sequences.
        // Not timed; this is where a scheduling or replay error shows up that a single pass hides.
        if (cfg.do_multistep && cfg.multistep_streams > 0 && cfg.multistep_steps > 0) {
            json ms = json::array();
            for (int mode : cfg.modes) for (int qz : cfg.seq_quants) {
                if (qz >= 0) sequence_set_quant(qz);
                json m = drv.multistep(mode, cfg.multistep_streams, cfg.multistep_steps);
                fprintf(stderr, "multistep mode %2d (%-24s) quant %-3s: %d streams x %d steps, seq len after %d\n",
                        mode, mode_name(mode), sequence_quant_name(sequence_quant()),
                        cfg.multistep_streams, cfg.multistep_steps, m["seq_len_after"].get<int>());
                ms.push_back(m);
            }
            run["multistep_runs"] = ms;
            std::ofstream(dir / "run.json") << run.dump(1) << "\n";
        }

        // ---- pass-width identity, once, after the timed rounds and outside every timed region.
        // Two widths of the same document must agree bit for bit on every logit they both produce.
        if (cfg.prefill_identity && cfg.prefill_rows.size() > 1) {
            json wi = json::array();
            for (int mode : cfg.modes) {
                std::string first;
                for (int rows : cfg.prefill_rows) {
                    json m = drv.prefill_identity(mode, rows);
                    if (first.empty()) first = m["logits_fnv64"].get<std::string>();
                    const bool same = m["logits_fnv64"].get<std::string>() == first;
                    m["matches_first_width"] = same;
                    fprintf(stderr, "identity mode %2d rows %3d: %zu logits in %d passes, %s nonfinite %zu, fnv %s%s\n",
                            mode, rows, m["logit_values"].get<size_t>(), m["passes"].get<int>(),
                            same ? "MATCH" : "DIFFERS", m["nonfinite"].get<size_t>(),
                            m["logits_fnv64"].get<std::string>().c_str(), same ? "" : "  <-- pass width changed the bits");
                    wi.push_back(m);
                }
            }
            run["prefill_identity"] = wi;
            std::ofstream(dir / "run.json") << run.dump(1) << "\n";
        }

        // ---- quality, once, after the timed rounds. Same fixed token inputs for every mode.
        if (cfg.do_quality) {
            json q = json::array();
            json inputs = json::object();
            json skipped = json::array();
            for (int rows : cfg.quality_rows) {
                for (const std::string & shape : cfg.quality_shapes) {
                    // A group we cannot serve is recorded with its reason, not dropped in silence.
                    std::string why;
                    if (shape == "decode" && rows > max_streams)
                        why = "decode shape needs one slot and one distinct prompt per row: " + std::to_string(rows) +
                              " rows, " + std::to_string(max_streams) + " available (raise --slots and the prompts file, "
                              "or measure this row count with --quality-shapes prefill)";
                    else if (shape == "prefill" && cfg.quality_ctx + rows > (int) drv.doc_tokens.size())
                        why = "prefill shape needs --quality-ctx + rows = " + std::to_string(cfg.quality_ctx + rows) +
                              " document tokens, " + std::to_string(drv.doc_tokens.size()) + " loaded (raise --prefill-tokens)";
                    if (!why.empty()) {
                        fprintf(stderr, "quality %s rows %d skipped: %s\n", shape.c_str(), rows, why.c_str());
                        skipped.push_back(json{ { "shape", shape }, { "rows", rows }, { "reason", why } });
                        continue;
                    }
                    for (int mode : cfg.modes) for (int gs : cfg.gdn_states) for (int gd : cfg.gdn_defers) for (int qz : cfg.seq_quants) {
                        if (gs >= 0) gdn_state_set_format(gs);
                        if (gd >= 0) gdn_defer_set_depth(gd);
                        if (qz >= 0) sequence_set_quant(qz);
                        if (e.batch_sequence) { sequence_batch_sync_quant(e.batch_sequence, e.stream); HIP_CHECK_H(hipStreamSynchronize(e.stream)); }
                        json m = drv.quality(mode, shape, rows, dir);
                        fprintf(stderr, "quality %s rows %3d mode %2d (%-24s) state %s quant %-3s: argmax[0]=%d\n",
                                shape.c_str(), rows, mode, mode_name(mode), gdn_state_format_name(gdn_state_format()),
                                sequence_quant_name(sequence_quant()), m["argmax"][0].get<int>());
                        q.push_back(m);
                    }
                }
                // teacher-forced inputs, recorded so the comparison is auditable
                json dec = json::array();
                for (int b = 0; b < std::min(rows, max_streams); b++)
                    dec.push_back(json{ { "stream", b }, { "prompt_tokens", (int) drv.prompt_tokens[b].size() }, { "row_token", drv.prompt_tokens[b].back() } });
                inputs[std::to_string(rows)] = json{
                    { "prefill_shape", json{ { "context_tokens", cfg.quality_ctx },
                                             { "rows", cfg.quality_ctx + rows <= (int) drv.doc_tokens.size()
                                                           ? std::vector<int>(drv.doc_tokens.begin() + cfg.quality_ctx, drv.doc_tokens.begin() + cfg.quality_ctx + rows)
                                                           : std::vector<int>{} } } },
                    { "decode_shape", dec } };
            }
            run["quality"] = q;
            run["quality_inputs"] = inputs;
            run["quality_skipped"] = skipped;
        }

        // ---- horizon, once, after the timed rounds. Every arm walks the same teacher-forced
        // tokens, so the pairing is exact and the state coordinate is the only difference.
        //
        // The activation precision belongs on this axis for the same reason the state coordinate
        // does, and it was missing: an activation quantiser is re-applied once per token per layer
        // in a generation step, so a 127-prediction teacher-forced window resolves it to +/-0.04
        // nats and the arms it is asked to rank differ by less than that. Switching the precision
        // permutes the sequence weight image in place, so the arm change needs the same
        // `sequence_batch_sync_quant` the quality loop below performs.
        if (cfg.do_horizon) {
            json h = json::array();
            const int ns = std::min(cfg.horizon_streams, max_streams);
            for (int mode : cfg.modes) for (int gs : cfg.gdn_states) for (int gd : cfg.gdn_defers) for (int qz : cfg.seq_quants) {
                if (gs >= 0) gdn_state_set_format(gs);
                if (gd >= 0) gdn_defer_set_depth(gd);
                if (qz >= 0) sequence_set_quant(qz);
                if (e.batch_sequence) { sequence_batch_sync_quant(e.batch_sequence, e.stream); HIP_CHECK_H(hipStreamSynchronize(e.stream)); }
                json m = drv.horizon(mode, ns, cfg.horizon_ctx, cfg.horizon_tokens, cfg.horizon_every);
                fprintf(stderr, "horizon mode %2d (%-20s) state %-5s defer %d quant %-8s: %.0f scored, NLL %.5f, top1 %.4f\n",
                        mode, mode_name(mode), gdn_state_format_name(gdn_state_format()), gdn_defer_effective(),
                        sequence_quant_name(sequence_quant()),
                        m["scored_predictions"].get<double>(), m["nll_mean"].get<double>(),
                        m["teacher_top1"].get<double>());
                h.push_back(m);
                run["horizon"] = h;
                run["horizon_config"] = json{ { "streams", ns }, { "ctx", cfg.horizon_ctx },
                    { "tokens", cfg.horizon_tokens }, { "score_every", cfg.horizon_every } };
                std::ofstream(dir / "run.json") << run.dump(1) << "\n";   // partial panels survive a timeout
            }
        }

        if (cfg.do_state_dump) {
            json d = json::array();
            for (int mode : cfg.modes) {
                json m = drv.state_dump(mode, dir);
                fprintf(stderr, "state dump mode %2d: %zu layers written\n", mode, m["files"].size());
                d.push_back(m);
            }
            run["state_dump"] = d;
            std::ofstream(dir / "run.json") << run.dump(1) << "\n";
        }

        if (!cfg.telemetry_cmd.empty()) run["telemetry_cmd_after"] = run_capture(cfg.telemetry_cmd);
        run["finished"] = [] { char b[32]; time_t t = time(nullptr); strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", localtime(&t)); return std::string(b); }();
        std::ofstream(dir / "run.json") << run.dump(1) << "\n";
        fprintf(stderr, "wrote %s\n", (dir / "run.json").c_str());
        printf("%s\n", (dir / "run.json").c_str());
    } catch (const std::exception & ex) {
        fprintf(stderr, "batch_compare failed: %s\n", ex.what());
        return 1;
    }
    return 0;
}
