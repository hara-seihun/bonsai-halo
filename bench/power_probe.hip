// What a spinning host thread costs the shader clock, with no model in the picture.
//
// One kernel, one shape, run for a fixed wall budget under each host-wait arm, with the
// socket rails and the delivered clock read out of `gpu_metrics` across each arm's own
// window. The kernel is a weight-stream read at the geometry this engine's matvec uses -
// 512-byte runs, one per wave - so the arms differ in exactly one thing: whether the
// thread that called hipStreamSynchronize was spinning while the GPU ran.
//
//   tools/run-batch-compare --exec ./bench/power_probe [--seconds N] [--arms LIST]
//
// Arms are `spin`, `block`, `yield` and `auto`, comma separated, and are run in the given
// order and then in reverse, so a monotone drift in clock or temperature shows up as a
// disagreement between the two orders rather than as a result.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "power_probe: %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

// A wave reads its own 512-byte run and folds it, which is the FFN/matvec weight-stream
// shape. The fold keeps the loads from being dead without adding a dependency that would
// make this an issue probe.
__global__ __launch_bounds__(256) void k_stream(const uint4 * __restrict__ w, unsigned * out, int runs, int stride) {
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    unsigned acc = 0;
    const uint4 * p = w + (size_t) wave * stride + lane;
    for (int r = 0; r < runs; r++) {
        uint4 v = p[0];
        acc += v.x ^ v.y ^ v.z ^ v.w;
        p += 32;
    }
    if (acc == 0xffffffffu) out[0] = acc;
}

// gpu_metrics v3.0. tools/gpu_clock.py's docstring records what pins each offset.
struct Rails { double socket, allcore, gfx; int clk, busy; unsigned thr[7]; };

static bool read_rails(Rails & r) {
    FILE * f = fopen("/sys/class/drm/card0/device/gpu_metrics", "rb");
    if (!f) return false;
    unsigned char b[264];
    size_t n = fread(b, 1, sizeof b, f);
    fclose(f);
    if (n != sizeof b) return false;
    auto u16 = [&](int o) { return (unsigned) (b[o] | (b[o + 1] << 8)); };
    auto u32 = [&](int o) { return (unsigned) (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | ((unsigned) b[o + 3] << 24)); };
    r.socket  = u32(112) / 1000.0;
    r.allcore = u32(132) / 1000.0;
    r.gfx     = u32(124) / 1000.0;
    r.clk     = (int) u16(174);
    static const int off[7] = { 228, 232, 236, 240, 244, 248, 252 };
    for (int i = 0; i < 7; i++) r.thr[i] = u32(off[i]);
    r.busy = 0;
    if (FILE * g = fopen("/sys/class/drm/card0/device/gpu_busy_percent", "r")) { if (fscanf(g, "%d", &r.busy) != 1) r.busy = 0; fclose(g); }
    return true;
}

// utime+stime of this process, in seconds.
static double self_cpu() {
    FILE * f = fopen("/proc/self/stat", "r");
    if (!f) return 0;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    char * p = strrchr(buf, ')');
    if (!p) return 0;
    long ut = 0, st = 0;
    int field = 2;
    for (char * t = strtok(p + 2, " "); t; t = strtok(nullptr, " ")) {
        field++;
        if (field == 14) ut = atol(t);
        if (field == 15) { st = atol(t); break; }
    }
    return (double) (ut + st) / sysconf(_SC_CLK_TCK);
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Result { double cpu_pct, clk, socket, allcore, gfx, gbps, thr_pct[7]; int rounds; };

// One arm, in its own process: hipSetDeviceFlags binds at the first device use, so a
// process cannot hold two of these, and a fork after HIP init does not inherit a usable
// context. The parent re-execs itself per arm and reads the result back over a pipe.
static int run_arm(const std::string & a, double seconds, size_t mb, int wfd) {
    unsigned flags = 0;
    if (a == "spin") flags = hipDeviceScheduleSpin;
    else if (a == "block") flags = hipDeviceScheduleBlockingSync;
    else if (a == "yield") flags = hipDeviceScheduleYield;
    if (flags) CK(hipSetDeviceFlags(flags));

    const size_t bytes = mb << 20;
    // 512-byte runs, one per wave, walked with a 512-byte stride: the contiguous band
    // bench/wstream measures at 194-218 GB/s, so this is not fighting an image order.
    const int waves = 4096, runs = (int) (bytes / ((size_t) waves * 512));
    const int stride = runs * 32;
    uint4 * w = nullptr; unsigned * out = nullptr;
    CK(hipMalloc(&w, (size_t) waves * stride * sizeof(uint4)));
    CK(hipMemset(w, 0x3c, (size_t) waves * stride * sizeof(uint4)));
    CK(hipMalloc(&out, 4));
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    // Warm the DPM ramp before the window opens, so an arm is not measured climbing.
    double warm = now();
    while (now() - warm < 1.0) {
        hipLaunchKernelGGL(k_stream, dim3(waves / 8), dim3(256), 0, s, w, out, runs, stride);
        CK(hipStreamSynchronize(s));
    }
    Rails r0, r1;
    read_rails(r0);
    double c0 = self_cpu(), t0 = now();
    long long iters = 0;
    double clk_sum = 0, sock_sum = 0, core_sum = 0, gfx_sum = 0; int samples = 0;
    while (now() - t0 < seconds) {
        for (int k = 0; k < 4; k++) hipLaunchKernelGGL(k_stream, dim3(waves / 8), dim3(256), 0, s, w, out, runs, stride);
        CK(hipStreamSynchronize(s));
        iters += 4;
        Rails rs;
        if (read_rails(rs) && rs.busy > 50) { clk_sum += rs.clk; sock_sum += rs.socket; core_sum += rs.allcore; gfx_sum += rs.gfx; samples++; }
    }
    double t1 = now(), c1 = self_cpu();
    read_rails(r1);
    Result res{};
    res.cpu_pct  = (c1 - c0) / (t1 - t0) * 100.0;
    res.clk      = samples ? clk_sum / samples : 0;
    res.socket   = samples ? sock_sum / samples : 0;
    res.allcore  = samples ? core_sum / samples : 0;
    res.gfx      = samples ? gfx_sum / samples : 0;
    res.gbps     = (double) iters * bytes / (t1 - t0) / 1e9;
    res.rounds   = (int) iters;
    for (int i = 0; i < 7; i++) res.thr_pct[i] = (r1.thr[i] - r0.thr[i]) / (t1 - t0) / 10.0;
    ssize_t ignored = write(wfd, &res, sizeof res);
    (void) ignored;
    return 0;
}

int main(int argc, char ** argv) {
    double seconds = 4.0;
    std::string arms = "spin,block", child;
    size_t mb = 768;
    int wfd = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--arms") && i + 1 < argc) arms = argv[++i];
        else if (!strcmp(argv[i], "--mb") && i + 1 < argc) mb = (size_t) atoll(argv[++i]);
        else if (!strcmp(argv[i], "--child") && i + 1 < argc) child = argv[++i];
        else if (!strcmp(argv[i], "--fd") && i + 1 < argc) wfd = atoi(argv[++i]);
    }
    if (!child.empty()) return run_arm(child, seconds, mb, wfd);

    std::vector<std::string> arm;
    for (size_t p = 0; p < arms.size();) {
        size_t c = arms.find(',', p);
        if (c == std::string::npos) c = arms.size();
        arm.push_back(arms.substr(p, c - p));
        p = c + 1;
    }
    std::vector<std::string> order = arm;
    for (size_t i = arm.size(); i-- > 0;) order.push_back(arm[i]);

    printf("power_probe: %zu MB stream, %.1f s per arm after a 1 s warm, arms %s, each forward then reverse\n",
           mb, seconds, arms.c_str());
    printf("%-8s %-5s %9s %8s %8s %8s %8s %9s  %s\n", "arm", "pass", "hostcpu%", "gfxclk", "gfxW", "cpuW", "socketW", "GB/s", "limit residency");
    fflush(stdout);
    for (size_t ai = 0; ai < order.size(); ai++) {
        const std::string & a = order[ai];
        int fds[2];
        if (pipe(fds) != 0) { perror("pipe"); return 1; }
        pid_t pid = fork();
        if (pid == 0) {
            close(fds[0]);
            char sec[32], mbs[32], fd[16];
            snprintf(sec, sizeof sec, "%g", seconds);
            snprintf(mbs, sizeof mbs, "%zu", mb);
            snprintf(fd, sizeof fd, "%d", fds[1]);
            char * av[] = { argv[0], (char *) "--child", (char *) a.c_str(), (char *) "--seconds", sec,
                            (char *) "--mb", mbs, (char *) "--fd", fd, nullptr };
            execv("/proc/self/exe", av);
            _exit(127);
        }
        close(fds[1]);
        Result res{};
        ssize_t got = read(fds[0], &res, sizeof res);
        close(fds[0]);
        int st = 0; waitpid(pid, &st, 0);
        if (got != (ssize_t) sizeof res) { printf("%-8s %-5s  (arm did not report, status %d)\n", a.c_str(), ai < arm.size() ? "fwd" : "rev", st); fflush(stdout); continue; }
        printf("%-8s %-5s %9.1f %8.0f %8.1f %8.1f %8.1f %9.1f  ", a.c_str(), ai < arm.size() ? "fwd" : "rev",
               res.cpu_pct, res.clk, res.gfx, res.allcore, res.socket, res.gbps);
        static const char * nm[7] = { "prochot", "spl", "fppt", "sppt", "thm_core", "thm_gfx", "thm_soc" };
        for (int i = 0; i < 7; i++) if (res.thr_pct[i] > 1.0) printf("%s=%.0f%% ", nm[i], res.thr_pct[i]);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
