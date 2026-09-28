// What `warp_sum` actually leaves in each lane, measured rather than modelled.
//
// `warp_sum` (kernels/device.hpp) is five DPP stages: `quad_perm(1,0,3,2)`, `quad_perm(2,3,0,1)`,
// `row_ror 4`, `row_ror 8`, `permlanex16`. Each stage is `v += dpp(v)`, so the value a lane ends
// with depends on which lane every stage handed it, and floating-point addition is not associative:
// the wave does NOT end with one bit pattern. `gdn_token` then applies lane `l`'s value to the
// state columns lane `l` owns, which makes the per-lane rounding part of the deployed numerical map.
//
// Any rewrite that moves a column between lanes has to reproduce that map. This probe is what makes
// that checkable instead of argued. It does two things, both on the device:
//
//   1. Measures each DPP stage as a lane permutation, by running it on the lane index.
//   2. Composes `warp_sum` on the host from those measured permutations and checks the composition
//      against the device's own `warp_sum` over random float32 waves, bit for bit.
//
// Then it prints the equivalence classes: how many distinct bit patterns the wave holds and which
// lanes hold each one. Reading a manual is not the instrument; a rotation's direction and the
// resulting pairing of quad sums are exactly the kind of thing a manual gets read wrong.
//
//   make tools/gdn_warpsum_probe && tools/run-batch-compare --exec tools/gdn_warpsum_probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "device.hpp"

using namespace halo;

// One stage's permutation: lane l reports which lane's value it received.
__global__ void perms(float * out) {
    const int lane = threadIdx.x & 31;
    const float v = (float) lane;
    out[0 * 32 + lane] = dppf<0xB1>(v);
    out[1 * 32 + lane] = dppf<0x4E>(v);
    out[2 * 32 + lane] = dppf<0x124>(v);
    out[3 * 32 + lane] = dppf<0x128>(v);
    out[4 * 32 + lane] = xrow16(v);
}

__global__ void sums(const float * in, float * out) {
    const int lane = threadIdx.x & 31;
    out[lane] = warp_sum(in[lane]);
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

int main() {
    float * dp, * din, * dout;
    if (hipMalloc(&dp, 5 * 32 * 4) != hipSuccess || hipMalloc(&din, 128) != hipSuccess ||
        hipMalloc(&dout, 128) != hipSuccess) { fprintf(stderr, "allocation failed\n"); return 1; }
    float hp[5 * 32];
    perms<<<1, 32>>>(dp);
    if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "perm launch failed\n"); return 1; }
    (void) hipMemcpy(hp, dp, sizeof hp, hipMemcpyDeviceToHost);
    const char * names[5] = { "quad_perm(1,0,3,2)", "quad_perm(2,3,0,1)", "row_ror 4", "row_ror 8", "permlanex16" };
    int src[5][32];
    for (int s = 0; s < 5; s++) {
        printf("%-20s lane <- ", names[s]);
        for (int l = 0; l < 32; l++) { src[s][l] = (int) hp[s * 32 + l]; printf("%d%s", src[s][l], l == 31 ? "\n" : ","); }
    }

    // Compose warp_sum from the measured permutations, as a symbolic float32 evaluation.
    std::mt19937 rng(20260921);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int matched = 0, cases = 0;
    std::map<std::string, std::vector<int>> classes;
    for (int c = 0; c < 2000; c++) {
        float h[32], got[32];
        for (int i = 0; i < 32; i++) h[i] = nd(rng) * (c % 3 == 0 ? 1e4f : c % 3 == 1 ? 1.0f : 1e-4f);
        if (c % 5 == 0) h[rng() % 32] *= 1e6f;
        (void) hipMemcpy(din, h, 128, hipMemcpyHostToDevice);
        sums<<<1, 32>>>(din, dout);
        if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "sum launch failed\n"); return 1; }
        (void) hipMemcpy(got, dout, 128, hipMemcpyDeviceToHost);
        float v[32];
        for (int l = 0; l < 32; l++) v[l] = h[l];
        for (int s = 0; s < 5; s++) {
            float n[32];
            for (int l = 0; l < 32; l++) n[l] = v[l] + v[src[s][l]];
            for (int l = 0; l < 32; l++) v[l] = n[l];
        }
        bool all = true;
        for (int l = 0; l < 32; l++) if (bits(v[l]) != bits(got[l])) all = false;
        matched += all;
        cases++;
        if (c == 0) {
            std::map<uint32_t, std::vector<int>> by;
            for (int l = 0; l < 32; l++) by[bits(got[l])].push_back(l);
            printf("\ncase 0: %zu distinct bit patterns across the wave\n", by.size());
            for (auto & kv : by) {
                printf("   %08x  lanes", kv.first);
                for (int l : kv.second) printf(" %d", l);
                printf("\n");
            }
        }
        // Which lanes agree, over every case: the equivalence classes of the numerical map.
        std::map<uint32_t, int> rep;
        std::string sig(32, '?');
        int next = 0;
        for (int l = 0; l < 32; l++) {
            auto it = rep.find(bits(got[l]));
            if (it == rep.end()) { rep[bits(got[l])] = next; sig[l] = (char) ('0' + next); next++; }
            else sig[l] = (char) ('0' + it->second);
        }
        classes[sig].push_back(c);
    }
    printf("\ncomposition of the measured permutations reproduces device warp_sum: %d/%d cases\n", matched, cases);
    printf("lane -> class signatures observed (lane 0 is always class 0):\n");
    for (auto & kv : classes) printf("   %s   %zu cases\n", kv.first.c_str(), kv.second.size());
    const bool ok = matched == cases;
    printf("verdict: %s\n", ok ? "PASS the permutation composition is the numerical map" : "FAIL");
    return ok ? 0 : 1;
}
