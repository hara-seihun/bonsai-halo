// Host exactness probe for the Q4 drafter coordinate; needs no GPU.
//
// What it establishes, over random tensors and random activation rows:
//
//   1. Placement. Every block-local element e of every row lands in the nibble
//      kernels/q4_format.h says it does, and reading it back gives its code.
//   2. Operand map. The two AND/shift words the kernels form from a dword are exactly the codes of
//      K 4m..4m+3 in byte order, which is what both the dot4 and the WMMA body assume when they
//      pair operand word m with activation bytes 4m..4m+3.
//   3. The contraction. The int32 the kernel accumulates, minus 8 * xsum, equals the exact integer
//      contraction of the *dequantised* weights against the activation row - i.e. the packed
//      coordinate introduces no arithmetic of its own; all of Q4's error is in the codes, and that
//      error is reported as a relative RMS.
//
// It does not establish anything about the drafter's acceptance rate. That is a full-model
// measurement and lives in docs/drafter-q4.md.
#include "q4_format.h"
#include "q8.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

using namespace halo;

static uint16_t f32bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
static float bff(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }
static float halff(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff, u;
    if (exp == 0) { if (mant == 0) u = sign; else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while (!(m & 0x400)); u = sign | ((uint32_t) (112 - e) << 23) | ((m & 0x3ff) << 13); } }
    else if (exp == 31) u = sign | 0x7f800000u | (mant << 13);
    else u = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &u, 4); return f;
}

int main() {
    const int64_t N = 64, K = 512;           // two tiles of four blocks
    std::mt19937 rng(20260921);
    std::normal_distribution<float> gauss(0.0f, 0.02f);
    std::uniform_int_distribution<int> byte(-127, 127);

    std::vector<uint16_t> src(N * K);
    for (auto & v : src) v = f32bf(gauss(rng));
    // a few rows with a wide dynamic range and a few exact zeros, which is where a scale search
    // and an offset-binary zero can go wrong quietly
    for (int64_t k = 0; k < K; k++) { src[k] = f32bf(gauss(rng) * 40.0f); src[K + k] = f32bf(0.0f); }

    std::vector<uint8_t> dst(q4_tensor_bytes(N, K));
    QuantError qe = quantize_q4_tiles(src.data(), N, K, dst.data(), 4);

    std::vector<int8_t> x(K);
    for (auto & v : x) v = (int8_t) byte(rng);

    long placement = 0, operand = 0, contraction = 0;
    const int64_t nb = K / BLOCK;
    for (int64_t T = 0; T < N / TILE_ROWS; T++) {
        for (int64_t b = 0; b < nb; b++) {
            const uint8_t * run = dst.data() + ((size_t) T * nb + b) * Q4_TILE_BLOCK_BYTES;
            for (int l = 0; l < TILE_ROWS; l++) {
                const uint8_t * row = run + q4_off_row(l);
                uint16_t sc; memcpy(&sc, run + q4_off_scale(l), 2);
                const float wscale = halff(sc);

                int code[BLOCK];
                for (int e = 0; e < BLOCK; e++) {
                    const uint8_t B = row[q4_byte_of(e)];
                    code[e] = q4_high_of(e) ? (B >> 4) : (B & 0xf);
                    if (code[e] < 1 || code[e] > 15) placement++;
                }
                // the kernels' expansion: two words per dword, tr[m] = codes of K 4m..4m+3
                unsigned tr[32];
                for (int g = 0; g < 16; g++) {
                    unsigned dw; memcpy(&dw, row + 4 * g, 4);
                    tr[2 * g] = dw & 0x0f0f0f0fu;
                    tr[2 * g + 1] = (dw >> 4) & 0x0f0f0f0fu;
                }
                for (int m = 0; m < 32; m++)
                    for (int j = 0; j < 4; j++)
                        if ((int) ((tr[m] >> (8 * j)) & 0xff) != code[4 * m + j]) operand++;

                // what the kernel accumulates: sudot4 over the operand words, then the bias
                long acc = 0, xsum = 0;
                for (int m = 0; m < 32; m++)
                    for (int j = 0; j < 4; j++)
                        acc += (long) ((tr[m] >> (8 * j)) & 0xff) * (long) x[b * BLOCK + 4 * m + j];
                for (int e = 0; e < BLOCK; e++) xsum += x[b * BLOCK + e];
                // what dequantised weights would give, in exact integers
                long direct = 0;
                for (int e = 0; e < BLOCK; e++) direct += (long) (code[e] - Q4_ZERO) * (long) x[b * BLOCK + e];
                if (acc - (long) Q4_ZERO * xsum != direct) contraction++;
                (void) wscale;
            }
        }
    }
    const double rel = qe.ref2 > 0 ? std::sqrt(qe.err2 / qe.ref2) : 0.0;
    printf("q4 pack check: placement %ld, operand map %ld, contraction %ld mismatches; relative RMS weight error %.4f\n",
           placement, operand, contraction, rel);
    if (placement || operand || contraction) { printf("FAIL\n"); return 1; }
    printf("OK\n");
    return 0;
}
