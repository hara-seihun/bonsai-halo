// Host exactness probe for the IU4 ternary operand map. No GPU.
//
// It runs kernels/mv_nibble.hpp itself -- the host arm of `nib_perm` reproduces the gfx1151
// selector rule and everything else is the same source the kernel compiles -- against
// src/halo_format.h's own encode/decode, which is the definition of the stored element order.
//
//   make kernels/mv_nibble_check && kernels/mv_nibble_check
//
// Three claims, in the order they would fail:
//   1. operand dword i, nibble j is the relabelled trit 2 - t[8i+j], for every element of a block.
//   2. code 8 = the trit pair (2,2) comes back as 0x00 from v_perm selector 8, so every pair of
//      trits in the alphabet survives -- checked over all 243 packed byte values.
//   3. the block's deployed accumulator `sum_k (t_k - 1) a_k` equals `xsum - (L + 16 H)` for the
//      nibble-split activations, which is the only thing mv_rows_t needs to be bit-identical.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "halo_format.h"
#include "mv_nibble.hpp"

using namespace halo;

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

static void load_block(const uint8_t * qs, const uint8_t * qh, uint4 & qa, uint2 & qb, unsigned & tail) {
    qa.x = *(const unsigned *) (qs + 0);  qa.y = *(const unsigned *) (qs + 4);
    qa.z = *(const unsigned *) (qs + 8);  qa.w = *(const unsigned *) (qs + 12);
    qb.x = *(const unsigned *) (qs + 16); qb.y = *(const unsigned *) (qs + 20);
    tail = (unsigned) qh[0] | ((unsigned) qh[1] << 8) | (0x3c00u << 16);   // scale half, unread here
}

int main() {
    int fail = 0;

    // ---- 1 and 2: element order and the nine-code alphabet, over random blocks --------------
    for (int trial = 0; trial < 4096 && !fail; trial++) {
        uint8_t trit[BLOCK];
        for (int e = 0; e < BLOCK; e++) trit[e] = (uint8_t) (trial < 3 ? (trial + e) % 3 : rnd() % 3u);
        uint8_t qs[QS_BYTES], qh[QH_BYTES], back[BLOCK];
        encode_block(trit, qs, qh);
        decode_block(qs, qh, back);
        for (int e = 0; e < BLOCK; e++)
            if (back[e] != trit[e]) { printf("FAIL round trip trial %d element %d\n", trial, e); fail = 1; break; }

        uint4 qa; uint2 qb; unsigned tail;
        load_block(qs, qh, qa, qb, tail);
        unsigned w[16];
        nib_block(qa, qb, tail, w);
        for (int i = 0; i < 16 && !fail; i++)
            for (int j = 0; j < 8; j++) {
                const unsigned got = (w[i] >> (4 * j)) & 0xfu;
                const unsigned want = 2u - trit[8 * i + j];
                if (got != want) {
                    printf("FAIL operand trial %d dword %d nibble %d: got %u want %u (trit %u)\n",
                           trial, i, j, got, want, trit[8 * i + j]);
                    fail = 1; break;
                }
            }
    }

    // ---- every packed byte value reaches the operand, including the (2,2) pair ---------------
    {
        int seen[9] = {0};
        for (unsigned t0 = 0; t0 < 3 && !fail; t0++)
            for (unsigned t1 = 0; t1 < 3; t1++)
                for (unsigned t2 = 0; t2 < 3; t2++)
                    for (unsigned t3 = 0; t3 < 3; t3++)
                        for (unsigned t4 = 0; t4 < 3; t4++) {
                            uint8_t t[5] = { (uint8_t) t0, (uint8_t) t1, (uint8_t) t2, (uint8_t) t3, (uint8_t) t4 };
                            const unsigned b = pack5(t);
                            const nib2 p = nib_pair(b | (b << 16));
                            const NibPeel r = nib_peel(p);
                            const unsigned A = (nib_word(r.a) >> 8) & 0xffu, B = (nib_word(r.b) >> 8) & 0xffu, C = (nib_word(r.c) >> 8) & 0xffu;
                            if (A != 3 * t0 + t1 || B != 3 * t2 + t3 || C != t4) {
                                printf("FAIL codes for %u%u%u%u%u: A %u B %u C %u\n", t0, t1, t2, t3, t4, A, B, C);
                                fail = 1;
                            }
                            seen[A] = 1;
                            const unsigned sel = nib_perm(nib_word(r.b), nib_word(r.a), NIB_AB);
                            const unsigned ops = nib_codes(sel);
                            if ((ops & 0xfu) != 2u - t0 || ((ops >> 4) & 0xfu) != 2u - t1) {
                                printf("FAIL expand for pair (%u,%u) code %u\n", t0, t1, A); fail = 1;
                            }
                        }
        for (int c = 0; c < 9; c++) if (!seen[c]) { printf("FAIL code %d never produced\n", c); fail = 1; }
    }

    // ---- 3: the accumulator identity the kernel relies on ------------------------------------
    for (int trial = 0; trial < 4096 && !fail; trial++) {
        uint8_t trit[BLOCK]; int8_t a[BLOCK];
        for (int e = 0; e < BLOCK; e++) { trit[e] = (uint8_t) (rnd() % 3u); a[e] = (int8_t) ((int) (rnd() % 255u) - 127); }
        if (trial == 0) for (int e = 0; e < BLOCK; e++) a[e] = (int8_t) (e & 1 ? 127 : -127);
        uint8_t qs[QS_BYTES], qh[QH_BYTES];
        encode_block(trit, qs, qh);
        uint4 qa; uint2 qb; unsigned tail;
        load_block(qs, qh, qa, qb, tail);
        unsigned w[16];
        nib_block(qa, qb, tail, w);

        int want = 0, xsum = 0;
        for (int e = 0; e < BLOCK; e++) { want += ((int) trit[e] - 1) * (int) a[e]; xsum += (int) a[e]; }

        int L = 0, H = 0;
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 8; j++) {
                const int v = (int) ((w[i] >> (4 * j)) & 0xfu);          // weight nibble, unsigned
                const int byte = (int) (uint8_t) a[8 * i + j];
                L += v * (byte & 0xf);                                   // low nibble, unsigned
                H += v * (((int8_t) (byte & 0xf0)) >> 4);                // high nibble, signed
            }
        const int got = xsum - (L + (H << 4));
        if (got != want) { printf("FAIL dot trial %d: got %d want %d\n", trial, got, want); fail = 1; }
    }

    printf(fail ? "mv_nibble_check: FAIL\n" : "mv_nibble_check: ok (operand order, nine-code alphabet, accumulator identity)\n");
    return fail;
}
