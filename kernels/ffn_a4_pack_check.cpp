// Host check of the optimized FFN weight path, against the shipped headers themselves:
// HALO block bytes -> two-bit codes -> lane-major words, pair codes and dense five-trit bytes ->
// the operand the matrix instruction sees.
//
// Modes 6, 7 and 8 claim to be modes 1, 3 and 2 with a different schedule and a different weight
// storage. That claim rests on every storage decoding to the same ternary weights, which is what
// this checks, on random weights through the real halo::encode_block. The packers and the peel are
// the shipped code compiled for the host; only v_perm_b32 is emulated, from the selector table
// Kelana measured on gfx1151 over all 256 selector values against twelve source vectors
// (research/ffn/batched/dense-consumer/results/perm-semantics.json). So this checks the packers,
// the slot maps and the peel arithmetic. It does not check the hardware permute, and it is not a
// substitute for comparing whole-FFN outputs on the device.
//
// Build and run (it is not in the Makefile's source lists, so it never enters the engine build):
//
//   hipcc -O2 -std=c++17 -Isrc -Ikernels -o /tmp/ffn_a4_pack_check kernels/ffn_a4_pack_check.cpp
//   /tmp/ffn_a4_pack_check 3000        # 3000 HALO block-runs = 12.3 M weights, about 0.15 s
//
// Exit status is nonzero if any representation disagrees with the weights it was built from.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>
#include "halo_format.h"
#include "ffn_a4_operands.hpp"

using namespace halo;
using namespace halo::ffnb;

// ---- v_perm_b32 on gfx1151 ----
//   sel 0..3 -> argB byte 0..3     sel 4..7  -> argA byte 0..3
//   sel 8..11-> sign of source byte 2*(sel-8)+1, replicated
//   sel 12   -> 0x00               sel 13.. -> 0xff
static uint32_t vperm(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned s = (sel >> (8 * i)) & 0xff;
        unsigned v;
        if (s < 4) v = (b >> (8 * s)) & 0xff;
        else if (s < 8) v = (a >> (8 * (s - 4))) & 0xff;
        else if (s < 12) {
            const unsigned src = 2 * (s - 8) + 1;
            const unsigned byte = src < 4 ? (b >> (8 * src)) & 0xff : (a >> (8 * (src - 4))) & 0xff;
            v = (byte & 0x80) ? 0xffu : 0x00u;
        } else if (s == 12) v = 0;
        else v = 0xff;
        out |= v << (8 * i);
    }
    return out;
}
static uint32_t expand_codes_h(uint32_t sel) { return vperm(PERM_S0, PERM_S1, sel); }
static void expand_pair_h(uint32_t d, uint32_t out[2]) {
    out[0] = expand_codes_h(d & 0x0f0f0f0fu);
    out[1] = expand_codes_h((d >> 4) & 0x0f0f0f0fu);
}
// the device peel and gathers, on plain integers
static void peel_pair_h(const unsigned b[2], unsigned A[2], unsigned B[2], unsigned c[2]) {
    for (int i = 0; i < 2; ++i) {
        unsigned m = b[i] * 9u; A[i] = m >> 8;
        m = (m & 0xff) * 9u;    B[i] = m >> 8;
        m = (m & 0xff) * 3u;    c[i] = m >> 8;
    }
}
static uint32_t pack2(const unsigned v[2]) { return (v[0] & 0xffff) | ((v[1] & 0xffff) << 16); }
struct DD { unsigned A0[2], B0[2], C0[2], A1[2], B1[2], C1[2]; };
static DD peel_dword_h(uint32_t d) {
    DD r;
    const unsigned ev[2] = { d & 0xffu, (d >> 16) & 0xffu };
    const unsigned od[2] = { (d >> 8) & 0xffu, (d >> 24) & 0xffu };
    peel_pair_h(ev, r.A0, r.B0, r.C0);
    peel_pair_h(od, r.A1, r.B1, r.C1);
    return r;
}
static void dense_slice_h(uint32_t src, uint32_t & cc, uint32_t out[2]) {
    const DD r = peel_dword_h(src);
    const uint32_t g = vperm(pack2(r.C1), pack2(r.C0), 0x06040200u);
    cc = g * 3u + (g >> 8);
    out[0] = expand_codes_h(vperm(pack2(r.B0), pack2(r.A0), 0x06040200u));
    out[1] = expand_codes_h(vperm(pack2(r.B1), pack2(r.A1), 0x06040200u));
}
static void dense_slice_cc_h(const uint32_t c[4], uint32_t out[2]) {
    out[0] = expand_codes_h(vperm(c[1], c[0], 0x06040200u));
    out[1] = expand_codes_h(vperm(c[3], c[2], 0x06040200u));
}
static void dense_slice7_h(uint32_t cc4, uint32_t cc5, uint32_t s2, uint32_t out[2]) {
    const unsigned bb[2] = { s2 & 0xffu, (s2 >> 8) & 0xffu };
    unsigned A[2], B[2], c[2];
    peel_pair_h(bb, A, B, c);
    out[0] = expand_codes_h(vperm(cc5, cc4, 0x06040200u));
    out[1] = expand_codes_h(vperm(pack2(B), pack2(A), 0x06040200u));
}
// what expand_i8 must produce, as a signed byte, and expand_i4 as an int4 nibble
static unsigned nib(int w) { return w == 0 ? 0u : (w > 0 ? 1u : 0xfu); }
static unsigned byt(int w) { return w == 0 ? 0u : (w > 0 ? 1u : 0xffu); }

int main(int argc, char ** argv) {
    const int runs = argc > 1 ? atoi(argv[1]) : 2000;   // each run is a 32-row HALO block-run
    std::mt19937 rng(19073);
    std::uniform_int_distribution<int> pick(0, 2);
    long weights = 0, bad_code = 0, bad_pair = 0, bad_dense = 0, bad_i8 = 0, bad_lane = 0;

    std::vector<uint8_t> run(TILE_BLOCK_BYTES);
    std::vector<int8_t> wt(32 * 128);
    for (int r = 0; r < runs; ++r) {
        // one HALO block-run: 32 rows of 128 trits, encoded exactly as the engine stores them
        for (int lane = 0; lane < 32; ++lane) {
            uint8_t trit[128];
            for (int i = 0; i < 128; ++i) { trit[i] = (uint8_t) pick(rng); wt[lane * 128 + i] = int8_t(int(trit[i]) - 1); }
            uint8_t qs[24], qh[2];
            halo::encode_block(trit, qs, qh);
            memcpy(run.data() + tile_off_qs_a(lane), qs, 16);
            memcpy(run.data() + tile_off_qs_b(lane), qs + 16, 8);
            uint8_t * tail = run.data() + tile_off_tail(lane);
            tail[0] = qh[0]; tail[1] = qh[1]; tail[2] = 0x34; tail[3] = 0x12;   // scale bits
        }

        for (int lane = 0; lane < 32; ++lane) {
            const uint4 qa = *(const uint4 *) (run.data() + tile_off_qs_a(lane));
            const uint2 qb = *(const uint2 *) (run.data() + tile_off_qs_b(lane));
            const unsigned tl = *(const unsigned *) (run.data() + tile_off_tail(lane));
            uint32_t w[8];
            halo_block_codes(qa, qb, tl, w);                       // shipped
            const int8_t * ref = &wt[lane * 128];

            for (int e = 0; e < 128; ++e) {
                const unsigned c = code_at(w, e);                  // shipped
                const int back = c == 0 ? 0 : (c == 1 ? 1 : -1);
                if (back != ref[e]) ++bad_code;
            }
            // the two-bit word is the same word in either layout, so lane-major storage is the
            // identity check: word s must hold codes of elements 16s..16s+15 in bit order
            for (int s = 0; s < 8; ++s)
                for (int j = 0; j < 16; ++j) {
                    const unsigned c = (w[s] >> (2 * j)) & 3u;
                    const int back = c == 0 ? 0 : (c == 1 ? 1 : -1);
                    if (back != ref[s * 16 + j]) ++bad_lane;
                }
            // expand_i8's byte table, applied to the stored codes
            for (int s = 0; s < 8; ++s)
                for (int j = 0; j < 16; ++j) {
                    const unsigned c = (w[s] >> (2 * j)) & 3u;
                    const unsigned got = c == 0 ? 0u : (c == 1 ? 1u : 0xffu);
                    if (got != byt(ref[s * 16 + j])) ++bad_i8;
                }

            uint32_t want[8][2];
            for (int s = 0; s < 8; ++s) {
                want[s][0] = want[s][1] = 0;
                for (int j = 0; j < 8; ++j) want[s][0] |= nib(ref[s * 16 + j]) << (4 * j);
                for (int j = 0; j < 8; ++j) want[s][1] |= nib(ref[s * 16 + 8 + j]) << (4 * j);
            }

            uint32_t pc[8];
            pack_pair_block(w, pc);                                // shipped
            for (int s = 0; s < 8; ++s) {
                uint32_t got[2];
                expand_pair_h(pc[s], got);
                if (got[0] != want[s][0] || got[1] != want[s][1]) ++bad_pair;
            }

            uint8_t db[26];
            pack_dense_block(w, db);                               // shipped
            uint32_t s0[4], s1[2], s2 = db[24] | (unsigned(db[25]) << 8);
            memcpy(s0, db, 16);
            memcpy(s1, db + 16, 8);
            uint32_t got[8][2], cc[6];
            for (int d = 0; d < 4; ++d) dense_slice_h(s0[d], cc[d], got[d]);
            dense_slice_cc_h(cc, got[6]);
            for (int d = 0; d < 2; ++d) dense_slice_h(s1[d], cc[4 + d], got[4 + d]);
            dense_slice7_h(cc[4], cc[5], s2, got[7]);
            for (int s = 0; s < 8; ++s)
                if (got[s][0] != want[s][0] || got[s][1] != want[s][1]) ++bad_dense;
            weights += 128;
        }
    }
    printf("weights checked            %ld\n", weights);
    printf("two-bit code errors        %ld\n", bad_code);
    printf("lane-major word errors     %ld\n", bad_lane);
    printf("IU8 byte table errors      %ld\n", bad_i8);
    printf("pair-code slice errors     %ld\n", bad_pair);
    printf("dense five-trit errors     %ld\n", bad_dense);
    return (bad_code || bad_lane || bad_i8 || bad_pair || bad_dense) ? 1 : 0;
}
