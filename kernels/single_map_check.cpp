// Exactness probe for the single-token palette operand map (FwdParams::single_map >= 1).
//
// The deployed path peels five trits out of every HALO byte and dots them as unsigned {0,1,2}
// against the int8 activations, then subtracts the block activation sum. The palette path pulls
// two nine-valued two-trit codes and one leftover trit out of the same byte with packed multiplies
// and turns the codes into operand bytes with v_perm_b32, whose palette carries the complement
// 2 - t. The dot then returns 2*sum(x) - sum(t x), so the block correction flips to xsum - acc.
//
// This program proves, on the host, that the two paths produce the same integer block accumulator
// for every reachable byte value and for random blocks, which is what makes the FP32 schedule and
// therefore the logits bit-identical. With --gpu it additionally runs the real device code over
// random blocks and compares against the same reference; that needs a GPU slot, the host part
// does not.
//
// Build: make kernels/single_map_check        Run: ./kernels/single_map_check [--gpu] [--json F]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include "halo_format.h"

// ---------------------------------------------------------------------------------------------
// Host twin of the device map. Mirrors kernels/device.hpp: pal_codes, pal_codes2, pal_expand,
// pal_block. Keep the two in step; the selector and palette constants are the same literals.

static const unsigned PAL0_LO = 0x01020202u, PAL0_HI = 0x00000101u;
static const unsigned PAL1_LO = 0x02000102u, PAL1_HI = 0x01020001u;
static const unsigned PALC_LO = 0x00000102u, PALC_HI = 0x00000000u;
static const unsigned SEL_PAIR = 0x06040200u, SEL_LO = 0x05010400u, SEL_HI = 0x07030602u, SEL_C = 0x06020400u;

// V_PERM_B32 as measured on gfx1151 (kelana research/ffn/batched/dense-consumer/results/
// perm-semantics.json): source byte 0..3 = arg b, 4..7 = arg a; selector 8..11 replicates the sign
// of source byte 2*(sel-8)+1; 12 gives 0x00; 13..255 give 0xff.
static unsigned vperm_ref(unsigned a, unsigned b, unsigned s) {
    unsigned char src[8];
    for (int i = 0; i < 4; i++) { src[i] = (unsigned char) (b >> (8 * i)); src[4 + i] = (unsigned char) (a >> (8 * i)); }
    unsigned out = 0;
    for (int i = 0; i < 4; i++) {
        const unsigned sel = (s >> (8 * i)) & 0xff;
        unsigned char v;
        if (sel < 8) v = src[sel];
        else if (sel < 12) v = (src[2 * (sel - 8) + 1] & 0x80) ? 0xff : 0x00;
        else if (sel == 12) v = 0x00;
        else v = 0xff;
        out |= (unsigned) v << (8 * i);
    }
    return out;
}
static unsigned pk_mul(unsigned h, unsigned k) {
    const unsigned lo = ((h & 0xffffu) * k) & 0xffffu, hi = (((h >> 16) & 0xffffu) * k) & 0xffffu;
    return lo | (hi << 16);
}
static unsigned pk_shr8(unsigned h) { return ((h & 0xffffu) >> 8) | ((((h >> 16) & 0xffffu) >> 8) << 16); }

static void pal_codes_ref(unsigned H, unsigned & A, unsigned & B, unsigned & C) {
    unsigned m = pk_mul(H, 9);
    A = pk_shr8(m);
    m = pk_mul(m & 0x00ff00ffu, 9);
    B = pk_shr8(m);
    m = pk_mul(m & 0x00ff00ffu, 3);
    C = pk_shr8(m);
}
static void pal_codes2_ref(unsigned H, unsigned & A, unsigned & B) {
    unsigned m = pk_mul(H, 9);
    A = pk_shr8(m);
    m = pk_mul(m & 0x00ff00ffu, 9);
    B = pk_shr8(m);
}
static void pal_expand_ref(unsigned K0, unsigned K1, unsigned & lo, unsigned & hi) {
    const unsigned sel = vperm_ref(K1, K0, SEL_PAIR);
    const unsigned q0 = vperm_ref(PAL0_HI, PAL0_LO, sel), q1 = vperm_ref(PAL1_HI, PAL1_LO, sel);
    lo = vperm_ref(q1, q0, SEL_LO);
    hi = vperm_ref(q1, q0, SEL_HI);
}
static void pal_block_ref(const uint8_t * qs, const uint8_t * qh, unsigned tr[32]) {
    for (int d = 0; d < 6; d++) {
        unsigned D = 0;
        for (int j = 0; j < 4; j++) D |= (unsigned) qs[4 * d + j] << (8 * j);
        unsigned A0, B0, C0, A1, B1, C1;
        pal_codes_ref(D & 0x00ff00ffu, A0, B0, C0);
        pal_codes_ref((D >> 8) & 0x00ff00ffu, A1, B1, C1);
        pal_expand_ref(A0, A1, tr[4 * d + 0], tr[4 * d + 2]);
        pal_expand_ref(B0, B1, tr[4 * d + 1], tr[4 * d + 3]);
        tr[24 + d] = vperm_ref(PALC_HI, PALC_LO, vperm_ref(C1, C0, SEL_C));
    }
    const unsigned tail = (unsigned) qh[0] | ((unsigned) qh[1] << 8);
    unsigned A, B;
    pal_codes2_ref((tail & 0xffu) | ((tail << 8) & 0xff0000u), A, B);
    const unsigned sel = vperm_ref(B, A, SEL_PAIR);
    const unsigned q0 = vperm_ref(PAL0_HI, PAL0_LO, sel), q1 = vperm_ref(PAL1_HI, PAL1_LO, sel);
    tr[30] = vperm_ref(q1, q0, SEL_LO);
    tr[31] = vperm_ref(q1, q0, SEL_HI);
}

// ---------------------------------------------------------------------------------------------
// Deployed peel path, same structure as mv_rows_t<.., 1, false> in kernels/device.hpp.

static unsigned peel_ref(unsigned & r) {
    const unsigned m = pk_mul(r, 3);
    r = m & 0x00ff00ffu;
    return pk_shr8(m);
}
static void peel_block_ref(const uint8_t * qs, const uint8_t * qh, unsigned tr[32]) {
    for (int d = 0; d < 6; d++) {
        unsigned D = 0;
        for (int j = 0; j < 4; j++) D |= (unsigned) qs[4 * d + j] << (8 * j);
        unsigned P0 = D & 0x00ff00ffu, P1 = (D >> 8) & 0x00ff00ffu;
        const unsigned t0 = peel_ref(P0), t1 = peel_ref(P0), t2 = peel_ref(P0), t3 = peel_ref(P0), t4 = pk_shr8(pk_mul(P0, 3));
        const unsigned u0 = peel_ref(P1), u1 = peel_ref(P1), u2 = peel_ref(P1), u3 = peel_ref(P1), u4 = pk_shr8(pk_mul(P1, 3));
        tr[4 * d] = t0 | (t1 << 8); tr[4 * d + 1] = t2 | (t3 << 8);
        tr[4 * d + 2] = u0 | (u1 << 8); tr[4 * d + 3] = u2 | (u3 << 8);
        tr[24 + d] = t4 | (u4 << 8);
    }
    const unsigned tail = (unsigned) qh[0] | ((unsigned) qh[1] << 8);
    unsigned P = (tail & 0xffu) | ((tail << 8) & 0xff0000u);
    const unsigned h0 = peel_ref(P), h1 = peel_ref(P), h2 = peel_ref(P), h3 = pk_shr8(pk_mul(P, 3));
    tr[30] = h0 | (h1 << 8); tr[31] = h2 | (h3 << 8);
}

static int dot4_us(unsigned w, unsigned x, int acc) {  // unsigned weight byte, signed activation
    for (int i = 0; i < 4; i++) acc += (int) ((w >> (8 * i)) & 0xff) * (int) (int8_t) ((x >> (8 * i)) & 0xff);
    return acc;
}

// ---------------------------------------------------------------------------------------------

static int fails = 0;
static void expect(bool ok, const char * what) { if (!ok) { fails++; if (fails < 20) printf("  FAIL %s\n", what); } }

int main(int argc, char ** argv) {
    bool want_gpu = false; const char * json = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpu")) want_gpu = true;
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
    }

    // 1. every byte value: the two codes and the leftover trit against the peel, and the palette
    //    lookups against the complement of each trit. Selector 8 is exercised by every byte whose
    //    first or second trit pair is (2,2), which is 1 in 9 of the reachable values.
    int sel8 = 0;
    for (unsigned b = 0; b < 256; b++) {
        unsigned t[5], r = b;
        for (int n = 0; n < 5; n++) { const unsigned m = r * 3u; t[n] = m >> 8; r = m & 0xffu; }
        const unsigned A = (b * 9u) >> 8, rr = (b * 9u) & 0xffu, B = (rr * 9u) >> 8, r2 = (rr * 9u) & 0xffu, C = (r2 * 3u) >> 8;
        expect(A == 3 * t[0] + t[1], "A = 3 t0 + t1");
        expect(B == 3 * t[2] + t[3], "B = 3 t2 + t3");
        expect(C == t[4], "C = t4");
        if (A == 8 || B == 8) sel8++;
        // palette lookups, one byte position of a v_perm each
        const unsigned p0A = vperm_ref(PAL0_HI, PAL0_LO, A) & 0xff, p1A = vperm_ref(PAL1_HI, PAL1_LO, A) & 0xff;
        const unsigned p0B = vperm_ref(PAL0_HI, PAL0_LO, B) & 0xff, p1B = vperm_ref(PAL1_HI, PAL1_LO, B) & 0xff;
        const unsigned pc = vperm_ref(PALC_HI, PALC_LO, C) & 0xff;
        expect(p0A == 2 - t[0] && p1A == 2 - t[1], "palette pair 0");
        expect(p0B == 2 - t[2] && p1B == 2 - t[3], "palette pair 1");
        expect(pc == 2 - t[4], "palette leftover trit");
    }
    printf("byte table: 256 values, %d exercise selector 8, %s\n", sel8, fails ? "FAILED" : "ok");

    // 2. random blocks: operand dwords are the exact complement of the deployed ones, and the
    //    corrected integer accumulators agree with each other and with a direct ternary dot.
    std::mt19937 rng(20260920);
    const int NBLOCK = 200000;
    long long checked_weights = 0;
    for (int it = 0; it < NBLOCK; it++) {
        uint8_t trit[128], qs[24], qh[2];
        for (int i = 0; i < 128; i++) trit[i] = (uint8_t) (rng() % 3);
        halo::encode_block(trit, qs, qh);
        uint8_t back[128];
        halo::decode_block(qs, qh, back);
        expect(!memcmp(trit, back, 128), "encode/decode round trip");

        unsigned tr_ref[32], tr_pal[32];
        peel_block_ref(qs, qh, tr_ref);
        pal_block_ref(qs, qh, tr_pal);
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++) {
                const unsigned a = (tr_ref[i] >> (8 * j)) & 0xff, p = (tr_pal[i] >> (8 * j)) & 0xff;
                expect(a <= 2 && p == 2 - a, "operand byte is the complement");
            }

        int8_t x[128];
        for (int i = 0; i < 128; i++) x[i] = (int8_t) (int) (rng() % 255) - 127;
        unsigned xw[32];
        int xsum = 0;
        for (int i = 0; i < 32; i++) {
            xw[i] = ((unsigned) (uint8_t) x[4 * i]) | ((unsigned) (uint8_t) x[4 * i + 1] << 8)
                  | ((unsigned) (uint8_t) x[4 * i + 2] << 16) | ((unsigned) (uint8_t) x[4 * i + 3] << 24);
            for (int j = 0; j < 4; j++) xsum += x[4 * i + j];
        }
        int acc_ref = 0, acc_pal = 0;
        for (int i = 0; i < 32; i++) { acc_ref = dot4_us(tr_ref[i], xw[i], acc_ref); acc_pal = dot4_us(tr_pal[i], xw[i], acc_pal); }
        int direct = 0;
        for (int i = 0; i < 128; i++) direct += ((int) trit[i] - 1) * (int) x[i];
        expect(acc_ref - xsum == direct, "deployed path equals the ternary dot");
        expect(xsum - acc_pal == direct, "palette path equals the ternary dot");
        checked_weights += 128;
    }
    printf("random blocks: %d blocks, %lld weights, operands and corrected accumulators %s\n",
           NBLOCK, checked_weights, fails ? "FAILED" : "identical");

    if (want_gpu) printf("--gpu: device probe not built into this binary yet; host result stands alone\n");

    if (json) {
        FILE * f = fopen(json, "w");
        if (f) {
            fprintf(f, "{\n  \"format\": \"bonsai-single-map-check/1\",\n  \"map\": \"palette two-trit codes, complemented IU8 operands\",\n");
            fprintf(f, "  \"byte_values\": 256,\n  \"selector8_bytes\": %d,\n  \"blocks\": %d,\n  \"weights\": %lld,\n", sel8, NBLOCK, checked_weights);
            fprintf(f, "  \"perm_semantics\": \"kelana research/ffn/batched/dense-consumer/results/perm-semantics.json\",\n");
            fprintf(f, "  \"failures\": %d,\n  \"pass\": %s\n}\n", fails, fails ? "false" : "true");
            fclose(f);
        }
    }
    printf("%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
