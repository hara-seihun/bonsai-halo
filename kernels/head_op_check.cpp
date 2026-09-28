// Host exactness probe for the head's three ternary operand maps. No GPU, no model.
//
// The maps in kernels/halo_expand.hpp are portable outside device compilation, so this runs the
// kernel's own source on the CPU. It checks two things that together make the device change exact:
//
//   1. THE PLACEMENT. The deployed peel puts natural block element `4i + p` in byte lane `p` of
//      operand dword `i`, for every i in [0,32). That follows from the stored trit order in
//      src/halo_format.h, and it is checked here against `halo::decode_block`, the reference peel
//      the encoder is defined by. So arm 0 is the deployed map and not merely self-consistent.
//
//   2. THE EQUIVALENCE. Arms 1 and 2 produce the same 32 dwords as arm 0, bit for bit, over random
//      blocks and over the corner blocks (all -1, all 0, all +1, and every byte value 0..255 in
//      every qs position). Equal operand dwords with an unchanged matrix instruction, accumulator
//      seed and reduction order is what makes the logit bits equal on the device.
//
//   make kernels/head_op_check && kernels/head_op_check
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "halo_expand.hpp"

using namespace halo;

namespace {

struct Block {
    unsigned dw[6];
    unsigned tail;
    uint8_t trit[128];
};

Block make_block(const uint8_t trit[128], uint16_t scale_bits) {
    uint8_t qs[24], qh[2];
    encode_block(trit, qs, qh);
    Block b{};
    for (int d = 0; d < 6; d++)
        b.dw[d] = (unsigned) qs[4 * d] | ((unsigned) qs[4 * d + 1] << 8)
                | ((unsigned) qs[4 * d + 2] << 16) | ((unsigned) qs[4 * d + 3] << 24);
    b.tail = (unsigned) qh[0] | ((unsigned) qh[1] << 8) | ((unsigned) scale_bits << 16);
    // What the device will actually read back is the *decoded* trit, and the five-trit pack is not
    // injective on every input: encode/decode round-trips the 243 representable five-trit groups,
    // so the reference decode is the ground truth for placement, not the input array.
    uint8_t dec[128];
    decode_block(qs, qh, dec);
    std::memcpy(b.trit, dec, 128);
    return b;
}

int failures = 0;

void check_block(const Block & b, const char * what) {
    unsigned tr0[32], tr1[32], tr2[32], w[8];
    hx_expand_peel(b.dw, b.tail, tr0);
    hx_expand_perm(b.dw, b.tail, tr1);
    spread_pack(tr0, w);
    spread_expand(w, tr2);
    for (int i = 0; i < 32; i++) {
        for (int p = 0; p < 4; p++) {
            const unsigned got = (tr0[i] >> (8 * p)) & 0xffu;
            const unsigned want = b.trit[4 * i + p];
            if (got != want) {
                if (failures++ < 8)
                    printf("FAIL placement %s: dword %d byte %d = %u, element %d is %u\n",
                           what, i, p, got, 4 * i + p, want);
            }
        }
        if (tr1[i] != tr0[i]) {
            if (failures++ < 8)
                printf("FAIL arm1 %s: dword %d is %08x, peel gives %08x\n", what, i, tr1[i], tr0[i]);
        }
        if (tr2[i] != tr0[i]) {
            if (failures++ < 8)
                printf("FAIL arm2 %s: dword %d is %08x, peel gives %08x\n", what, i, tr2[i], tr0[i]);
        }
    }
}

} // namespace

int main() {
    // corner blocks: one trit value everywhere
    for (int v = 0; v < 3; v++) {
        uint8_t trit[128];
        for (int i = 0; i < 128; i++) trit[i] = (uint8_t) v;
        check_block(make_block(trit, 0x3c00), "uniform");
    }
    // every stored byte value, in every qs and qh position: drive the packed bytes directly, so the
    // arms are compared over all 256 inputs of every source lane rather than over reachable trits.
    for (unsigned v = 0; v < 256; v++) {
        Block b{};
        for (int d = 0; d < 6; d++) b.dw[d] = v * 0x01010101u;
        b.tail = v | (v << 8) | (0x3c00u << 16);
        uint8_t qs[24], qh[2];
        for (int i = 0; i < 24; i++) qs[i] = (uint8_t) v;
        qh[0] = qh[1] = (uint8_t) v;
        decode_block(qs, qh, b.trit);
        check_block(b, "byte sweep");
    }
    std::mt19937 rng(20260921);
    std::uniform_int_distribution<int> trit3(0, 2), bits16(0, 65535);
    for (int n = 0; n < 4000; n++) {
        uint8_t trit[128];
        for (int i = 0; i < 128; i++) trit[i] = (uint8_t) trit3(rng);
        check_block(make_block(trit, (uint16_t) bits16(rng)), "random");
    }
    // the spread word's own documented bit map, independently of the arms
    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 4; k++)
            for (int p = 0; p < 4; p++) {
                unsigned tr[32] = {}, w[8];
                tr[4 * j + k] = 1u << (8 * p);      // element 16j + 4k + p is trit 1
                spread_pack(tr, w);
                if (w[j] != (1u << (8 * p + 2 * k))) {
                    printf("FAIL spread bit map: element %d landed at %08x\n", 16 * j + 4 * k + p, w[j]);
                    failures++;
                }
            }
    if (failures) { printf("head_op_check: %d failures\n", failures); return 1; }
    printf("head_op_check: placement, arm 1, arm 2 and the spread bit map agree over "
           "3 uniform, 256 byte-sweep and 4000 random blocks\n");
    return 0;
}
