// Host exactness probe for the sequence projections' three stored code orders. No GPU, no model.
//
// The two projections read one weight word per K16 slice and turn it into the matrix instruction's
// operand. Three alphabets can carry that word in the same 2.000 bits per weight, and which one the
// image holds is a free choice of the packer:
//
//   spread  code k at bit 8*(k&3) + 2*(k>>2)     -> sixteen signed bytes for iu8
//   nibble  code k at bit 4*(k&7) + 2*(k>>3)     -> sixteen signed nibbles for iu4, by arithmetic
//   pair    a nine-valued code per nibble        -> the same sixteen nibbles, by one permute each
//
// This checks the three things that together make the device change exact:
//
//   1. THE REFERENCE. For the deployed word (code k at bits 2k, 0 = weight 0, 1 = +1, 3 = -1), the
//      iu4 operand nibble for K slot k must be 0, 1 and 0xf respectively, at nibble k of dword 0
//      for k < 8 and nibble k-8 of dword 1 for k >= 8. Both iu4 maps are checked against that
//      statement of the weights, not against each other alone.
//
//   2. THE EQUIVALENCE. `expand_i4_pair(pair_codes(w))` equals `expand_i4_nib(nibble_codes(w))`
//      bit for bit, over every code value in every one of the sixteen slots and over random words.
//      Equal operand dwords with an unchanged matrix instruction, accumulator seed, block sum and
//      scale drain is what makes the projected values equal on the device.
//
//   3. THE REVERSIBILITY. `seq_to_deployed(seq_from_deployed(w, o), o) == w` for all three orders,
//      which is what lets `reorder_codes` move a live image between any two of them without a
//      second copy of the weights.
//
//   make kernels/seq_op_check && kernels/seq_op_check
#include <cstdio>
#include <cstdlib>
#include <random>
#include "sequence_operands.hpp"

using namespace halo::ffnb;

namespace {

int failures = 0;

void fail(const char * what, uint32_t w, unsigned a, unsigned b) {
    if (++failures <= 8) printf("  FAIL %-28s w=%08x %08x != %08x\n", what, w, a, b);
}

// The weights a deployed word names, as the signed nibble the iu4 instruction reads.
unsigned nibble_of_code(unsigned c) { return c == 0u ? 0u : (c == 1u ? 1u : 0xfu); }

// The operand the deployed word must produce, built from the weights and nothing else.
void reference_operand(uint32_t w, unsigned out[2]) {
    out[0] = out[1] = 0;
    for (int k = 0; k < 16; ++k) {
        const unsigned n = nibble_of_code((w >> (2 * k)) & 3u);
        out[k >> 3] |= n << (4 * (k & 7));
    }
}

// Deployed words with only the legal codes 0, 1 and 3 in every slot.
uint32_t legal_word(std::mt19937 & rng) {
    static const unsigned code[3] = { 0u, 1u, 3u };
    uint32_t w = 0;
    for (int k = 0; k < 16; ++k) w |= code[rng() % 3u] << (2 * k);
    return w;
}

void check_word(uint32_t w) {
    unsigned ref[2];
    reference_operand(w, ref);
    const int2v nib = expand_i4_nib(nibble_codes(w));
    const int2v pai = expand_i4_pair(pair_codes(w));
    for (int j = 0; j < 2; ++j) {
        if ((unsigned) nib[j] != ref[j]) fail("nibble order vs weights", w, (unsigned) nib[j], ref[j]);
        if ((unsigned) pai[j] != ref[j]) fail("pair order vs weights", w, (unsigned) pai[j], ref[j]);
    }
    for (int o = SEQ_ORD_SPREAD; o <= SEQ_ORD_PAIR; ++o) {
        const uint32_t back = seq_to_deployed(seq_from_deployed(w, o), o);
        if (back != w) fail("order round trip", w, back, w);
    }
}

} // namespace

int main() {
    // Every code value in every slot, with the other fifteen slots held at each of the three codes.
    static const unsigned code[3] = { 0u, 1u, 3u };
    long swept = 0;
    for (int fill = 0; fill < 3; ++fill) {
        uint32_t base = 0;
        for (int k = 0; k < 16; ++k) base |= code[fill] << (2 * k);
        check_word(base);
        ++swept;
        for (int k = 0; k < 16; ++k)
            for (int c = 0; c < 3; ++c) {
                check_word((base & ~(3u << (2 * k))) | (code[c] << (2 * k)));
                ++swept;
            }
    }
    // Every value of one byte pair, so all 81 four-weight combinations a stored byte can carry.
    for (unsigned lo = 0; lo < 3u; ++lo)
        for (unsigned a = 0; a < 3u; ++a)
            for (unsigned b = 0; b < 3u; ++b)
                for (unsigned c = 0; c < 3u; ++c) {
                    check_word(code[lo] | (code[a] << 2) | (code[b] << 16) | (code[c] << 18));
                    ++swept;
                }
    std::mt19937 rng(20260921u);
    long random = 0;
    for (int i = 0; i < 200000; ++i) { check_word(legal_word(rng)); ++random; }

    printf("seq_op_check: %ld swept + %ld random words, %d failures\n", swept, random, failures);
    if (failures) { printf("  (only the first eight are printed)\n"); return 1; }
    printf("  nibble and pair orders agree with the weights and with each other; "
           "all three orders round trip\n");
    return 0;
}
