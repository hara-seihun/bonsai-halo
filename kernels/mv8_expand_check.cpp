// Host acceptance for the eight-row matrix body's operand map: the incremental expansion in
// `kernels/mv_wmma8.hpp` must produce the deployed operand dwords, byte for byte.
//
// The body expands one source dword straight into the K slice that consumes it instead of building
// all thirty-two operand dwords first, which is where twenty of its registers come from. That is a
// reordering of `hx_expand_perm`, and a reordering is exactly the kind of change that is right on
// paper and off by one byte lane in the tree. This checks it against two independent references:
// the deployed gather, and `decode_block` from the format header, which is what the packer's own
// trits are.
//
//   make kernels/mv8_expand_check && kernels/mv8_expand_check
#include "halo_expand.hpp"
#include "../src/halo_format.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

using namespace halo;

// The device body's per-dword expansion, in the host form of the same three helpers.
static void mv8_expand_dw_host(unsigned d, unsigned A[4], unsigned & fifth) {
    const unsigned m0 = hx_mul3(d & HX_LOW);
    const unsigned m1 = hx_mul3(m0 & HX_LOW);
    const unsigned m2 = hx_mul3(m1 & HX_LOW);
    const unsigned m3 = hx_mul3(m2 & HX_LOW);
    const unsigned m4 = hx_mul3(m3 & HX_LOW);
    const unsigned n0 = hx_mul3((d >> 8) & HX_LOW);
    const unsigned n1 = hx_mul3(n0 & HX_LOW);
    const unsigned n2 = hx_mul3(n1 & HX_LOW);
    const unsigned n3 = hx_mul3(n2 & HX_LOW);
    const unsigned n4 = hx_mul3(n3 & HX_LOW);
    A[0] = hx_gather(m1, m0);
    A[1] = hx_gather(m3, m2);
    A[2] = hx_gather(n1, n0);
    A[3] = hx_gather(n3, n2);
    fifth = hx_gather(n4, m4);
}
static void mv8_expand_tail_host(unsigned tail, unsigned & t0, unsigned & t1) {
    const unsigned P = (tail & 0xffu) | ((tail << 8) & 0xff0000u);
    const unsigned h0 = hx_mul3(P);
    const unsigned h1 = hx_mul3(h0 & HX_LOW);
    const unsigned h2 = hx_mul3(h1 & HX_LOW);
    const unsigned h3 = hx_mul3(h2 & HX_LOW);
    t0 = hx_gather(h1, h0);
    t1 = hx_gather(h3, h2);
}

int main() {
    std::mt19937 rng(20260921);
    std::uniform_int_distribution<int> trit(0, 2);
    long blocks = 0, slices = 0;
    for (int it = 0; it < 20000; it++) {
        uint8_t t[128];
        for (int i = 0; i < 128; i++) t[i] = (uint8_t) trit(rng);
        uint8_t qs[24], qh[2];
        encode_block(t, qs, qh);
        // What the kernel loads: six operand dwords and the tail dword (qh0, qh1, fp16 scale).
        unsigned dw[6];
        memcpy(dw, qs, 24);
        const unsigned tail = (unsigned) qh[0] | ((unsigned) qh[1] << 8) | (0xabcdu << 16);

        unsigned ref[32];
        hx_expand_perm(dw, tail, ref);

        // The body's order: slice kb takes tr[4kb..4kb+3]; kb 6 and 7 are the fifth trits.
        unsigned got[32] = {0};
        unsigned fifth[6];
        for (int d = 0; d < 6; d++) {
            unsigned A[4];
            mv8_expand_dw_host(dw[d], A, fifth[d]);
            for (int j = 0; j < 4; j++) got[4 * d + j] = A[j];
        }
        unsigned h0, h1;
        mv8_expand_tail_host(tail, h0, h1);
        // Slice 6 = fifths of dwords 0..3, slice 7 = fifths of 4,5 then the two tail dwords.
        got[24] = fifth[0]; got[25] = fifth[1]; got[26] = fifth[2]; got[27] = fifth[3];
        got[28] = fifth[4]; got[29] = fifth[5]; got[30] = h0; got[31] = h1;

        for (int i = 0; i < 32; i++)
            if (got[i] != ref[i]) {
                printf("FAIL iter %d operand dword %d: %08x against %08x\n", it, i, got[i], ref[i]);
                return 1;
            }
        // And the operand bytes are the packer's own trits, in the element order the format fixes.
        uint8_t back[128];
        decode_block(qs, qh, back);
        for (int i = 0; i < 128; i++) {
            const unsigned byte = (got[i >> 2] >> (8 * (i & 3))) & 0xffu;
            if (byte != back[i] || byte != t[i]) {
                printf("FAIL iter %d element %d: operand byte %u, decoded %u, source %u\n", it, i, byte, back[i], t[i]);
                return 1;
            }
        }
        blocks++; slices += 8;
    }
    printf("mv8 operand map: %ld blocks, %ld K slices, every operand dword equal to hx_expand_perm "
           "and every operand byte equal to the packed trit\n", blocks, slices);
    return 0;
}
