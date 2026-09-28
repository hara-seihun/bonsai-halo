#!/usr/bin/env python3
"""Host check of the four-bit sequence operand pair: the same sixteen products, in nibbles.

The eight-bit path stores a code word in the spread order and expands it to sixteen signed bytes
whose byte lane is the K slot; the activation fragment is sixteen bytes in natural K order. The
four-bit path stores the same codes in the nibble order and expands them to sixteen signed nibbles;
the prep writes the activation fragment as sixteen nibbles in the same order. Both are 2.000 bits
per weight in storage and the K order inside a 128-block is a free relabelling, so the two paths
must agree on every dot product. This checks that they do, and that the order switch is reversible.

Mirrors kernels/sequence_operands.hpp and the PREP_QUANT branch of kernels/phases.hpp; no GPU.
"""
import random

M32 = 0xFFFFFFFF


def spread_codes(w):
    out = 0
    for k in range(16):
        out |= ((w >> (2 * k)) & 3) << (8 * (k & 3) + 2 * (k >> 2))
    return out


def unspread_codes(d):
    out = 0
    for k in range(16):
        out |= ((d >> (8 * (k & 3) + 2 * (k >> 2))) & 3) << (2 * k)
    return out


def nibble_codes(w):
    out = 0
    for k in range(16):
        out |= ((w >> (2 * k)) & 3) << (4 * (k & 7) + 2 * (k >> 3))
    return out


def unnibble_codes(d):
    out = 0
    for k in range(16):
        out |= ((d >> (4 * (k & 7) + 2 * (k >> 3))) & 3) << (2 * k)
    return out


def perm(a, b, sel):
    """__builtin_amdgcn_perm: output byte i is source byte sel[i], sources b[0:4] then a[4:8]."""
    src = [(b >> (8 * i)) & 0xFF for i in range(4)] + [(a >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        out |= src[(sel >> (8 * i)) & 0x7] << (8 * i)
    return out


def expand_i8_spread(d):
    """Four dwords of signed bytes; dword j byte i is the weight of K slot 4j+i."""
    return [perm(0, 0xFF000100, (d >> (2 * j)) & 0x03030303) for j in range(4)]


def sext_nibbles(x):
    h = x & 0x22222222
    return (x | (h << 1) | (h << 2)) & M32


def expand_i4_nib(w):
    """Two dwords of signed nibbles; dword j nibble n is the weight of K slot 8j+n."""
    return [sext_nibbles(w & 0x33333333), sext_nibbles((w >> 2) & 0x33333333)]


def s8(v):
    return v - 256 if v >= 128 else v


def s4(v):
    return v - 16 if v >= 8 else v


def code_value(c):
    return {0: 0, 1: 1, 3: -1}[c]


def main():
    rng = random.Random(20260921)
    for trial in range(4000):
        codes = [rng.choice((0, 1, 3)) for _ in range(16)]
        acts = [rng.randint(-7, 7) for _ in range(16)]           # four-bit activation codes
        deployed = 0
        for k, c in enumerate(codes):
            deployed |= c << (2 * k)

        # storage words and their inverses
        sp, nb = spread_codes(deployed), nibble_codes(deployed)
        assert unspread_codes(sp) == deployed, "spread order is not reversible"
        assert unnibble_codes(nb) == deployed, "nibble order is not reversible"
        assert nibble_codes(unspread_codes(sp)) == nb, "spread -> nibble switch"
        assert spread_codes(unnibble_codes(nb)) == sp, "nibble -> spread switch"
        assert bin(sp).count("1") == bin(nb).count("1"), "both orders carry the same code bits"

        # eight-bit operand pair: A from the spread word, B sixteen bytes in K order
        a8 = expand_i8_spread(sp)
        b8 = [acts[4 * j + i] & 0xFF for j in range(4) for i in range(4)]
        acc8 = 0
        for j in range(4):
            for i in range(4):
                acc8 += s8((a8[j] >> (8 * i)) & 0xFF) * s8(b8[4 * j + i])

        # four-bit operand pair: A from the nibble word, B sixteen nibbles the prep writes as four
        # 16-bit halves, half c holding K slots 4c..4c+3 in nibbles 4c..4c+3 of dword c >> 1
        a4 = expand_i4_nib(nb)
        halves = [sum((acts[4 * c + t] & 0xF) << (4 * t) for t in range(4)) for c in range(4)]
        b4 = [halves[0] | (halves[1] << 16), halves[2] | (halves[3] << 16)]
        acc4 = 0
        for j in range(2):
            for n in range(8):
                acc4 += s4((a4[j] >> (4 * n)) & 0xF) * s4((b4[j] >> (4 * n)) & 0xF)

        reference = sum(code_value(c) * x for c, x in zip(codes, acts))
        assert acc8 == reference, f"eight-bit operand pair {acc8} != {reference}"
        assert acc4 == reference, f"four-bit operand pair {acc4} != {reference}"
        assert trial >= 0

    # the sign extension is only ever fed the three codes the packer emits
    for c in (0, 1, 3):
        x = sum(c << (4 * n) for n in range(8))
        want = sum((c if c != 3 else 0xF) << (4 * n) for n in range(8))
        assert sext_nibbles(x) == want, f"sign extension of code {c}"
    print("seq_a4_pack_check: 4000 random K16 slices agree on every dot product; both orders reversible")


if __name__ == "__main__":
    main()
