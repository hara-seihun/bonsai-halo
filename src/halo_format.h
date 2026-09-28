// HALO tile format: the on-device weight layout for ternary matrices.
//
//
// Trit order inside a row block (block-local element index e in [0,128)) is defined by the
// packed-16 peel used by the kernel. A qs dword d (bytes 4d..4d+3) is split into two byte pairs
//   pair w=0: (byte 4d,   byte 4d+2)   pair w=1: (byte 4d+1, byte 4d+3)
// and peel step n of a pair yields dword [t_n(first), 0, t_n(second), 0]. With p = 2d + w:
//   e = 8p + {0,1,2,3} <- t_0(first), t_1(first), t_0(second), t_1(second)
//   e = 8p + {4,5,6,7} <- t_2(first), t_3(first), t_2(second), t_3(second)
//   e = 96 + 4d + j    <- t_4(byte 4d + j)
// The two qh bytes form one pair (qh0, qh1) with four peel steps:
//   e = 120 + {0,1,2,3} <- t_0(qh0), t_1(qh0), t_0(qh1), t_1(qh1)
//   e = 124 + {0,1,2,3} <- t_2(qh0), t_3(qh0), t_2(qh1), t_3(qh1)
// Trits are stored as {0,1,2} meaning {-1,0,+1}.
#pragma once
#include <cstdint>
#include <cstddef>
#ifndef __HIP_DEVICE_COMPILE__
#ifndef __host__
#define __host__
#define __device__
#endif
#endif

namespace halo {

constexpr int BLOCK        = 128;
constexpr int TILE_ROWS    = 32;
constexpr int QS_BYTES     = 24;
constexpr int QH_BYTES     = 2;
constexpr int TILE_QS_A    = TILE_ROWS * 16;         // 512: qs bytes 0..15 of each row
constexpr int TILE_QS_B    = TILE_ROWS * 8;          // 256: qs bytes 16..23 of each row
constexpr int TILE_TAIL    = TILE_ROWS * 4;          // 128: qh0 qh1 scale(fp16) per row
constexpr int TILE_BLOCK_BYTES = TILE_QS_A + TILE_QS_B + TILE_TAIL; // 896
__host__ __device__ inline size_t tile_off_qs_a(int lane) { return (size_t) lane * 16; }
__host__ __device__ inline size_t tile_off_qs_b(int lane) { return TILE_QS_A + (size_t) lane * 8; }
__host__ __device__ inline size_t tile_off_tail(int lane) { return TILE_QS_A + TILE_QS_B + (size_t) lane * 4; }

inline size_t halo_tensor_bytes(int64_t N, int64_t K) { return (size_t) (N / TILE_ROWS) * (K / BLOCK) * TILE_BLOCK_BYTES; }

// Pack 5 trits (t0 peeled first) into one byte, 4 trits for qh bytes.
inline uint8_t pack5(const uint8_t * t) {
    unsigned q = ((((t[0] * 3u + t[1]) * 3u + t[2]) * 3u + t[3]) * 3u + t[4]);
    return (uint8_t) ((q * 256u + 242u) / 243u);
}
inline uint8_t pack4(const uint8_t * t) {
    unsigned q = ((((t[0] * 3u + t[1]) * 3u + t[2]) * 3u + t[3]) * 3u);
    return (uint8_t) ((q * 256u + 242u) / 243u);
}

// Encode 128 trits (values 0..2 in natural element order) into 24 qs bytes and 2 qh bytes.
inline void encode_block(const uint8_t * trit, uint8_t * qs, uint8_t * qh) {
    for (int d = 0; d < 6; d++) {
        for (int j = 0; j < 4; j++) {
            int w = j & 1, half = j >> 1;      // half: 0 = first byte of the pair, 1 = second
            int p = 2 * d + w;
            uint8_t t[5] = {
                trit[8 * p + 2 * half + 0],
                trit[8 * p + 2 * half + 1],
                trit[8 * p + 4 + 2 * half + 0],
                trit[8 * p + 4 + 2 * half + 1],
                trit[96 + 4 * d + j],
            };
            qs[4 * d + j] = pack5(t);
        }
    }
    for (int h = 0; h < 2; h++) {
        uint8_t t[4] = { trit[120 + 2 * h], trit[121 + 2 * h], trit[124 + 2 * h], trit[125 + 2 * h] };
        qh[h] = pack4(t);
    }
}

// Reference peel (what the GPU does), for self-tests.
inline void decode_block(const uint8_t * qs, const uint8_t * qh, uint8_t * trit) {
    for (int d = 0; d < 6; d++) {
        for (int j = 0; j < 4; j++) {
            int w = j & 1, half = j >> 1, p = 2 * d + w;
            unsigned b = qs[4 * d + j];
            uint8_t t[5];
            for (int n = 0; n < 5; n++) { unsigned m = b * 3u; t[n] = (uint8_t) (m >> 8); b = m & 0xff; }
            trit[8 * p + 2 * half + 0] = t[0];
            trit[8 * p + 2 * half + 1] = t[1];
            trit[8 * p + 4 + 2 * half + 0] = t[2];
            trit[8 * p + 4 + 2 * half + 1] = t[3];
            trit[96 + 4 * d + j] = t[4];
        }
    }
    for (int h = 0; h < 2; h++) {
        unsigned b = qh[h];
        uint8_t t[4];
        for (int n = 0; n < 4; n++) { unsigned m = b * 3u; t[n] = (uint8_t) (m >> 8); b = m & 0xff; }
        trit[120 + 2 * h] = t[0]; trit[121 + 2 * h] = t[1]; trit[124 + 2 * h] = t[2]; trit[125 + 2 * h] = t[3];
    }
}

// GGUF source block decoders -> 128 trits in natural order + fp16 scale bits.
// PTQ1_0: 24 qs (stages 16 then 8 bytes, 5 trits each), 2 qh (4 trits each), fp16 d.
inline void decode_gguf_ptq1_0(const uint8_t * blk, uint8_t * trit, uint16_t * scale) {
    const uint8_t * qs = blk; const uint8_t * qh = blk + 24;
    int e = 0;
    // stage c=16: bytes 0..15, then c=8: bytes 16..23
    const int stages[2][2] = { {0, 16}, {16, 8} };
    for (int s = 0; s < 2; s++) {
        int j0 = stages[s][0], c = stages[s][1];
        for (int n = 0; n < 5; n++)
            for (int m = 0; m < c; m++) {
                unsigned q = qs[j0 + m];
                for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
                trit[e++] = (uint8_t) ((q * 3u) >> 8);
            }
    }
    for (int n = 0; n < 4; n++)
        for (int h = 0; h < 2; h++) {
            unsigned q = qh[h];
            for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
            trit[e++] = (uint8_t) ((q * 3u) >> 8);
        }
    *scale = (uint16_t) (blk[26] | (blk[27] << 8));
}

// PQ2_0: fp16 d, 32 bytes of 2-bit codes (00=-1, 01=0, 10=+1). Codes map directly to our trit values.
inline void decode_gguf_pq2_0(const uint8_t * blk, uint8_t * trit, uint16_t * scale) {
    *scale = (uint16_t) (blk[0] | (blk[1] << 8));
    const uint8_t * qs = blk + 2;
    for (int j = 0; j < 128; j++) trit[j] = (qs[j >> 2] >> ((j & 3) * 2)) & 3;
}

} // namespace halo
