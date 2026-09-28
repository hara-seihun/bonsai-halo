// Q4 drafter tiles: the four-bit weight coordinate for the DFlash2 drafter.
//
// A lossless drafter proposes tokens and the target model verifies every one of them, so the
// drafter's arithmetic cannot change a single output bit - it changes only how many proposals
// survive. That makes it the one place in this engine where a coarser weight coordinate costs
// nothing but acceptance, and the drafter's time is bytes and nothing else: a Q8 block is 4160
// bytes with no peel and the phase runs at about 200 GB/s of a 242 GB/s roof.
//
// Layout. Same tile/row mapping as Q8 and HALO: 32 weight rows x 128 K per block, lane == row.
//
//   run + row * 64                    64 bytes of packed nibbles, K 0..127 of that row
//   run + TILE_ROWS * 64 + row * 2    fp16 scale for that row and block
//
// 2112 bytes a block against Q8's 4160, so the drafter image is 0.5077x.
//
// Nibble placement is chosen so that unpacking is two instructions per four weights and needs no
// permute. Within a row, K is cut into groups of eight and each group is one dword:
//
//   dword g = bytes 4g..4g+3 of the row, g = 0..15
//   low  nibble of byte 4g+j  = code(K = 8g + j)      j = 0..3
//   high nibble of byte 4g+j  = code(K = 8g + 4 + j)
//
// so `P & 0x0F0F0F0F` is exactly the four codes of K 8g..8g+3 in byte order and `(P >> 4) &
// 0x0F0F0F0F` is exactly the four codes of K 8g+4..8g+7. Those are the two operand words the dot4
// and WMMA bodies want, in the K order the activation row already has: the unpack is an AND, a
// shift and an AND, amortised over all TT activation rows of the block.
//
// Codes are offset binary, c = q + 8 with q in [-7, 7], and the bias is free. The matvec already
// carries `xsum`, the int32 sum of a block's activation bytes, for the ternary path's +1 offset;
// the Q4 path subtracts 8 * xsum from the same int32 accumulator. No unbiasing happens per weight.
#pragma once
#include "halo_format.h"

namespace halo {

constexpr int Q4_ROW_BYTES        = BLOCK / 2;                          // 64
constexpr int Q4_TILE_BLOCK_BYTES = TILE_ROWS * Q4_ROW_BYTES + TILE_ROWS * 2; // 2112
constexpr int Q4_ZERO             = 8;                                  // offset-binary zero code

__host__ __device__ inline size_t q4_off_row(int lane) { return (size_t) lane * Q4_ROW_BYTES; }
__host__ __device__ inline size_t q4_off_scale(int lane) { return (size_t) TILE_ROWS * Q4_ROW_BYTES + (size_t) lane * 2; }

// Byte holding block-local element e of a row, and whether it is the high nibble.
__host__ __device__ inline int  q4_byte_of(int e) { return (e / 8) * 4 + (e % 4); }
__host__ __device__ inline bool q4_high_of(int e) { return (e % 8) >= 4; }

inline size_t q4_tensor_bytes(int64_t N, int64_t K) {
    return (size_t) (N / TILE_ROWS) * (K / BLOCK) * Q4_TILE_BLOCK_BYTES;
}

} // namespace halo
