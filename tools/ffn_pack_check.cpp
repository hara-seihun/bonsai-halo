// CPU self-check for the batched FFN's weight repack.
//
// halo_block_codes() is the only genuinely new piece of the weight path: it turns a HALO 896-byte
// block-run into the two-bit code words the projections read, without materialising trits. It is
// __host__ __device__ so this runs the shipped function itself, on the CPU, with no GPU involved.
//
//   hipcc -O2 -std=c++17 -Isrc -Ikernels tools/ffn_pack_check.cpp -o build/ffn_pack_check && ./build/ffn_pack_check
//
// Checks, over random blocks:
//   - every one of the 32 rows x 128 elements decodes to the trit that was encoded,
//   - the code convention holds (0 -> 0, +1 -> 1, -1 -> 3),
//   - the row's FP16 scale bits land where the tile-major scale image expects them.
#include "halo_format.h"
#include "ffn_operands.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace halo;
using namespace halo::ffnb;

int main() {
    std::mt19937 rng(12345);
    const int BLOCKS = 64;
    long checked = 0, bad = 0;

    for (int b = 0; b < BLOCKS && bad == 0; ++b) {
        // One HALO block-run: 32 rows of 128 trits, each row with its own FP16 scale.
        std::vector<uint8_t> run(TILE_BLOCK_BYTES);
        uint8_t trit[32][128];
        uint16_t scale[32];
        for (int lane = 0; lane < 32; ++lane) {
            for (int e = 0; e < 128; ++e) trit[lane][e] = uint8_t(rng() % 3);
            scale[lane] = uint16_t(rng() & 0xffffu);
            uint8_t qs[24], qh[2];
            encode_block(trit[lane], qs, qh);
            std::memcpy(run.data() + tile_off_qs_a(lane), qs, 16);
            std::memcpy(run.data() + tile_off_qs_b(lane), qs + 16, 8);
            uint8_t tail[4] = { qh[0], qh[1], uint8_t(scale[lane] & 0xff), uint8_t(scale[lane] >> 8) };
            std::memcpy(run.data() + tile_off_tail(lane), tail, 4);
        }

        for (int lane = 0; lane < 32; ++lane) {
            uint4 qa; uint2 qb; unsigned tail;
            std::memcpy(&qa, run.data() + tile_off_qs_a(lane), 16);
            std::memcpy(&qb, run.data() + tile_off_qs_b(lane), 8);
            std::memcpy(&tail, run.data() + tile_off_tail(lane), 4);

            uint32_t w[8];
            halo_block_codes(qa, qb, tail, w);

            for (int e = 0; e < 128; ++e) {
                const unsigned code = (w[e >> 4] >> (2 * (e & 15))) & 3u;
                const int weight = code == 0 ? 0 : code == 1 ? 1 : code == 3 ? -1 : 99;
                const int want = int(trit[lane][e]) - 1;   // stored 0,1,2 means -1,0,+1
                ++checked;
                if (weight != want) {
                    printf("FAIL block %d lane %d element %d: code %u -> %d, want %d\n",
                           b, lane, e, code, weight, want);
                    if (++bad > 8) break;
                }
            }
            if ((unsigned) (tail >> 16) != scale[lane]) {
                printf("FAIL block %d lane %d: scale bits %04x, want %04x\n",
                       b, lane, unsigned(tail >> 16), scale[lane]);
                ++bad;
            }
        }
    }

    // Storage accounting the projections depend on.
    if (codes_bytes(16, 128) != BLOCK_BYTES) { printf("FAIL codes_bytes\n"); ++bad; }
    if (scale_elems(16, 128) != 16) { printf("FAIL scale_elems\n"); ++bad; }

    printf("%s: %ld weights checked over %d blocks, %ld mismatches\n",
           bad ? "FAIL" : "ok", checked, BLOCKS, bad);
    return bad ? 1 : 0;
}
