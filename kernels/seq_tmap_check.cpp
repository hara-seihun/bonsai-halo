// What a token map costs, before any of it reaches a GPU.
//
// `project`'s block loop gives a lane TT activation fragments per K16 slice, and which token sits
// in column `col` of tile `t` is a labelling the kernel chooses: `v_wmma_i32_16x16x16` reduces over
// K alone and never across its sixteen B columns, so any bijection from (tile, lane) onto the
// group's TT*16 tokens computes the same outputs from the same products. This program enumerates
// the three maps `kernels/sequence_batch.hip` carries and reports the two things that decide
// whether one is cheaper than another:
//
//   * that it is a bijection onto the same token set - the correctness condition, and the whole
//     bit-identity argument;
//   * how many distinct 64-byte cache lines ONE INSTRUCTION's sixteen lanes touch, and how much of
//     each it uses - which docs/seq-operand-coord.md measures as the only term that moves the phase.
//
//   make kernels/seq_tmap_check && kernels/seq_tmap_check      # under a second, no GPU
#include <cstdio>
#include <set>
#include <vector>

namespace {

constexpr int LINE = 64;

// The three maps, kept as one expression each so they can be read against the kernel's source.
int token_of(int tmap, int TT, int t, int col) {
    const int m = TT < 2 ? 0 : tmap;          // at one tile every map is `token = first + col`
    if (m == 1) return col * TT + t;
    if (m == 2) return (t >> 1) * 32 + col * 2 + (t & 1);
    return t * 16 + col;
}

struct Fetch { int lines; long bytes; };

// One fragment instruction under this map: at four-bit activations a lane reads FB bytes per tile,
// and the wide maps read a lane's tiles 2t and 2t+1 as one 2*FB piece. The upper sixteen lanes of
// the wave read the same addresses (WMMA replicates A and B across the halves), so the wave's
// request is what these sixteen lanes cover.
Fetch first_instruction(int tmap, int TT, int FB) {
    const int m = TT < 2 ? 0 : tmap;
    const int width = m == 0 ? FB : 2 * FB;
    std::set<int> lines;
    long bytes = 0;
    for (int col = 0; col < 16; ++col) {
        const int off = token_of(tmap, TT, 0, col) * FB;
        const int lo = m == 2 ? (token_of(tmap, TT, 0, col) * FB) : off;
        for (int b = lo; b < lo + width; ++b) lines.insert(b / LINE);
        bytes += width;
    }
    return { (int) lines.size(), bytes };
}

}  // namespace

int main() {
    const char * name[3] = { "0 tile-major  t*16+col", "1 lane-major  col*TT+t", "2 pair-major  (t>>1)*32+col*2+(t&1)" };
    int bad = 0;
    printf("%-38s %4s %10s %7s %7s %8s\n", "map", "TT", "bijection", "lines", "bytes", "used");
    for (int TT : { 1, 2, 4, 8 }) {
        for (int tmap = 0; tmap < 3; ++tmap) {
            std::set<int> seen;
            for (int t = 0; t < TT; ++t)
                for (int col = 0; col < 16; ++col) seen.insert(token_of(tmap, TT, t, col));
            bool onto = (int) seen.size() == TT * 16 && *seen.begin() == 0 && *seen.rbegin() == TT * 16 - 1;
            const Fetch f = first_instruction(tmap, TT, 8);
            printf("%-38s %4d %10s %7d %7ld %7.0f%%\n", name[tmap], TT, onto ? "yes" : "NO",
                   f.lines, f.bytes, 100.0 * f.bytes / (double) (LINE * f.lines));
            if (!onto) ++bad;
        }
    }
    // The floor: 512 bytes of a (slice, 64-token group) is eight lines, so eight lookups is the
    // least any map can pay for them. Map 0 pays four instructions of two lines and map 2 pays two
    // of four; map 1 pays two of eight, each half used, and measures 16% of the phase for it.
    printf("\n%s\n", bad ? "A MAP IS NOT A BIJECTION - it does not compute the projection at all"
                         : "every map is a bijection onto the same TT*16 tokens, so all three are bit-identical by construction");
    return bad ? 1 : 0;
}
