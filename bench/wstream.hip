// What a weight stream costs as a function of how many streams there are.
//
// Every weight image in this engine is tile-major: [row tile][K block][512 B]. A wave owns one row
// tile and walks its blocks, so at any instant the 320 to 2048 resident waves each sit 512 bytes
// into their own stream and those streams are nb*512 bytes apart - 24 kB for the sequence output
// projection, 20 kB for its input. The one phase that reaches this device's 242 GB/s roof,
// resident_state, is also the only one that reads tens of kilobytes contiguously per unit.
//
// This prices that difference with nothing else in the picture: no model, no matrix instructions,
// no scales. Same bytes, same wave count, same loads per wave; only which wave reads which run.
//
//   tile-major   wave w visits (w*nvisit + v)       - the deployed weight image
//   block-major  wave w visits (v*waves  + w)       - every resident wave inside one contiguous run
//   padded tile  wave w visits (w*(nvisit+1) + v)   - the same walk, one run of stride padding
//
// The third arm is what splits the cause. Tile-major and padded tile-major have the same locality
// per wave and the same distance between waves up to one run; all that changes is whether the
// stream stride carries the alignment the device's channel hash is sensitive to. Block-major
// changes where the waves are relative to each other and keeps no per-wave locality at all.
//
// RUN is the second axis and it is independent: how many bytes one visit takes, which is what a
// kernel would change by processing two or four K blocks per iteration instead of one.
// WORK is the ALU the real block loop issues between two visits (209 slots at one token tile, 369
// at four), which sets how many streams are actually in flight.
//
// DUP is the third axis and it is the one a matrix kernel takes without meaning to. A wave32 WMMA
// operand is addressed by `lane & 15`, so a projection kernel's weight load covers 256 distinct
// bytes in a 32-lane instruction where this probe's `lane * 16` covers 512. Same bytes, same order,
// same streams, twice the instructions, each asking for half as much - which is exactly the shape
// k_proj_opt reads its five-trit and pair images in. Nothing in the tile-major/block-major/padded
// axis can see it, because all three keep the full-lane form.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)

// MODE 0 tile-major, 1 block-major, 2 padded tile-major. RUN bytes per visit per wave.
// WORK dependent VALU ops per visit.
// `share` waves read the same stream, which is what the shared row-tile ownership does.
// Same body, two waves per workgroup so a workgroup is one wave per SIMD32 of a CU, with a static
// LDS block that admits exactly OCC workgroups per CU. That is the only knob this probe has for
// resident waves: the body is twenty registers and nothing else would stop the hardware filling
// every slot. 65536 / (OCC + 1) + 1 bytes leaves room for OCC workgroups and not OCC + 1; the
// arithmetic is checked against hipOccupancyMaxActiveBlocksPerMultiprocessor and printed, so the
// axis is labelled by what the device admits rather than by this comment.
template <int MODE, int RUN, int WORK, int OCC>
__global__ __launch_bounds__(64) void k_stream_occ(const uint8_t * __restrict__ src, int nvisit, int nstream,
                                                   int share, unsigned * __restrict__ sink) {
    __shared__ unsigned char pad[65536 / (OCC + 1) + 1];
    const int lane = threadIdx.x & 31;
    const int wave = int(blockIdx.x) * 2 + (int(threadIdx.x) >> 5);
    const int stream = wave / share;
    unsigned acc = lane;
    for (int v = 0; v < nvisit; ++v) {
        const size_t run = MODE == 0 ? (size_t) stream * nvisit + v
                         : MODE == 1 ? (size_t) v * nstream + stream
                                     : (size_t) stream * (nvisit + 1) + v;
        const uint8_t * p = src + run * RUN + lane * 16;
        #pragma unroll
        for (int r = 0; r < RUN / 512; ++r) {
            const uint4 a = *(const uint4 *) (p + r * 512);
            acc ^= a.x ^ a.y ^ a.z ^ a.w;
        }
        #pragma unroll
        for (int i = 0; i < WORK; ++i) acc = (acc ^ (acc << 7)) + 0x9e3779b9u;
    }
    if (acc == 0xdeadbeefu) { pad[0] = (unsigned char) acc; sink[0] = acc + pad[1]; }
}

// The A4 dense arm's own read, byte for byte: one wave-block is two 416-byte runs (gate and up, or
// the down matrix's two halves) read as dense_load0/1/2 off `lane & 15`, and a block of the real
// loop is about 600 issue slots. Every other arm in this probe reads 512-byte runs addressed by a
// full lane, which is the shape no projection kernel in this engine actually has.
template <int MODE, int WORK, int OCC>
__global__ __launch_bounds__(64) void k_stream_dense(const uint8_t * __restrict__ src, int nvisit, int nstream,
                                                    int half_off, unsigned * __restrict__ sink) {
    __shared__ unsigned char pad[65536 / (OCC + 1) + 1];
    const int col = threadIdx.x & 15;
    const int wave = int(blockIdx.x) * 2 + (int(threadIdx.x) >> 5);
    unsigned acc = col;
    for (int v = 0; v < nvisit; ++v) {
        const size_t run = MODE == 0 ? (size_t) wave * nvisit + v
                         : MODE == 1 ? (size_t) v * nstream + wave
                                     : (size_t) wave * (nvisit + 1) + v;
        const uint8_t * p = src + run * 416;
        const uint8_t * q = p + half_off;
        const uint4 a0 = *(const uint4 *) (p + col * 16);
        const uint2 a1 = *(const uint2 *) (p + 256 + col * 8);
        const unsigned a2 = *(const unsigned short *) (p + 384 + col * 2);
        const uint4 b0 = *(const uint4 *) (q + col * 16);
        const uint2 b1 = *(const uint2 *) (q + 256 + col * 8);
        const unsigned b2 = *(const unsigned short *) (q + 384 + col * 2);
        acc ^= a0.x ^ a0.y ^ a0.z ^ a0.w ^ a1.x ^ a1.y ^ a2 ^ b0.x ^ b0.y ^ b0.z ^ b0.w ^ b1.x ^ b1.y ^ b2;
        #pragma unroll
        for (int i = 0; i < WORK; ++i) acc = (acc ^ (acc << 7)) + 0x9e3779b9u;
    }
    if (acc == 0xdeadbeefu) { pad[0] = (unsigned char) acc; sink[0] = acc + pad[1]; }
}

// nstream waves of nvisit blocks, two 416-byte runs a block, the second `half` runs away.
template <int MODE, int WORK, int OCC>
double run_dense(const uint8_t * buf, size_t bytes, int nstream, int nvisit, int rounds) {
    const size_t half = (size_t) nstream * (MODE == 2 ? nvisit + 1 : nvisit) * 416;
    const size_t region = 2 * half;
    const size_t read = 2 * (size_t) nstream * nvisit * 416;
    const int launches = int(bytes / region);
    if (launches < 2) return 0.0;
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
        CHECK(hipEventRecord(a));
        for (int l = 0; l < launches; ++l)
            k_stream_dense<MODE, WORK, OCC><<<nstream / 2, 64>>>(buf + (size_t) l * region, nvisit, nstream, (int) half, nullptr);
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms = 0; CHECK(hipEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b));
    return (double) (read * launches) / (best * 1e-3) / 1e9;
}

template <int MODE, int RUN, int WORK, int OCC>
int occ_blocks() {
    int n = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *) k_stream_occ<MODE, RUN, WORK, OCC>, 64, 0));
    return n;
}

template <int MODE, int RUN, int WORK, bool DUP = false>
__global__ __launch_bounds__(256) void k_stream(const uint8_t * __restrict__ src, int nvisit, int nstream,
                                                int share, unsigned * __restrict__ sink) {
    const int lane = threadIdx.x & 31;
    const int wave = int(blockIdx.x) * (int(blockDim.x) >> 5) + (int(threadIdx.x) >> 5);
    const int stream = wave / share;
    unsigned acc = lane;
    for (int v = 0; v < nvisit; ++v) {
        const size_t run = MODE == 0 ? (size_t) stream * nvisit + v
                         : MODE == 1 ? (size_t) v * nstream + stream
                                     : (size_t) stream * (nvisit + 1) + v;
        const uint8_t * p = src + run * RUN + (DUP ? (lane & 15) : lane) * 16;
        constexpr int STEP = DUP ? 256 : 512;
        #pragma unroll
        for (int r = 0; r < RUN / STEP; ++r) {
            const uint4 a = *(const uint4 *) (p + r * STEP);
            acc ^= a.x ^ a.y ^ a.z ^ a.w;
        }
        #pragma unroll
        for (int i = 0; i < WORK; ++i) acc = (acc ^ (acc << 7)) + 0x9e3779b9u;  // full-rate filler
    }
    if (acc == 0xdeadbeefu) sink[0] = acc;   // keeps the loads live, never stores
}

// One geometry: `nstream` concurrent weight streams of `nvisit` runs each, `share` waves per
// stream, RUN bytes per run. The stream stride tile-major sees is nvisit*RUN.
struct Geom { const char * name; int nstream, nvisit, share; };
struct Point { Geom g; int work; double tile, block, pad, dup, duptile; };

template <int MODE, int RUN, int WORK, bool DUP = false>
double run_one(const uint8_t * buf, size_t bytes, const Geom & g, int rounds) {
    // The padded arm stores one more run per stream and never reads it, so its region is larger
    // and its GB/s counts the bytes it actually read - the same bytes the other two arms read.
    const size_t region = (size_t) g.nstream * (MODE == 2 ? g.nvisit + 1 : g.nvisit) * RUN;
    const size_t read = (size_t) g.nstream * g.nvisit * RUN;
    const int launches = int(bytes / region);
    if (launches < 2) return 0.0;
    const int waves = g.nstream * g.share;
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
        CHECK(hipEventRecord(a));
        for (int l = 0; l < launches; ++l)     // successive launches over distinct regions, as layers do
            k_stream<MODE, RUN, WORK, DUP><<<waves / 8, 256>>>(buf + (size_t) l * region, g.nvisit, g.nstream, g.share, nullptr);
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms = 0; CHECK(hipEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b));
    return (double) (read * launches) / (best * 1e-3) / 1e9;   // distinct bytes only
}

// One geometry, one work level, resident waves as the axis.
template <int MODE, int RUN, int WORK, int OCC>
double run_occ(const uint8_t * buf, size_t bytes, const Geom & g, int rounds) {
    const size_t region = (size_t) g.nstream * (MODE == 2 ? g.nvisit + 1 : g.nvisit) * RUN;
    const size_t read = (size_t) g.nstream * g.nvisit * RUN;
    const int launches = int(bytes / region);
    if (launches < 2) return 0.0;
    const int waves = g.nstream * g.share;
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
        CHECK(hipEventRecord(a));
        for (int l = 0; l < launches; ++l)
            k_stream_occ<MODE, RUN, WORK, OCC><<<waves / 2, 64>>>(buf + (size_t) l * region, g.nvisit, g.nstream, g.share, nullptr);
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms = 0; CHECK(hipEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b));
    return (double) (read * launches) / (best * 1e-3) / 1e9;
}

template <int WORK>
void measure(const uint8_t * buf, size_t bytes, const std::vector<Geom> & gs, int rounds, std::vector<Point> & out) {
    for (const Geom & g : gs) {
        Point p{ g, WORK, run_one<0, 512, WORK>(buf, bytes, g, rounds),
                 run_one<1, 512, WORK>(buf, bytes, g, rounds),
                 run_one<2, 512, WORK>(buf, bytes, g, rounds),
                 run_one<2, 512, WORK, true>(buf, bytes, g, rounds),
                 run_one<0, 512, WORK, true>(buf, bytes, g, rounds) };
        if (p.tile > 0) out.push_back(p);
    }
}

int main(int argc, char ** argv) {
    size_t mb = 1024; int rounds = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mb") && i + 1 < argc) mb = (size_t) atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
    }
    const size_t bytes = mb << 20;
    uint8_t * buf = nullptr;
    CHECK(hipMalloc(&buf, bytes));
    CHECK(hipMemset(buf, 0x5a, bytes));
    CHECK(hipDeviceSynchronize());

    // The deployed geometries, then the same stream count at neighbouring block counts. nb decides
    // the stride tile-major puts between two concurrent streams: 40 blocks is 20480 bytes and 48 is
    // 24576, both with a 2^12 or 2^13 factor.
    const std::vector<Geom> gs{
        { "seq-output 32r  (320 tiles x 2 waves, nb 48)", 320, 48, 2 },
        { "seq-output 128r (320 tiles x 2 waves, nb 48)", 320, 48, 2 },
        { "seq-input  32r  (1024 tiles, nb 40)",          1024, 40, 1 },
        { "seq-input  attn (896 tiles, nb 40)",            896, 40, 1 },
        { "ffn gate/up     (2176 tiles, nb 40)",          2176, 40, 1 },
        { "ffn down        (640 tiles, nb 136)",           640, 136, 1 },
        { "head            (7760 tiles, nb 40)",          7760, 40, 1 },
        { "stride control  (320 tiles, nb 47)",            320, 47, 2 },
        { "stride control  (320 tiles, nb 49)",            320, 49, 2 },
        { "stride control  (1024 tiles, nb 41)",          1024, 41, 1 },
    };
    std::vector<Point> pts;
    measure<192>(buf, bytes, gs, rounds, pts);
    measure<0>(buf, bytes, gs, rounds, pts);

    // The occupancy axis, on the geometry the A4 FFN gate/up stage reads, padded run order, at the
    // work level of its block loop and with none.
    const Geom gu = gs[4];
    const int occs[] = { 16, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1 };
    double occ_w192[11], occ_w0[11];
    int occ_seen[11];
    occ_seen[0]  = occ_blocks<2, 512, 192, 16>();
    occ_seen[1]  = occ_blocks<2, 512, 192, 12>();
    occ_seen[2]  = occ_blocks<2, 512, 192, 10>();
    occ_seen[3]  = occ_blocks<2, 512, 192, 8>();
    occ_seen[4]  = occ_blocks<2, 512, 192, 7>();
    occ_seen[5]  = occ_blocks<2, 512, 192, 6>();
    occ_seen[6]  = occ_blocks<2, 512, 192, 5>();
    occ_seen[7]  = occ_blocks<2, 512, 192, 4>();
    occ_seen[8]  = occ_blocks<2, 512, 192, 3>();
    occ_seen[9]  = occ_blocks<2, 512, 192, 2>();
    occ_seen[10] = occ_blocks<2, 512, 192, 1>();
    occ_w192[0]  = run_occ<2, 512, 192, 16>(buf, bytes, gu, rounds);
    occ_w192[1]  = run_occ<2, 512, 192, 12>(buf, bytes, gu, rounds);
    occ_w192[2]  = run_occ<2, 512, 192, 10>(buf, bytes, gu, rounds);
    occ_w192[3]  = run_occ<2, 512, 192, 8>(buf, bytes, gu, rounds);
    occ_w192[4]  = run_occ<2, 512, 192, 7>(buf, bytes, gu, rounds);
    occ_w192[5]  = run_occ<2, 512, 192, 6>(buf, bytes, gu, rounds);
    occ_w192[6]  = run_occ<2, 512, 192, 5>(buf, bytes, gu, rounds);
    occ_w192[7]  = run_occ<2, 512, 192, 4>(buf, bytes, gu, rounds);
    occ_w192[8]  = run_occ<2, 512, 192, 3>(buf, bytes, gu, rounds);
    occ_w192[9]  = run_occ<2, 512, 192, 2>(buf, bytes, gu, rounds);
    occ_w192[10] = run_occ<2, 512, 192, 1>(buf, bytes, gu, rounds);
    occ_w0[0]  = run_occ<2, 512, 0, 16>(buf, bytes, gu, rounds);
    occ_w0[1]  = run_occ<2, 512, 0, 12>(buf, bytes, gu, rounds);
    occ_w0[2]  = run_occ<2, 512, 0, 10>(buf, bytes, gu, rounds);
    occ_w0[3]  = run_occ<2, 512, 0, 8>(buf, bytes, gu, rounds);
    occ_w0[4]  = run_occ<2, 512, 0, 7>(buf, bytes, gu, rounds);
    occ_w0[5]  = run_occ<2, 512, 0, 6>(buf, bytes, gu, rounds);
    occ_w0[6]  = run_occ<2, 512, 0, 5>(buf, bytes, gu, rounds);
    occ_w0[7]  = run_occ<2, 512, 0, 4>(buf, bytes, gu, rounds);
    occ_w0[8]  = run_occ<2, 512, 0, 3>(buf, bytes, gu, rounds);
    occ_w0[9]  = run_occ<2, 512, 0, 2>(buf, bytes, gu, rounds);
    occ_w0[10] = run_occ<2, 512, 0, 1>(buf, bytes, gu, rounds);

    // The dense arm's own geometry. Gate/up is 1088 waves of 40 blocks, two matrices a wave; down
    // is 320 waves of 136. The work levels are the two block loops: about 600 slots and about 300.
    const double gu_t16 = run_dense<0, 200, 16>(buf, bytes, 1088, 40, rounds);
    const double gu_t7  = run_dense<0, 200, 7>(buf, bytes, 1088, 40, rounds);
    const double gu_t4  = run_dense<0, 200, 4>(buf, bytes, 1088, 40, rounds);
    const double gu_b16 = run_dense<1, 200, 16>(buf, bytes, 1088, 40, rounds);
    const double gu_b7  = run_dense<1, 200, 7>(buf, bytes, 1088, 40, rounds);
    const double gu_b4  = run_dense<1, 200, 4>(buf, bytes, 1088, 40, rounds);
    const double dn_t16 = run_dense<0, 100, 16>(buf, bytes, 320, 136, rounds);
    const double dn_t7  = run_dense<0, 100, 7>(buf, bytes, 320, 136, rounds);
    const double dn_t4  = run_dense<0, 100, 4>(buf, bytes, 320, 136, rounds);
    const double dn_b16 = run_dense<1, 100, 16>(buf, bytes, 320, 136, rounds);
    const double dn_b7  = run_dense<1, 100, 7>(buf, bytes, 320, 136, rounds);
    const double dn_b4  = run_dense<1, 100, 4>(buf, bytes, 320, 136, rounds);

    printf("{\"schema\":\"bonsai-wstream/4\",\"buffer_mb\":%zu,\"rounds\":%d,\"run_bytes\":512,\"points\":[\n", mb, rounds);
    for (size_t i = 0; i < pts.size(); ++i)
        printf("  {\"geom\":\"%s\",\"streams\":%d,\"blocks\":%d,\"share\":%d,\"stride\":%d,\"work\":%d,"
               "\"tile_major_gb_s\":%.1f,\"block_major_gb_s\":%.1f,\"padded_tile_gb_s\":%.1f,"
               "\"padded_half_lane_gb_s\":%.1f,\"tile_half_lane_gb_s\":%.1f}%s\n",
               pts[i].g.name, pts[i].g.nstream, pts[i].g.nvisit, pts[i].g.share, pts[i].g.nvisit * 512,
               pts[i].work, pts[i].tile, pts[i].block, pts[i].pad, pts[i].dup, pts[i].duptile,
               i + 1 == pts.size() ? "" : ",");
    printf("],\n \"occupancy\":{\"geom\":\"%s\",\"order\":\"padded\",\"waves_per_simd32\":[", gu.name);
    for (int i = 0; i < 11; ++i) printf("%d%s", occ_seen[i], i == 10 ? "" : ",");
    printf("],\"waves_asked\":[");
    for (int i = 0; i < 11; ++i) printf("%d%s", occs[i], i == 10 ? "" : ",");
    printf("],\"work192_gb_s\":[");
    for (int i = 0; i < 11; ++i) printf("%.1f%s", occ_w192[i], i == 10 ? "" : ",");
    printf("],\"work0_gb_s\":[");
    for (int i = 0; i < 11; ++i) printf("%.1f%s", occ_w0[i], i == 10 ? "" : ",");
    printf("]},\n \"dense_arm_shape\":{\"runs\":\"2 x 416 B per block, lane&15, dense_load0/1/2\","
           "\"gate_up_1088_waves_nb40\":{\"tile_major\":[%.1f,%.1f,%.1f],\"block_major\":[%.1f,%.1f,%.1f],"
           "\"waves_per_simd32\":[16,7,4],\"work\":200},"
           "\"down_320_waves_nb136\":{\"tile_major\":[%.1f,%.1f,%.1f],\"block_major\":[%.1f,%.1f,%.1f],"
           "\"waves_per_simd32\":[16,7,4],\"work\":100}}}\n",
           gu_t16, gu_t7, gu_t4, gu_b16, gu_b7, gu_b4, dn_t16, dn_t7, dn_t4, dn_b16, dn_b7, dn_b4);

    fprintf(stderr, "\n work | stream stride | %-46s | tile-major  block-major  padded  pad/tile  half-lane(pad)  half/pad\n", "geometry");
    for (const Point & p : pts)
        fprintf(stderr, " %4d | %13d | %-46s | %9.1f %11.1f %7.1f %8.2fx %14.1f %8.2fx\n", p.work, p.g.nvisit * 512, p.g.name,
                p.tile, p.block, p.pad, p.pad / p.tile, p.dup, p.dup / p.pad);

    fprintf(stderr, "\n resident waves per SIMD32, %s, padded order\n asked | device | work 192 | work 0\n", gu.name);
    for (int i = 0; i < 11; ++i)
        fprintf(stderr, " %5d | %6d | %8.1f | %6.1f\n", occs[i], occ_seen[i], occ_w192[i], occ_w0[i]);

    fprintf(stderr, "\n the A4 dense arm's own read: two 416 B runs a block, lane&15\n");
    fprintf(stderr, " stage                          | waves | tile-major | block-major\n");
    fprintf(stderr, " gate/up 1088 waves nb 40 w200  |    16 | %10.1f | %11.1f\n", gu_t16, gu_b16);
    fprintf(stderr, " gate/up 1088 waves nb 40 w200  |     7 | %10.1f | %11.1f\n", gu_t7, gu_b7);
    fprintf(stderr, " gate/up 1088 waves nb 40 w200  |     4 | %10.1f | %11.1f\n", gu_t4, gu_b4);
    fprintf(stderr, " down     320 waves nb 136 w100 |    16 | %10.1f | %11.1f\n", dn_t16, dn_b16);
    fprintf(stderr, " down     320 waves nb 136 w100 |     7 | %10.1f | %11.1f\n", dn_t7, dn_b7);
    fprintf(stderr, " down     320 waves nb 136 w100 |     4 | %10.1f | %11.1f\n", dn_t4, dn_b4);
    CHECK(hipFree(buf));
    return 0;
}
