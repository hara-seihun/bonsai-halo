// Which counter the multi-row ternary matvec waits on, priced with no model in the picture.
//
// A TT = 8 block of `mv_rows_t` streams the same 896 weight bytes a TT = 1 block does and pays
// sixteen `s_waitcnt lgkmcnt(0)` against its one, because SMEM returns out of order on gfx11 and
// that counter can only be drained whole. This benchmark runs the deployed body and the arms that
// change where the activations come from over a real weight image, at a chosen resident-wave count,
// and reports achieved GB/s of distinct weight bytes plus an FNV-64 of the outputs.
//
//   bench/mvsched                 # the ladder, at the deployed ten waves per SIMD32
//   bench/mvsched --waves 16      # the same arms with no occupancy ballast
//
// The digest is the exactness control: every arm moves the same bytes through the same dot products
// in the same order, so a differing digest is a bug, not a trade.
#include "../kernels/halo_kernels.h"
#include "../kernels/device.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace halo;
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e)); exit(1);}}while(0)

namespace rejected {
// TWO REJECTED ARMS, kept executable here and out of the runtime header.
//
// Arm 2 puts the activations on the vector path. It is the fastest body measured at ten waves per
// SIMD32 (+5.2%) and it CANNOT BE PAID FOR in the persistent kernel: one row of activations is 32
// VGPRs, and under `HALO_ROWS_WPE = 12` the deployed `k_forward_rows<8, 0, 0, 0, false, 12>` is
// already 120 registers with 7 spilled, where this arm spills 370. It becomes shippable the day
// that kernel gets its registers back, or the day the matvec phase is its own kernel.
//
// DEPTH is the weight cursor: more than one block in flight. Measured null at two and -2.9% at
// three, so the runtime carries no depth parameter at all.
using namespace halo;
// Activations on the VECTOR path, which is the counter this loop should have been using.
//
// `vmcnt` is IN ORDER and partially waitable: `s_waitcnt vmcnt(N)` means "at most N outstanding",
// so a wave can hold a row of activations in flight and consume them one load at a time without
// ever draining the rest. A uniform `global_load_b128` is one sixteen-byte request the coalescer
// serves to all thirty-two lanes at once, and `v_dot4_i32_iu8` takes that VGPR as its activation
// operand exactly as it takes an SGPR. So the same bytes arrive through L0 with the same values in
// the same order, and the block's only remaining scalar request is the batch of `xs`/`xsum` drain
// scalars at the top, whose single wait the 229-slot trit expansion covers.
//
// The registers are there: `mv_rows_t<5, 8, false>` compiled alone is 61 VGPR against the 135 of
// `k_forward_rows<8>`, so this phase can hold a whole row of activations - 32 dwords - without
// touching the kernel's peak or its ten resident waves.
template <int NBLK, int TT, int ACC, int DEPTH, int AR>
__device__ __forceinline__ void mv_rows_vec(const uint8_t * __restrict__ run, int lane, const int * __restrict__ xg, int xstride,
                                            cfptr xs, cptr xsum, int xsstride, int nrows, float y[TT]) {
    constexpr int BB = TILE_BLOCK_BYTES;
    constexpr int DEP = DEPTH < NBLK ? DEPTH : NBLK;
    MvBlk pipe[DEP];
    #pragma unroll
    for (int i = 0; i < DEP; i++) pipe[i] = mv_load_blk(run + i * BB, lane);
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        MvBlk nxt = pipe[DEP - 1];
        if (b + DEP < NBLK) nxt = mv_load_blk(run + (b + DEP) * BB, lane);
        // Every drain scalar of the block in one scalar request. Rows past `nrows` read row
        // `nrows - 1`, in bounds and discarded, exactly as the deployed loop clamps its rows.
        float sc[TT]; int sm[TT];
        #pragma unroll
        for (int r = 0; r < TT; r++) {
            const int rr = min(r, nrows - 1);
            sc[r] = xs[rr * xsstride + b];
            sm[r] = xsum[rr * xsstride + b];
        }
        __builtin_amdgcn_sched_barrier(0);
        unsigned tr[32];
        const float wscale = mv_expand(pipe[0], tr);
        // AR slots of activation registers, so AR - 1 rows stay in flight across a row's dot
        // products: row r's request was issued AR rows ago and the wait before its first dot is
        // `vmcnt(8 * (AR - 1))`, which is "the later rows are still coming". That pipeline is what
        // the scalar counter cannot express at any depth, because draining it drains everything.
        int4 buf[AR][8];
        #pragma unroll
        for (int j = 0; j < AR; j++) {
            const int4 * __restrict__ p = (const int4 *) (xg + min(j, nrows - 1) * xstride + b * 32);
            #pragma unroll
            for (int k = 0; k < 8; k++) buf[j][k] = p[k];
        }
        #pragma unroll
        for (int r = 0; r < TT; r++) {
            if (r < nrows) {
                int xa[32];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    const int4 v = buf[r % AR][k];
                    xa[4 * k] = v.x; xa[4 * k + 1] = v.y; xa[4 * k + 2] = v.z; xa[4 * k + 3] = v.w;
                }
                int part[ACC];
                #pragma unroll
                for (int q = 0; q < ACC; q++) part[q] = 0;
                #pragma unroll
                for (int i = 0; i < 32; i++) part[i % ACC] = __builtin_amdgcn_sudot4(false, (int) tr[i], true, xa[i], part[i % ACC], false);
                int acc = part[0];
                #pragma unroll
                for (int q = 1; q < ACC; q++) acc += part[q];
                y[r] = fmaf((float) (acc - sm[r]), wscale * sc[r], y[r]);
            }
            // The slot is free only now, which is what keeps this a pipeline and not a clobber.
            if (r + AR < TT) {
                const int4 * __restrict__ p = (const int4 *) (xg + min(r + AR, nrows - 1) * xstride + b * 32);
                #pragma unroll
                for (int k = 0; k < 8; k++) buf[r % AR][k] = p[k];
            }
        }
        #pragma unroll
        for (int i = 0; i + 1 < DEP; i++) pipe[i] = pipe[i + 1];
        pipe[DEP - 1] = nxt;
    }
}


// The shipped row batch, with the rejected weight cursor added back: DEPTH blocks in flight.
template <int NBLK, int TT, int ACC, int RB, int DEPTH>
__device__ __forceinline__ void mv_rows_batched_d(const uint8_t * __restrict__ run, int lane, cptr xq, int xstride,
                                                  cfptr xs, cptr xsum, int xsstride, int nrows, float y[TT]) {
    constexpr int BB = TILE_BLOCK_BYTES;
    constexpr int DEP = DEPTH < NBLK ? DEPTH : NBLK;
    MvBlk pipe[DEP];
    #pragma unroll
    for (int i = 0; i < DEP; i++) pipe[i] = mv_load_blk(run + i * BB, lane);
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        MvBlk nxt = pipe[DEP - 1];
        if (b + DEP < NBLK) nxt = mv_load_blk(run + (b + DEP) * BB, lane);
        int xa[RB][32]; float sc[RB]; int sm[RB];
        #pragma unroll
        for (int j = 0; j < RB; j++) {
            const int rr = min(j, nrows - 1);
            cptr xr = xq + rr * xstride + b * 32;
            #pragma unroll
            for (int i = 0; i < 32; i++) xa[j][i] = xr[i];
            sc[j] = xs[rr * xsstride + b];
            sm[j] = xsum[rr * xsstride + b];
        }
        __builtin_amdgcn_sched_barrier(0);
        unsigned tr_shared[32];
        const float wscale_shared = mv_expand(pipe[0], tr_shared);
        #pragma unroll
        for (int base = 0; base < TT; base += RB) {
            if (base > 0) {
                #pragma unroll
                for (int j = 0; j < RB; j++) {
                    const int rr = min(base + j, nrows - 1);
                    cptr xr = xq + rr * xstride + b * 32;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) xa[j][i] = xr[i];
                    sc[j] = xs[rr * xsstride + b];
                    sm[j] = xsum[rr * xsstride + b];
                }
                __builtin_amdgcn_sched_barrier(0);
            }
            #pragma unroll
            for (int j = 0; j < RB; j++) {
                if (base + j < nrows) {
                    int part[ACC];
                    #pragma unroll
                    for (int q = 0; q < ACC; q++) part[q] = 0;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) part[i % ACC] = __builtin_amdgcn_sudot4(false, (int) tr_shared[i], true, xa[j][i], part[i % ACC], false);
                    int acc = part[0];
                    #pragma unroll
                    for (int q = 1; q < ACC; q++) acc += part[q];
                    y[base + j] = fmaf((float) (acc - sm[j]), wscale_shared * sc[j], y[base + j]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        #pragma unroll
        for (int i = 0; i + 1 < DEP; i++) pipe[i] = pipe[i + 1];
        pipe[DEP - 1] = nxt;
    }
}
// ARM 3, REJECTED AND MEASURED: G OUTPUT TILES SHARE ONE ACTIVATION BATCH.
//
// `RB` buys weight bytes per scalar wait from the SGPR file and ran out at two rows: four rows of
// activations is 128 SGPRs against a 107-SGPR file and loses 12%. This is the same purchase made
// from the VGPR file instead, which the matvec has and the persistent kernel is not using -
// `mv_rows_t<5, 8, false>` compiled alone is 61 VGPR where `k_forward_rows<8>` allocates 141 for
// its other phases.
//
// A wave carries G weight tiles instead of one and dots every activation batch against all of them
// before letting it go. The wait count per block is unchanged, but a block now streams `G * 896`
// weight bytes, so the drain density that separates 216 GB/s from 115 falls by G: eight rows over
// two tiles is the wait-per-weight-byte of a four-row pass at the row count of an eight-row one.
// The activation side is bought too - the same 40 kB window feeds G tiles per read instead of one.
//
// WHAT DOES NOT MOVE: a wave keeps its own K blocks in the same order, the same int32 dot chains,
// the same `acc - xsum`, the same `fmaf(.., wscale * xs, y)` per (tile, row), and the eight-wave
// LDS reduction of a tile is untouched. Only how many tiles one unit carries moves, and a tile is
// computed entirely inside one workgroup, so the output bits are the deployed ones by construction.
// docs/mv-tile-fold.md.
//
// `runs[g]` is tile g's weight run; a unit with fewer than G tiles left repeats its last pointer
// and discards that tile's accumulator, which keeps the inner loop branch-free.
template <int NBLK, int TT, int G, int ACC, int RB, int EXP = HALO_MV_EXPAND>
__device__ __forceinline__ void mv_rows_fold(const uint8_t * const __restrict__ * runs, int lane, cptr xq, int xstride,
                                             cfptr xs, cptr xsum, int xsstride, int nrows, float y[G][TT]) {
    constexpr int BB = TILE_BLOCK_BYTES;
    MvBlk cur[G];
    #pragma unroll
    for (int g = 0; g < G; g++) cur[g] = mv_load_blk(runs[g], lane);
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        MvBlk nxt[G];
        #pragma unroll
        for (int g = 0; g < G; g++) {
            nxt[g] = cur[g];
            if (b + 1 < NBLK) nxt[g] = mv_load_blk(runs[g] + (b + 1) * BB, lane);
        }
        int xa[RB][32]; float sc[RB]; int sm[RB];
        #pragma unroll
        for (int j = 0; j < RB; j++) {
            const int rr = min(j, nrows - 1);
            cptr xr = xq + rr * xstride + b * 32;
            #pragma unroll
            for (int i = 0; i < 32; i++) xa[j][i] = xr[i];
            sc[j] = xs[rr * xsstride + b];
            sm[j] = xsum[rr * xsstride + b];
        }
        __builtin_amdgcn_sched_barrier(0);
        unsigned tr[G][32]; float ws[G];
        #pragma unroll
        for (int g = 0; g < G; g++) ws[g] = mv_expand<EXP>(cur[g], tr[g]);
        #pragma unroll
        for (int base = 0; base < TT; base += RB) {
            if (base > 0) {
                #pragma unroll
                for (int j = 0; j < RB; j++) {
                    const int rr = min(base + j, nrows - 1);
                    cptr xr = xq + rr * xstride + b * 32;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) xa[j][i] = xr[i];
                    sc[j] = xs[rr * xsstride + b];
                    sm[j] = xsum[rr * xsstride + b];
                }
                __builtin_amdgcn_sched_barrier(0);
            }
            #pragma unroll
            for (int j = 0; j < RB; j++) {
                if (base + j < nrows) {
                    #pragma unroll
                    for (int g = 0; g < G; g++) {
                        int part[ACC];
                        #pragma unroll
                        for (int q = 0; q < ACC; q++) part[q] = 0;
                        #pragma unroll
                        for (int i = 0; i < 32; i++) part[i % ACC] = __builtin_amdgcn_sudot4(false, (int) tr[g][i], true, xa[j][i], part[i % ACC], false);
                        int acc = part[0];
                        #pragma unroll
                        for (int q = 1; q < ACC; q++) acc += part[q];
                        y[g][base + j] = fmaf((float) (acc - sm[j]), ws[g] * sc[j], y[g][base + j]);
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        #pragma unroll
        for (int g = 0; g < G; g++) cur[g] = nxt[g];
    }
}
}  // namespace rejected

// LDS per workgroup sets how many 256-thread workgroups a WGP holds, which is the only knob here
// that moves resident waves without touching the body: 24 kB is five workgroups, ten waves per
// SIMD32, exactly what `k_forward_rows<8>` gets at 141 VGPR.
template <int TT, int PW, int MVS, int AR, int DEPTH, int LDSB, int RB = 2, int EXP = HALO_MV_EXPAND>
__global__ void __launch_bounds__(256) kmv(const uint8_t* w, const int8_t* xq_g, const float* xs_g, const int* xsum_g,
                                          float* out, int nb, int nrows) {
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int8_t* xq_l = xq_g; asm volatile("" : "+s"(xq_l)); cptr xq = (cptr) xq_l;
    const float* xs_l = xs_g; asm volatile("" : "+s"(xs_l)); cfptr xs = (cfptr) xs_l;
    const int* xsum_l = xsum_g; asm volatile("" : "+s"(xsum_l)); cptr xsum = (cptr) xsum_l;
    __shared__ float red[8][TT][32];
    // `red` is itself 1024 * TT bytes, so the ballast is what makes TOTAL LDS - and therefore
    // resident waves - the same number at every row count. That confound is worth stating: at a
    // 24 kB ballast the TT = 8 arms ran at eight waves a SIMD32 and the TT = 1 control at ten.
    __shared__ char ballast[LDSB > 1024 * TT ? LDSB - 1024 * TT : 1];
    if (out == nullptr) ballast[threadIdx.x] = 1;   // never taken; keeps the allocation
    const int wb = wave * PW;
    const uint8_t* run = w + ((size_t) blockIdx.x * nb + wb) * TILE_BLOCK_BYTES;
    float y[TT];
    for (int r = 0; r < TT; r++) y[r] = 0.f;
    if constexpr (MVS == 2)
        rejected::mv_rows_vec<PW, TT, (TT > 1 ? HALO_MV_ACC : 1), DEPTH, (AR < TT ? AR : TT)>(
            run, lane, (const int*) xq_g + wb * 32, nb * 32, xs + wb, xsum + wb, nb, nrows, y);
    else if constexpr (MVS == 1 && TT > 1 && DEPTH > 1)
        rejected::mv_rows_batched_d<PW, TT, HALO_MV_ACC, RB, DEPTH>(
            run, lane, xq + wb * 32, nb * 32, xs + wb, xsum + wb, nb, nrows, y);
    else if constexpr (MVS == 1)
        mv_rows_t<PW, TT, false, (TT > 1 ? HALO_MV_ACC : 1), 1, RB, EXP>(
            run, lane, xq + wb * 32, nb * 32, xs + wb, xsum + wb, nb, nrows, y);
    else
        mv_rows_t<PW, TT, false, (TT > 1 ? HALO_MV_ACC : 1), 0, 2, EXP>(
            run, lane, xq + wb * 32, nb * 32, xs + wb, xsum + wb, nb, nrows, y);
    if (wave > 0) for (int r = 0; r < TT; r++) red[wave][r][lane] = y[r];
    __syncthreads();
    if (wave == 0) for (int r = 0; r < TT; r++) {
        float v = y[r];
        for (int ww = 1; ww < 8; ww++) v += red[ww][r][lane];
        out[(size_t) r * gridDim.x * 32 + blockIdx.x * 32 + lane] = v;
    }
}

// THE FOLD: one workgroup owns G consecutive weight tiles and every wave dots its activation batch
// against all G before letting it go. Same grid of waves, same blocks per wave, same K order, G
// times the weight bytes between two scalar waits. The drain runs G times through the same LDS, so
// this arm's occupancy ballast means what the other arms' does.
template <int TT, int PW, int G, int LDSB, int RB = 2, int EXP = HALO_MV_EXPAND>
__global__ void __launch_bounds__(256) kmvf(const uint8_t* w, const int8_t* xq_g, const float* xs_g, const int* xsum_g,
                                            float* out, int nb, int nrows) {
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int8_t* xq_l = xq_g; asm volatile("" : "+s"(xq_l)); cptr xq = (cptr) xq_l;
    const float* xs_l = xs_g; asm volatile("" : "+s"(xs_l)); cfptr xs = (cfptr) xs_l;
    const int* xsum_l = xsum_g; asm volatile("" : "+s"(xsum_l)); cptr xsum = (cptr) xsum_l;
    __shared__ float red[8][TT][32];
    __shared__ char ballast[LDSB > 1024 * TT ? LDSB - 1024 * TT : 1];
    if (out == nullptr) ballast[threadIdx.x] = 1;
    const int wb = wave * PW;
    const uint8_t* runs[G];
    #pragma unroll
    for (int g = 0; g < G; g++) runs[g] = w + ((size_t) (blockIdx.x * G + g) * nb + wb) * TILE_BLOCK_BYTES;
    float y[G][TT];
    #pragma unroll
    for (int g = 0; g < G; g++)
        #pragma unroll
        for (int r = 0; r < TT; r++) y[g][r] = 0.f;
    rejected::mv_rows_fold<PW, TT, G, (TT > 1 ? HALO_MV_ACC : 1), RB, EXP>(
        runs, lane, xq + wb * 32, nb * 32, xs + wb, xsum + wb, nb, nrows, y);
    const int ntiles = gridDim.x * G;
    #pragma unroll
    for (int g = 0; g < G; g++) {
        if (wave > 0) for (int r = 0; r < TT; r++) red[wave][r][lane] = y[g][r];
        __syncthreads();
        if (wave == 0) for (int r = 0; r < TT; r++) {
            float v = y[g][r];
            for (int ww = 1; ww < 8; ww++) v += red[ww][r][lane];
            out[(size_t) r * ntiles * 32 + (blockIdx.x * G + g) * 32 + lane] = v;
        }
        __syncthreads();
    }
}

static unsigned long long fnv(const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*) p;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    int N = 34816, K = 5120, reps = 40, waves = 10;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) N = atoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) K = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--waves" && i + 1 < argc) waves = atoi(argv[++i]);
    }
    const int nb = K / 128;
    size_t bytes = halo_tensor_bytes(N, K);
    std::vector<uint8_t> h(bytes);
    srand(12345);
    for (size_t i = 0; i < bytes; i++) h[i] = rand() & 0xff;
    int copies = (int) ((96u << 20) / bytes) + 1;
    std::vector<uint8_t*> ws(copies);
    for (int c = 0; c < copies; c++) { CK(hipMalloc(&ws[c], bytes)); CK(hipMemcpy(ws[c], h.data(), bytes, hipMemcpyHostToDevice)); }

    // Real operand values, so the output digest is a witness and not a constant.
    std::vector<int8_t> hxq(8 * K); std::vector<float> hxs(8 * nb); std::vector<int> hxsum(8 * nb);
    for (int i = 0; i < 8 * K; i++) hxq[i] = (int8_t) ((rand() & 0xff) - 128);
    for (int i = 0; i < 8 * nb; i++) { hxs[i] = (float) ((rand() % 2001) - 1000) / 8192.0f; hxsum[i] = (rand() % 8192) - 4096; }
    int8_t* xq; float* xs; int* xsum; float* out;
    CK(hipMalloc(&xq, 8 * K)); CK(hipMalloc(&xs, 8 * nb * 4)); CK(hipMalloc(&xsum, 8 * nb * 4));
    CK(hipMalloc(&out, (size_t) 8 * N * 4));
    CK(hipMemcpy(xq, hxq.data(), 8 * K, hipMemcpyHostToDevice));
    CK(hipMemcpy(xs, hxs.data(), 8 * nb * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(xsum, hxsum.data(), 8 * nb * 4, hipMemcpyHostToDevice));
    std::vector<float> hout((size_t) 8 * N);
    hipStream_t st; CK(hipStreamCreate(&st));

    auto run = [&](const char* nm, auto launch) {
        for (int i = 0; i < 3; i++) launch(ws[0]);
        CK(hipStreamSynchronize(st));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        CK(hipEventRecord(e0, st));
        for (int i = 0; i < reps; i++) launch(ws[i % copies]);
        CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        CK(hipMemcpy(hout.data(), out, (size_t) 8 * N * 4, hipMemcpyDeviceToHost));
        printf("%-22s %8.1f us %6.1f GB/s  digest %020llu\n", nm, 1000 * ms / reps,
               bytes * (double) reps / ms / 1e6, fnv(hout.data(), hout.size() * 4));
        fflush(stdout);
    };

    // A 256-thread workgroup is two waves a SIMD32, so LDS per workgroup sets resident waves in
    // steps of two: 26,000 B is five workgroups a WGP and the ten waves `k_forward_rows<8>` gets.
    #define ARME(nm, TT, MVS, AR, DEPTH, ROWS, RB, EXP)                                              \
        do {                                                                                         \
            if (waves == 6)       run(nm, [&](uint8_t* w) { kmv<TT, 5, MVS, AR, DEPTH, 43000, RB, EXP><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else if (waves == 8)  run(nm, [&](uint8_t* w) { kmv<TT, 5, MVS, AR, DEPTH, 32700, RB, EXP><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else if (waves == 10) run(nm, [&](uint8_t* w) { kmv<TT, 5, MVS, AR, DEPTH, 26000, RB, EXP><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else if (waves == 12) run(nm, [&](uint8_t* w) { kmv<TT, 5, MVS, AR, DEPTH, 21800, RB, EXP><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else                  run(nm, [&](uint8_t* w) { kmv<TT, 5, MVS, AR, DEPTH, 16300, RB, EXP><<<N / 32, 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); });  \
        } while (0)
    #define ARMR(nm, TT, MVS, AR, DEPTH, ROWS, RB) ARME(nm, TT, MVS, AR, DEPTH, ROWS, RB, HALO_MV_EXPAND)
    #define ARM(nm, TT, MVS, AR, DEPTH, ROWS) ARMR(nm, TT, MVS, AR, DEPTH, ROWS, 2)
    #define ARMF(nm, TT, G, ROWS, RB)                                                                \
        do {                                                                                         \
            if (waves == 6)       run(nm, [&](uint8_t* w) { kmvf<TT, 5, G, 43000, RB><<<N / (32 * G), 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else if (waves == 8)  run(nm, [&](uint8_t* w) { kmvf<TT, 5, G, 32700, RB><<<N / (32 * G), 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else if (waves == 10) run(nm, [&](uint8_t* w) { kmvf<TT, 5, G, 26000, RB><<<N / (32 * G), 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else if (waves == 12) run(nm, [&](uint8_t* w) { kmvf<TT, 5, G, 21800, RB><<<N / (32 * G), 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); }); \
            else                  run(nm, [&](uint8_t* w) { kmvf<TT, 5, G, 16300, RB><<<N / (32 * G), 256, 0, st>>>(w, xq, xs, xsum, out, nb, ROWS); });  \
        } while (0)

    printf("# rows=%d K=%d nb=%d image=%.1f MB copies=%d reps=%d waves=%d\n",
           N, K, nb, bytes / 1048576.0, copies, reps, waves);
    // Palindrome order: every arm is measured before and after every other one.
    const char* order[] = { "deployed", "batch-rb1", "batch-rb2", "batch-rb4", "batch-d2", "vec-ar2",
                            "fold-g2", "fold-g2-rb1", "fold-g4" };
    constexpr int NARM = 9;
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < NARM; i++) {
            const char* nm = order[pass ? NARM - 1 - i : i];
            char label[64]; snprintf(label, sizeof label, "TT=8 %s%s", nm, pass ? " (rev)" : "");
            if (!strcmp(nm, "deployed"))   ARM(label, 8, 0, 1, 1, 8);
            else if (!strcmp(nm, "batch-rb2")) ARM(label, 8, 1, 1, 1, 8);
            else if (!strcmp(nm, "batch-rb1")) ARMR(label, 8, 1, 1, 1, 8, 1);
            else if (!strcmp(nm, "batch-rb4")) ARMR(label, 8, 1, 1, 1, 8, 4);
            else if (!strcmp(nm, "batch-d2"))  ARM(label, 8, 1, 1, 2, 8);
            else if (!strcmp(nm, "vec-ar2"))   ARM(label, 8, 2, 2, 1, 8);
            else if (!strcmp(nm, "fold-g2"))   ARMF(label, 8, 2, 8, 2);
            else if (!strcmp(nm, "fold-g2-rb1")) ARMF(label, 8, 2, 8, 1);
            else if (!strcmp(nm, "fold-g4"))   ARMF(label, 8, 4, 8, 2);
        }
    }
    // The fold at the row counts below eight, where the drain density it buys is already cheap.
    ARMF("TT=4 fold-g2", 4, 2, 4, 2);
    ARMF("TT=2 fold-g2", 2, 2, 2, 2);
    ARMF("TT=1 fold-g2", 1, 2, 1, 1);
    // The in-panel control: one row takes the deployed body in every arm and must not move.
    ARM("TT=1 deployed", 1, 0, 1, 1, 1);
    ARM("TT=4 deployed", 4, 0, 1, 1, 4);
    ARM("TT=4 batch-rb2", 4, 1, 1, 1, 4);
    ARMR("TT=4 batch-rb1", 4, 1, 1, 1, 4, 1);
    ARM("TT=2 deployed", 2, 0, 1, 1, 2);
    ARM("TT=2 batch-rb2", 2, 1, 1, 1, 2);

    // THE EXPANSION AXIS. `peel` and `perm` write the same 32 operand dwords out of the same 26
    // bytes and differ by 64 issue slots a block, so their digests must agree to the bit; `none`
    // is the ablation that deletes the expansion altogether and computes a wrong answer on
    // purpose, which is the only way to price what a slot in this loop is worth. Palindrome order
    // again, and the shapes are the two the engine serves: the eight-row verify pass and the
    // single-token decode.
    printf("# expansion axis: peel = 232 slots a block, perm = 168, none = 0 (wrong on purpose)\n");
    {
        const char* ex[] = { "peel", "perm", "none" };
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i < 3; i++) {
                const int e = pass ? 2 - i : i;
                char label[64];
                snprintf(label, sizeof label, "TT=8 %s%s", ex[e], pass ? " (rev)" : "");
                if (e == 0)      ARME(label, 8, 1, 1, 1, 8, 2, 0);
                else if (e == 1) ARME(label, 8, 1, 1, 1, 8, 2, 1);
#ifdef HALO_MV_EXPAND_PROBE
                else             ARME(label, 8, 1, 1, 1, 8, 2, 2);
#endif
                snprintf(label, sizeof label, "TT=1 %s%s", ex[e], pass ? " (rev)" : "");
                if (e == 0)      ARME(label, 1, 0, 1, 1, 1, 2, 0);
                else if (e == 1) ARME(label, 1, 0, 1, 1, 1, 2, 1);
#ifdef HALO_MV_EXPAND_PROBE
                else             ARME(label, 1, 0, 1, 1, 1, 2, 2);
#endif
            }
    }
    return 0;
}
