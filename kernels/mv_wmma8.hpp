// The eight-row matvec through the matrix instruction, at a register budget the deployed grid can
// afford.
//
// WHY THIS FILE EXISTS. `docs/matvec-crossover.md` put both matvec bodies on the same 60-workgroup
// grid and the scalar-fed dot4 body LOST, 62.8 ms against 53.0 on the drafted verify pass. dot4
// ships anyway for one reason, written down in that document: it holds `k_forward_rows<8>` in 136
// VGPR where `ph_matvec_w` holds it in 217, and on `bbfca1db`'s measured ladder (VGPRs round up to
// 24 of 1536 per SIMD32, a 256-thread workgroup is two waves per SIMD32) that is five workgroups
// per WGP against three - grid 100 against grid 60. The eight-row pass wants the grid more than it
// wants the body, so the faster body has never run on the deployed shape.
//
// 217 is not a property of the arithmetic. It is the live range of a block: `tr[32]` for the whole
// 128-wide block, the double-buffered weight cursor, two int32 accumulator pairs, the eight
// activation fragments the scheduler hoists above the first matrix instruction, and `y1`/`y2` held
// across the cross-wave drain. This body computes the same values with a smaller live set:
//
//   THE EXPANSION IS INCREMENTAL. `src/halo_format.h` fixes the trit order so that source dword `d`
//   carries operand dwords `tr[4d .. 4d+3]` - exactly the K slice `kb = d` consumes - plus one
//   fifth-trit dword `tr[24+d]`. Expanding a dword straight into the matrix instruction that eats
//   it holds four operand dwords plus the six fifths instead of thirty-two.
//
//   ONE ACCUMULATOR PAIR, ONE COLUMN GROUP. A 32-row output tile is two 16-row matrix fragments
//   (`A` and its half swap), so `C1`/`C2` are the tile, not a latency device: `a4fc28a5`'s
//   `bench/wmma_chain` measures `v_wmma_i32_16x16x16_iu8` at 17.14 / 17.12 / 17.10 / 17.10 slots
//   per SIMD32 at one, two, four and eight independent chains, so a dependent chain is free here
//   and no accumulator is bought for rate.
//
// WHAT DOES NOT MOVE, AND WHY THE OUTPUT IS BIT-IDENTICAL TO THE DEPLOYED dot4 BODY. Same image,
// same bytes, same blocks in the same order, same K order inside a block. A block's contribution to
// an output element is one exact `int32` - trit codes are unsigned and every output owes one
// `-sum(x)`, which both bodies apply once per block - and `int32` addition is associative and wraps
// identically, so the grouping the matrix instruction uses cannot change the word. That word is
// folded with the same `fmaf((float) acc, wscale * xscale, y)` in the same block order and the same
// left-to-right wave order in the drain. Only which lane holds an output row moves.
//
// THIS HEADER IS INCLUDED FROM INSIDE `kernels/phases.hpp`'s anonymous namespace, after `Ctx`,
// `MvR`, `Geom` and the unit deal it builds on, so it opens no namespace and includes no header of
// its own.
#pragma once

// Fence the activation fragment of each K slice against the one before it. Without it the
// scheduler hoists all eight `global_load_dwordx4` of a block above the first matrix instruction,
// which is 32 VGPRs of fragment live at once for an operand stream `a4fc28a5` measured at 1.59% of
// its phase. 0 keeps the compiler's own order and is the control arm.
#ifndef HALO_MV8_FENCE
#define HALO_MV8_FENCE 1
#endif
// Build a K slice's A operand one slice ahead of the matrix instruction that reads it, so the
// `v_perm`/`v_permlanex16` that writes it is not the instruction in front of it.
#ifndef HALO_MV8_PIPE
#define HALO_MV8_PIPE 0
#endif
// Publish all NW partials to LDS instead of letting wave 0 keep its own in registers. The
// register-side reason is `a4fc28a5`'s: in `ph_matvec_w` the register form keeps `y1`/`y2` live
// across the barrier and the LDS reads and cost their body 21 VGPR. IT DOES NOT REPRODUCE HERE -
// this body is 192 VGPR with the seven-slot form and 195 with the eighth slot, and the eighth slot
// also costs 1 kB of the shared block - so 0 ships and the arm stays for the next body in here.
#ifndef HALO_MV8_DRAIN8
#define HALO_MV8_DRAIN8 0
#endif

// Spread the cross-wave reduction over the eight waves that computed it instead of letting wave 0
// sum all of it: accumulator element `r` belongs to wave `r`. Bit-identical - the sum is the same
// eight partials in the same left-to-right order - and it is the mechanism `docs/matvec-drain.md`
// measured at 4.4% of a 1088-unit phase and 8.9% of a 320-unit K-split one in the dot4 body, which
// this body pays TWICE per unit because a 32-row tile is two matrix fragments.
#ifndef HALO_MV8_SPREAD
#define HALO_MV8_SPREAD 1
#endif

// The wave's scale table, and wave 0's drain slot under DRAIN8. One allocation for every
// instantiation of the phase below: a `__shared__` array inside the template is allocated per
// instantiation, and a forward pass instantiates this body three times (40, 48 and 136 blocks).
__device__ __forceinline__ float * mv8_shared() {
    __shared__ __attribute__((aligned(16))) float buf[(HALO_MV8_DRAIN8 ? 8 * 32 : 0) + NW * 32];
    return buf;
}

typedef int mv8_v4i __attribute__((ext_vector_type(4)));
typedef int mv8_v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ int mv8_swap16(int v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
}

// One source dword to the four operand dwords of its K slice, plus the fifth-trit dword that the
// last two slices are built from. Same products, same `v_perm_b32` gather and same operand bytes as
// `hx_expand_perm`; only the order they are produced in moves.
// `HX_FN` rather than `__device__`: the host pass of this translation unit parses the body, and
// `hx_mul3`/`hx_gather` are host-only there. It also lets `kernels/mv8_expand_check.cpp` hold the
// same arithmetic against `hx_expand_perm` with no GPU.
HX_FN void mv8_expand_dw(unsigned d, mv8_v4i & A, unsigned & fifth) {
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
    A.x = (int) hx_gather(m1, m0);
    A.y = (int) hx_gather(m3, m2);
    A.z = (int) hx_gather(n1, n0);
    A.w = (int) hx_gather(n3, n2);
    fifth = hx_gather(n4, m4);
}

// The two operand dwords the `qh` tail carries, which close out K slice 7.
HX_FN void mv8_expand_tail(unsigned tail, unsigned & t0, unsigned & t1) {
    const unsigned P = (tail & 0xffu) | ((tail << 8) & 0xff0000u);
    const unsigned h0 = hx_mul3(P);
    const unsigned h1 = hx_mul3(h0 & HX_LOW);
    const unsigned h2 = hx_mul3(h1 & HX_LOW);
    const unsigned h3 = hx_mul3(h2 & HX_LOW);
    t0 = hx_gather(h1, h0);
    t1 = hx_gather(h3, h2);
}

// One 32-row tile against up to eight activation rows, `NBLK` blocks of a K run.
// `y1`/`y2` are the two matrix fragments of the tile: `y1` holds even rows in the low lane half and
// odd rows + 17 in the high half, `y2` the same pair shifted by sixteen, exactly as `ph_matvec_w`
// lays them out, because the drain below writes with the same expression.
template <int NBLK>
__device__ __forceinline__ void mv8_rows(const uint8_t * __restrict__ run, int lane,
                                         const int8_t * __restrict__ xrow, const float * xs, const int * xsum,
                                         int xsstride, float * scl, float (&y1)[8], float (&y2)[8]) {
    constexpr int BB = TILE_BLOCK_BYTES;
    const int half = lane >> 4, col = lane & 15, arow = col & 7;
    uint4 qa = *(const uint4 *) (run + tile_off_qs_a(lane));
    uint2 qb = *(const uint2 *) (run + tile_off_qs_b(lane));
    unsigned tail = *(const unsigned *) (run + tile_off_tail(lane));
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        uint4 nqa = qa; uint2 nqb = qb; unsigned ntail = tail;
        if (b + 1 < NBLK) {
            const uint8_t * nrun = run + (b + 1) * BB;
            nqa = *(const uint4 *) (nrun + tile_off_qs_a(lane));
            nqb = *(const uint2 *) (nrun + tile_off_qs_b(lane));
            ntail = *(const unsigned *) (nrun + tile_off_tail(lane));
        }
        // The weight scale reaches the lanes that hold its output rows through this wave's own
        // scale table; the fragment layout puts row 2m in table slot [0][m] and row 2m+1 in [1][m].
        scl[(lane & 1) * 16 + (lane >> 1)] = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
        const float4 * s1p = (const float4 *) (scl + half * 16 + 8 * half);
        const float4 * s2p = (const float4 *) (scl + half * 16 + 8 * (half ^ 1));
        const int8_t * xb = xrow + b * 128;
        // Seed both accumulators with the negated activation block sum: the operand carries
        // unsigned trit codes, so every output owes one -sum(x), and integer addition wraps
        // identically wherever it is applied. This is the seed `ph_matvec_w` already ships.
        const int xsc = xsum[arow * xsstride + b];
        mv8_v8i C1 = {-xsc, -xsc, -xsc, -xsc, -xsc, -xsc, -xsc, -xsc}, C2 = C1;
        // K slice order 0,1,2,3,6,4,5,7. A source dword carries its own slice plus one fifth-trit
        // dword, and slice 6 is built from the fifths of dwords 0..3, so taking it as soon as they
        // exist retires four of them instead of carrying six to the end of the block. The slices
        // meet in one `int32` accumulator, which is exact and associative, so the order is free.
        unsigned fifth[6];
        constexpr int kb_of[8] = { 0, 1, 2, 3, 6, 4, 5, 7 };
        // One K slice's A operand, from the source dword that carries it.
        auto slice_a = [&](int ki, mv8_v4i & A) {
            const int kb = kb_of[ki];
            if (kb < 6) {
                const unsigned dw = kb == 0 ? qa.x : kb == 1 ? qa.y : kb == 2 ? qa.z : kb == 3 ? qa.w : kb == 4 ? qb.x : qb.y;
                mv8_expand_dw(dw, A, fifth[kb]);
            } else if (kb == 6) {
                A.x = (int) fifth[0]; A.y = (int) fifth[1]; A.z = (int) fifth[2]; A.w = (int) fifth[3];
            } else {
                unsigned t0, t1;
                mv8_expand_tail(tail, t0, t1);
                A.x = (int) fifth[4]; A.y = (int) fifth[5]; A.z = (int) t0; A.w = (int) t1;
            }
        };
#if HALO_MV8_PIPE
        // The A-operand hazard: `d9b78400`'s `bench/mvblock` prices a matrix instruction whose A
        // dword was written by the `v_perm` or `v_permlanex16` in front of it at about three cycles,
        // 9% of the instruction. So a slice's operand is built one slice ahead, behind the two
        // matrix instructions that are already issued. It costs one more `mv8_v4i` pair live.
        mv8_v4i A, As;
        slice_a(0, A);
        As.x = mv8_swap16(A.x); As.y = mv8_swap16(A.y); As.z = mv8_swap16(A.z); As.w = mv8_swap16(A.w);
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            const int4 bx = *(const int4 *) (xb + kb_of[ki] * 16);
            const mv8_v4i B = { bx.x, bx.y, bx.z, bx.w };
            C1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, A, true, B, C1, false);
            C2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, As, true, B, C2, false);
            if (ki + 1 < 8) {
                slice_a(ki + 1, A);
                As.x = mv8_swap16(A.x); As.y = mv8_swap16(A.y); As.z = mv8_swap16(A.z); As.w = mv8_swap16(A.w);
            }
#if HALO_MV8_FENCE
            __builtin_amdgcn_sched_barrier(0);
#endif
        }
#else
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            mv8_v4i A;
            slice_a(ki, A);
            const int4 bx = *(const int4 *) (xb + kb_of[ki] * 16);
            const mv8_v4i B = { bx.x, bx.y, bx.z, bx.w };
            C1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, A, true, B, C1, false);
            const mv8_v4i As = { mv8_swap16(A.x), mv8_swap16(A.y), mv8_swap16(A.z), mv8_swap16(A.w) };
            C2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, As, true, B, C2, false);
#if HALO_MV8_FENCE
            __builtin_amdgcn_sched_barrier(0);
#endif
        }
#endif
        // Four scale floats live at a time rather than sixteen: the table is this wave's own LDS and
        // a `ds_read_b128` at the point of use is cheaper than the registers holding all of it.
        const float xsb = xs[arow * xsstride + b];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            const float4 s = (q < 2) ? s1p[q] : s2p[q - 2];
            float * y = (q < 2) ? y1 : y2;
            const mv8_v8i & C = (q < 2) ? C1 : C2;
            const int r0 = (q & 1) * 4;
            y[r0 + 0] = fmaf((float) C[r0 + 0], s.x * xsb, y[r0 + 0]);
            y[r0 + 1] = fmaf((float) C[r0 + 1], s.y * xsb, y[r0 + 1]);
            y[r0 + 2] = fmaf((float) C[r0 + 2], s.z * xsb, y[r0 + 2]);
            y[r0 + 3] = fmaf((float) C[r0 + 3], s.w * xsb, y[r0 + 3]);
        }
        qa = nqa; qb = nqb; tail = ntail;
    }
}

// The phase, with `ph_matvec`'s unit decomposition exactly: one 32-row tile per unit, `KS` K parts,
// the same segment walk, the same dynamic deal and the same output expression.
template <int NB, int KS, int TT>
__device__ __forceinline__ void ph_matvec_m(Ctx & c, const MvR & m, const int8_t * xq, const float * xs, const int * xsum, int nrows) {
    static_assert(TT <= 8, "this body carries one sixteen-column group and the caller has eight rows");
    using G = Geom<NB, KS>;
    constexpr int BB = TILE_BLOCK_BYTES;
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int half = lane >> 4, col = lane & 15;
    const int total = m.total_tiles * KS;
    // The cross-wave drain, and the per-wave scale table beside it. `rows_lds_floats` sizes the
    // shared block for the dot4 drain, which is the same [NW-1][8][32] this body stages, so the
    // extra rows here are the scale table and - under DRAIN8 - wave 0's own slot. They live in
    // this body's own allocation rather than in `c.lds`, whose size is another engineer's number.
    float * mv8_lds = mv8_shared();
    float * red = c.lds;                                 // [NW-1][8][32], waves 1..NW-1
    float * red0 = mv8_lds;                              // [8][32], wave 0 under DRAIN8
    float * scl = mv8_lds + (HALO_MV8_DRAIN8 ? 8 * 32 : 0) + wave * 32;
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int part = u / m.total_tiles;
        int tile = u - part * m.total_tiles;
        MvSegR seg = m.seg[0];
        if (m.nseg > 1 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[1]; if (m.nseg > 2 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[2]; } }
        const int pw = (part == 0 || !G::SPECIAL) ? G::PW_A : G::PW_B;
        const int wb = (part == 0 ? 0 : (G::SPECIAL ? G::A_BLOCKS : part * G::PART)) + wave * pw;
        const uint8_t * run = seg.w + ((size_t) tile * NB + wb) * BB;
        const int8_t * xrow = xq + (size_t) (col & 7) * NB * 128 + wb * 128;
        float y1[8], y2[8];
        #pragma unroll
        for (int r = 0; r < 8; r++) { y1[r] = 0.0f; y2[r] = 0.0f; }
        if (part == 0 || !G::SPECIAL) mv8_rows<G::PW_A>(run, lane, xrow, xs + wb, xsum + wb, NB, scl, y1, y2);
        else                          mv8_rows<G::PW_B>(run, lane, xrow, xs + wb, xsum + wb, NB, scl, y1, y2);
        const int N = seg.ntiles * 32;
        // Two passes over one reduction buffer, one per matrix fragment of the tile. The sum is
        // y_0[r] + y_1[r] + ... + y_7[r] left to right in every form.
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            float * y = pass ? y2 : y1;
#if HALO_MV8_SPREAD
            // Accumulator element `r` belongs to wave `r`, which is `ph_matvec`'s spread drain in
            // this body's coordinate: eight partials fit the seven slots because an element's owner
            // reads its own from a register and wave 0 writes into the slot that frees. Without it
            // one wave reduces 8 elements x 7 partials while seven wait, twice per unit, and this
            // body pays that where the dot4 body pays it once - which is what the blocks-per-wave
            // ordering of the losing panel is. docs/matvec-drain.md.
            if (wave > 0) {
                #pragma unroll
                for (int r = 0; r < 8; r++) if (r != wave) red[((wave - 1) * 8 + r) * 32 + lane] = y[r];
            } else {
                #pragma unroll
                for (int r = 1; r < 8; r++) red[((r - 1) * 8 + r) * 32 + lane] = y[r];
            }
            __syncthreads();
            if (col < nrows) {
                const int r = wave;
                float v = r ? red[((r - 1) * 8 + r) * 32 + lane] : y[0];
                #pragma unroll
                for (int w = 1; w < NW; w++) v += (w == r) ? y[r] : red[((w - 1) * 8 + r) * 32 + lane];
                const int row = pass ? (half ? 2 * r + 1 : 2 * r + 16) : (half ? 2 * r + 17 : 2 * r);
                float * o = seg.out + (size_t) col * N + (size_t) tile * 32 + row;
                if (KS > 1) atomicAdd(o, v);
                else if (seg.add) *o += v;
                else *o = v;
            }
#elif HALO_MV8_DRAIN8
            if (wave > 0) {
                #pragma unroll
                for (int r = 0; r < 8; r++) red[((wave - 1) * 8 + r) * 32 + lane] = y[r];
            } else {
                #pragma unroll
                for (int r = 0; r < 8; r++) red0[r * 32 + lane] = y[r];
            }
            __syncthreads();
            if (wave == 0 && col < nrows) {
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float v = red0[r * 32 + lane];
                    #pragma unroll
                    for (int w = 1; w < NW; w++) v += red[((w - 1) * 8 + r) * 32 + lane];
                    const int row = pass ? (half ? 2 * r + 1 : 2 * r + 16) : (half ? 2 * r + 17 : 2 * r);
                    float * o = seg.out + (size_t) col * N + (size_t) tile * 32 + row;
                    if (KS > 1) atomicAdd(o, v);
                    else if (seg.add) *o += v;
                    else *o = v;
                }
            }
#else
            if (wave > 0) {
                #pragma unroll
                for (int r = 0; r < 8; r++) red[((wave - 1) * 8 + r) * 32 + lane] = y[r];
            }
            __syncthreads();
            if (wave == 0 && col < nrows) {
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float v = y[r];
                    #pragma unroll
                    for (int w = 1; w < NW; w++) v += red[((w - 1) * 8 + r) * 32 + lane];
                    const int row = pass ? (half ? 2 * r + 1 : 2 * r + 16) : (half ? 2 * r + 17 : 2 * r);
                    float * o = seg.out + (size_t) col * N + (size_t) tile * 32 + row;
                    if (KS > 1) atomicAdd(o, v);
                    else if (seg.add) *o += v;
                    else *o = v;
                }
            }
#endif
            __syncthreads();
        }
    }
    end_phase(c);
}

