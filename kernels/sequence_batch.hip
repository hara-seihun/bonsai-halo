#include "sequence_batch.h"
#include "ffn_operands.hpp"
#include "sequence_operands.hpp"
#include "sequence_stream.hpp"
#include "prep_batch.h"
#include "device.hpp"
#include <algorithm>
#include <cstdio>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <vector>
namespace halo {
using namespace ffnb;

// Which four-bit operand map the two projections' block loop compiles, and whether the one it
// replaced is compiled beside it.
//
// The map and the stored code order are one choice: an image written in the nine-valued pair
// alphabet cannot be read by the nibble expansion and the other way round, so `project` takes the
// map as a template argument and `sequence_batch_sync_quant` reorders the image to match it. Both
// arms produce the same operand bits, so the arm is a cost axis and never a numerical one - and it
// is a PROCESS-WIDE one, because an image holds one order at a time and the reorder costs about
// 30 ms, so a process that both ingests and decodes cannot take a different map for each.
//
// 1 IS SELECTED AND THE CENSUS SAYS IT SHOULD NOT BE. The pair map is 57 fewer work slots a block
// (325 -> 267 at four token tiles, 236 -> 179 at two, 191 -> 134 at one) for the same bytes, the
// same loads, the same matrix instructions and two fewer registers, and it measures **+1.9% on both
// projection phases** at the 128-row shape the product ingests on, over six samples an arm with the
// phases it cannot reach as the control and one residual FNV-64 across every sample. At a 32-stream
// decode step it is -1.5% and -1.1%, inside the same panel's own control drift. The block loop does
// not pay for these instructions; docs/seq-pair-operand.md has the panel, the four-cell barrier
// control and the compiled issue orders that explain it.
//
// The default build carries the selected arm alone, so the losing map is in no shipped kernel.
// `-DHALO_SEQ_I4OP_CONTROL=1` compiles both and lets `tools/batch_profile --seq-i4op 1,2` walk them
// in ONE process against one warmed device and one clock - which this box needs, because it moves
// up to 15% between processes (docs/row-band-fit.md).
#ifndef HALO_SEQ_I4OP
#define HALO_SEQ_I4OP 1
#endif
#ifndef HALO_SEQ_I4OP_CONTROL
#define HALO_SEQ_I4OP_CONTROL 0
#endif
// The cost ablation that says what the block loop is waiting for, and it is a build-time
// instrument rather than a runtime one: at its default of 0 no line of the selected arm changes.
//
//   1  the WEIGHT stream's block index is masked, so every block reads block 0's codes and scales
//   2  the ACTIVATION fragment's block offset is masked, so every block reads block 0's fragments
//   3  both
//
// Same instructions, same request counts, same expansion, same matrix work - only which address a
// request names stops walking, so the stream collapses onto one block and hits in cache. The arms
// compute the wrong numbers by construction and exist to separate issue from latency. A literal
// zero would let the compiler hoist the whole stream out of the block loop and the arm would then
// measure a kernel nobody runs, so the zero is an external `__device__` int written once from the
// host; `ffn_batch.hip` learned that the same way.
// Which token a (token tile, lane) position of the block loop carries. The wmma's sixteen B
// columns are independent - the instruction reduces over K alone and never across columns - so
// WHICH token sits in column `col` of tile `t` is a labelling this kernel chooses, not something
// the image, the weights or the arithmetic fix. Both maps cover the same `TT * 16` tokens of a
// group and every output element takes the same products in the same order; only the addresses
// the lane computes move. docs/seq-operand-coord.md.
//
// A map is worth what its WAVE reads, not what its lane reads: sixteen lanes issue one address
// each and the coalescer turns them into cache lines, so a map that makes a lane's own bytes
// contiguous can still make the wave's instruction straddle four times the lines. All three below
// fetch the same 512 bytes per (slice, token group) and differ only in how those bytes are dealt
// across the sixteen lanes of one instruction.
//
//   0  token = first + t*16 + col   the deployed map. A lane's TT fragments are 16*FB bytes
//                                   apart, so at four-bit activations (FB = 8) the wave issues
//                                   TT*8 `global_load_b64` per 128-K block - 32 of the 39
//                                   requests in a block that carries 32 `v_wmma`. Each of them is
//                                   sixteen lanes over 128 contiguous bytes: two lines, fully used.
//   1  token = first + col*TT + t   a lane's TT fragments become one contiguous TT*FB run, so the
//                                   wave reads them as `global_load_b128` and the request count
//                                   halves. MEASURED AND REJECTED, +15.8% of the input projection:
//                                   the lanes are now TT*FB apart, so one instruction touches
//                                   eight lines and uses half of each, and the pair of them costs
//                                   twice the line lookups the deployed four cost.
//   2  token = first + (t>>1)*32    the pair map: a lane still gets its tiles two at a time in one
//              + col*2 + (t&1)      16-byte load, and the sixteen lanes of that load are
//                                   CONTIGUOUS - 256 bytes, four lines, fully used. Half the
//                                   requests of the deployed map at exactly its line traffic.
//                                   Needs an even TT; TT = 1 runs map 0.
//
// No image byte moves under any of them and no writer changes: `capture`, the wide prep, the
// attention combine and the head all keep writing `[K16 slice][token]`, and at TT = 1 every map is
// the same expression, so that width is untouched code.
//
// `-DHALO_SEQ_TMAP_CONTROL=1` compiles all three and lets `tools/batch_profile --seq-tmap 0,1,2`
// walk them in one process on one clock; the default build resolves to a single instantiation.
#ifndef HALO_SEQ_TMAP
#define HALO_SEQ_TMAP 0
#endif
#ifndef HALO_SEQ_TMAP_CONTROL
#define HALO_SEQ_TMAP_CONTROL 0
#endif
#if HALO_SEQ_TMAP != 0 && HALO_SEQ_TMAP != 1 && HALO_SEQ_TMAP != 2
#error "HALO_SEQ_TMAP must be 0 (tile-major), 1 (lane-major) or 2 (pair-major) token map"
#endif
#ifndef HALO_SEQ_PIN
#define HALO_SEQ_PIN 0
#endif
// The same ablation reached at runtime, so all four arms live in one process against one warmed
// device and one clock. `-DHALO_SEQ_PIN=n` bakes an arm in and leaves the deployed block loop
// alone; `-DHALO_SEQ_PIN_CONTROL=1` instead multiplies the block index by a device word the host
// writes, so `--seq-pin 0,1,2,3` walks the arms without a rebuild. The multiply costs the control
// build one `v_mul_lo_u32` per block that the default build does not emit, so read the control
// build's arms against each other and never against a default build's absolute time.
//
// Both words are the ablation's opaque one: a literal would let the compiler fold the whole walk
// out of the block loop and the arm would then measure a kernel nobody runs.
#ifndef HALO_SEQ_PIN_CONTROL
#define HALO_SEQ_PIN_CONTROL 0
#endif
__device__ int halo_seq_probe_zero = 0;
__device__ int halo_seq_pin_w = 1;   // 0 pins the WEIGHT stream's block index
__device__ int halo_seq_pin_a = 1;   // 0 pins the ACTIVATION fragment's block offset
#if HALO_SEQ_I4OP != 1 && HALO_SEQ_I4OP != 2
#error "HALO_SEQ_I4OP must be 1 (nibble sign extension) or 2 (nine-valued pair codes)"
#endif

namespace {
void check(hipError_t e) { if(e!=hipSuccess)throw std::runtime_error(std::string("sequence_batch: ")+hipGetErrorString(e)); }
// `tstep` and `bstep` are the run index's two steps: a run is one (row tile, 128-block) pair,
// 512 bytes of codes and sixteen halves of scale, and its index is `tile*tstep + b*bstep`.
//
//   tile-major   tstep = nb, bstep = 1     the deployed image: a wave's own blocks are contiguous
//   block-major  tstep = 1,  bstep = ntiles  every resident wave's block b inside one run
//
// Same bytes, same values, same order of use. Only which 512-byte run a pair lands on moves, so
// every output bit is the deployed one whichever order the image is in.
// docs/weight-stream-order.md has why the choice is measurable at all.
struct Matrix { uint8_t *codes=nullptr; uint16_t *scales=nullptr; int n=0, nb=0, n0=0, n1=0; int tstep=0, bstep=1; };
struct Layer { Matrix input,output; bool recurrent=false; };
inline void image_steps(Matrix &w,int image) {
    if(image==SEQ_IMAGE_BLOCK){w.tstep=1;w.bstep=w.n/16;} else {w.tstep=w.nb;w.bstep=1;}
}
// SPREAD writes the byte-lane order expand_i8_spread reads (sequence_operands.hpp); the deployed
// order stays for the scaled-FP16 control, whose expansion reads four codes per byte anyway.
template<bool SPREAD> __global__ void repack(const uint8_t *src, uint8_t *dst, uint16_t *sc, int nb,int n,int rowoff,int tstep,int bstep) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)n*nb)return;
    int lane=i%32, b=i/32%nb, tile=i/(32*nb);
    const uint8_t *p=src+((size_t)tile*nb+b)*TILE_BLOCK_BYTES;
    auto qa=*(const uint4*)(p+tile_off_qs_a(lane)); auto qb=*(const uint2*)(p+tile_off_qs_b(lane));
    unsigned tail=*(const unsigned*)(p+tile_off_tail(lane)), w[8]; halo_block_codes(qa,qb,tail,w);
    int t=tile*2+(lane>>4)+rowoff/16, col=lane&15;
    size_t block=(size_t)t*tstep+(size_t)b*bstep;
#pragma unroll
    for(int s=0;s<8;++s)*(unsigned*)(dst+block*512+col*32+s*4)=SPREAD?spread_codes(w[s]):w[s];
    sc[block*16+col]=tail>>16;
}
// Move one matrix between the two run orders. A run moves whole, so this composes with
// `reorder_codes`' permutation inside a code word and with anything else that treats the image as
// a flat array of runs.
__global__ void permute_runs(const uint8_t *src_c,const uint16_t *src_s,uint8_t *dst_c,uint16_t *dst_s,
                             int ntiles,int nb,int stile,int sblk,int dtile,int dblk) {
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    const size_t runs=(size_t)ntiles*nb;
    if(i>=runs*32)return;                       // one thread per 16 bytes of a run's codes
    const size_t r=i>>5; const int part=int(i&31);
    const int t=int(r/(size_t)nb), b=int(r-(size_t)t*nb);
    const size_t s=(size_t)t*stile+(size_t)b*sblk, d=(size_t)t*dtile+(size_t)b*dblk;
    *(uint4*)(dst_c+d*512+part*16)=*(const uint4*)(src_c+s*512+part*16);
    if(part<16)dst_s[d*16+part]=src_s[s*16+part];
}
template<bool SCALED> __global__ void capture(void *B,float *sc,float *meta,const int8_t *xq,const float *xs,const float *ab,const float *ninv,
                        int nb,int count,int offset,int npad,bool save_meta) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count*nb*8) {
        int s=i%8,b=i/8%nb,r=i/(8*nb);
        size_t dst=(size_t)(b*8+s)*npad+offset+r;
        const int8_t *src=xq+(size_t)r*nb*128+b*128+s*16;
        if constexpr(SCALED) {
            half16 v; float scale=xs[r*nb+b];
#pragma unroll
            for(int j=0;j<16;++j)v[j]=(_Float16)(float(src[j])*scale);
            ((half16*)B)[dst]=v;
        } else ((int4v*)B)[dst]=*(const int4v*)src;
        if(s==0)sc[(size_t)(offset+r)*nb+b]=xs[r*nb+b];
    }
    if(save_meta&&i<count*(2*HV+1)) {
        int r=i/(2*HV+1),j=i%(2*HV+1);
        meta[(size_t)(offset+r)*(2*HV+1)+j]=j==2*HV?ninv[r]:ab[r*2*HV+j];
    }
}
__global__ void restore(const float *data,const float *meta,FwdParams p,int offset,int width,bool recurrent) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<p.nrows*width) {
        int r=i/width,j=i%width; float v=data[(size_t)(offset+r)*width+j];
        if(recurrent) { if(j<QKV_OUT)p.big[(size_t)r*QKV_OUT+j]=v; else p.zbuf[(size_t)r*VDIM+j-QKV_OUT]=v; }
        else { if(j<Q_OUT)p.qfull[(size_t)r*Q_OUT+j]=v; else if(j<Q_OUT+KV_OUT)p.kbuf[(size_t)r*KV_OUT+j-Q_OUT]=v; else p.vbuf[(size_t)r*KV_OUT+j-Q_OUT-KV_OUT]=v; }
    }
    if(recurrent&&i<p.nrows*(2*HV+1)) {
        int r=i/(2*HV+1),j=i%(2*HV+1);float v=meta[(size_t)(offset+r)*(2*HV+1)+j];
        if(j==2*HV)p.ninv[r]=v;else p.ab[r*2*HV+j]=v;
    }
}
// Row-tile ownership, the axis docs/ffn-schedule.md measured for the batched FFN, on the wide
// sequence projections.
//
//   SHARE = false  a workgroup's WV waves hold WV different row tiles and the same token group.
//                  This is the deployed shape: one wave carries the whole batch in its
//                  accumulators, which at 128 rows is eight token tiles and 252 VGPRs, and the
//                  hardware then grants five waves per SIMD32.
//   SHARE = true   the whole workgroup holds one row tile and each wave takes its own token group.
//                  The tokens covered per weight read grow with the wave count instead of with the
//                  accumulator width, so the same 128 tokens per weight read cost TT = 8 / WV
//                  accumulator sets. An output element is still owned end to end by one wave:
//                  nothing is reduced across waves and no summation order moves.
//
// Both forms walk the same K blocks in the same order and drain each block against the same two
// scales, so every output bit is the deployed one.
//
// RT is the third way to spend the batch, and it is the only one that changes how many activation
// bytes the kernel reads. TT and WV both leave the fragment stream at one whole read of a token
// tile's K row per weight row tile - 1024 row tiles times 80 kB times eight tiles is 671 MB per
// recurrent layer - because neither lets one fragment reach two row tiles. RT does: the wave holds
// RT adjacent row tiles and spends one loaded fragment against all of them, so the fragment bytes
// per matrix instruction fall from 256 to 256/RT while the weight stream, the accumulator width
// and every product stay where they were. It costs one A expansion per (slice, row tile) instead
// of per slice, which is the trade docs/sequence-fragment-reuse.md prices and measures.
//
// SB places the total scheduling barrier that sat after every second K16 slice from the day this
// kernel was written, with no recorded reason. docs/ffn-pair-lookahead.md measured the same
// construct in k_proj_opt and found it fitted at one of the two shapes it fired on; the same is
// true here, and the split is by accumulator width.
//
//   1  the fitted rule and the default: barrier at every width except four token tiles.
//   2  barrier after every second slice, the prior behaviour, kept reachable as the control.
//   0  no barrier at any width.
//
// At four token tiles a block holds thirty-two fragment loads, and letting the compiler look ahead
// costs eight registers and no wave slot (154 -> 162 VGPR, nine waves per SIMD32 either way). It
// is a null on the input projection, which has 2048 waves to hide a load with, and a 4-6% gain on
// the output projection, whose 320 row tiles give it exactly 640. At one token tile the same
// removal costs 82 -> 117 VGPR for a block holding eight loads and measures +33%.
//
// Four is where it is dropped and nowhere else, because four is where it was measured. Eight token
// tiles is the one width the panel did not reach and the compiler says it is worth reaching: the
// barrier costs that shape 205 -> 242 VGPR and seven waves per SIMD32 down to five, so removing it
// there may be free twice over. It is left alone because `--seq-sched 0` selects it and every
// published ownership table used it as the control arm.
// docs/sequence-fragment-reuse.md has the panel.
//
// LA is how the wave reaches its weight stream and when, and it is the only axis here that is
// about memory rather than about how the batch is spent. 1 is the deployed path: a per-lane
// pointer, so three loads a block carry 64-bit VGPR address pairs, and each is issued at the head
// of the block that consumes it. 2 reaches the same bytes from a wave-uniform base. 3 adds one
// 128-block of lookahead on top of the base form. kernels/sequence_stream.hpp carries the
// derivation, the ladder and the two instruments that predicted where it pays.
//
// I4 runs the same map on v_wmma_i32_16x16x16_iu4. Everything that decides a number is unchanged -
// the same weights against the same activations, the same K blocks in the same order, the same
// int32 accumulator per block and the same two scales in the same FP32 drain - and three things
// that decide a cost move: the weight word arrives in a four-bit order (sequence_operands.hpp),
// the activation fragment is eight bytes instead of sixteen, and the matrix instruction retires in
// half the cycles. It is reachable only with four-bit activations, which is a different numerical
// map than the deployed one and carries its own quality evidence; docs/sequence-input-a4.md.
//
// Its VALUE is 0 for the eight-bit operand and, on four-bit activations, WHICH ternary alphabet the
// stored word carries. Both four-bit arms emit the same sixteen signed nibbles in the same
// positions out of the same 512 bytes, so they differ in issue slots and in nothing else:
//
//   1  the nibble order. `w & 0x33333333` is the codes already in their nibbles and a two-step sign
//      extension turns code 3 into 0xf. Twelve instructions a K16 slice, 96 of the deployed
//      128-row block loop's 325 work slots - its largest class, against 32 `v_wmma`.
//   2  the nine-valued pair alphabet ffn_a4_operands.hpp stores for the A4 FFN, where a code names
//      a WEIGHT PAIR and is therefore the byte selector of the permute that expands it. A mask, a
//      shift and two `v_perm_b32`: five instructions a slice, 40 slots a block.
//
// docs/seq-pair-operand.md; `kernels/seq_op_check` proves both against the weights with no GPU.
template<int OP> __device__ __forceinline__ int2v seq_i4_operand(unsigned cw) {
    if constexpr(OP==2)return expand_i4_pair(cw);
    else return expand_i4_nib(cw);
}
template<int TT,bool ADD,bool PLANAR,int WV,bool SHARE,int RT=1,int SB=2,int LA=SEQ_OP_LANE,int I4=0,int TM=0>
__global__ __launch_bounds__(WV*32) void project(Matrix w,const typename std::conditional<I4!=0,int2v,int4v>::type *B,const float *sc,
                                        float *out,int rows,int npad) {
    using Frag=typename std::conditional<I4!=0,int2v,int4v>::type;
    constexpr unsigned FB=I4!=0?8u:16u;        // activation bytes per (token, K16 slice)
    // Maps 1 and 2 need at least two token tiles to say anything; at one tile every map is
    // `token = first + col` and this instantiation is the deployed one instruction for instruction.
    constexpr int TMAP=TT<2?0:TM;
    // The token a (tile, lane) position carries, split into its lane part and its tile part so the
    // lane part can stay a wave-uniform base's lane offset and the tile part an immediate.
    constexpr unsigned LSTEP=TMAP==0?FB:TMAP==1?(unsigned)TT*FB:2u*FB;   // bytes per lane step
    const int lane=threadIdx.x&31,wave=threadIdx.x>>5,col=lane&15,half=lane>>4;
    const int group=SHARE?int(blockIdx.x):int(blockIdx.x)*WV+wave;
    const int tile=group*RT;
    if(tile+RT>w.n/16)return;
    const int G=(npad+TT*16-1)/(TT*16), TGW=SHARE?WV:1;
    // Uniform for the whole wave; the lane's own column is a fixed offset inside the block.
    // The row tile is wave-uniform by construction - it is built from blockIdx and, off the shared
    // path, from the wave index - but divergence analysis has to be told, and telling it is what
    // lets the weight loads keep a scalar base and an unsigned lane offset instead of a 64-bit
    // VGPR address pair. Same form the activation fragment already uses two loops below.
    const int utile=__builtin_amdgcn_readfirstlane(tile);
    // Run-index steps, hoisted out of every loop: the block walk multiplies by these and the row
    // tile offsets by those, so the emitted addressing is the deployed one with `nb` replaced by a
    // field that happens to equal `nb` in the deployed order.
    const size_t tcode=(size_t)w.tstep*512,tscale=(size_t)w.tstep*16;
    const size_t bcode=(size_t)w.bstep*512,bscale=(size_t)w.bstep*16;
    const uint16_t *scale_base=w.scales+(size_t)utile*tscale;
    const uint8_t *code_base=w.codes+(size_t)utile*tcode;
    const unsigned code_lane=(unsigned)col*32u,scale_lane=(unsigned)col;
    // The deployed per-lane form, built from `tile` rather than its readfirstlane, so the default
    // instantiation is the canonical one down to the instruction.
    const uint16_t *lane_scale=w.scales+(size_t)tile*tscale+col;
    const uint8_t *lane_code=w.codes+(size_t)tile*tcode+col*32;
    int row_base=tile*16,stride=w.n;
    // Segment boundaries are multiples of 1024 rows and RT is at most 4 tiles, so a wave's whole
    // row-tile group lands in one destination plane; the launcher refuses any other geometry.
    if constexpr(PLANAR) {
        if(row_base>=w.n0+w.n1) {
            out+=(size_t)npad*(w.n0+w.n1);stride=w.n-w.n0-w.n1;row_base-=w.n0+w.n1;
        } else if(row_base>=w.n0) {
            out+=(size_t)npad*w.n0;stride=w.n1;row_base-=w.n0;
        } else stride=w.n0;
    }
    for(int g=int(blockIdx.y)*TGW+(SHARE?wave:0);g<G;g+=int(gridDim.y)*TGW) {
        // The token group is wave-uniform by construction, but it is built from threadIdx.x and
        // divergence analysis has to assume otherwise. Naming the uniformity is what lets a
        // fragment or scale load keep a scalar base and a 32-bit lane offset: without it the
        // activation scale costs a 64-bit multiply per (token tile, block) and the fragment walk a
        // carry pair per slice. Same form the vocabulary head and the FFN projection now use.
        const int first=__builtin_amdgcn_readfirstlane(g*TT*16);
        const char *bbase=(const char*)(B+first);       // wave-uniform
        const float *scp=sc+(size_t)first*w.nb;         // wave-uniform
        const unsigned blane=(unsigned)col*LSTEP,slane=(unsigned)(col*(TMAP==0?1:TMAP==1?TT:2)*w.nb);
        float y[RT][TT][8]{};
        SeqStream<LA,RT> stream(code_base,scale_base,code_lane,scale_lane,tcode,tscale,(unsigned)bcode,(unsigned)bscale);
        for(int b0=0;b0<w.nb;++b0) {
            // Identity in every shipped build; the ablation masks one or both streams' block index.
#if HALO_SEQ_PIN_CONTROL
            const int b=b0*halo_seq_pin_w;
            const int bx=b0*halo_seq_pin_a;
#else
            const int b=(HALO_SEQ_PIN&1)?b0*halo_seq_probe_zero:b0;
            const int bx=(HALO_SEQ_PIN&2)?b0*halo_seq_probe_zero:b0;
#endif
            SeqBlock blk[RT];
            float wr[RT][8];
            // The deployed operand path is kept verbatim rather than expressed through the stream,
            // so the default build emits the kernel canonical main emits - 374 block instructions,
            // 135 VGPR, ten waves per SIMD32 - and the named arms are measured against the thing
            // they would replace, not against a re-expression of it.
            if constexpr(LA==SEQ_OP_LANE) {
#pragma unroll
                for(int u=0;u<RT;++u) {
                    float ws=half_bits_to_float(lane_scale[(size_t)b*bscale+(size_t)u*tscale]);
#pragma unroll
                    for(int r=0;r<8;++r)wr[u][r]=__shfl(ws,2*r+half,32);
                }
                const uint8_t *p[RT];
                BlockWords<LAYOUT_LANE> words[RT];
#pragma unroll
                for(int u=0;u<RT;++u) {
                    p[u]=lane_code+(size_t)b*bcode+(size_t)u*tcode;
                    words[u]=BlockWords<LAYOUT_LANE>(p[u]);
                }
#pragma unroll
                for(int u=0;u<RT;++u)
#pragma unroll
                    for(int s=0;s<8;++s)blk[u].w[s]=words[u].take(p[u],s);
            } else {
                stream.take(blk,b,w.nb);
#pragma unroll
                for(int u=0;u<RT;++u) {
                    float ws=half_bits_to_float(blk[u].scale);
#pragma unroll
                    for(int r=0;r<8;++r)wr[u][r]=__shfl(ws,2*r+half,32);
                }
            }
            // One zero octet seeds every accumulator through the first slice's src2, instead of
            // RT * TT octets of v_mov per block.
            int8v acc[RT][TT];
            const int8v zero{};
            unsigned bo=blane+(unsigned)bx*(8u*(unsigned)npad*FB);
#pragma unroll
            for(int s=0;s<8;++s) {
                Frag x[TT];
                if constexpr(TMAP!=0&&I4!=0&&(TT%2)==0) {
                    // Both non-deployed maps put a lane's tiles 2t and 2t+1 side by side, so the
                    // wave reads them in 16-byte pieces: the same bytes into the same registers,
                    // half the requests, every offset an immediate off one wave-uniform base.
                    const char *q=bbase+bo;
#pragma unroll
                    for(int t=0;t<TT;t+=2) {
                        const int4v v=*(const int4v*)(q+(TMAP==1?t:(t>>1)*32)*int(FB));
                        x[t][0]=v[0];x[t][1]=v[1];x[t+1][0]=v[2];x[t+1][1]=v[3];
                    }
                } else {
#pragma unroll
                    for(int t=0;t<TT;++t)x[t]=*(const Frag*)(bbase+bo+t*int(16u*FB));
                }
#pragma unroll
                for(int u=0;u<RT;++u) {
                    const unsigned cw=blk[u].w[s];
                    if constexpr(I4!=0) {
                        const int2v a=seq_i4_operand<I4>(cw);
#pragma unroll
                        for(int t=0;t<TT;++t)
                            acc[u][t]=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true,a,true,x[t],s?acc[u][t]:zero,false);
                    } else {
                        const int4v a=expand_i8_spread(cw);
#pragma unroll
                        for(int t=0;t<TT;++t)
                            acc[u][t]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,x[t],s?acc[u][t]:zero,false);
                    }
                }
                bo+=(unsigned)npad*FB;
                if constexpr(SB==2||(SB==1&&TT!=4))if(s&1)__builtin_amdgcn_sched_barrier(0);
            }
#pragma unroll
            for(int t=0;t<TT;++t) {
                // Uniform pointer, divergent 32-bit index: the scale image is [token][block], so
                // the lane stride is w.nb floats and only the lane part belongs in the offset.
                float a=(scp+((TMAP==0?t*16:TMAP==1?t:(t>>1)*32+(t&1))*w.nb+b))[slane];
#pragma unroll
                for(int u=0;u<RT;++u)
#pragma unroll
                    for(int r=0;r<8;++r)y[u][t][r]=__builtin_fmaf(float(acc[u][t][r]),wr[u][r]*a,y[u][t][r]);
            }
        }
#pragma unroll
        for(int t=0;t<TT;++t) {
            int token=first+(TMAP==0?t*16+col:TMAP==1?col*TT+t:(t>>1)*32+col*2+(t&1));if(token>=rows)continue;
#pragma unroll
            for(int u=0;u<RT;++u)
#pragma unroll
                for(int r=0;r<8;++r) {
                    size_t o=(size_t)token*stride+row_base+16*u+2*r+half;
                    if constexpr(ADD)out[o]=out[o]+y[u][t][r];else out[o]=y[u][t][r];
                }
        }
    }
}
template<int TT,bool ADD> __global__ __launch_bounds__(128) void project_scaled(Matrix w,const half16 *B,
                                        float *out,int rows,int npad) {
    int lane=threadIdx.x&31,col=lane&15,half=lane>>4,tile=blockIdx.x*4+(threadIdx.x>>5);
    if(tile>=w.n/16)return;
    int first=blockIdx.y*TT*16;
    float8 acc[TT]{};
    for(int b=0;b<w.nb;++b) {
        const size_t run=(size_t)tile*w.tstep+(size_t)b*w.bstep;
        unsigned lo,hi;scale_tables(w.scales[run*16+col],lo,hi);
        const uint8_t *p=w.codes+run*512+col*32;
        BlockWords<LAYOUT_LANE> words(p);
#pragma unroll
        for(int s=0;s<8;++s) {
            half16 a=expand_scaled(words.take(p,s),lo,hi);
#pragma unroll
            for(int t=0;t<TT;++t) {
                half16 x=B[(size_t)(b*8+s)*npad+first+t*16+col];
                acc[t]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,x,acc[t]);
            }
            if(s&1)__builtin_amdgcn_sched_barrier(0);
        }
    }
#pragma unroll
    for(int t=0;t<TT;++t) {
        int token=first+t*16+col;if(token>=rows)continue;
#pragma unroll
        for(int r=0;r<8;++r) {
            size_t o=(size_t)token*w.n+tile*16+2*r+half;
            if constexpr(ADD)out[o]=out[o]+acc[t][r];else out[o]=acc[t][r];
        }
    }
}
}
// A batch of `ntiles` 16-token tiles is spent two ways here: TT tiles held in one wave's
// accumulators, and W waves of a workgroup owning different token groups of the same weight row
// tile. TT is accumulator width and costs 16 VGPRs per tile; W is grid parallelism that costs no
// registers and re-reads no weights, because the W waves stream the same bytes together.
//
//   SEQ_SCHED_TILE    the deployed shape: W = 1 inside a workgroup of four waves on four different
//                     row tiles, TT as wide as the batch allows.
//   SEQ_SCHED_SHARE   waves per shared row tile is the larger of what the accumulator budget needs
//                     to cover the batch in one weight read and what this projection's row-tile
//                     grid needs to fill the device. W = 1 degenerates to the control.
//   SEQ_SCHED_WIDE    the same rule against a budget of two token tiles instead of four.
//
// The budget is a register cliff, not a preference. The compiler allocates 242 VGPRs for eight
// token tiles and the hardware grants five waves per SIMD32; four tiles are 153 and nine, two are
// 95 and twelve. Four is the widest shape that still holds eight waves, so it is what the rule
// spends before it starts buying waves.
//
// The two terms are what makes the rule stage-aware, and the projections need that: the input
// matrix owns 1024 row tiles and the output matrix 320. At 32 rows the input grid already holds
// 1024 waves and sharing only halves the tokens each expansion serves, which measured +20.6%; the
// output grid holds 320, under the 640 this device wants, and the same sharing measured -5.9%.
//   SEQ_SCHED_PAIR    the share rule with two row tiles per wave and half the accumulator width,
//                     so the wave covers the same token tiles per weight read and spends one
//                     fragment read on two row tiles. This is the only arm that changes how many
//                     activation bytes the projection reads; docs/sequence-fragment-reuse.md
//                     prices it.
//
// The three ownerships all carry the fitted slice barrier. Two arms reach past it so the next
// engineer can re-fit it on another shape rather than inherit this one:
//   SEQ_SCHED_FREE    the selected ownership with no slice barrier at any width, and
//   SEQ_SCHED_HOLD    the selected ownership with the prior barrier after every second slice.
//   SEQ_SCHED_LANE    the selected ownership on the deployed operand path, and
//   SEQ_SCHED_BASE    the same with the weight loads reached from a wave-uniform base.
//
// Those last two are the memory axis rather than the batch-spending one, and they stay reachable
// as named arms whatever the rule does, so a panel taken before the rule was fitted and one taken
// after mean the same thing.
enum SeqSched { SEQ_SCHED_TILE=0, SEQ_SCHED_SHARE=1, SEQ_SCHED_WIDE=2, SEQ_SCHED_PAIR=3,
                SEQ_SCHED_FREE=4, SEQ_SCHED_HOLD=5, SEQ_SCHED_LANE=6, SEQ_SCHED_BASE=7 };
struct SeqShape { int w; int tt; bool share; int rt; int sb; int la; };
int ceil_div_i(int a,int b){return (a+b-1)/b;}
int floor_pow2(int v){int r=1;while(r*2<=v)r*=2;return r;}
constexpr int SEQ_WAVES_WANTED=640;   // eight waves per SIMD32 on this device's 80 units

// `tiles` is the projection's 16-row tile count, which is what decides whether its grid fills the
// device on its own. The overrides let a sweep cross the two axes without a rebuild and cannot
// reach a width the launcher does not carry.
// THE TOKEN-TILE AND BARRIER AXES OF THIS RULE ARE BOTH MEASURED AND BOTH AT THEIR OPTIMUM, and
// docs/row-band-fit.md holds the ladder so the next engineer does not re-walk it. The rule asks
// for as many token tiles as the accumulator budget allows once the device holds
// SEQ_WAVES_WANTED waves, which reads like an oversight against the register table - `project`
// takes 73-78 VGPRs at one token tile, 90-95 at two, 133-144 at four and 213-240 at eight, with no
// spill anywhere, so four tiles costs a third of the wave slots (16 per SIMD32 down to 10) and
// eight costs more than half. Clamping the rule at two tiles measured -2.6 to -4.8% of the two
// projection phases under the eight-bit operand and +10.3 to +16.6% under the four-bit operand
// this route now defaults to, because the narrower fragment halves what a token tile costs to
// fetch and the amortisation the wave slots were paying for arrives for free. The second slice
// barrier is the same story from the other side: it takes the four-bit TT=4 arm from 180 VGPR at
// eight waves to 133 at ten, and that wave slot measured +4.6 to +10.9%. Neither arm is in the
// tree. If you change the operand width, re-walk the ladder before you trust either constant.
SeqShape sequence_shape(int npad,int sched,int tiles,int tt_override,int w_override,int rt_override,int sb_override=-1,int la_override=0) {
    const int ntiles=std::max(1,npad/16);
    const int deployed=npad<=16?1:npad<=32?2:npad<=64?4:8;   // the deployed width rule, unchanged
    SeqShape s{4,deployed,false,1,1,1};
    if(sched!=SEQ_SCHED_TILE) {
        const int cap=sched==SEQ_SCHED_WIDE?2:4;
        int wv=std::min(4,std::max(ceil_div_i(ntiles,cap),ceil_div_i(SEQ_WAVES_WANTED,std::max(1,tiles))));
        if(wv==3)wv=2;   // the launcher carries 1, 2 and 4
        const int tt=floor_pow2(std::max(1,std::min(cap,ceil_div_i(ntiles,wv))));
        wv=std::min(wv,std::max(1,ceil_div_i(ntiles,tt)));   // no more waves than token groups
        s=wv<=1?SeqShape{4,deployed,false,1,1,1}:SeqShape{wv,tt,true,1,1,1};
    }
    if(sched==SEQ_SCHED_PAIR)s.sb=2;   // the measured pair arm carried the prior barrier
    if(sched==SEQ_SCHED_FREE)s.sb=0;
    if(sched==SEQ_SCHED_HOLD)s.sb=2;
    // Two row tiles per wave against one fragment read, at the accumulator budget the wave already
    // holds: the token tiles per wave halve so RT * TT is what it was, and only the activation
    // traffic and the expansion count move.
    if(sched==SEQ_SCHED_PAIR){s.rt=2;s.tt=std::max(1,s.tt/2);}
    if(w_override>0) {
        const int wv=w_override>=4?4:w_override>=2?2:1;
        s.share=wv>1;s.w=wv>1?wv:4;
        if(tt_override<=0)s.tt=wv>1?floor_pow2(std::max(1,std::min(4,ceil_div_i(ntiles,wv)))):deployed;
    }
    if(rt_override>0)s.rt=rt_override>=2?2:1;
    if(sb_override>=0)s.sb=sb_override;
    // The deployed operand path is what every shape selects. The wave-uniform base is a named arm
    // rather than a rule because its sign depends on the instantiation and not on anything the
    // launcher can see: docs/sequence-weight-address.md has the four shapes and the two nulls.
    if(sched==SEQ_SCHED_LANE)s.la=SEQ_OP_LANE;
    if(sched==SEQ_SCHED_BASE)s.la=SEQ_OP_BASE;
    if(la_override>0)s.la=std::min(2,la_override);
    if(s.rt>1)s.la=SEQ_OP_LANE;   // the row-tile pair arm carries one operand path
    if(tt_override>0)s.tt=std::max(1,std::min(floor_pow2(tt_override),8));
    if(s.rt>1&&s.tt>4)s.tt=4;   // the launcher carries RT = 2 at one, two and four token tiles
    return s;
}

// The stored order of a projection matrix's code words. All three orders are the same 2.000 bits
// per weight in the same 512 bytes per (row tile, block); only which bit a code sits on moves, and
// each expansion reads the order it wants for free. Switching is a pure bit permutation, so it is
// reversible and loses nothing. Each stage carries its own order because each stage carries its
// own activation coordinate and its own operand map.
__global__ void reorder_codes(uint8_t *codes,size_t ndwords,int from,int to) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=ndwords)return;
    unsigned *p=(unsigned*)codes+i;
    *p=seq_from_deployed(seq_to_deployed(*p,from),to);
}
struct SequenceBatch {
    Layer layers[NLAYER]; int capacity; size_t bytes=0;
    uint8_t *B=nullptr;float *sc=nullptr,*projected=nullptr,*meta=nullptr;
    bool scaled=false,direct=false,resident=false;
    // `repack` writes the spread order, so that is where every image starts.
    int input_order=SEQ_ORD_SPREAD,output_order=SEQ_ORD_SPREAD;
    // Which four-bit operand map the block loop runs, and therefore which order the image holds
    // when a stage is on four-bit activations. 2 is the nine-valued pair alphabet (selected); 1 is
    // the nibble sign extension it replaced, reachable only in a control build.
    int i4op=HALO_SEQ_I4OP;
    // Which token a (tile, lane) position of the block loop carries; see the map comment above
    // `project`. Only a -DHALO_SEQ_TMAP_CONTROL=1 build carries both.
    int tmap=HALO_SEQ_TMAP;
    // Which route wrote the output operand for the layer about to be projected. The four-bit
    // output coordinate is a property of the wide writers (`resident_output` and the wide
    // attention combine); the per-slice half-layer inside the persistent kernel has no nibble
    // store, so a pass that falls back to it cannot be read as nibbles.
    bool out_wide=true;
    int image=SEQ_IMAGE_BLOCK;   // measured default; HALO_SEQUENCE_IMAGE=tile is the control
    int state_split=-1;   // -1 lets gdn_resident_split_for pick by grid depth; HALO_GDN_SPLIT pins one
    int sched=SEQ_SCHED_SHARE,tt_override=0,w_override=0,rt_override=0,sb_override=-1,la_override=0;
    float *state_tokens=nullptr,*state_output=nullptr;
    std::vector<void*> allocations;
    template<class T> void alloc(T *&p,size_t n) {
        void *q=nullptr;check(hipMalloc(&q,n*sizeof(T)));allocations.push_back(q);p=(T*)q;bytes+=n*sizeof(T);check(hipMemset(q,0,n*sizeof(T)));
    }
    ~SequenceBatch(){for(void *p:allocations)(void)hipFree(p);}
};
size_t sequence_batch_required_bytes(const LayerW *layers,int capacity) {
    if(!layers||capacity<1||capacity>PASSMAX)throw std::runtime_error("sequence capacity must be 1..PASSMAX");
    const int pad=(capacity+15)/16*16;
    const char *operand=getenv("HALO_SEQUENCE_OPERAND");
    if(operand&&std::string(operand)!="int8"&&std::string(operand)!="scaled-f16")
        throw std::runtime_error("HALO_SEQUENCE_OPERAND must be int8 or scaled-f16");
    const bool scaled=operand&&std::string(operand)=="scaled-f16";
    const char *layout=getenv("HALO_SEQUENCE_LAYOUT");
    if(layout&&std::string(layout)!="staged"&&std::string(layout)!="direct")
        throw std::runtime_error("HALO_SEQUENCE_LAYOUT must be staged or direct");
    const bool direct=layout?std::string(layout)=="direct":!scaled;
    if(direct&&scaled)throw std::runtime_error("direct sequence layout currently requires int8 operands");
    bool resident=direct;
    if(const char *v=getenv("HALO_GDN_RESIDENT")) {
        std::string value(v);if(value!="0"&&value!="1")throw std::runtime_error("HALO_GDN_RESIDENT must be 0 or 1");
        resident=value=="1";
    }
    if(resident&&!direct)throw std::runtime_error("resident GDN requires direct integer sequence layout");
    size_t bytes=0;
    if(resident)bytes+=(size_t)pad*(CONV_CH+2*HV)*sizeof(float)+(size_t)pad*VDIM*sizeof(float);
    bytes+=((size_t)48*8*pad+128)*(scaled?32:16);
    bytes+=(size_t)(pad+128)*48*sizeof(float);
    bytes+=(size_t)pad*(QKV_OUT+VDIM)*sizeof(float);
    bytes+=(size_t)pad*(2*HV+1)*sizeof(float);
    for(int l=0;l<NLAYER;++l) {
        const int input=layers[l].recurrent?QKV_OUT+VDIM:Q_OUT+2*KV_OUT;
        bytes+=(size_t)input*NB_D*(sizeof(uint8_t)*32+sizeof(uint16_t));
        bytes+=(size_t)D*NB_V*(sizeof(uint8_t)*32+sizeof(uint16_t));
    }
    return bytes;
}
SequenceBatch *create_sequence_batch(const LayerW *layers,int capacity) {
    if(capacity<1||capacity>PASSMAX)throw std::runtime_error("sequence capacity must be 1..PASSMAX");
    auto *s=new SequenceBatch; s->capacity=capacity;
    try {
        int pad=(capacity+15)/16*16;
        const char *operand=getenv("HALO_SEQUENCE_OPERAND");
        if(operand&&std::string(operand)!="int8"&&std::string(operand)!="scaled-f16")
            throw std::runtime_error("HALO_SEQUENCE_OPERAND must be int8 or scaled-f16");
        s->scaled=operand&&std::string(operand)=="scaled-f16";
        const char *layout=getenv("HALO_SEQUENCE_LAYOUT");
        if(layout&&std::string(layout)!="staged"&&std::string(layout)!="direct")
            throw std::runtime_error("HALO_SEQUENCE_LAYOUT must be staged or direct");
        s->direct=layout?std::string(layout)=="direct":!s->scaled;
        if(s->direct&&s->scaled)throw std::runtime_error("direct sequence layout currently requires int8 operands");
        s->resident=s->direct;
        if(const char *v=getenv("HALO_GDN_RESIDENT")) {
            std::string value(v);if(value!="0"&&value!="1")throw std::runtime_error("HALO_GDN_RESIDENT must be 0 or 1");
            s->resident=value=="1";
        }
        if(const char *v=getenv("HALO_GDN_SPLIT")) {
            std::string value(v);if(!gdn_split_valid(atoi(v))||value!=std::to_string(atoi(v)))throw std::runtime_error("HALO_GDN_SPLIT must be 2, 4, 8 or 16");
            s->state_split=atoi(v);
        }
        if(s->resident&&!s->direct)throw std::runtime_error("resident GDN requires direct integer sequence layout");
        if(s->resident){s->alloc(s->state_tokens,(size_t)pad*(CONV_CH+2*HV));s->alloc(s->state_output,(size_t)pad*VDIM);}
        s->alloc(s->B,((size_t)48*8*pad+128)*(s->scaled?32:16));s->alloc(s->sc,(size_t)(pad+128)*48);
        s->alloc(s->projected,(size_t)pad*(QKV_OUT+VDIM));s->alloc(s->meta,(size_t)pad*(2*HV+1));
        auto matrix=[&](Matrix &w,int n,int nb) {
            w.n=n;w.nb=nb;image_steps(w,s->image);
            s->alloc(w.codes,(size_t)n*nb*32);s->alloc(w.scales,(size_t)n*nb);
        };
        if(const char *v=getenv("HALO_SEQUENCE_TT")) {
            std::string value(v);
            if(value!="1"&&value!="2"&&value!="4"&&value!="8")throw std::runtime_error("HALO_SEQUENCE_TT must be 1, 2, 4 or 8");
            s->tt_override=atoi(v);
        }
        if(const char *v=getenv("HALO_SEQUENCE_W")) {
            std::string value(v);
            if(value!="1"&&value!="2"&&value!="4")throw std::runtime_error("HALO_SEQUENCE_W must be 1, 2 or 4");
            s->w_override=atoi(v);
        }
        if(const char *v=getenv("HALO_SEQUENCE_RT")) {
            std::string value(v);
            if(value!="1"&&value!="2")throw std::runtime_error("HALO_SEQUENCE_RT must be 1 or 2");
            s->rt_override=atoi(v);
        }
        if(const char *v=getenv("HALO_SEQUENCE_SB")) {
            std::string value(v);
            if(value!="0"&&value!="1"&&value!="2")throw std::runtime_error("HALO_SEQUENCE_SB must be 0 (none), 1 (below four token tiles) or 2 (every second slice)");
            s->sb_override=atoi(v);
        }
        if(const char *v=getenv("HALO_SEQUENCE_LA")) {
            std::string value(v);
            if(value!="1"&&value!="2")throw std::runtime_error("HALO_SEQUENCE_LA must be 1 (per-lane pointer) or 2 (wave-uniform base)");
            s->la_override=atoi(v);
        }
        if(const char *v=getenv("HALO_SEQUENCE_IMAGE")) {
            std::string value(v);
            if(value!="tile"&&value!="block")throw std::runtime_error("HALO_SEQUENCE_IMAGE must be tile or block");
            s->image=value=="block"?SEQ_IMAGE_BLOCK:SEQ_IMAGE_TILE;
        }
        if(const char *v=getenv("HALO_SEQUENCE_SCHED")) {
            const int n=atoi(v);
            if(n<SEQ_SCHED_TILE||n>SEQ_SCHED_BASE)throw std::runtime_error("HALO_SEQUENCE_SCHED must be 0 (tile), 1 (share), 2 (wide), 3 (row-tile pair), 4 (no slice barrier), 5 (barrier every second slice), 6 (deployed operand path) or 7 (wave-uniform weight base)");
            s->sched=n;
        }
        if(const char *v=getenv("HALO_SEQ_I4OP")) {
            const std::string value(v);
            if(value!="1"&&value!="2")throw std::runtime_error("HALO_SEQ_I4OP must be 1 (nibble sign extension) or 2 (pair codes)");
            sequence_batch_set_i4op(s,atoi(v));
        }
        if(const char *v=getenv("HALO_SEQ_TMAP")) {
            const std::string value(v);
            if(value!="0"&&value!="1"&&value!="2")throw std::runtime_error("HALO_SEQ_TMAP must be 0 (tile-major), 1 (lane-major) or 2 (pair-major) token map");
            sequence_batch_set_tmap(s,atoi(v));
        }
        // The scaled-FP16 control reads the deployed word order; the integer operand reads the
        // byte-lane order its expansion wants. Same image size either way.
        // The ablation's opaque zero, written once so the device compiler cannot fold it.
        { const int z=0; check(hipMemcpyToSymbol(HIP_SYMBOL(halo_seq_probe_zero),&z,sizeof z)); }
        sequence_set_pin(0);
        const bool spread=!s->scaled;
        auto pack=[&](Matrix &w,const uint8_t *src,int n,int off) {
            const int grid=(n*w.nb+255)/256;
            if(spread)repack<true><<<grid,256>>>(src,w.codes,w.scales,w.nb,n,off,w.tstep,w.bstep);
            else repack<false><<<grid,256>>>(src,w.codes,w.scales,w.nb,n,off,w.tstep,w.bstep);
        };
        for(int l=0;l<NLAYER;++l) {
            auto &a=s->layers[l];const auto &w=layers[l];a.recurrent=w.recurrent;
            matrix(a.input,a.recurrent?QKV_OUT+VDIM:Q_OUT+2*KV_OUT,NB_D);matrix(a.output,D,NB_V);
            a.input.n0=a.recurrent?QKV_OUT:Q_OUT;a.input.n1=a.recurrent?VDIM:KV_OUT;
            if(a.recurrent){pack(a.input,w.qkv,QKV_OUT,0);pack(a.input,w.z,VDIM,QKV_OUT);pack(a.output,w.ssm_out,D,0);}
            else {pack(a.input,w.q,Q_OUT,0);pack(a.input,w.k,KV_OUT,Q_OUT);pack(a.input,w.v,KV_OUT,Q_OUT+KV_OUT);pack(a.output,w.o,D,0);}
        }
        if(s->bytes!=sequence_batch_required_bytes(layers,capacity))throw std::runtime_error("sequence memory estimate mismatch");
        check(hipDeviceSynchronize());return s;
    }catch(...){delete s;throw;}
}
void destroy_sequence_batch(SequenceBatch *s){delete s;}
size_t sequence_batch_bytes(const SequenceBatch *s){return s?s->bytes:0;}
bool sequence_direct_layout(const SequenceBatch *s){return s&&s->direct;}
bool sequence_resident_enabled(const SequenceBatch *s){return s&&s->resident;}
bool sequence_resident_layer(const SequenceBatch *s,int layer){return s&&s->resident&&layer>=0&&layer<NLAYER&&s->layers[layer].recurrent;}
void sequence_run_resident(SequenceBatch *s,FwdParams p,int layer,const ResidentSeq *seqs,int nseq,int rows,hipStream_t stream) {
    if(!s||!s->resident||!p.commit_state||nseq<1||nseq>128)throw std::runtime_error("resident GDN requires a committed sequence batch");
    GdnResidentParams a{};p.nrows=std::min(rows,RMAX);
    sequence_bind(s,p,layer,0,rows,2);p.o=s->state_output;
    a.p=p;a.nseq=nseq;a.nrows=rows;a.tokens=s->state_tokens;
    int count=0;
    for(int i=0;i<nseq;++i) {
        if(seqs[i].row0!=count||seqs[i].nrows<1||seqs[i].n_replay<0||seqs[i].n_replay>RMAX)
            throw std::runtime_error("invalid resident sequence rows or replay prefix");
        a.seqs[i]=seqs[i];count+=seqs[i].nrows;
    }
    if(count!=rows)throw std::runtime_error("resident sequence rows do not cover the batch");
    static bool probed=false;
    // The sliced reference commits its state once per eight-row slice and the resident path once
    // per pass, so the two agree bit for bit only while the state is exact: a packed coordinate
    // rounds a different number of times on each route. This diagnostic is an fp32 check.
    // The diagnostic reads the state region in place, so it wants a committed one: a deferring
    // process keeps part of the state in its pending list and this would compare a base.
    const bool probe=getenv("HALO_GDN_COMPARE")&&!probed&&layer==0&&nseq==1&&seqs[0].slot==0&&seqs[0].n_replay==0
                     &&gdn_state_format()==GDN_STATE_F32&&gdn_defer_effective()<2;
    struct Temps {std::vector<void*> ptrs;~Temps(){for(void *p:ptrs)(void)hipFree(p);}} temp;
    auto allocate=[&](size_t n) {void *q;check(hipMalloc(&q,n));temp.ptrs.push_back(q);return q;};
    float *ref_tokens=nullptr,*ref_output=nullptr,*ref_state=nullptr,*ref_ring=nullptr;
    if(probe) {
        ref_tokens=(float*)allocate((size_t)rows*(CONV_CH+2*HV)*4);
        ref_output=(float*)allocate((size_t)rows*VDIM*4);
        ref_state=(float*)allocate(GDN_STATE_FLOATS*4);ref_ring=(float*)allocate(GDN_RING_FLOATS*4);
        check(hipMemcpyAsync(ref_state,p.gdn_state,GDN_STATE_FLOATS*4,hipMemcpyDeviceToDevice,stream));
        check(hipMemcpyAsync(ref_ring,p.conv_ring,GDN_RING_FLOATS*4,hipMemcpyDeviceToDevice,stream));
        unsigned *sync=(unsigned*)allocate(32*4);
        int parity=seqs[0].parity;
        for(int offset=0;offset<rows;offset+=RMAX) {
            FwdParams q=p;int count=std::min(RMAX,rows-offset);
            q.nseq=1;q.nrows=count;q.seqs[0]={0,count,0,parity};
            for(int j=0;j<count;++j)q.rows[j].seq=0;
            q.gdn_state=ref_state;q.conv_ring=ref_ring;q.o=ref_output+(size_t)offset*VDIM;
            q.big=p.big+(size_t)offset*QKV_OUT;q.zbuf=p.zbuf+(size_t)offset*VDIM;
            q.ab=p.ab+(size_t)offset*2*HV;q.ninv=p.ninv+offset;q.sequence_offset=offset;
            q.bar=sync;q.work=sync+1;q.prof=nullptr;
            check(hipMemsetAsync(sync,0,32*4,stream));launch_sequence_part(q,layer,2,stream);
            const float *src=q.blk_cache+(size_t)parity*48*RMAX*BLK_TOKEN_FLOATS+CONV_CH;
            check(hipMemcpy2DAsync(ref_tokens+(size_t)offset*(CONV_CH+2*HV),(CONV_CH+2*HV)*4,
                src,BLK_TOKEN_FLOATS*4,(CONV_CH+2*HV)*4,count,hipMemcpyDeviceToDevice,stream));
            parity^=1;
        }
    }
    launch_gdn_resident(a,layer,s->state_split,stream);check(hipGetLastError());
    if(probe) {
        check(hipStreamSynchronize(stream));
        size_t mismatches=0;
        // Row-strided so a token record can be compared over a prefix of its columns.
        auto compare=[&](const char *name,const float *x,const float *y,size_t cols,size_t nrow,size_t stride) {
            const size_t n=cols*nrow;
            std::vector<unsigned> hx(n),hy(n);
            check(hipMemcpy2D(hx.data(),cols*4,x,stride*4,cols*4,nrow,hipMemcpyDeviceToHost));
            check(hipMemcpy2D(hy.data(),cols*4,y,stride*4,cols*4,nrow,hipMemcpyDeviceToHost));
            size_t dif=0,first=n;for(size_t i=0;i<n;++i)if(hx[i]!=hy[i]){++dif;if(first==n)first=i;}
            fprintf(stderr,"resident compare %s: %zu/%zu differ, first %zu (%08x vs %08x)\n",name,dif,n,first,first<n?hx[first]:0,first<n?hy[first]:0);
            mismatches+=dif;
        };
        // The last 2*HV columns hold either the raw alpha and beta projections or the evaluated
        // gates, and the two producers decide independently: the sliced reference by this build's
        // block-cache form (HALO_GDN_BLK_GATE), the resident pass by its own gate arm. They are two
        // representations of the same quantity, so compare those columns only when the two forms
        // match, and rely on state, ring and recurrent output - all compared whole, below - to catch
        // a wrong gate one step later.
        const bool res_pregate=gdn_resident_gate_for(rows,nseq)!=0;
        const size_t token_cols=gdn_blk_pregate()==res_pregate?(size_t)(CONV_CH+2*HV):(size_t)CONV_CH;
        compare("tokens",ref_tokens,s->state_tokens,token_cols,(size_t)rows,(size_t)(CONV_CH+2*HV));
        compare("output",ref_output,s->state_output,(size_t)rows*VDIM,1,(size_t)rows*VDIM);
        compare("state",ref_state,p.gdn_state,GDN_STATE_FLOATS,1,GDN_STATE_FLOATS);
        compare("ring",ref_ring,p.conv_ring,GDN_RING_FLOATS,1,GDN_RING_FLOATS);
        probed=true;
        if(mismatches)throw std::runtime_error("resident GDN diagnostic differs from the sliced path");
    }
}
bool sequence_bind(SequenceBatch *s,FwdParams &p,int layer,int offset,int total,int part) {
    if(!s||layer<0||layer>=NLAYER||offset<0||p.nrows<1||offset+p.nrows>total||total>s->capacity||(part!=1&&part!=2))
        throw std::runtime_error("invalid direct sequence binding");
    if(!s->direct)return false;
    const auto &a=s->layers[layer];int npad=(total+15)/16*16;
    p.sequence_q=(unsigned*)s->B;p.sequence_scales=s->sc;
    p.sequence_npad=npad;p.sequence_offset=offset;
    p.ab=s->meta+(size_t)offset*2*HV;
    p.ninv=s->meta+(size_t)((s->capacity+15)/16*16)*2*HV+offset;
    if(part==2) {
        if(a.recurrent) {
            p.big=s->projected+(size_t)offset*QKV_OUT;
            p.zbuf=s->projected+(size_t)npad*QKV_OUT+(size_t)offset*VDIM;
        } else {
            p.qfull=s->projected+(size_t)offset*Q_OUT;
            p.kbuf=s->projected+(size_t)npad*Q_OUT+(size_t)offset*KV_OUT;
            p.vbuf=s->projected+(size_t)npad*(Q_OUT+KV_OUT)+(size_t)offset*KV_OUT;
        }
    }
    return true;
}
void sequence_capture(SequenceBatch *s,const FwdParams &p,int layer,int offset,int total,int part,hipStream_t stream) {
    const auto &a=s->layers[layer];int nb=part==1?NB_D:NB_V;
    int count=std::max(p.nrows*nb*8,p.nrows*(2*HV+1));
#define CAPTURE(S) capture<S><<<(count+255)/256,256,0,stream>>>(s->B,s->sc,s->meta,p.xq,p.xs,p.ab,p.ninv,nb,p.nrows,offset,(total+15)/16*16,part==1&&a.recurrent)
    if(s->scaled){CAPTURE(true);}else{CAPTURE(false);}
#undef CAPTURE
}
void sequence_restore(SequenceBatch *s,const FwdParams &p,int layer,int offset,hipStream_t stream) {
    auto &a=s->layers[layer];int count=p.nrows*a.input.n;
    restore<<<(count+255)/256,256,0,stream>>>(s->projected,s->meta,p,offset,a.input.n,a.recurrent);
}
// A stage whose operand this pass cannot have written in seven levels. A coordinate the process
// was told is a contract and fails; the default drops that stage to the exact eight-bit operand,
// once, out loud, and the caller re-syncs the image so the codes match the kernel that reads them.
static int quant_or_fall_back(int q,int dropped,const char *stage,const char *route) {
    if(sequence_quant_explicit())
        throw std::runtime_error(std::string("a four-bit sequence ")+stage+" projection needs "+route+
            ", and HALO_SEQ_QUANT="+sequence_quant_name(q)+" asked for one this pass cannot carry");
    static bool said=false;
    if(!said) {
        said=true;
        fprintf(stderr,"bonsai-halo: this route writes the sequence %s operand without %s, which has no "
                       "seven-level store, so the %s projection falls back from %s to %s for this process. "
                       "HALO_SEQ_QUANT=%s demands it and fails instead.\n",
                stage,route,stage,sequence_quant_name(q),sequence_quant_name(dropped),sequence_quant_name(q));
    }
    sequence_quant_fall_back_to(dropped);
    return dropped;
}
void sequence_project(SequenceBatch *s,int layer,int total,int part,float *residual,hipStream_t stream) {
    if(!s||layer<0||layer>=NLAYER||total<1||total>s->capacity||(part!=1&&part!=2))throw std::runtime_error("invalid sequence projection");
    auto w=part==1?s->layers[layer].input:s->layers[layer].output;
    const int npad=(total+15)/16*16,tiles=w.n/16;
    float *out=part==1?s->projected:residual;
    // Four-bit activations exist only where a wide kernel writes the operand: that kernel is the
    // quantiser, and the per-slice route inside the persistent kernel has no four-bit store.
    // Refusing here is what keeps a mismatched operand from being read as numbers - and the weight
    // image is in the matching code order for the whole process, so a silent fallback to the
    // eight-bit kernel would read nibble-ordered codes as spread ones.
    int q=sequence_quant();
    if(part==1&&sequence_quant_in_seven(q)&&(!s->direct||!sequence_prep_wide_enabled()))
        q=quant_or_fall_back(q,sequence_quant_drop_in(q),"input","the direct layout and the wide input prep");
    if(part==2&&sequence_quant_out_seven(q)&&(!s->direct||!s->out_wide))
        q=quant_or_fall_back(q,sequence_quant_drop_out(q),"output",
                             "the direct layout and a wide writer (resident GDN for a recurrent layer, "
                             "wide attention for an attention one, which wants a pass wider than RMAX rows)");
    // The stored code order is a process-wide property of the image, so the image has to follow
    // the coordinate on whichever call changed it - including a part-2 fallback, which is the one
    // case this used to leave to part 1 of the next layer.
    sequence_batch_sync_quant(s,stream);
    const bool i4=part==1?sequence_quant_in_nibble(q):sequence_quant_out_nibble(q);
    if(s->scaled) {
        const int tt=s->tt_override>0?s->tt_override:sequence_shape(npad,SEQ_SCHED_TILE,tiles,0,0,0).tt;
        dim3 grid((tiles+3)/4,(npad+tt*16-1)/(tt*16));
#define RUN_SCALED(T) do { \
        if(part==1)project_scaled<T,false><<<grid,128,0,stream>>>(w,(const half16*)s->B,out,total,npad); \
        else project_scaled<T,true><<<grid,128,0,stream>>>(w,(const half16*)s->B,out,total,npad); } while(0)
        switch(tt){case 1:RUN_SCALED(1);break;case 2:RUN_SCALED(2);break;case 4:RUN_SCALED(4);break;default:RUN_SCALED(8);}
#undef RUN_SCALED
        check(hipGetLastError());
        return;
    }
    SeqShape sh=sequence_shape(npad,s->sched,tiles,s->tt_override,s->w_override,s->rt_override,s->sb_override,s->la_override);
    // A wave's row-tile group has to sit inside one destination plane and inside the matrix.
    if(sh.rt>1&&(tiles%sh.rt||(part==1&&s->direct&&(w.n0%(sh.rt*16)||w.n1%(sh.rt*16)))))sh.rt=1;
    // Token groups the batch needs, spread over the waves that share a row tile.
// The four-bit arm, at whichever operand map this process selected. Only the control build carries
// both; the default one resolves this to a single instantiation at compile time.
// The token map rides inside the four-bit arms only: at eight-bit activations a lane's fragment is
// already a whole 16-byte load and the map buys no request, so every eight-bit instantiation stays
// the deployed one instruction for instruction and the control build stays one compile.
#if HALO_SEQ_TMAP_CONTROL
#define RUN_TM(I4V,TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV) do { \
    if(s->tmap==1)project<TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV,I4V,1><<<grid,(WVV)*32,0,stream>>>(w,(const int2v*)s->B,s->sc,out,total,npad); \
    else if(s->tmap==2)project<TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV,I4V,2><<<grid,(WVV)*32,0,stream>>>(w,(const int2v*)s->B,s->sc,out,total,npad); \
    else project<TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV,I4V,0><<<grid,(WVV)*32,0,stream>>>(w,(const int2v*)s->B,s->sc,out,total,npad); } while(0)
#else
#define RUN_TM(I4V,TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV) \
    project<TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV,I4V,HALO_SEQ_TMAP><<<grid,(WVV)*32,0,stream>>>(w,(const int2v*)s->B,s->sc,out,total,npad)
#endif
#if HALO_SEQ_I4OP_CONTROL
#define RUN_I4(TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV) do { \
    if(s->i4op==1)RUN_TM(1,TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV); \
    else RUN_TM(2,TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV); } while(0)
#else
#define RUN_I4(TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV) RUN_TM(HALO_SEQ_I4OP,TTV,ADDV,PLV,WVV,SHV,RTV,SBV,LAV)
#endif
#define RUN_INT(TTV,WVV,SHV,RTV,SBV,LAV) do { \
    const int groups=ceil_div_i(npad,(TTV)*16),rowgroups=ceil_div_i(tiles,RTV); \
    dim3 grid((unsigned)((SHV)?rowgroups:ceil_div_i(rowgroups,WVV)),(unsigned)ceil_div_i(groups,(SHV)?(WVV):1)); \
    if(part==1) { \
        if(i4)RUN_I4(TTV,false,true,WVV,SHV,RTV,SBV,LAV); \
        else if(s->direct)project<TTV,false,true,WVV,SHV,RTV,SBV,LAV><<<grid,(WVV)*32,0,stream>>>(w,(const int4v*)s->B,s->sc,out,total,npad); \
        else project<TTV,false,false,WVV,SHV,RTV,SBV,LAV><<<grid,(WVV)*32,0,stream>>>(w,(const int4v*)s->B,s->sc,out,total,npad); \
    } else if(i4)RUN_I4(TTV,true,false,WVV,SHV,RTV,SBV,LAV); \
    else project<TTV,true,false,WVV,SHV,RTV,SBV,LAV><<<grid,(WVV)*32,0,stream>>>(w,(const int4v*)s->B,s->sc,out,total,npad); \
} while(0)
#define BY_TT(WVV,SHV,SBV,LAV) do { \
    switch(sh.tt){case 1:RUN_INT(1,WVV,SHV,1,SBV,LAV);break;case 2:RUN_INT(2,WVV,SHV,1,SBV,LAV);break; \
                  case 4:RUN_INT(4,WVV,SHV,1,SBV,LAV);break;default:RUN_INT(8,WVV,SHV,1,SBV,LAV);} } while(0)
#define BY_TT_PAIR(WVV,SHV) do { \
    switch(sh.tt){case 1:RUN_INT(1,WVV,SHV,2,2,1);break;case 2:RUN_INT(2,WVV,SHV,2,2,1);break; \
                  default:RUN_INT(4,WVV,SHV,2,2,1);} } while(0)
#define BY_SHAPE(SBV,LAV) do { \
    if(!sh.share)    BY_TT(4,false,SBV,LAV); \
    else if(sh.w>=4) BY_TT(4,true,SBV,LAV); \
    else             BY_TT(2,true,SBV,LAV); } while(0)
#define BY_SB(LAV) do { \
    switch(sh.sb){case 0:BY_SHAPE(0,LAV);break;case 1:BY_SHAPE(1,LAV);break;default:BY_SHAPE(2,LAV);} } while(0)
    if(sh.rt>1) {
        if(!sh.share)    BY_TT_PAIR(4,false);
        else if(sh.w>=4) BY_TT_PAIR(4,true);
        else             BY_TT_PAIR(2,true);
    }
    else switch(sh.la){case SEQ_OP_BASE:BY_SB(SEQ_OP_BASE);break;default:BY_SB(SEQ_OP_LANE);}
#undef BY_SB
#undef BY_SHAPE
#undef BY_TT_PAIR
#undef BY_TT
#undef RUN_INT
#undef RUN_I4
    check(hipGetLastError());
}
int sequence_batch_image(const SequenceBatch *s){return s?s->image:SEQ_IMAGE_TILE;}
// Move every matrix between the two run orders in place, so one process can order the two images
// against each other the way --seq-quant orders the two code orders. One scratch copy of the
// largest matrix, two passes over 1.9 GB, about 30 ms: an inter-arm cost, never inside a timing.
void sequence_batch_set_image(SequenceBatch *s,int image,hipStream_t stream) {
    if(!s)return;
    if(image!=SEQ_IMAGE_TILE&&image!=SEQ_IMAGE_BLOCK)
        throw std::runtime_error("sequence weight image must be 0 (tile-major) or 1 (block-major)");
    if(image==s->image)return;
    size_t widest=0;
    for(int l=0;l<NLAYER;++l)for(const Matrix *m:{&s->layers[l].input,&s->layers[l].output})
        widest=std::max(widest,(size_t)m->n*m->nb);
    uint8_t *tc=nullptr;uint16_t *ts=nullptr;
    check(hipMalloc((void**)&tc,widest*32));
    if(hipMalloc((void**)&ts,widest*2)!=hipSuccess){(void)hipFree(tc);throw std::runtime_error("sequence image scratch");}
    try {
        for(int l=0;l<NLAYER;++l)for(Matrix *m:{&s->layers[l].input,&s->layers[l].output}) {
            const int stile=m->tstep,sblk=m->bstep;
            image_steps(*m,image);
            const size_t runs=(size_t)m->n/16*m->nb;
            check(hipMemcpyAsync(tc,m->codes,runs*512,hipMemcpyDeviceToDevice,stream));
            check(hipMemcpyAsync(ts,m->scales,runs*32,hipMemcpyDeviceToDevice,stream));
            const size_t threads=runs*32;
            permute_runs<<<(unsigned)((threads+255)/256),256,0,stream>>>(tc,ts,m->codes,m->scales,
                m->n/16,m->nb,stile,sblk,m->tstep,m->bstep);
        }
        check(hipGetLastError());
        check(hipStreamSynchronize(stream));
    } catch(...) {(void)hipFree(tc);(void)hipFree(ts);throw;}
    (void)hipFree(tc);(void)hipFree(ts);
    s->image=image;
}
void sequence_batch_sync_quant(SequenceBatch *s,hipStream_t stream) {
    if(!s||!s->direct)return;
    const int q=sequence_quant();
    // A stage on eight-bit activations reads the spread order; a stage on four-bit activations reads
    // whichever four-bit alphabet this process's operand map compiles. Both are one enum here, so a
    // change of quantiser and a change of operand map take the same single reorder pass.
    const int in=sequence_quant_in_nibble(q)?s->i4op:SEQ_ORD_SPREAD;
    const int out=sequence_quant_out_nibble(q)?s->i4op:SEQ_ORD_SPREAD;
    if(in==s->input_order&&out==s->output_order)return;
    for(int l=0;l<NLAYER;++l) {
        if(in!=s->input_order) {
            const Matrix &m=s->layers[l].input; const size_t nd=(size_t)m.n*m.nb*8;
            reorder_codes<<<(unsigned)((nd+255)/256),256,0,stream>>>(m.codes,nd,s->input_order,in);
        }
        if(out!=s->output_order) {
            const Matrix &m=s->layers[l].output; const size_t nd=(size_t)m.n*m.nb*8;
            reorder_codes<<<(unsigned)((nd+255)/256),256,0,stream>>>(m.codes,nd,s->output_order,out);
        }
    }
    check(hipGetLastError());
    s->input_order=in;s->output_order=out;
}
// The four-bit operand map, as a runtime axis so a panel can walk both arms in one process. It is
// refused outside a control build, where only the selected arm is instantiated, and it takes effect
// on the next `sequence_project`, which syncs the image to it.
void sequence_batch_set_i4op(SequenceBatch *s,int op) {
    if(!s)return;
    if(op!=SEQ_ORD_NIBBLE&&op!=SEQ_ORD_PAIR)
        throw std::runtime_error("sequence four-bit operand map must be 1 (nibble sign extension) or 2 (pair codes)");
#if !HALO_SEQ_I4OP_CONTROL
    if(op!=HALO_SEQ_I4OP)
        throw std::runtime_error("this build carries one sequence four-bit operand map; rebuild with DEFS='-DHALO_SEQ_I4OP_CONTROL=1' to walk both");
#endif
    s->i4op=op;
}
int sequence_batch_i4op(const SequenceBatch *s){return s?s->i4op:HALO_SEQ_I4OP;}
bool sequence_batch_i4op_control(){return HALO_SEQ_I4OP_CONTROL!=0;}
// The token map, as a runtime axis. It changes no stored byte - the image, the scales and every
// writer are untouched - so unlike the operand map this one needs no image sync and takes effect
// on the very next `sequence_project`.
void sequence_batch_set_tmap(SequenceBatch *s,int map) {
    if(!s)return;
    if(map<0||map>2)
        throw std::runtime_error("sequence token map must be 0 (tile-major), 1 (lane-major) or 2 (pair-major)");
#if !HALO_SEQ_TMAP_CONTROL
    if(map!=HALO_SEQ_TMAP)
        throw std::runtime_error("this build carries one sequence token map; rebuild with DEFS='-DHALO_SEQ_TMAP_CONTROL=1' to walk both");
#endif
    s->tmap=map;
}
int sequence_batch_tmap(const SequenceBatch *s){return s?s->tmap:HALO_SEQ_TMAP;}
bool sequence_batch_tmap_control(){return HALO_SEQ_TMAP_CONTROL!=0;}
// Which route is about to write the output operand for this layer. See SequenceBatch::out_wide.
void sequence_set_output_wide(SequenceBatch *s,bool wide){if(s)s->out_wide=wide;}
int sequence_batch_sched(const SequenceBatch *s){return s?s->sched:0;}
// The shape overrides, at runtime rather than only at construction. `sequence_shape` is evaluated
// on every `sequence_project` call, so these take effect on the next launch and a sweep can walk
// the (waves, token tiles) ladder inside one process against one warmed device and one clock. The
// 33..127-row band was fitted by interpolation because that ladder had no in-process axis:
// HALO_SEQUENCE_TT/W are read in `create_sequence_batch`, so every point cost its own process and
// this box moves 15% between them. Zero restores the fitted rule. docs/row-band-fit.md.
void sequence_batch_set_shape(SequenceBatch *s,int tt,int wv) {
    if(!s)return;
    if(tt!=0&&tt!=1&&tt!=2&&tt!=4&&tt!=8)throw std::runtime_error("sequence shape TT override must be 0 (rule), 1, 2, 4 or 8");
    if(wv!=0&&wv!=1&&wv!=2&&wv!=4)throw std::runtime_error("sequence shape W override must be 0 (rule), 1, 2 or 4");
    s->tt_override=tt;s->w_override=wv;
}
int sequence_batch_shape_tt(const SequenceBatch *s){return s?s->tt_override:0;}
int sequence_batch_shape_w(const SequenceBatch *s){return s?s->w_override:0;}
// Which of the two streams still walks its blocks. Bit 0 pins the weight stream, bit 1 the
// activation fragment; both collapse onto block 0 and hit in cache, with the same instructions,
// the same request counts, the same expansion and the same matrix work. A pinned arm computes the
// wrong numbers by construction and exists only to split a phase into issue and round trip.
void sequence_set_pin(int mask) {
    if(mask<0||mask>3)throw std::runtime_error("sequence pin mask must be 0 (neither), 1 (weights), 2 (activations) or 3 (both)");
    if(mask&&!HALO_SEQ_PIN_CONTROL)
        throw std::runtime_error("this build cannot pin a sequence stream at runtime; rebuild with DEFS='-DHALO_SEQ_PIN_CONTROL=1'");
    const int w=(mask&1)?0:1,a=(mask&2)?0:1;
    check(hipMemcpyToSymbol(HIP_SYMBOL(halo_seq_pin_w),&w,sizeof w));
    check(hipMemcpyToSymbol(HIP_SYMBOL(halo_seq_pin_a),&a,sizeof a));
}
bool sequence_pin_control(){return HALO_SEQ_PIN_CONTROL!=0;}
void sequence_batch_set_sched(SequenceBatch *s,int sched) {
    if(!s||sched<SEQ_SCHED_TILE||sched>SEQ_SCHED_BASE)
        throw std::runtime_error("sequence projection schedule must be 0 (tile), 1 (share), 2 (wide), 3 (row-tile pair), 4 (no slice barrier), 5 (barrier every second slice), 6 (deployed operand path) or 7 (wave-uniform weight base)");
    s->sched=sched;
}
}
