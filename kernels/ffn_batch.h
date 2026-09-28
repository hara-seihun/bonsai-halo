// Batched FFN: full residual-in / residual-out feed-forward for up to 128 rows at once.
//
// The deployed per-pass path walks eight rows through the persistent kernel and re-reads the whole
// weight stream for each pass. At batch it is the weight traffic that dominates, so this module
// runs the FFN as four ordinary kernels over a whole slice of rows: producer, gate/up with fused
// SiLU, hidden producer, down onto the incoming residual. It is a peer of the deployed FFN, not a
// replacement: mode 0 is reserved for the engine's own path and this module never claims it.
//
// The weights are the deployed ternary values with their deployed FP16 block scales, repacked once
// at create() into compact codes on the device (kernels/ffn_operands.hpp, kernels/ffn_a4_operands.hpp).
// Activation buffers are shared across all 64 layers and all modes; run() allocates nothing and
// copies nothing.
//
// Modes differ in how the weights meet the activations, and in the schedule that brings them
// together. Three control modes and three optimized ones:
//
//   1  IU8 / A8     v_wmma_i32_16x16x16_iu8 against the deployed eight-bit activation quantiser.
//                   The honest control: same arithmetic the engine deploys, different schedule.
//   2  IU4 / A4     v_wmma_i32_16x16x16_iu4, which issues at half the cost on gfx1151, against
//                   four-bit activations. Approximate through the activation width only.
//   3  scaled F16   the block scale folded into the FP16 A operand by byte selection, eight-bit
//      / A8         activation values carried as FP16, one FP32 accumulator across the whole K and
//                   no per-block epilogue.
//
//   6  IU8 / A8     mode 1's arithmetic on wide weight loads and the adaptive shared-tile schedule.
//   7  scaled F16   mode 3's arithmetic on wide loads up to 32 rows and the adaptive shared-tile
//      / A8         schedule above.
//   8  IU4 / A4     mode 2's arithmetic on dense five-trit storage up to 32 rows and nine-valued
//                   pair codes above.
//
// Numerical labelling. Weights are never approximated in any mode. Mode 1 differs from the deployed
// engine only in summation order. Mode 2 quantises activations to four bits (deterministic,
// symmetric, per (row, 128-block), scale amax/7, round-to-nearest-even, clamped to [-7, +7]); the
// Kelana measurements put that at about 1% relative RMS on the FFN output against the engine. Mode
// 3 keeps the eight-bit activation grid and rounds the dequantised value to FP16, and reassociates
// the sum: one FP32 accumulation over all 5120 or 17408 terms instead of per-128 integer sums
// combined in FP32.
//
// Modes 6, 7 and 8 are 1, 3 and 2 with a different schedule and a different weight storage, and
// nothing else. They run the same producer instantiation, build the same A operand values, sum the
// same products in the same order into the same accumulator, and write the same output elements
// from the same wave. Every output bit is the control's; what changes is how many times the weight
// stream is read and how many loads a lane's 128-block costs. Kelana measured identical whole-FFN
// output hashes across each family at 32, 128 and 256 rows on two layers.
//
// One deliberate difference from the Kelana producers, which use 1e-5: the RMS norm epsilon here is
// the engine's halo::NORM_EPS (1e-6). Matching the engine matters more than matching the research
// harness, so this module is not bit-comparable to stored Kelana residuals.
//
// Provenance: algorithms from Kelana research/ffn/batched, at commit 1adc35e for the control modes
// (arithmetic/ and compact-scaled/) and 15e0524 for the optimized ones (tile-ownership/ and
// dense-consumer/). The source lives here; there is no build or runtime dependency on that tree.
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include "halo_kernels.h"

namespace halo {

enum FfnBatchMode {
    FFN_BATCH_ENGINE          = 0,   // reserved: the deployed per-pass FFN, not served by this module
    FFN_BATCH_IU8_A8          = 1,
    FFN_BATCH_IU4_A4          = 2,
    FFN_BATCH_SCALED_A8       = 3,
    FFN_BATCH_IU8_A8_OPT      = 6,
    FFN_BATCH_SCALED_A8_OPT   = 7,
    FFN_BATCH_IU4_A4_OPT      = 8,
};

// Which modes a state serves. A mode not named at create() or at ffn_batch_prepare() has no weight
// image, and run_ffn_batch() rejects it rather than silently running something else.
constexpr unsigned ffn_batch_mode_bit(int mode) { return 1u << mode; }
constexpr unsigned FFN_BATCH_CONTROLS  = (1u << 1) | (1u << 2) | (1u << 3);
constexpr unsigned FFN_BATCH_OPTIMIZED = (1u << 6) | (1u << 7) | (1u << 8);
constexpr unsigned FFN_BATCH_ALL       = FFN_BATCH_CONTROLS | FFN_BATCH_OPTIMIZED;

// Options ride in the same word as the mode bits. Mode 8 otherwise holds both of its images and
// picks per call; either restriction keeps one image and uses it at every row count, which costs
// about 13% at exactly 32 rows (wide only) or 35-40% at 128 and above (dense only).
constexpr unsigned FFN_BATCH_A4_WIDE_ONLY  = 1u << 28;
constexpr unsigned FFN_BATCH_A4_DENSE_ONLY = 1u << 29;
constexpr unsigned FFN_BATCH_OPTION_MASK   = FFN_BATCH_A4_WIDE_ONLY | FFN_BATCH_A4_DENSE_ONLY;

// The weight images this module can hold. Several modes share one image where the bytes are the
// same; ffn_batch_image_bytes() prices each of them, so a report can state resident cost without
// double counting a shared image.
//
//   SCALES  FP16 block scales, tile-major. Layout-independent, so one copy serves every mode.
//   SLICE2  two-bit ternary codes, [slice][row]: modes 1, 2, 3, and mode 7 above 32 rows.
//   LANE2   the same codes, [row][slice], so a lane's 128-block is two uint4 loads: mode 6 at every
//           row count, mode 7 at 32 rows and below.
//   PAIR    nine-valued two-trit codes, one per nibble, lane-major: mode 8 above 32 rows.
//   DENSE5  the stored HALO five-trit bytes, 26 per (row, 128-block): mode 8 at 32 rows and below.
enum FfnBatchImage {
    FFN_IMAGE_SCALES = 0,
    FFN_IMAGE_SLICE2 = 1,
    FFN_IMAGE_LANE2  = 2,
    FFN_IMAGE_PAIR   = 3,
    FFN_IMAGE_DENSE5 = 4,
    FFN_IMAGE_COUNT  = 5,
};

struct FfnBatch;

// Repacks every layer's gate, up and down weights for the requested modes and allocates the shared
// activation workspace. `host_layers` is the host-side LayerW table (NLAYER entries) whose pointers
// are device HALO images; `signs17408` is the device sign vector applied to the hidden activation.
// The table is kept, so it must outlive the state for ffn_batch_prepare() to work.
//
// Untimed: one decode pass per matrix fills every image being built. max_rows is clamped to at
// least 16 and rounded up to 16.
//
// Throws std::runtime_error if the device will not admit the images those modes need, naming the
// bytes required, the bytes free, the mode mask and the images. Nothing is left allocated in that
// case. Which images a mode set needs is a fixed cost of that set, so a caller that catches this
// has one useful move: ask for fewer modes, or for the same ones with FFN_BATCH_A4_WIDE_ONLY.
FfnBatch * create_ffn_batch(const LayerW * host_layers, const float * signs17408, int max_rows,
                            unsigned modes = FFN_BATCH_ALL);

// Adds the images further modes need, for a caller that would rather not hold every representation
// from the start. Idempotent, synchronises the device, returns the device bytes it added. It
// allocates and writes gigabytes, so call it before a timed region, never inside one. The A4
// options are fixed at create() and ignored here.
//
// Throws std::runtime_error the same way, and an exception leaves the state exactly as it was: the
// modes it already served still run, ffn_batch_bytes() still equals what it holds, and this call's
// allocations are given back. So a caller may narrow its mode set and try again on the same state.
size_t ffn_batch_prepare(FfnBatch * state, unsigned modes);

// What a preflight needs to decide before either of those. ffn_batch_prepare_bytes() is what a
// prepare() call would newly allocate given what this state already holds, and is zero when every
// image is resident; ffn_batch_workspace_bytes() is the shared activation workspace create() takes,
// which is the part ffn_batch_weight_bytes() does not cover. Compare their sum against
// hipMemGetInfo(). Both are arithmetic on the model geometry and touch no device state.
size_t ffn_batch_prepare_bytes(const FfnBatch * state, unsigned modes);
size_t ffn_batch_workspace_bytes(int max_rows);

// One layer's FFN for `rows` rows. `input` and `output` are row-major [rows][D] FP32 with row
// stride D; `input` is the incoming residual and `output` receives residual + down(...). Passing
// the same pointer for both is supported and is the intended use: every element is read and written
// by one thread, and the producer has finished with the residual before the down projection stores.
//
// Any row count from 1 to max_rows works; a batch whose tile count the kernel's token-tile width
// does not divide reads fragment slack past the batch and discards it at the row guard.
//
// Returns false, without touching `output`, for mode 0, an unprepared or unsupported mode, an
// out-of-range layer, or rows outside [1, max_rows]. Everything is enqueued on `stream`; nothing
// synchronises.
bool run_ffn_batch(FfnBatch * state, int layer, int mode, int rows,
                   const float * input, float * output, hipStream_t stream);

// Whether run_ffn_batch() would accept this mode: it was named at create() or prepare().
bool ffn_batch_supports(const FfnBatch * state, int mode);

// Row-tile ownership for mode 8's pair-code arm: 0 gives each wave its own weight row tile, which
// is the schedule this mode shipped with; 1 shares one row tile across the waves of a workgroup,
// as modes 6 and 7 do; 2 shares it against eight waves per SIMD32 instead of four, which is the
// selected shape. All three build the same A operand values, issue the same products in the same
// order and store the same output elements from the same wave, so every output bit is the same.
// Only which workgroup owns a row tile, and therefore how many times the weight stream is read,
// changes. Settable at runtime so one process can interleave the arms of a comparison instead of
// pairing separate processes, whose clocks on this device differ by more than these steps.
int  ffn_batch_a4_sched(const FfnBatch * state);
void ffn_batch_set_a4_sched(FfnBatch * state, int sched);

// Which weight image mode 8 reads, per stage, crossed with the ownership above. 0 lets the row
// count choose for each stage; 1 forces the dense five-trit image on both; 2 forces pair codes on
// both; 3 gives gate/up the dense image and down the pair codes, and 4 the reverse. Both images
// decode to the same weights through the same nine-valued code alphabet, so a projection built on
// either produces the same bits and only the operand build and the bytes read change.
//
// The stages are separate because they are limited by different things: gate/up owns 1088 weight
// row tiles and fills the device, down owns 160 and runs two waves per SIMD32. The row-count rule
// was also fitted on prefill shapes, and a generation step is 32 rows however many sequences it
// advances, so this axis is how the arms are ordered inside one process rather than across two.
int  ffn_batch_a4_image(const FfnBatch * state);
void ffn_batch_set_a4_image(FfnBatch * state, int image);

// Where a (row tile, 128-block) pair lands in the stored weight image. A run is that pair's codes
// and its sixteen FP16 scales, and its index is `tile*tstep + blk*bstep`:
//
//   0  tile-major        tstep = nb,     bstep = 1        the deployed image. A wave's own blocks
//                                                         are contiguous and the resident waves
//                                                         stand one stream stride apart: 20,480
//                                                         bytes for a 512-byte gate/up image,
//                                                         16,640 for the dense five-trit one.
//   1  padded tile-major tstep = nb + 1, bstep = 1        the same locality with the stride's
//                                                         power-of-two factor broken. Costs one
//                                                         run per row tile, so the images must be
//                                                         allocated for it (HALO_FFN_RUN_ORDER=1).
//   2  block-major       tstep = 1,      bstep = ntiles   every resident wave's block b inside one
//                                                         contiguous run.
//   3  the stride rule    padded where a matrix's stream stride is a multiple of 4096, tile-major
//                         where it is not. THE DEFAULT. Measured at 32 prefill rows with both A4
//                         images forced: the pair-code image (stride 20,480) is -17% padded and
//                         the dense five-trit one (16,640) is a null, while block-major costs the
//                         dense image 6% of its phase. The deployed 128-row pass, which reads pair
//                         codes on both stages, is -4.6%.
//
// Same bytes, same values, same order of use and the same block loop, so every order produces the
// same bits; only the addresses the resident waves ask for at one instant change. Setting it
// rewrites every resident image and the shared scale image from the HALO source weights, which
// takes a second or so and synchronizes, so it belongs between measured passes and not inside one.
// HALO_FFN_RUN_ORDER picks the order a process starts in; only order 1 makes the images larger,
// and only a process that names it before create() can select it later.
int  ffn_batch_run_order(const FfnBatch * state);
void ffn_batch_set_run_order(FfnBatch * state, int order, hipStream_t stream = nullptr);
const char * ffn_batch_run_order_name(int order);

// Operand liveness inside one 128-block of mode 8's pair-code gate/up projection, in token tiles
// per accumulator group: 0 expands a K16 slice and spends it against every token tile the wave
// holds, which keeps all TT int32 accumulator sets live; 1, 2 and 4 expand the block's sixteen
// fragments once and run that many tiles at a time. Two is selected. Each output element accumulates the same
// slices in the same order and drains into the same FP32 chain once per block in all three, so
// every output bit is the same; only the register allocation, the resident wave count and the
// number of independent matrix chains change. Settable at runtime so one process can interleave
// the arms.
// The dense five-trit arm's load schedule, which is what serves a generation step and every pass of
// 32 rows or fewer. Both terms are scheduling, so all four settings issue the same instructions
// against the same bytes and produce the same bits; only when a load is issued and what the peel's
// scheduling barriers are allowed to fence changes.
//
//   0  barriers fence every class, weight cursor two blocks deep. The shape the cursor shipped with.
//   1  barriers let VMEM reads cross, cursor two deep.
//   2  barriers let VMEM reads cross, cursor three deep.
//   3  barriers fence every class, cursor three deep.
//
// Settable at runtime so one process can interleave the arms; a generation step is 32 rows and the
// steps between these shapes are smaller than this box's between-process drift.
int  ffn_batch_a4_dense(const FfnBatch * state);
void ffn_batch_set_a4_dense(FfnBatch * state, int schedule);

int  ffn_batch_a4_order(const FfnBatch * state);
void ffn_batch_set_a4_order(FfnBatch * state, int order);

// Where a block's activation fragments come from on the ternary arms.
//
//   0  sixteen global loads per wave per 128-block, each wave of a workgroup loading the same
//      2 kB the other three load. The shape this replaced.
//   1  one workgroup-shared LDS stage per block, filled a block ahead: one staging request per
//      thread and ds_read in the block body, with one barrier per block.
//   2  the same stage held per wave instead of per workgroup: no barrier, WV times the LDS.
//
// Bit-identical: the same 8-byte fragments reach the same lanes in the same order, through LDS
// instead of through L1. Settable at runtime so one process can interleave the arms.
// docs/ffn-b-operand.md.
int  ffn_batch_b_stage(const FfnBatch * state);
void ffn_batch_set_b_stage(FfnBatch * state, int mode);

// Where the block's two scalar operands are read: the weight scale word each matrix spends and the
// activation scale each token tile spends.
//
//   0  in the block that spends them, which is where the source put them and where the compiler
//      leaves them: the scale pair is waited for 26 slots after its request and each token scale
//      3 to 5 slots after its own, so a 625-slot block stalls three times on memory it asked for
//      itself.
//   1  a block ahead, held in TT + 2 registers and rotated after the drain has spent them, so
//      every wait in the block loop is for a request made in the previous block.
//
// Bit-identical: the same words, the same `half_bits_to_float`, the same broadcast positions and
// the same `fmaf` chain. Only when a load is issued moves. Settable at runtime so one process can
// interleave the arms. docs/ffn-block-pipeline.md.
int  ffn_batch_spipe(const FfnBatch * state);
void ffn_batch_set_spipe(FfnBatch * state, int mode);

// How many of the down projection's two weight matrices one wave owns, on the dense five-trit arm.
//
//   0, 1  one matrix per wave: 320 waves instead of 160, the same weight bytes, and half the
//         accumulator budget, which is what lets this stage hold four token tiles.
//   2     the pair, which is the shape this kernel was written with.
//
// The two arms are bit-identical: an output element is one wave's dot product over the same K
// order either way, and only which wave owns a row tile moves. Settable at runtime so one process
// can order the arms. docs/ffn-down-waves.md.
int  ffn_batch_dn_mats(const FfnBatch * state);
void ffn_batch_set_dn_mats(FfnBatch * state, int mats);

void destroy_ffn_batch(FfnBatch * state);

// Device bytes owned by this state: every prepared weight image plus the shared workspace.
size_t ffn_batch_bytes(const FfnBatch * state);

// Bytes one image takes for all 64 layers, and the bytes a mode set needs, neither of which needs
// a state. Use the second for a budget check before create(): FFN_BATCH_CONTROLS is 4.55 GB,
// FFN_BATCH_ALL is 16.58 GB, and FFN_BATCH_ALL | FFN_BATCH_A4_WIDE_ONLY is 13.10 GB.
// `order` is the stored run order the price is for; the padded one spends one extra run per row
// tile and the other two hold the same runs (see ffn_batch_set_run_order).
size_t ffn_batch_image_bytes(int image, int order = 0);
size_t ffn_batch_weight_bytes(unsigned modes = FFN_BATCH_ALL, int order = 0);

// Bytes an image actually occupies in this state, zero if it was never prepared.
size_t ffn_batch_resident_image_bytes(const FfnBatch * state, int image);

const char * ffn_batch_image_name(int image);
const char * ffn_batch_mode_name(int mode);

} // namespace halo
