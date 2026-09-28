#include "engine.h"
#include "batch_route.h"
#include "tokenizer.h"
#include "ffn_batch.h"
#include "sequence_batch.h"
#include "head_batch.h"
#include "attn_batch.h"
#include "prep_batch.h"
#include "../vendor/nlohmann/json.hpp"
#include "halo_env.hpp"
#include "executable_provenance.hpp"
#include <fstream>
#include <chrono>
#include <random>
#include <numeric>
#include <map>
#include <filesystem>
using namespace halo;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static std::string command(const char * cmd) {
    FILE * p=popen(cmd,"r"); if(!p) throw std::runtime_error("cannot collect source provenance");
    std::string result; char b[4096]; while(fgets(b,sizeof(b),p))result+=b;
    if(pclose(p))throw std::runtime_error("source provenance command failed");
    while(!result.empty()&&result.back()=='\n')result.pop_back(); return result;
}
// The A4 projection's operand schedule, as one axis, because they are one kernel's load schedule
// and a panel wants them interleaved in one process: 0..5 select the weight cursor's depth and
// fences, 8..13 the ablation ladder (HALO_FFN_PROBE builds only), and 16/17 the activation stage -
// 16 workgroup-shared LDS, 17 wave-private - which run on the selected cursor, and 18/19 the
// block's scalar operands, 18 read in the block that spends them and 19 a block ahead.
// docs/ffn-b-operand.md, docs/ffn-block-pipeline.md.
static void apply_ffn_dense(FfnBatch * f, int dense) {
    if (dense < 0) return;
    if (dense >= 18) { ffn_batch_set_spipe(f, dense - 18); return; }     // 18 scalars in the block, 19 a block ahead
    if (dense >= 15) { ffn_batch_set_b_stage(f, dense - 15); return; }   // 15 off, 16 workgroup, 17 wave
    ffn_batch_set_a4_dense(f, dense);
}

static std::vector<int> list(const std::string & s) {
    std::vector<int> v; std::stringstream in(s); std::string x;
    while (std::getline(in,x,',')) v.push_back(std::stoi(x)); return v;
}
int main(int argc,char **argv) { try {
    std::string out="../../data/bonsai2/batch-comparison/phase-profile.json";
    std::vector<int> modes{5,10,11}, rows{32,128}, heads{0,1}, traces{0,1}, shares{-1}, images{-1}, pregates{-1}, orders{-1}, seq_scheds{-1}, gdn_states{-1}, gdn_defers{-1}, denses{-1}, dnmats{-1}, seq_quants{-1}, ffn_slices{-1}, seq_images{-1}, attn_coords{-1}, ffn_runs{-1}, gdn_splits{-1}, gdn_colss{-1}; int rounds=3, context=128, warmup_ms=2000;
    // --prefill-scan N ingests an N-token document in `rows`-row passes with the tracer on and
    // emits one sample per pass. It is the only shape here whose cost moves with position: a pass
    // at position p reads p keys and values per (row group, head) and writes p/ACHUNK chunk
    // partials per (row, head), while every other phase is priced per row. Every published prefill
    // number in this lane is a 384-token document, which is one point of that curve.
    int prefill_scan=0;
    // Rows per attention unit group, walked in one process. 8 is what an eight-row slice handed the
    // phase; 16 merges neighbouring groups so a key chunk is read once for twice the rows.
    std::vector<int> attn_tts{-1}, wide_attns{-1};
    // Score-unit schedule. 0 is the deployed body, 1 the rescheduled one; they compute the same
    // floats, so this axis is the clean way to order a schedule change against the clock drift
    // this box has on a prefill arm.
    std::vector<int> attn_scheds{-1};
    // How many rows of the pass the vocabulary head produces: 1 is the tail row alone, which is
    // what prompt ingestion reads, and 0 is every row of the pass. Walked as a case axis because
    // the two arms differ by a whole phase and a cross-process pair on this box drifts further
    // than the result.
    std::vector<int> head_rows{1};
    std::vector<int> head_ops{-1};
    // The sequence projections' four-bit operand map: 1 is the nibble sign extension, 2 the
    // nine-valued pair codes. Both are compiled only in a control build; the selected arm alone is
    // in the shipped one and asking for the other there fails loudly. It changes the stored code
    // order, which `sequence_batch_sync_quant` below pays for outside every timed region.
    std::vector<int> seq_i4ops{-1};
    std::vector<int> seq_tmaps{-1};
    // Which of the two projection streams stops walking its blocks: 1 the weights, 2 the activation
    // fragments, 3 both. A pinned arm is numerically invalid by construction and splits the phase
    // into what its instructions cost and what its memory round trip costs. Needs
    // DEFS='-DHALO_SEQ_PIN_CONTROL=1'. docs/seq-block-roundtrip.md.
    std::vector<int> seq_pins{-1};
    std::vector<int> head_tts{-1};
    // Which operand the deployed FFN slice's block loop requests first. Both arms are in every build
    // and compute the same bits, so this is a schedule axis inside one process.
    std::vector<int> mvw_ords{-1};
    std::vector<int> mvw_arms{-1};
    // Decode shape: `streams` independent slots, one row each, the aggregate generation workload.
    // The default prefill shape puts every row in one sequence, which is not how a decode step
    // distributes its state, attention and gather work.
    //
    // --decode-tokens walks the other axis of that shape: how many rows each sequence contributes
    // to one step. k > 1 is the verify half of a speculative step, and it is the only lever that
    // adds tokens without adding recurrent-state traffic, because the state round trip is per
    // (sequence, layer) and not per token. Walked as a case axis so every k shares one build,
    // one warmed device and one clock; a pair of processes cannot order effects this size.
    int decode_streams=0;
    // Steps run between the setup prefill and the timed one. A deferred state commit makes
    // consecutive steps different work - most of them append a rank-1 update and one writes the
    // state back - so a single timed step measures whichever one it lands on. This picks.
    std::vector<int> decode_primes{0};
    // The position axis of a generation step, and the decode counterpart of --prefill-scan. Every
    // other phase in a step is priced per row and per sequence; attention is priced by where the
    // row sits, so a panel that never leaves a 64-token prefix cannot see that term. Walked as a
    // case axis inside one process: each case re-prefills every slot to its own prefix, so the two
    // dispatch arms meet at the same position under one clock.
    std::vector<int> decode_prompts{64};
    std::vector<int> decode_tokens{1};
    // Which route ingests the decode context. -1 is the measured mode, which is what every panel
    // before this flag ran. Naming it lets a long-context panel reach mode 0 at all: the eight-row
    // route ingests at about 155 tok/s, so a 4096-token prefix is half a minute of GPU lock per
    // case, and a panel of four cases would never finish inside one bounded call.
    int decode_setup_mode=-1;
    // The sequence projections' (token tiles, waves) shape, as an in-process case axis. Both are
    // -1 for "the fitted rule" so a panel that does not name them measures what ships.
    std::vector<int> seq_tts{-1}, seq_ws{-1};
    // Rows at or above which an automatic mode takes the wide schedule (src/batch_route.h). The
    // decode shape is the one that cannot widen itself - a 16-stream step has 16 rows and no more
    // to give - so this axis is how a step below the floor is priced against the same step above
    // it, in one process, on one clock, with the residual hash as the map check.
    std::vector<int> wide_mins{-1};
    // Which attention score instantiation a row group gets: 1 the narrowest that covers it, 0 the
    // wide-only dispatch, -1 leave the process default. Both arms compute the same bits, so this is
    // a pure timing axis. It is the only phase in a generation step that is priced by position, so
    // --decode-context raises the KV allocation and lets --decode-prompt walk that position.
    std::vector<int> attn_narrows{-1};
    // Query heads of one KV group per score unit on a one-row-group launch: 1 the deployed unit,
    // 6 the whole KV group. Only the bytes per unit change, so this is the byte axis of the same
    // phase --attn-narrow walks the slot axis of.
    std::vector<int> attn_hgs{-1};
    std::vector<int> attn_folds{-1};
    // Megabytes of chunk partials one score launch may leave in flight. -1 leaves the process
    // default, 0 is the deployed one-pair-per-pass schedule, and a positive value makes the driver
    // run score and combine over fewer row groups at a time. Exact at every value.
    std::vector<int> attn_part_mbs{-1};
    int decode_context=0;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i]; if(i+1==argc) throw std::runtime_error("missing argument");
        std::string b=argv[++i];
        if(a=="--out")out=b; else if(a=="--modes")modes=list(b); else if(a=="--rows")rows=list(b);
        else if(a=="--prefill-scan")prefill_scan=std::stoi(b);
        else if(a=="--attn-tt")attn_tts=list(b);
        else if(a=="--attn-sched")attn_scheds=list(b);
        else if(a=="--wide-attn")wide_attns=list(b);
        else if(a=="--rounds")rounds=std::stoi(b); else if(a=="--context")context=std::stoi(b);
        else if(a=="--heads")heads=list(b); else if(a=="--traces")traces=list(b); else if(a=="--warmup-ms")warmup_ms=std::stoi(b);
        else if(a=="--head-rows")head_rows=list(b);
        // Which ternary operand map the wide head builds: 0 the peel, 1 the product gather. Both
        // read the same image and produce the same logit bits, so this is an in-process case axis
        // and the residual hash is its control. -1 leaves the process default (HALO_HEAD_OP).
        else if(a=="--head-op")head_ops=list(b);
        else if(a=="--seq-i4op")seq_i4ops=list(b);
        else if(a=="--seq-tmap")seq_tmaps=list(b);
        else if(a=="--seq-pin")seq_pins=list(b);
        // Token columns per head workgroup, as an in-process case axis: 1, 2, 4, 8 or -1 for the
        // shape rule. Pure schedule, same logits, different number of passes over the 278 MB image.
        else if(a=="--head-tt")head_tts=list(b);
        else if(a=="--decode-streams")decode_streams=std::stoi(b); else if(a=="--decode-prompt")decode_prompts=list(b);
        else if(a=="--decode-context")decode_context=std::stoi(b);
        else if(a=="--decode-setup-mode")decode_setup_mode=std::stoi(b);
        else if(a=="--attn-narrow")attn_narrows=list(b);
        else if(a=="--attn-hg")attn_hgs=list(b);
        else if(a=="--attn-fold")attn_folds=list(b);
        else if(a=="--attn-part-mb")attn_part_mbs=list(b);
        else if(a=="--ffn-sched")shares=list(b);
        else if(a=="--ffn-image")images=list(b);
        else if(a=="--ffn-run")ffn_runs=list(b);
        else if(a=="--gdn-pregate")pregates=list(b);
        else if(a=="--gdn-cols")gdn_colss=list(b);
        else if(a=="--ffn-order")orders=list(b);
        else if(a=="--ffn-dense")denses=list(b);
        // How many of the down projection's two weight matrices one wave owns on the dense arm.
        // The stage owns DH/16 = 160 row tiles, so pairing them is what leaves it at two waves per
        // SIMD32; splitting buys 320 waves at identical weight bytes. Bit-identical, so the panel
        // checks the residual hash rather than assuming it.
        else if(a=="--ffn-dn-mats")dnmats=list(b);
        else if(a=="--decode-tokens")decode_tokens=list(b);
        else if(a=="--wide-min")wide_mins=list(b);
        else if(a=="--seq-sched")seq_scheds=list(b);
        // The projections' shape ladder: token tiles per wave and waves per workgroup. A shape
        // moves which wave computes which output element, not the K order, the int32 block sums or
        // the FP32 drain, so every cell is bit-identical and the residual hash says so.
        // docs/row-band-fit.md walks it at 32, 64, 128 and 256 rows under both operand widths.
        else if(a=="--seq-tt")seq_tts=list(b);
        else if(a=="--seq-w")seq_ws=list(b);
        // The stored order of the sequence weight image's (row tile, block) runs. Permuted in place
        // between cases, so both orders share one process, one warmed device and one clock, and the
        // residual hash is identical by construction. docs/weight-stream-order.md.
        else if(a=="--seq-image")seq_images=list(b);
        // The state's storage coordinate. Every case re-prefills its slots, so a format change is
        // safe between cases and the arms share one build, one warmed device and one clock.
        else if(a=="--gdn-state")gdn_states=list(b);
        else if(a=="--attn-coord")attn_coords=list(b);
        // How many steps share one state write-back; 0 runs the deferred kernels and commits every
        // step, which separates the instantiation from the deferral.
        else if(a=="--gdn-defer")gdn_defers=list(b);
        else if(a=="--gdn-split")gdn_splits=list(b);
        else if(a=="--decode-prime")decode_primes=list(b);
        // Activation precision of the sequence input projection. The weight order follows it and
        // the switch is applied between cases, outside every timed region.
        else if(a=="--seq-quant")seq_quants=list(b);
        // Rows per deployed-FFN slice: how many times a pass re-reads the FFN weight stream. Every
        // width computes each output element from the same weights in the same K order, so the arms
        // share one build, one warmed device and one clock. Modes 4 and 20 run that kernel.
        else if(a=="--ffn-slice")ffn_slices=list(b);
        // 0 requests each group's operands inside the loop that consumes them, 1 requests the first
        // operand of every group after the first at the top of the block. docs/ffn-slice-issue-order.md
        else if(a=="--mvw-order")mvw_ords=list(b);
        // Which unit shape the FFN slice runs: 0 the predecessor's wave-0 drain, 1 the drain spread
        // over the waves that computed it (bit-identical), 2/3/4 the activation, weight and joint
        // footprint ablations, which need -DHALO_MVW_ABL=1 and produce wrong numbers by design.
        // docs/ffn-slice-activation.md.
        else if(a=="--mvw-arm")mvw_arms=list(b);
        else throw std::runtime_error("unknown option "+a);
    }
    if(!getenv("BONSAI_BENCH_SERVER_MASKED")) throw std::runtime_error("use tools/run-batch-compare --profile-tool");
    for(auto key:{"HALO_STOP","HALO_LAYERS"})if(getenv(key)&&atoi(getenv(key)))throw std::runtime_error("partial model environment");
    if(modes.empty()||rows.empty()||heads.empty()||traces.empty()||shares.empty()||images.empty()||pregates.empty()||orders.empty()||seq_scheds.empty()||gdn_states.empty()||denses.empty()||dnmats.empty()||seq_quants.empty()||ffn_slices.empty()||seq_tts.empty()||seq_ws.empty())throw std::runtime_error("empty case list");
    for(int n:ffn_slices)if(n!=-1&&(n<1||n>FMAX))throw std::runtime_error("ffn-slice must be -1 (leave as configured) or 1.."+std::to_string(FMAX)+" rows per deployed FFN slice");
    for(int n:seq_quants)if(n<-1||n>SEQ_QUANT_MAX)throw std::runtime_error("seq-quant must be -1 (leave as configured), 0 (a8), 1 (a4e: input stage on the eight-bit instruction), 2 (a4: input stage), 3 (a4e-out), 4 (a4-out), 5 (a4e-both) or 6 (a4-both)");
    if(gdn_defers.empty()||decode_primes.empty())throw std::runtime_error("empty case list");
    for(int n:decode_primes)if(n<0||n>64)throw std::runtime_error("decode-prime must be 0..64 untimed steps before the timed one");
    for(int n:gdn_defers)if(n<-1||n>GDN_DEFER_MAX)throw std::runtime_error("gdn-defer must be -1 (leave as configured), 0 (deferred kernels, commit every step) or 1.."+std::to_string(GDN_DEFER_MAX)+" steps per state write-back");
    for(int n:gdn_states)if(n<-1||n>GDN_STATE_FMT_MAX)throw std::runtime_error("gdn-state must be -1 (leave as configured), 0 (f32), 1 (f16), 2 (i16), 3 (i8) or 4 (f32 through the packed kernels)");
    for(int n:seq_images)if(n<-1||n>1)throw std::runtime_error("seq-image must be -1 (leave as configured), 0 (tile-major) or 1 (block-major)");
    for(int n:seq_scheds)if(n<-1||n>7)throw std::runtime_error("seq-sched must be -1 (leave as configured), 0 (tile), 1 (share), 2 (wide), 3 (row-tile pair), 4 (no slice barrier), 5 (barrier at four tiles), 6 (deployed operand path) or 7 (wave-uniform weight base)");
    for(int n:orders)if(n!=-1&&n!=0&&n!=1&&n!=2&&n!=4)throw std::runtime_error("ffn-order must be -1 (leave as configured), 0 (slice-outer) or 1,2,4 token tiles per accumulator group");
    for(int n:shares)if(n<-1||n>1)throw std::runtime_error("ffn-sched must be -1 (leave as configured), 0 (tile), 1 (adapt) or 2 (wide)");
    for(int n:images)if(n<-1||n>4)throw std::runtime_error("ffn-image must be -1 (leave as configured), 0 (rule), 1 (dense), 2 (pair), 3 (dense gate/up + pair down) or 4 (pair gate/up + dense down)");
    for(int n:ffn_runs)if(n<-1||n>3)throw std::runtime_error("ffn-run must be -1 (leave as configured), 0 (tile-major), 1 (padded tile-major), 2 (block-major) or 3 (the stride rule, the default); the unconditional pad needs HALO_FFN_RUN_ORDER=1 in the environment so every image is allocated for it");
    // 8, 9 and 10 are the ablation ladder (activations pinned, weights pinned, both) and exist only
    // in a -DHALO_FFN_PROBE build; ffn_batch_set_a4_dense refuses them in a shipped one.
    for(int n:denses)if(n<-1||(n>5&&(n<8||n>13)&&(n<15||n>19)))throw std::runtime_error("ffn-dense must be -1 (leave as configured), 0 (fenced, cursor 2), 1 (loads may cross, cursor 2), 2 (loads may cross, cursor 3), 3 (fenced, cursor 3), 4 (down cursor 4), 5 (down cursor 6), 15 (activation stage off), 16 (workgroup activation stage), 17 (wave activation stage), 18 (block scalars in the spending block), 19 (block scalars a block ahead), or 8..13 for the ablation ladder in a HALO_FFN_PROBE build");
    // 2 is the paired shape the message has always named and the bound has always refused, so the
    // 160-wave arm of the down stage was unreachable from this tool; docs/ffn-stream-ceiling.md
    // needed it as the wave-count control and found the bound instead.
    for(int n:dnmats)if(n<-1||n>2)throw std::runtime_error("ffn-dn-mats must be -1 (leave as configured), 0 (the wave-count rule), 1 (one matrix per wave) or 2 (the pair)");
    for(int n:pregates)if(n<-1||n>1)throw std::runtime_error("gdn-pregate must be -1 (leave as configured), 0 (gate in the token loop) or 1 (gate once per row and head)");
    if(gdn_splits.empty())throw std::runtime_error("empty case list");
    for(int n:gdn_splits)if(n!=-1&&!gdn_split_valid(n))throw std::runtime_error("gdn-split must be -1 (leave as configured) or 2, 4, 8, 16 row groups per head");
    if(gdn_colss.empty())throw std::runtime_error("empty case list");
    for(int n:gdn_colss)if(n<-1||n>1)throw std::runtime_error("gdn-cols must be -1 (leave as configured), 0 (the deployed rows-per-lane state units) or 1 (one lane per 32 columns of one row); the column arm needs fp32 state and a committed pass");
    for(int n:rows)if(n<1||n>PASSMAX)throw std::runtime_error("rows must be 1.."+std::to_string(PASSMAX));
    for(int n:attn_tts)if(n!=-1&&n!=8&&n!=16)throw std::runtime_error("attn-tt must be -1 (leave as configured), 8 or 16");
    for(int n:attn_scheds)if(n<-1||n>2)throw std::runtime_error("attn-sched must be -1 (leave as configured), 0 (deployed body), 1 (rescheduled) or 2 (rescheduled, reached through the fused kernel that also compiles the lane-per-key body: the delivery control)");
    if(attn_scheds.empty())throw std::runtime_error("empty case list");
    for(int n:wide_attns)if(n<-1||n>1)throw std::runtime_error("wide-attn must be -1 (leave as configured), 0 (per-slice) or 1 (one launch per layer)");
    if(attn_tts.empty()||wide_attns.empty()||head_rows.empty())throw std::runtime_error("empty case list");
    for(const auto & choices:{heads,traces,head_rows})for(int n:choices)if(n!=0&&n!=1)throw std::runtime_error("heads/traces/head-rows must be 0 or 1");
    // The scan owns the position axis, so it sets its own engine context and the 512-token bound
    // below does not apply to it.
    if(prefill_scan<0||prefill_scan>MAXCTX-PASSMAX)throw std::runtime_error("prefill-scan must be 0.."+std::to_string(MAXCTX-PASSMAX));
    // Attention is the only phase on the position axis, so a case-axis panel has to be able to sit
    // where it costs something: the engine context follows --context instead of a fixed 512.
    if(!prefill_scan&&!decode_streams&&(context<0||context+*std::max_element(rows.begin(),rows.end())>MAXCTX))throw std::runtime_error("invalid geometry");
    if(rounds<1||warmup_ms<0)throw std::runtime_error("invalid geometry");
    // A decode step's own context, which every shape here used to share with the 512 the prefill
    // shapes use. Attention is the one phase in the step that grows with it, so a panel that never
    // leaves 512 cannot see that term at all; a KV slot is 90112 bytes per token per sequence, so
    // 32 streams at 2048 is 5.9 GB and the budget is the reason this is opt-in.
    const int decode_ctx=decode_context>0?decode_context:512;
    if(decode_streams<0||decode_streams>128)throw std::runtime_error("invalid decode shape");
    if(decode_context&&(decode_context<512||decode_context>MAXCTX))throw std::runtime_error("decode-context must be 512.."+std::to_string(MAXCTX));
    if(decode_tokens.empty()||attn_narrows.empty()||attn_hgs.empty()||decode_prompts.empty()||wide_mins.empty())throw std::runtime_error("empty case list");
    for(int v:attn_narrows)if(v<-1||v>1)throw std::runtime_error("attn-narrow must be -1 (leave as configured), 0 or 1");
    for(int v:attn_hgs)if(v!=-1&&v!=1&&v!=NH/NKV)throw std::runtime_error("attn-hg must be -1 (leave as configured), 1 or "+std::to_string(NH/NKV));
    if(attn_folds.empty()||attn_part_mbs.empty())throw std::runtime_error("empty case list");
    for(int v:attn_part_mbs)if(v<-1)throw std::runtime_error("attn-part-mb must be -1 (leave as configured), 0 (one score/combine pair per pass) or a positive megabyte window");
    if(seq_i4ops.empty())throw std::runtime_error("empty case list");
    for(int n:seq_i4ops)if(n!=-1&&n!=1&&n!=2)throw std::runtime_error("seq-i4op must be -1 (leave as configured), 1 (nibble sign extension) or 2 (pair codes)");
    if(seq_tmaps.empty())throw std::runtime_error("empty case list");
    for(int n:seq_tmaps)if(n<-1||n>2)throw std::runtime_error("seq-tmap must be -1 (leave as configured), 0 (tile-major), 1 (lane-major) or 2 (pair-major) token map");
    if(seq_tmaps.size()>1&&!sequence_batch_tmap_control())
        throw std::runtime_error("this build carries one sequence token map; rebuild with DEFS='-DHALO_SEQ_TMAP_CONTROL=1' to walk both");
    if(seq_pins.empty())throw std::runtime_error("empty case list");
    for(int n:seq_pins)if(n<-1||n>3)throw std::runtime_error("seq-pin must be -1 (leave as configured), 0 (neither stream pinned), 1 (weights), 2 (activations) or 3 (both)");
    for(int n:seq_pins)if(n>0&&!sequence_pin_control())
        throw std::runtime_error("this build cannot pin a sequence stream; rebuild with DEFS='-DHALO_SEQ_PIN_CONTROL=1'");
    if(seq_i4ops.size()>1&&!sequence_batch_i4op_control())
        throw std::runtime_error("this build carries one sequence four-bit operand map; rebuild with DEFS='-DHALO_SEQ_I4OP_CONTROL=1' to walk both");
    // 8, 9 and 10 are the pinned-chunk ablation and exist only in a -DHALO_ATTN_PROBE build, where
    // they compute the wrong attention on purpose. A default build reports attn_fold 1 for them, so
    // a panel that asked for an ablation it did not have says so in its own samples.
    for(int v:attn_folds)if(v!=-1&&v!=0&&v!=1&&v!=2&&v!=3&&v!=6&&(v<8||v>10))throw std::runtime_error("attn-fold must be -1 (leave as configured), 0 (the shape rule), 1 (one head per score unit), 2 (two heads, one leaf walk), 3 (two heads, a leaf walk each), 6 (a whole KV group, one-row groups only) or 8/9/10 (the pinned-key, pinned-value and both ablations of a probe build). The fold needs the leaf coordinate: --attn-coord 1");
    for(int p:decode_prompts)for(int k:decode_tokens) {
        if(k<1)throw std::runtime_error("decode-tokens must be at least 1");
        if(p<1)throw std::runtime_error("decode-prompt must be at least 1");
        // forward_batch caps a pass at PASSMAX rows, and the step has to fit the context it prefilled.
        if(decode_streams*k>PASSMAX)throw std::runtime_error("decode streams x tokens exceeds the "+std::to_string(PASSMAX)+"-row pass limit");
        // The prefix, the measured step and the three warm-up steps that precede it all have to fit
        // the allocated KV positions.
        if(p+4*k+4>decode_ctx)throw std::runtime_error("decode prompt plus step and warm-up tokens exceeds the context");
    }
    const int decode_prompt_max=decode_prompts.empty()?64:*std::max_element(decode_prompts.begin(),decode_prompts.end());
    const std::string model="../../data/bonsai2/PTQ1_0.gguf";
    // A KV slot is 90112 bytes per token across the 22 cache slots, so a scan to 8192 costs 738 MB
    // on one sequence. Every other shape keeps the 512 it always had.
    const int engine_ctx=std::max(prefill_scan?prefill_scan+PASSMAX:(decode_streams?decode_ctx:512),
        (context+*std::max_element(rows.begin(),rows.end())+255)/256*256);
    // Reserve every state coordinate the case list will select, before the engine sizes the regions
    // each arm will address; a list that names only packed ones starts the process in one too, so
    // the fp32 default does not reserve three times the memory for an arm this panel never runs.
    if(!gdn_states.empty()&&std::none_of(gdn_states.begin(),gdn_states.end(),[](int g){return g<0;}))
        gdn_state_boot_format(gdn_states.front());
    for(int gs:gdn_states)if(gs>=0)gdn_state_reserve_format(gs);
    // A deferred write-back keeps its pending triples beside the state, which an exact coordinate
    // has no room for unless the process says so before it allocates. Reserving here is what lets
    // `--gdn-defer 1,4` walk the axis at f32 in one process.
    for(int gd:gdn_defers)if(gd>=0)gdn_defer_reserve_depth(gd);
    Engine e; e.load(model,16,std::max(decode_streams,1),engine_ctx);
    unsigned mask=0;
    // The ingestion route is a mode this process runs, so it reserves what it needs like any other.
    std::vector<int> prep_modes=modes;
    if(decode_setup_mode>=0&&decode_streams)prep_modes.push_back(decode_setup_mode);
    for(int m:prep_modes) {
        if(m!=0&&m!=4&&m!=5&&m!=9&&m!=10&&m!=11&&(m<16||m>20))throw std::runtime_error("profile modes: 0,4,5,9,10,11,16,17,18,19,20");
        // Mode 20 is the wide schedule over the deployed FFN, so like mode 4 it reads no image.
        // Mode 0 is the persistent eight-row kernel and reads no image either. It is here because
        // it is the route this engine serves solo requests and every drafted verify pass on, and
        // until now it was the one route that could not be measured beside another inside one
        // process: a route question had to be answered across two builds and two clock states.
        // `--decode-streams 1 --decode-tokens 8` is the drafted verify shape. Mode 0 samples DO
        // carry a residual hash: the persistent route writes the engine's own `x` rather than
        // `batch_x`, and a step of at most RMAX rows is one slice, so the hash below reads `x` and
        // `residual_rows` records how many rows it covered.
        if(m!=0&&m!=4&&m!=20)mask |= ffn_batch_mode_bit(m>=16?(m==19?8:7):m==5?1:m-3);
    }
    // The capacity covers the widest shape this process measures, never below 128: the decode and
    // quality setup ingests prompts in 128-row passes whatever the measured rows are.
    int capacity=std::max(128,*std::max_element(rows.begin(),rows.end()));
    for(int k:decode_tokens)capacity=std::max(capacity,decode_streams*k);
    e.prepare_batch(capacity,mask?mask:Engine::BATCH_WORKSPACE_ONLY);
    if(std::any_of(prep_modes.begin(),prep_modes.end(),[](int m){return m>=17;}))e.prepare_sequence();
    e.set_batch_profile(true); e.set_batch_profile(false);
    Tokenizer tk; tk.load(model);
    std::ifstream f("PLAN.md"); std::string text((std::istreambuf_iterator<char>(f)),{});
    auto tokens=tk.encode(text,false); if(tokens.size()<512)throw std::runtime_error("short document");
    // Attention cost is set by position, not by token value: no phase here branches on a value, and
    // the score loop runs `count` keys whatever they are. So a scan longer than the document tiles
    // it, which keeps the shape exactly reproducible instead of depending on which file is longest
    // in the tree this hour.
    const size_t doc_tokens=tokens.size();
    while((int)tokens.size()<std::max(prefill_scan,context+*std::max_element(rows.begin(),rows.end()))+PASSMAX)tokens.insert(tokens.end(),tokens.begin(),tokens.begin()+doc_tokens);
    // Same argument for the decode prefixes: each stream takes a distinct slice at `i * prompt`, so
    // the document has to hold one stream's prefix plus the offset of the last stream.
    if(decode_streams)while((int)tokens.size()<(decode_streams+1)*decode_prompt_max+2)tokens.insert(tokens.end(),tokens.begin(),tokens.begin()+doc_tokens);
    Engine::Seq seq; std::vector<Engine::Seq*> sv{&seq};
    std::vector<Engine::Seq> dseq((size_t)std::max(decode_streams,0));
    for(size_t i=0;i<dseq.size();++i)dseq[i].slot=(int)i;
    std::vector<Engine::Seq*> dsv; for(auto & s:dseq)dsv.push_back(&s);
    // Each stream gets a distinct slice of the document, so no two slots share a state trajectory.
    // `setup_mode` ingests the context, `step_mode` is what the timed step runs. They are the same
    // unless --decode-setup-mode says otherwise, and the reason to separate them is mode 0: a
    // 4096-token context ingested through the eight-row route costs half a minute of lock per case,
    // where the same context ingested wide costs three seconds. The ingested VALUES differ between
    // routes, and the step's cost does not depend on them - no phase of a decode step branches on a
    // state or cache value - so a panel that pins one setup mode across its arms compares the arms
    // on identical state. The sample records both.
    auto decode_setup=[&](int setup_mode,int step_mode,int decode_prompt) {
        const int mode=setup_mode;
        e.set_batch_profile(false); e.batch_mode=mode;
        for(size_t i=0;i<dseq.size();++i) {
            e.reset_seq(dseq[i]);
            const size_t base=(i*decode_prompt)%(tokens.size()-decode_prompt-1);
            std::vector<Engine::Seq*> one{&dseq[i]};
            // One pass admits at most the prepared capacity, so a prefix longer than that is
            // ingested the way Engine::prefill ingests a prompt: consecutive passes over one slot.
            for(int off=0;off<decode_prompt;off+=capacity) {
                const int k=std::min(capacity,decode_prompt-off);
                e.forward_batch(one,{std::vector<int>(tokens.begin()+base+off,tokens.begin()+base+off+k)},false);
            }
        }
        e.batch_mode=step_mode;
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
    };
    auto setup=[&](int mode) {
        e.set_batch_profile(false); e.batch_mode=mode; e.reset_seq(seq);
        for(int n=0;n<context;n+=128) {
            int k=std::min(128,context-n);
            e.forward_batch(sv,{std::vector<int>(tokens.begin()+n,tokens.begin()+n+k)},false);
        }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
    };
    json result={{"schema","bonsai-phase-profile/1"},{"model",model},{"context",context},{"rounds",rounds},
        {"device_bytes",e.device_bytes},{"modes",modes},{"rows",rows},{"kernel_clock_hz",100000000},
        {"notes","Paired tracing on/off; setup excluded. Events bracket stream launches, barrier stamps include waits. Wall time includes trace collection when enabled; device span excludes its readback. Residual hashing is outside timing."},{"samples",json::array()}};
    result["git_revision"]=command("git rev-parse HEAD");
    result["source_sha256_rollup"]=command("git ls-files --cached --others --exclude-standard -z -- src kernels Makefile tools/batch_profile.cpp tools/batch_profile.py tools/executable_provenance.hpp tools/halo_env.hpp tools/run-batch-compare | xargs -0 sha256sum | sha256sum");
    result["binary_sha256"]=executable_sha256();
    result["warmup_ms"]=warmup_ms;
    result["halo_env"]=halo_env_snapshot();
    result["gdn_resident"]=sequence_resident_enabled(e.batch_sequence);
    result["wide_head"]=head_batch_capacity(e.batch_head)>0;
    result["wide_head_tile_width"]=getenv("HALO_HEAD_TT")?getenv("HALO_HEAD_TT"):"auto";
    result["gdn_state_split"]=getenv("HALO_GDN_SPLIT")?getenv("HALO_GDN_SPLIT"):"4";
    result["sequence_layout"]=sequence_direct_layout(e.batch_sequence)?"direct":"staged";
    result["sequence_operand"]=getenv("HALO_SEQUENCE_OPERAND")?getenv("HALO_SEQUENCE_OPERAND"):"int8";
    result["sequence_tile_width"]=getenv("HALO_SEQUENCE_TT")?getenv("HALO_SEQUENCE_TT"):"auto";
    result["ffn_a4_sched"]=ffn_batch_a4_sched(e.batch_ffn);
    result["ffn_scheds"]=shares;
    result["ffn_a4_image"]=ffn_batch_a4_image(e.batch_ffn);
    result["ffn_images"]=images;
    result["head_rows"]=head_rows;
    result["ffn_a4_dense"]=ffn_batch_a4_dense(e.batch_ffn);
    result["ffn_denses"]=denses;
    result["ffn_dn_mats_cases"]=dnmats;
    result["gdn_pregate"]=gdn_resident_pregate();
    result["gdn_pregates"]=pregates;
    result["gdn_splits"]=gdn_splits;
    result["gdn_cols"]=gdn_resident_cols();
    result["gdn_colss"]=gdn_colss;
    result["ffn_a4_order"]=ffn_batch_a4_order(e.batch_ffn);
    result["ffn_orders"]=orders;
    result["gdn_state"]=gdn_state_format_name(gdn_state_format());
    result["gdn_states"]=gdn_states;
    result["attn_coords"]=attn_coords;
    result["gdn_defer"]=gdn_defer_depth();result["gdn_defer_effective"]=gdn_defer_effective();result["gdn_defers"]=gdn_defers;
    result["sequence_sched"]=e.batch_sequence?sequence_batch_sched(e.batch_sequence):-1;
    result["seq_tt_cases"]=seq_tts;result["seq_w_cases"]=seq_ws;
    result["sequence_scheds"]=seq_scheds;
    result["sequence_image"]=e.batch_sequence?sequence_batch_image(e.batch_sequence):-1;
    result["sequence_images"]=seq_images;
    result["ffn_runs"]=ffn_runs;
    result["ffn_slice_rows"]=batch_ffn_slice_rows();
    result["ffn_slice_cases"]=ffn_slices;result["mvw_orders"]=mvw_ords;result["mvw_arms"]=mvw_arms;
    result["sequence_wave_share"]=getenv("HALO_SEQUENCE_W")?getenv("HALO_SEQUENCE_W"):"auto";
    // Warm the same model route before recording.
    result["decode_streams"]=decode_streams; result["decode_prompt_cases"]=decode_prompts;
    result["decode_context"]=decode_ctx; result["attn_narrows"]=attn_narrows;
    result["decode_token_cases"]=decode_tokens;
    auto warm=Clock::now(); while(std::chrono::duration<double,std::milli>(Clock::now()-warm).count()<warmup_ms) { setup(modes.front()); e.forward_batch(sv,{std::vector<int>(tokens.begin()+context,tokens.begin()+context+128)},true); HIP_CHECK_H(hipStreamSynchronize(e.stream)); }
    if(prefill_scan) {
        // The position axis. One document, ingested in `rows`-row passes with the tracer on, one
        // sample per pass. A pass is charged where it lands: `pos` is the position of its first
        // row, so the phase-by-position curve and the whole-prompt wall time come out of the same
        // walk rather than two panels.
        result["prefill_scan"]=prefill_scan; result["doc_tokens"]=(int)doc_tokens;
        result["engine_context"]=engine_ctx; result["attn_tts"]=attn_tts; result["wide_attns"]=wide_attns;
        result["attn_scheds"]=attn_scheds; result["attn_folds"]=attn_folds; result["attn_part_mbs"]=attn_part_mbs;
        for(int round=0;round<rounds;++round)for(int mode:modes)for(int n:rows)for(int att:attn_tts)for(int wa:wide_attns)for(int as:attn_scheds)for(int af:attn_folds)for(int apw:attn_part_mbs)for(int ac:attn_coords)for(int anarrow:attn_narrows) {
            // The coordinate is a property of what is already in the K cache, so it can only move
            // where the walk is about to rewrite the cache from position 0 - which is exactly what
            // the `reset_seq` below does. That is why it is an axis of this loop and not of a pass.
            if(ac>=0&&ac!=attn_coord())attn_coord_set(ac);
            if(anarrow>=0)e.set_attn_narrow(anarrow);
            if(apw>=0)attn_batch_set_part_window_mb(e.batch_attn,apw);
            if(af>=0)attn_batch_set_fold(e.batch_attn,af);
            if(att>0)attn_batch_set_group_rows(e.batch_attn,att);
            if(wa>=0)attn_batch_set_enabled(e.batch_attn,wa!=0);
            if(as>=0)attn_batch_set_sched(e.batch_attn,as);
            e.set_batch_profile(false); e.batch_mode=mode; e.reset_seq(seq);
            e.set_batch_profile(true);
            double prompt_wall=0.0;
            for(int pos=0;pos<prefill_scan;pos+=n) {
                const int k=std::min(n,prefill_scan-pos);
                std::vector<std::vector<int>> tv{std::vector<int>(tokens.begin()+pos,tokens.begin()+pos+k)};
                auto start=Clock::now(); e.forward_batch(sv,tv,pos+k>=prefill_scan); HIP_CHECK_H(hipStreamSynchronize(e.stream));
                const double wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                prompt_wall+=wall;
                // Per-pass phase totals rather than every launch: a 64-pass scan would otherwise
                // write a quarter of a million trace entries into run.json.
                std::map<std::string,double> by; std::map<std::string,int> cnt;
                for(auto & t:e.batch_trace) { by[t.kind]+=t.end_ms-t.start_ms; cnt[t.kind]++; }
                json phases=json::object(), launches=json::object();
                for(auto & kv:by)phases[kv.first]=kv.second;
                for(auto & kv:cnt)launches[kv.first]=kv.second;
                result["samples"].push_back({{"round",round},{"mode",mode},{"shape","prefill-scan"},
                    {"rows",k},{"pos",pos},{"scan_tokens",prefill_scan},{"logits",pos+k>=prefill_scan},
                    {"attn_tt",attn_batch_group_rows(e.batch_attn)},{"wide_attn",attn_batch_enabled(e.batch_attn)},
                    {"attn_sched",attn_batch_sched(e.batch_attn)},{"attn_narrow",e.attn_narrow()},
                    {"attn_fold",attn_batch_fold(e.batch_attn)},{"attn_coord",attn_coord()},
                    {"attn_part_mb",attn_batch_part_window_mb(e.batch_attn)},
                    {"profile",true},{"wall_ms",wall},{"device_span_ms",e.batch_trace_span_ms},
                    {"phases",phases},{"launches",launches}});
            }
            std::filesystem::create_directories(std::filesystem::path(out).parent_path()); std::ofstream(out)<<result.dump(2)<<"\n";
            fprintf(stderr,"scan round %d mode %d rows %d attn-tt %d attn-fold %d coord %d wide-attn %d tokens %d: %.1f ms  %.1f tok/s\n",
                round,mode,n,attn_batch_group_rows(e.batch_attn),attn_batch_fold(e.batch_attn),attn_coord(),(int)attn_batch_enabled(e.batch_attn),
                prefill_scan,prompt_wall,prefill_scan*1000.0/prompt_wall);
            result["samples"].push_back({{"round",round},{"mode",mode},{"shape","prefill-scan-total"},
                {"rows",n},{"scan_tokens",prefill_scan},{"attn_tt",attn_batch_group_rows(e.batch_attn)},
                {"attn_sched",attn_batch_sched(e.batch_attn)},{"attn_fold",attn_batch_fold(e.batch_attn)},{"attn_coord",attn_coord()},
                {"attn_part_mb",attn_batch_part_window_mb(e.batch_attn)},
                {"wide_attn",attn_batch_enabled(e.batch_attn)},{"attn_narrow",e.attn_narrow()},
                {"wall_ms",prompt_wall},{"prompt_tok_s",prefill_scan*1000.0/prompt_wall}});
        }
        std::ofstream(out)<<result.dump(2)<<"\n";
        e.batch_profiler.reset();
        HIP_CHECK_H(hipStreamDestroy(e.stream));
        HIP_CHECK_H(hipDeviceReset());
        return 0;
    }
    if(decode_streams) {
        // One profiled decode step per case, after a settled prefill of every slot. The step is a
        // real generation step: one row per stream, logits and argmax on every row.
        decode_setup(decode_setup_mode>=0?decode_setup_mode:modes.front(),modes.front(),decode_prompts.front());
        // Warm every distinct step width; a k-row step takes different kernel instantiations from
        // a one-row step, and a cold instantiation would be charged to whichever case ran first.
        // Both dispatch arms are warmed for the same reason: the narrow instantiation is its own
        // code path in the same kernel.
        for(int wm:wide_mins)for(int an:attn_narrows)for(int hg:attn_hgs)for(int k:decode_tokens)for(int i=0;i<3;++i) { if(wm>=0)batch_set_wide_min(wm); if(an>=0)e.set_attn_narrow(an); if(hg>=1)e.set_attn_hg(hg); std::vector<std::vector<int>> tv(dseq.size(),std::vector<int>(k,1000)); e.forward_batch(dsv,tv,true,nullptr,nullptr); }
        HIP_CHECK_H(hipStreamSynchronize(e.stream));
        for(int round=0;round<rounds;++round)for(int mode:modes)for(int image:images)for(int dense:denses)for(int dnm:dnmats)for(int share:shares)for(int pregate:pregates)for(int gstate:gdn_states)for(int acoord:attn_coords)for(int gdefer:gdn_defers)for(int prime:decode_primes)for(int squant:seq_quants)for(int simg:seq_images)for(int frun:ffn_runs)for(int as:attn_scheds)for(int af:attn_folds)for(int dprompt:decode_prompts)for(int dtok:decode_tokens)for(int anarrow:attn_narrows)for(int ahg:attn_hgs)for(int gsplit:gdn_splits)for(int gcols:gdn_colss)for(int wmin:wide_mins)for(int si4:seq_i4ops)for(int stm:seq_tmaps)for(int spin:seq_pins)for(int tracei:traces) {
            if(spin>=0)sequence_set_pin(spin);
            if(wmin>=0)batch_set_wide_min(wmin);
            if(si4>0&&e.batch_sequence)sequence_batch_set_i4op(e.batch_sequence,si4);
            // No stored byte moves with the token map, so it needs no image sync and takes effect
            // on the next projection.
            if(stm>=0&&e.batch_sequence)sequence_batch_set_tmap(e.batch_sequence,stm);
            // The unit width owns no state, so it moves between steps with nothing to re-prefill.
            // Set unconditionally: -1 has to CLEAR a pin the previous case left, or the arm that
            // asks for the shape rule silently inherits its neighbour's width.
            gdn_resident_set_split(gsplit);
            // Rewrites every resident weight image from the source weights, so each arm of a walk
            // pays the same cold start. Between passes only, never inside one.
            if(frun>=0)ffn_batch_set_run_order(e.batch_ffn,frun,e.stream);
            const bool trace=tracei!=0;
            if(as>=0)attn_batch_set_sched(e.batch_attn,as);
            if(ahg>=1)e.set_attn_hg(ahg);
            if(af>=0)attn_batch_set_fold(e.batch_attn,af);
            if(simg>=0&&e.batch_sequence)sequence_batch_set_image(e.batch_sequence,simg,e.stream);
            if(anarrow>=0)e.set_attn_narrow(anarrow);
            if(squant>=0)sequence_set_quant(squant);
            if(e.batch_sequence){sequence_batch_sync_quant(e.batch_sequence,e.stream);HIP_CHECK_H(hipStreamSynchronize(e.stream));}
            if(image>=0)ffn_batch_set_a4_image(e.batch_ffn,image);
            apply_ffn_dense(e.batch_ffn,dense);
            if(dnm>=0)ffn_batch_set_dn_mats(e.batch_ffn,dnm);
            if(share>=0)ffn_batch_set_a4_sched(e.batch_ffn,share);
            if(pregate>=0)gdn_resident_set_pregate(pregate);
            gdn_resident_set_cols(gcols);   // -1 clears a previous case's pin and restores the shape rule
            // Before decode_setup: the prefill that follows writes every slot's state in this
            // coordinate, so the measured step reads back what it wrote.
            if(gstate>=0)gdn_state_set_format(gstate);
            // The K cache holds one coordinate at a time, so changing it invalidates every settled
            // slot, and the streams have to be re-prefilled into the new one.
            if(acoord>=0&&acoord!=attn_coord())attn_coord_set(acoord);   // decode_setup below re-prefills into it
            if(gdefer>=0)gdn_defer_set_depth(gdefer);
            decode_setup(decode_setup_mode>=0?decode_setup_mode:mode,mode,dprompt);
            // The sequence-projection schedule is the one axis that was reachable in the prefill
            // loop and not here, so no arm of it had ever been run at a decode shape. It selects a
            // kernel instantiation and touches no state, which is why it can sit inside one
            // `decode_setup` instead of paying for its own: the arms then run seconds apart on one
            // prefilled batch rather than a full setup apart, and the residual hash checks it.
            for(int sqs:seq_scheds)for(int stt:seq_tts)for(int sw:seq_ws) {
            if(sqs>=0&&e.batch_sequence)sequence_batch_set_sched(e.batch_sequence,sqs);
            // Shape overrides sit beside the schedule for the same reason: they select a kernel
            // instantiation and touch no state, so the arms run seconds apart on one prefilled
            // batch and the residual hash checks that the choice moved no value.
            if(e.batch_sequence&&(stt>=0||sw>=0))sequence_batch_set_shape(e.batch_sequence,stt>=0?stt:0,sw>=0?sw:0);
            {   // untraced priming steps, so the timed step is the one this case is about
                std::vector<std::vector<int>> pv(dseq.size(),std::vector<int>(dtok,1000));
                for(int i=0;i<prime;++i)e.forward_batch(dsv,pv,true,nullptr,nullptr);
                if(prime)HIP_CHECK_H(hipStreamSynchronize(e.stream));
            }
            e.set_batch_profile(trace);
            std::vector<std::vector<int>> tv(dseq.size(),std::vector<int>(dtok,1000));
            const int step_rows=(int)dseq.size()*dtok;
            auto start=Clock::now(); e.forward_batch(dsv,tv,true,nullptr,nullptr); HIP_CHECK_H(hipStreamSynchronize(e.stream));
            double wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            // Mode 0 runs the persistent kernel one slice at a time through `Engine::forward`, which
            // leaves the residual in the engine's own `x` rather than in the batch buffer. A decode
            // step of at most RMAX rows is a single slice, so `x` holds every row of it; a wider
            // mode-0 step keeps only its last slice and `residual_rows` says so.
            const int hash_rows=mode?step_rows:std::min(step_rows,RMAX);
            std::vector<float> residual((size_t)hash_rows*D); HIP_CHECK_H(hipMemcpy(residual.data(),mode?e.batch_x:e.x,residual.size()*4,hipMemcpyDeviceToHost));
            uint64_t hash=14695981039346656037ull;
            auto bytes=reinterpret_cast<const unsigned char*>(residual.data());
            for(size_t i=0;i<residual.size()*sizeof(float);++i)hash=(hash^bytes[i])*1099511628211ull;
            json sample={{"round",round},{"mode",mode},{"shape","decode"},{"rows",step_rows},{"streams",(int)dseq.size()},
                {"decode_tokens",dtok},{"logits",true},{"profile",trace},{"attn_narrow",e.attn_narrow()},
                {"attn_hg",e.attn_hg()},{"decode_prompt",dprompt},{"decode_context",decode_ctx},
                {"ffn_a4_sched",ffn_batch_a4_sched(e.batch_ffn)},{"ffn_a4_image",ffn_batch_a4_image(e.batch_ffn)},{"ffn_a4_dense",ffn_batch_a4_dense(e.batch_ffn)},{"ffn_b_stage",ffn_batch_b_stage(e.batch_ffn)},{"ffn_spipe",ffn_batch_spipe(e.batch_ffn)},
                {"ffn_dn_mats",ffn_batch_dn_mats(e.batch_ffn)},
                {"gdn_pregate",gdn_resident_pregate()},{"gdn_split",gdn_resident_split()},{"gdn_cols",gdn_resident_cols()},{"gdn_state",gdn_state_format_name(gdn_state_format())},{"attn_coord",attn_coord()},
                {"seq_quant",sequence_quant_name(sequence_quant())},{"attn_sched",attn_batch_sched(e.batch_attn)},{"attn_fold",attn_batch_fold(e.batch_attn)},
                {"seq_i4op",e.batch_sequence?sequence_batch_i4op(e.batch_sequence):-1},
                {"seq_tmap",e.batch_sequence?sequence_batch_tmap(e.batch_sequence):-1},
                {"residual_rows",hash_rows},{"setup_mode",decode_setup_mode>=0?decode_setup_mode:mode},
                {"gdn_defer",gdn_defer_effective()},{"decode_prime",prime},
                {"seq_sched",e.batch_sequence?sequence_batch_sched(e.batch_sequence):-1},
                {"seq_tt",e.batch_sequence?sequence_batch_shape_tt(e.batch_sequence):-1},
                {"seq_w",e.batch_sequence?sequence_batch_shape_w(e.batch_sequence):-1},
                {"seq_image",e.batch_sequence?sequence_batch_image(e.batch_sequence):-1},
                {"ffn_run",ffn_batch_run_order(e.batch_ffn)},
                {"wide_min",batch_wide_min()},{"seq_pin",spin},
                {"wall_ms",wall},{"residual_fnv64",std::to_string(hash)}};
            if(trace) {
                sample["device_span_ms"]=e.batch_trace_span_ms; sample["trace"]=json::array();
                for(auto & t:e.batch_trace)sample["trace"].push_back({{"kind",t.kind},{"layer",t.layer},{"offset",t.offset},{"rows",t.rows},{"start_ms",t.start_ms},{"end_ms",t.end_ms},{"phase_us",t.phase_us}});
            }
            result["samples"].push_back(sample);
            std::filesystem::create_directories(std::filesystem::path(out).parent_path()); std::ofstream(out)<<result.dump(2)<<"\n";
            fprintf(stderr,"decode round %d mode %d streams %d prompt %d tok %d rows %d wmin %d narrow %d hg %d seqsched %d seqimg %d image %d dense %d sched %d pregate %d state %s defer %d prime %d quant %s trace %d: %.3f ms  %.1f tok/s\n",
                round,mode,(int)dseq.size(),dprompt,dtok,step_rows,batch_wide_min(),e.attn_narrow(),e.attn_hg(),
                e.batch_sequence?sequence_batch_sched(e.batch_sequence):-1,
                e.batch_sequence?sequence_batch_image(e.batch_sequence):-1,
                ffn_batch_a4_image(e.batch_ffn),ffn_batch_a4_dense(e.batch_ffn),ffn_batch_a4_sched(e.batch_ffn),gdn_resident_pregate(),
                gdn_state_format_name(gdn_state_format()),gdn_defer_effective(),prime,sequence_quant_name(sequence_quant()),trace,wall,step_rows*1000.0/wall);
            }
        }
        e.batch_profiler.reset();
        HIP_CHECK_H(hipStreamDestroy(e.stream));
        HIP_CHECK_H(hipDeviceReset());
        return 0;
    }
    std::mt19937 rng(71093);
    for(int round=0;round<rounds;++round) {
        std::vector<std::tuple<int,int,bool,bool,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int>> cases;
        for(int m:modes)for(int n:rows)for(bool head:heads)for(bool trace:traces)for(int sh:shares)for(int im:images)for(int pg:pregates)for(int od:orders)for(int sq:seq_scheds)for(int gs:gdn_states)for(int gd:gdn_defers)for(int dn:denses)for(int dm:dnmats)for(int hr:head_rows)for(int qz:seq_quants)for(int fs:ffn_slices)for(int as:attn_scheds)for(int si:seq_images)for(int an:attn_narrows)for(int ac:attn_coords)for(int fr:ffn_runs)for(int gsp:gdn_splits)for(int gc:gdn_colss)for(int wm:wide_mins)for(int ho:head_ops)for(int ht:head_tts)for(int af:attn_folds)for(int mo:mvw_ords)for(int ma:mvw_arms)for(int s4:seq_i4ops)for(int stm:seq_tmaps)for(int sp:seq_pins)cases.emplace_back(m,n,head,trace,sh,im,pg,od,sq,gs,gd,dn,dm,hr,qz,fs,as,si,an,ac,fr,gsp,gc,ho,ht,wm,af,mo,ma,s4,stm,sp);
        std::shuffle(cases.begin(),cases.end(),rng);
        for(auto [mode,n,head,trace,share,image,pregate,order,seq_sched,gstate,gdefer,dense,dnm,hrows,squant,ffn_slice,asched,seq_image,anarrow,acoord,ffn_run,gsplit,gcols,hop,htt,wmin,afold,mvw_ord,mvw_arm,seq_i4op,seq_tmap,seq_pin]:cases) {
            // Collapses one or both projection streams onto block 0; owns no state and takes
            // effect on the next projection.
            if(seq_pin>=0)sequence_set_pin(seq_pin);
            // No stored byte moves with the token map, so it needs no image sync and takes
            // effect on the next projection.
            if(seq_tmap>=0&&e.batch_sequence)sequence_batch_set_tmap(e.batch_sequence,seq_tmap);
            // Selects a kernel instantiation and the stored code order that feeds it. Set before
            // the image sync below, which is outside every timed region, so an arm switch costs a
            // reorder pass between cases and nothing inside one.
            if(seq_i4op>0&&e.batch_sequence)sequence_batch_set_i4op(e.batch_sequence,seq_i4op);
            // Which operand the FFN slice's block loop requests first; -1 leaves the process default.
            if(mvw_ord>=0)ffn_slice_set_order(mvw_ord);
            // Selects a kernel instantiation and nothing else; the launcher caches one grid per arm
            // and HALO_FFN_GRID=1 prints it, so an arm that costs a workgroup says so.
            if(mvw_arm>=0)ffn_slice_set_arm(mvw_arm);
            // The prefill shape reaches the floor too: a prompt whose last chunk is 9..31 rows, or
            // a short prompt that is one such pass, routes on exactly this constant.
            if(wmin>=0)batch_set_wide_min(wmin);
            head_set_operand_arm(hop);   // -1 clears a previous case's pin
            head_set_tile_width(htt);
            gdn_resident_set_split(gsplit);   // -1 clears a previous case's pin, see the decode loop
            if(acoord>=0)attn_coord_set(acoord);   // setup() re-prefills into the new coordinate
            // Rewrites every resident weight image from the source weights; every arm pays it.
            if(ffn_run>=0)ffn_batch_set_run_order(e.batch_ffn,ffn_run,e.stream);
            if(asched>=0)attn_batch_set_sched(e.batch_attn,asched);
            if(afold>=0)attn_batch_set_fold(e.batch_attn,afold);
            if(anarrow>=0)e.set_attn_narrow(anarrow);
            if(ffn_slice>=0)batch_set_ffn_slice_rows(ffn_slice);
            if(seq_image>=0&&e.batch_sequence)sequence_batch_set_image(e.batch_sequence,seq_image,e.stream);
            if(squant>=0)sequence_set_quant(squant);
            if(e.batch_sequence){sequence_batch_sync_quant(e.batch_sequence,e.stream);HIP_CHECK_H(hipStreamSynchronize(e.stream));}
            if(share>=0)ffn_batch_set_a4_sched(e.batch_ffn,share);
            if(image>=0)ffn_batch_set_a4_image(e.batch_ffn,image);
            apply_ffn_dense(e.batch_ffn,dense);
            if(dnm>=0)ffn_batch_set_dn_mats(e.batch_ffn,dnm);
            if(pregate>=0)gdn_resident_set_pregate(pregate);
            gdn_resident_set_cols(gcols);   // -1 clears a previous case's pin and restores the shape rule
            if(order>=0)ffn_batch_set_a4_order(e.batch_ffn,order);
            if(seq_sched>=0&&e.batch_sequence)sequence_batch_set_sched(e.batch_sequence,seq_sched);
            if(gstate>=0)gdn_state_set_format(gstate);   // setup() re-prefills into the new coordinate
            if(gdefer>=0)gdn_defer_set_depth(gdefer);
            setup(mode); e.set_batch_profile(trace);
            std::vector<std::vector<int>> tv{std::vector<int>(tokens.begin()+context,tokens.begin()+context+n)};
            // A prompt pass asks for the tail row the way `Engine::prefill` does, or for every row,
            // which is what the head did before the selection existed.
            auto start=Clock::now(); e.forward_batch(sv,tv,head,nullptr,nullptr,
                hrows?Engine::LogitRows::Tail:Engine::LogitRows::All); HIP_CHECK_H(hipStreamSynchronize(e.stream));
            double wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            std::vector<float> residual((size_t)n*D); HIP_CHECK_H(hipMemcpy(residual.data(),e.batch_x,residual.size()*4,hipMemcpyDeviceToHost));
            uint64_t hash=14695981039346656037ull;
            auto bytes=reinterpret_cast<const unsigned char*>(residual.data());
            for(size_t i=0;i<residual.size()*sizeof(float);++i)hash=(hash^bytes[i])*1099511628211ull;
            json sample={{"round",round},{"mode",mode},{"rows",n},{"logits",head},{"head_rows",hrows?"tail":"all"},{"head_op",head_operand_arm()},{"head_tt",htt},{"profile",trace},{"attn_narrow",e.attn_narrow()},{"attn_hg",e.attn_hg()},
                {"ffn_a4_sched",ffn_batch_a4_sched(e.batch_ffn)},{"ffn_a4_image",ffn_batch_a4_image(e.batch_ffn)},{"ffn_a4_dense",ffn_batch_a4_dense(e.batch_ffn)},{"ffn_b_stage",ffn_batch_b_stage(e.batch_ffn)},{"ffn_spipe",ffn_batch_spipe(e.batch_ffn)},
                {"ffn_dn_mats",ffn_batch_dn_mats(e.batch_ffn)},
                {"gdn_pregate",gdn_resident_pregate()},{"gdn_split",gdn_resident_split()},{"ffn_a4_order",ffn_batch_a4_order(e.batch_ffn)},
                {"gdn_pregate",gdn_resident_pregate()},{"ffn_a4_order",ffn_batch_a4_order(e.batch_ffn)},
                {"gdn_cols",gdn_resident_cols()},{"gdn_cols_available",gdn_resident_cols_available()},
                {"ffn_run",ffn_batch_run_order(e.batch_ffn)},
                {"seq_sched",e.batch_sequence?sequence_batch_sched(e.batch_sequence):-1},
                {"seq_image",e.batch_sequence?sequence_batch_image(e.batch_sequence):-1},
                {"gdn_state",gdn_state_format_name(gdn_state_format())},{"attn_coord",attn_coord()},
                {"seq_quant",sequence_quant_name(sequence_quant())},{"ffn_slice_rows",batch_ffn_slice_rows()},
                {"seq_i4op",e.batch_sequence?sequence_batch_i4op(e.batch_sequence):-1},
                {"seq_tmap",e.batch_sequence?sequence_batch_tmap(e.batch_sequence):-1},
                {"mvw_order",ffn_slice_order()},{"mvw_arm",ffn_slice_arm()},
                {"gdn_defer",gdn_defer_effective()},{"attn_sched",attn_batch_sched(e.batch_attn)},{"wide_min",batch_wide_min()},
                {"attn_fold",attn_batch_fold(e.batch_attn)},{"seq_pin",seq_pin},
                {"wall_ms",wall},{"residual_fnv64",std::to_string(hash)}};
            if(trace) {
                sample["device_span_ms"]=e.batch_trace_span_ms; sample["trace"]=json::array();
                for(auto & t:e.batch_trace)sample["trace"].push_back({{"kind",t.kind},{"layer",t.layer},{"offset",t.offset},{"rows",t.rows},{"start_ms",t.start_ms},{"end_ms",t.end_ms},{"phase_us",t.phase_us}});
            }
            result["samples"].push_back(sample);
            std::filesystem::create_directories(std::filesystem::path(out).parent_path()); std::ofstream(out)<<result.dump(2)<<"\n";
            fprintf(stderr,"round %d mode %d rows %d head %d(%s) op %d tt %d trace %d sched %d image %d dense %d pregate %d order %d seq %d seqimg %d state %s quant %s: %.3f ms\n",round,mode,n,head,hrows?"tail":"all",head_operand_arm(),htt,trace,
                ffn_batch_a4_sched(e.batch_ffn),ffn_batch_a4_image(e.batch_ffn),ffn_batch_a4_dense(e.batch_ffn),gdn_resident_pregate(),
                ffn_batch_a4_order(e.batch_ffn),e.batch_sequence?sequence_batch_sched(e.batch_sequence):-1,
                e.batch_sequence?sequence_batch_image(e.batch_sequence):-1,
                gdn_state_format_name(gdn_state_format()),sequence_quant_name(sequence_quant()),wall);
        }
    }
    e.batch_profiler.reset();
    HIP_CHECK_H(hipStreamDestroy(e.stream));
    HIP_CHECK_H(hipDeviceReset());
    return 0;
} catch(const std::exception & ex) { fprintf(stderr,"batch_profile: %s\n",ex.what()); return 1; } }
