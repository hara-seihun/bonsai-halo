#include "engine.h"
#include "ffn_batch.h"
#include "head_batch.h"
#include "device_budget.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
#include <map>
#include <algorithm>
#include <random>
#include <cmath>

namespace halo {

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); } while (0)

void * Engine::dmalloc(size_t bytes) {
    admit_device_bytes(device_bytes, bytes, "device allocation");
    void * p = nullptr;
    const char * managed = getenv("HALO_MANAGED_ALLOC");
    if (managed && atoi(managed)) {
        HIP_CHECK(hipMallocManaged(&p, bytes));
        device_bytes += bytes;
        return p;
    }
    hipError_t e = hipMalloc(&p, bytes);
    if (e != hipSuccess) {
        (void) hipGetLastError();
        HIP_CHECK(hipMallocManaged(&p, bytes));
    }
    device_bytes += bytes;
    return p;
}

template <typename T> T * Engine::upload(const void * src, size_t bytes) {
    T * d = (T *) dmalloc(bytes);
    HIP_CHECK(hipMemcpy(d, src, bytes, hipMemcpyHostToDevice));
    return d;
}

const float * Engine::upload_f32(const GgufTensor & t) {
    if (t.type != GGML_F32) throw std::runtime_error("expected f32 tensor: " + t.name);
    return upload<float>(t.data, t.nbytes);
}

const unsigned short * Engine::upload_bf16_pair(const GgufTensor & a, const GgufTensor & b) {
    if (a.type != GGML_BF16 || b.type != GGML_BF16) throw std::runtime_error("expected bf16: " + a.name);
    std::vector<uint8_t> h(a.nbytes + b.nbytes);
    memcpy(h.data(), a.data, a.nbytes);
    memcpy(h.data() + a.nbytes, b.data, b.nbytes);
    return upload<unsigned short>(h.data(), h.size());
}

// norm weight (or none) times the sign vector, precomputed for the fused prep: one constant load per element
const float * Engine::upload_folded(const GgufTensor * norm, const std::vector<float> & signs, bool per_head_128) {
    std::vector<float> h(signs.size());
    const float * nw = norm ? (const float *) norm->data : nullptr;
    for (size_t i = 0; i < h.size(); i++) h[i] = signs[i] * (nw ? nw[per_head_128 ? (i & 127) : i] : 1.0f);
    return upload<float>(h.data(), h.size() * 4);
}

const uint8_t * Engine::upload_halo(const std::string & name, int64_t N, int64_t K) {
    const HaloCacheEntry & e = cache.entry(name);
    if (e.N != N || e.K != K) throw std::runtime_error("shape mismatch for " + name);
    return upload<uint8_t>(cache.data(e), e.bytes);
}

void Engine::load(const std::string & path, int nthreads, int n_slots, int context_capacity) {
    halo_format_selftest();
    if (n_slots < 1 || n_slots > MAXSLOTS) throw std::runtime_error("sequence slots must be 1.." + std::to_string(MAXSLOTS));
    if (context_capacity < 256 || context_capacity > MAXCTX || context_capacity % 256)
        throw std::runtime_error("context capacity must be a multiple of 256 in 256..32768");
    nslots = n_slots;
    context = context_capacity;
    if (const char * gb = getenv("HALO_SNAPSHOT_KV_GB")) { const double v = atof(gb); if (v >= 0) snap_kv_budget = (size_t) (v * (double) (1ull << 30)); }
    auto t0 = std::chrono::steady_clock::now();
    g.open(path);
    if (g.get("general.architecture").as_string() != "qwen35") throw std::runtime_error("not a qwen35 model");
    if (g.get("qwen35.block_count").as_int() != NLAYER || g.get("qwen35.embedding_length").as_int() != D ||
        g.get("qwen35.feed_forward_length").as_int() != FF) throw std::runtime_error("model geometry is not Bonsai 27B");
    if (!g.has("prism.hadamard.version")) throw std::runtime_error("model has no prism.hadamard metadata");
    if (g.get("prism.hadamard.block_size").as_int() != HAD) throw std::runtime_error("unexpected hadamard block size");

    cache.open_or_build(g, path, nthreads);

    hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    if (prop.warpSize != 32) throw std::runtime_error("kernels assume wave32");
    // ROCm's active wait spins this thread for as long as the GPU is busy, which reads
    // as a fully busy core. It is not worth changing: measured, that core costs 0 to 2 W
    // of a 120 W socket and blocking on the completion interrupt instead is +0.09% on a
    // 32-stream step. docs/power-budget.md holds the panel and the arm.
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // sign vectors
    {
        auto widths = g.get("prism.hadamard.sign_widths").as_int_array();
        auto values = g.get("prism.hadamard.sign_values").as_int_array();
        size_t off = 0;
        for (int64_t w : widths) {
            std::vector<float> s(w);
            for (int64_t i = 0; i < w; i++) s[i] = (float) values.at(off + i);
            off += w;
            const float * d = upload<float>(s.data(), w * 4);
            if (w == 5120) { signs5120 = d; h_signs5120 = s; } else if (w == 6144) { signs6144 = d; h_signs6144 = s; } else if (w == 17408) signs17408 = d;
            else throw std::runtime_error("unexpected sign width");
        }
        if (!signs5120 || !signs6144 || !signs17408) throw std::runtime_error("missing sign vectors");
    }

    tok_embd = upload_halo("token_embd.weight", VOCAB, D);
    output = upload_halo("output.weight", VOCAB, D);
    output_norm = upload_f32(g.tensor("output_norm.weight"));
    output_norm_s = upload_folded(&g.tensor("output_norm.weight"), h_signs5120, false);

    layers.resize(NLAYER);
    int kv_slot = 0;
    for (int l = 0; l < NLAYER; l++) {
        LayerDev & L = layers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        L.recurrent = ((l + 1) % 4) != 0;
        L.attn_norm = upload_f32(g.tensor(p + "attn_norm.weight"));
        L.post_norm = upload_f32(g.tensor(p + "post_attention_norm.weight"));
        L.attn_norm_s = upload_folded(&g.tensor(p + "attn_norm.weight"), h_signs5120, false);
        L.post_norm_s = upload_folded(&g.tensor(p + "post_attention_norm.weight"), h_signs5120, false);
        if (L.recurrent) {
            L.qkv = upload_halo(p + "attn_qkv.weight", QKV_OUT, D);
            L.z = upload_halo(p + "attn_gate.weight", VDIM, D);
            L.ssm_out = upload_halo(p + "ssm_out.weight", D, VDIM);
            L.conv_w = upload_f32(g.tensor(p + "ssm_conv1d.weight"));
            L.ssm_a = upload_f32(g.tensor(p + "ssm_a"));
            L.ssm_dt = upload_f32(g.tensor(p + "ssm_dt.bias"));
            L.ssm_norm = upload_f32(g.tensor(p + "ssm_norm.weight"));
            L.ssm_norm_s = upload_folded(&g.tensor(p + "ssm_norm.weight"), h_signs6144, true);
            L.alpha_beta = upload_bf16_pair(g.tensor(p + "ssm_alpha.weight"), g.tensor(p + "ssm_beta.weight"));
        } else {
            L.q = upload_halo(p + "attn_q.weight", Q_OUT, D);
            L.k = upload_halo(p + "attn_k.weight", KV_OUT, D);
            L.v = upload_halo(p + "attn_v.weight", KV_OUT, D);
            L.o = upload_halo(p + "attn_output.weight", D, ATTN_OUT);
            L.q_norm = upload_f32(g.tensor(p + "attn_q_norm.weight"));
            L.k_norm = upload_f32(g.tensor(p + "attn_k_norm.weight"));
            L.kv_slot = kv_slot++;
        }
        L.gate = upload_halo(p + "ffn_gate.weight", FF, D);
        L.up = upload_halo(p + "ffn_up.weight", FF, D);
        L.down = upload_halo(p + "ffn_down.weight", D, FF);
        fprintf(stderr, "\rloaded layer %d/%d (%.2f GB on device)", l + 1, NLAYER, device_bytes / 1e9);
    }
    fprintf(stderr, "\n");
    if (kv_slot != NATTN) throw std::runtime_error("unexpected attention layer count");

    // activations (row-major, RMAX rows)
    x = (float *) dmalloc((size_t) RMAX * D * 4); xn = (float *) dmalloc((size_t) RMAX * D * 4); tmp = (float *) dmalloc((size_t) RMAX * D * 4);
    // widest quantised activation row: the drafter's 5 x D captured features. FMAX rows, because the
    // FFN slice fills all sixteen columns of its matrix instruction and quantises that many rows.
    xq = (int8_t *) dmalloc((size_t) FMAX * NCAP * D); xs = (float *) dmalloc((size_t) FMAX * (NCAP * D / BLOCK) * 4); xsum = (int *) dmalloc((size_t) FMAX * (NCAP * D / BLOCK) * 4);
    big = (float *) dmalloc((size_t) RMAX * QKV_OUT * 4); zbuf = (float *) dmalloc((size_t) RMAX * VDIM * 4); conv_out = (float *) dmalloc(CONV_CH * 4);
    ab = (float *) dmalloc((size_t) RMAX * 2 * HV * 4); y = (float *) dmalloc((size_t) RMAX * VDIM * 4); gu = (float *) dmalloc((size_t) 2 * FMAX * FF * 4);
    qfull = (float *) dmalloc((size_t) RMAX * Q_OUT * 4); kbuf = (float *) dmalloc((size_t) RMAX * KV_OUT * 4); vbuf = (float *) dmalloc((size_t) RMAX * KV_OUT * 4); qrot = (float *) dmalloc((size_t) RMAX * ATTN_OUT * 4);
    partials = (AttnPartial *) dmalloc(sizeof(AttnPartial) * RMAX * DF_NH * AMAX_CHUNKS); // 32 heads covers both geometries
    logits = (float *) dmalloc((size_t) RMAX * VOCAB * 4);
    tok_out = (int *) dmalloc(4); pos = (int *) dmalloc(4);
    HIP_CHECK(hipHostMalloc((void **) &h_ring, TOK_RING * 4, hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer((void **) &d_ring, h_ring, 0));
    obuf = (float *) dmalloc((size_t) RMAX * VDIM * 4);
    bar = (unsigned *) dmalloc(4096); work = (unsigned *) dmalloc(4096 * 4);
    amax_val = (float *) dmalloc((size_t) RMAX * 1024 * 4); amax_idx = (int *) dmalloc((size_t) RMAX * 1024 * 4); argmax_out = (int *) dmalloc(RMAX * 4);
    hcap = (float *) dmalloc((size_t) CAPMAX * NCAP * D * 4); hfinal = (float *) dmalloc((size_t) CAPMAX * D * 4); ninv = (float *) dmalloc(RMAX * 4);
    if (getenv("HALO_PROFILE")) prof = (unsigned long long *) dmalloc(1024 * 8);
    // per-slot state, plus KV-less areas for the prompt-prefix snapshots
    // The stride freezes here, at the widest coordinate this process reserved, and every kernel
    // reads the same number from `r_gdn_region`. An int8 process allocates 1.152 MiB per
    // (slot, layer) instead of 3.000, which is 55 MiB of a sequence slot instead of 144.
    const size_t gdn_region = gdn_region_floats();
    gdn_state = (float *) dmalloc((size_t) (nslots + (snap_kv_budget ? SNAP_AREAS : 0)) * 48 * gdn_region * 4);
    conv_state = (float *) dmalloc((size_t) 3 * CONV_CH * 4 * 48);
    conv_ring = (float *) dmalloc((size_t) (nslots + (snap_kv_budget ? SNAP_AREAS : 0)) * 48 * GDN_RING_FLOATS * 4);
    blk_cache = (float *) dmalloc((size_t) (nslots + (snap_kv_budget ? SNAP_AREAS : 0)) * 2 * 48 * RMAX * BLK_TOKEN_FLOATS * 4);
    kcache = (__half *) dmalloc((size_t) nslots * KV_SLOTS * NKV * context * HD * 2);
    vcache = (__half *) dmalloc((size_t) nslots * KV_SLOTS * NKV * context * HD * 2);
    HIP_CHECK(hipMemset(kcache, 0, (size_t) nslots * KV_SLOTS * NKV * context * HD * 2));
    HIP_CHECK(hipMemset(vcache, 0, (size_t) nslots * KV_SLOTS * NKV * context * HD * 2));
    {
        std::vector<LayerW> lw(NLAYER);
        for (int l = 0; l < NLAYER; l++) {
            const LayerDev & L = layers[l]; LayerW & w = lw[l];
            w.recurrent = L.recurrent; w.kv_slot = L.kv_slot;
            w.qkv = L.qkv; w.z = L.z; w.ssm_out = L.ssm_out; w.q = L.q; w.k = L.k; w.v = L.v; w.o = L.o;
            w.gate = L.gate; w.up = L.up; w.down = L.down;
            w.attn_norm = L.attn_norm; w.post_norm = L.post_norm; w.q_norm = L.q_norm; w.k_norm = L.k_norm;
            w.attn_norm_s = L.attn_norm_s; w.post_norm_s = L.post_norm_s; w.ssm_norm_s = L.ssm_norm_s;
            w.conv_w = L.conv_w; w.ssm_a = L.ssm_a; w.ssm_dt = L.ssm_dt; w.ssm_norm = L.ssm_norm; w.alpha_beta = L.alpha_beta;
        }
        host_layers = lw;
        set_layer_table_rows(lw.data());
    }
    fwd.context = context;
    // Narrow row groups get the instantiation that fits them. The environment pins it for a process
    // that cannot call set_attn_narrow(); the setter overrides it, so one process can interleave
    // both dispatches under one clock. Every params struct the engine builds - the batch slices,
    // the MTP head and the DFlash2 drafter - is copied from `fwd`, so this one assignment covers
    // them all.
    if (const char * e = getenv("HALO_ATTN_NARROW")) fwd.attn_narrow = atoi(e) != 0;
    if (const char * e = getenv("HALO_ATTN_HG")) fwd.attn_hg = atoi(e) > 1 ? atoi(e) : 1;
    fwd.k_tiled = attn_coord();
    fwd.tok_embd = tok_embd; fwd.output = output; fwd.output_norm = output_norm;
    fwd.signs5120 = signs5120; fwd.signs6144 = signs6144; fwd.signs17408 = signs17408; fwd.output_norm_s = output_norm_s;
    fwd.x = x; fwd.xn = xn; fwd.tmp = tmp; fwd.xq = xq; fwd.xs = xs; fwd.xsum = xsum;
    fwd.big = big; fwd.zbuf = zbuf; fwd.ab = ab; fwd.o = obuf; fwd.gu = gu; fwd.qfull = qfull; fwd.kbuf = kbuf; fwd.vbuf = vbuf; fwd.logits = logits;
    fwd.partials = partials; fwd.gdn_state = gdn_state; fwd.conv_ring = conv_ring; fwd.kcache = kcache; fwd.vcache = vcache;
    fwd.tok_ring = d_ring; fwd.pos = pos; fwd.tok_out = tok_out; fwd.bar = bar; fwd.work = work; fwd.amax_val = amax_val; fwd.amax_idx = amax_idx; fwd.prof = prof;
    fwd.blk_cache = blk_cache; fwd.qrot = qrot; fwd.hcap = nullptr; fwd.hfinal = hfinal; fwd.argmax_out = argmax_out; fwd.ninv = ninv;
    // Register budget for the persistent kernel, read once. Every FwdParams the batch path launches
    // is a copy of this one, so setting it here reaches the wide route as well as forward_rows, and
    // set_pk_occ() overrides it per call for an in-process pair.
    if (const char * pko = getenv("HALO_PK_OCC")) { const int v = atoi(pko); fwd.pk_occ = v < 0 ? -1 : (v > 2 ? 2 : v); }
    if (pk_occ_forced >= -1) fwd.pk_occ = pk_occ_forced;
    reset();
    HIP_CHECK(hipDeviceSynchronize());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "model ready: %.2f GB on device, %.1f s\n", device_bytes / 1e9, s);
}

void Engine::reset() {
    // The host's pending-term count belongs to each slot's state, which this reset discards.
    for (int slot = 0; slot < nslots; slot++) gdn_defer_note_reset(slot);
    HIP_CHECK(hipMemsetAsync(gdn_state, 0, (size_t) (nslots + (snap_kv_budget ? SNAP_AREAS : 0)) * 48 * gdn_region_floats() * 4, stream));
    HIP_CHECK(hipMemsetAsync(conv_state, 0, (size_t) 3 * CONV_CH * 4 * 48, stream));
    HIP_CHECK(hipMemsetAsync(conv_ring, 0, (size_t) (nslots + (snap_kv_budget ? SNAP_AREAS : 0)) * 48 * GDN_RING_FLOATS * 4, stream));
    launch_set_int(pos, 0, stream);
    n_pos = 0;
    seq0 = Seq{};
    HIP_CHECK(hipStreamSynchronize(stream));
}

void Engine::reset_seq(Seq & s) {
    // A zeroed region is an empty pending list over a zero state in every coordinate; this is the
    // host's half of the same fact, which is what decides whether a later narrow pass commits.
    gdn_defer_note_reset(s.slot);
    HIP_CHECK(hipMemsetAsync(gdn_state + (size_t) s.slot * 48 * gdn_region_floats(), 0, (size_t) 48 * gdn_region_floats() * 4, stream));
    HIP_CHECK(hipMemsetAsync(conv_ring + (size_t) s.slot * 48 * GDN_RING_FLOATS, 0, (size_t) 48 * GDN_RING_FLOATS * 4, stream));
    s.len = 0; s.keep = 0; s.parity = 0; s.last_n = 0; s.rollbackable = true;
}

void Engine::forward(std::vector<Seq *> & seqs, const std::vector<std::vector<int>> & toks, bool with_logits, std::vector<int> * argmax) {
    fwd.k_tiled = attn_coord();
    int n = 0;
    fwd.nseq = (int) seqs.size();
    if (fwd.nseq > MAXSEQ || seqs.size() != toks.size()) throw std::runtime_error("invalid sequence count in one pass");
    for (size_t i = 0; i < seqs.size(); i++) {
        Seq & s = *seqs[i];
        const int k = (int) toks[i].size();
        if (s.slot < 0 || s.slot >= nslots || k <= 0) throw std::runtime_error("invalid sequence slot or empty row group");
        if (n + k > RMAX) throw std::runtime_error("too many rows in one pass");
        if (s.len + k > context) throw std::runtime_error("context full");
        fwd.seqs[i] = { n, k, s.keep, s.parity };
        for (int j = 0; j < k; j++) fwd.rows[n + j] = { toks[i][j], s.slot, s.len + j, 0 };
        n += k;
        s.len += k; s.keep = fwd.commit_state ? 0 : k; s.last_n = k; s.parity ^= 1;
        s.rollbackable = !fwd.commit_state;
    }
    fwd.nrows = n;
    { static int want = getenv("HALO_PROFILE_ROWS") ? atoi(getenv("HALO_PROFILE_ROWS")) : 0; fwd.prof = (prof && (!want || want == n)) ? prof : nullptr; }
    fwd.with_logits = with_logits ? 1 : 0;
    fwd.debug_layers = getenv("HALO_LAYERS") ? atoi(getenv("HALO_LAYERS")) : 0;
    fwd.debug_stop = getenv("HALO_STOP") ? atoi(getenv("HALO_STOP")) : 0;
    // Persistent-kernel workgroup count. The environment pins it for a process that cannot call
    // set_grid_rows(); the setter overrides it, so one process can interleave grids.
    { static const char * e = getenv("HALO_GRID_ROWS");
      if (e && grid_rows_forced < 0) fwd.grid_rows = atoi(e); }
    if (grid_rows_forced >= 0) fwd.grid_rows = grid_rows_forced;
    if (pk_occ_forced >= -1) fwd.pk_occ = pk_occ_forced;
    HIP_CHECK(hipMemsetAsync(bar, 0, 4, stream));
    HIP_CHECK(hipMemsetAsync(work, 0, 4096 * 4, stream));
    launch_forward_rows(fwd, stream);
    if (argmax) {
        argmax->resize(n);
        HIP_CHECK(hipMemcpyAsync(argmax->data(), argmax_out, n * 4, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
}

void Engine::dump(const char * name, const float * dev, size_t n) {
    if (dump_dir.empty()) return;
    std::vector<float> h = read(dev, n);
    std::string fn = dump_dir + "/" + name + ".bin";
    FILE * f = fopen(fn.c_str(), "wb"); if (!f) throw std::runtime_error("cannot write " + fn);
    fwrite(h.data(), 4, n, f); fclose(f);
}

void Engine::forward_token(bool with_logits) {
    hipStream_t st = stream;
    // This route reads the state in place and has no rebuild, so a pending write-back has to be
    // committed first; it costs nothing when there is none. See docs/gdn-defer-exact.md.
    gdn_defer_flush_pending(gdn_state, 0, st);
    char nm[64];
#define DUMP(name, ptr, n) do { if (!dump_dir.empty()) { snprintf(nm, sizeof nm, "%s-%d", name, l); dump(nm, ptr, n); } } while (0)
    const int8_t * xq_c = xq; const float * xs_c = xs; const int * xsum_c = xsum;

    // embedding
    launch_embed_row(tok_embd, d_ring, pos, tmp, st);
    { PrepArgs p{}; p.x = tmp; p.signs = signs5120; p.n = D; p.flags = PREP_HADAMARD | PREP_SIGN_AFTER | PREP_STORE_F32; p.out_f32 = x; launch_prep(p, st); }
    if (!dump_dir.empty()) { dump("model.input_embed", x, D); dump("cs0_start", conv_state, 3 * CONV_CH); }

    int gdn_idx = 0;
    for (int l = 0; l < NLAYER; l++) {
        const LayerDev & L = layers[l];
        if (L.recurrent) {
            { PrepArgs p{}; p.x = x; p.norm_w = L.attn_norm; p.eps = NORM_EPS; p.signs = signs5120; p.n = D;
              p.flags = PREP_NORM | PREP_SIGN | PREP_HADAMARD | PREP_QUANT; p.xq = xq; p.xs = xs; p.xsum = xsum; p.out_norm = xn; launch_prep(p, st); }
            DUMP("attn_norm", xn, D);
            { MvArgs a{}; a.nseg = 2; a.seg[0] = { L.qkv, big, QKV_OUT / 32, 0 }; a.seg[1] = { L.z, zbuf, VDIM / 32, 0 };
              a.xq = xq_c; a.xs = xs_c; a.xsum = xsum_c; launch_matvec(a, D, st); }
            DUMP("linear_attn_qkv_mixed", big, QKV_OUT); DUMP("z", zbuf, VDIM);
            launch_matvec_bf16(L.alpha_beta, xn, ab, 2 * HV, D, st);
            DUMP("alpha", ab, HV); DUMP("beta", ab + HV, HV);
            DUMP("conv_state_pre", conv_state + (size_t) gdn_idx * 3 * CONV_CH, 3 * CONV_CH);
            launch_gdn_conv(big, L.conv_w, conv_state + (size_t) gdn_idx * 3 * CONV_CH, conv_out, st);
            DUMP("conv_output_silu", conv_out, CONV_CH); DUMP("conv_state", conv_state + (size_t) gdn_idx * 3 * CONV_CH, 3 * CONV_CH);
            { GdnArgs a{}; a.conv_out = conv_out; a.alpha = ab; a.beta = ab + HV; a.z = zbuf; a.ssm_a = L.ssm_a; a.dt_bias = L.ssm_dt; a.ssm_norm = L.ssm_norm;
              // Slot 0's regions at this process's stride. The stride is the region and not the
              // value count: a deferring process reserves room for a pending list after the
              // values, and this route reads the state directly rather than through the codec.
              a.state = gdn_state + (size_t) gdn_idx * gdn_region_floats(); a.y = y; a.eps = NORM_EPS; launch_gdn(a, st); }
            DUMP("final_output", y, VDIM);
            { PrepArgs p{}; p.x = y; p.signs = signs6144; p.n = VDIM; p.flags = PREP_PERMUTE_GDN | PREP_SIGN | PREP_HADAMARD | PREP_QUANT;
              p.xq = xq; p.xs = xs; p.xsum = xsum; launch_prep(p, st); }
            { MvArgs a{}; a.nseg = 1; a.seg[0] = { L.ssm_out, x, D / 32, 1 }; a.xq = xq_c; a.xs = xs_c; a.xsum = xsum_c; launch_matvec(a, VDIM, st); }
            DUMP("attn_residual", x, D);
            gdn_idx++;
        } else {
            { PrepArgs p{}; p.x = x; p.norm_w = L.attn_norm; p.eps = NORM_EPS; p.signs = signs5120; p.n = D;
              p.flags = PREP_NORM | PREP_SIGN | PREP_HADAMARD | PREP_QUANT; p.xq = xq; p.xs = xs; p.xsum = xsum; launch_prep(p, st); }
            { MvArgs a{}; a.nseg = 3; a.seg[0] = { L.q, qfull, Q_OUT / 32, 0 }; a.seg[1] = { L.k, kbuf, KV_OUT / 32, 0 }; a.seg[2] = { L.v, vbuf, KV_OUT / 32, 0 };
              a.xq = xq_c; a.xs = xs_c; a.xsum = xsum_c; launch_matvec(a, D, st); }
            DUMP("Qcur_full", qfull, Q_OUT); DUMP("Kcur", kbuf, KV_OUT); DUMP("Vcur", vbuf, KV_OUT);
            __half * kc = kcache + (size_t) L.kv_slot * NKV * context * HD;
            __half * vc = vcache + (size_t) L.kv_slot * NKV * context * HD;
            { AttnPreArgs a{}; a.context = context; a.q_full = qfull; a.k = kbuf; a.v = vbuf; a.q_norm = L.q_norm; a.k_norm = L.k_norm; a.eps = NORM_EPS; a.pos = pos;
              a.kcache = kc; a.vcache = vc; a.q_out = qrot; a.k_tiled = attn_coord(); launch_attn_pre(a, st); }
            { AttnArgs a{}; a.context = context; a.q = qrot; a.q_full = qfull; a.kcache = kc; a.vcache = vc; a.pos = pos; a.partials = partials; a.y = y; a.k_tiled = attn_coord(); launch_attn(a, st); }
            DUMP("attn_gated", y, ATTN_OUT);
            { PrepArgs p{}; p.x = y; p.signs = signs6144; p.n = ATTN_OUT; p.flags = PREP_SIGN | PREP_HADAMARD | PREP_QUANT;
              p.xq = xq; p.xs = xs; p.xsum = xsum; launch_prep(p, st); }
            { MvArgs a{}; a.nseg = 1; a.seg[0] = { L.o, x, D / 32, 1 }; a.xq = xq_c; a.xs = xs_c; a.xsum = xsum_c; launch_matvec(a, ATTN_OUT, st); }
            DUMP("attn_residual", x, D);
        }
        // FFN
        { PrepArgs p{}; p.x = x; p.norm_w = L.post_norm; p.eps = NORM_EPS; p.signs = signs5120; p.n = D;
          p.flags = PREP_NORM | PREP_SIGN | PREP_HADAMARD | PREP_QUANT; p.xq = xq; p.xs = xs; p.xsum = xsum; launch_prep(p, st); }
        { MvArgs a{}; a.nseg = 2; a.seg[0] = { L.gate, gu, FF / 32, 0 }; a.seg[1] = { L.up, gu + FF, FF / 32, 0 };
          a.xq = xq_c; a.xs = xs_c; a.xsum = xsum_c; launch_matvec(a, D, st); }
        { PrepArgs p{}; p.x = gu; p.x2 = gu + FF; p.signs = signs17408; p.n = FF; p.flags = PREP_SILU_MUL | PREP_SIGN | PREP_HADAMARD | PREP_QUANT;
          p.xq = xq; p.xs = xs; p.xsum = xsum; launch_prep(p, st); }
        { MvArgs a{}; a.nseg = 1; a.seg[0] = { L.down, x, D / 32, 1 }; a.xq = xq_c; a.xs = xs_c; a.xsum = xsum_c; launch_matvec(a, FF, st); }
        DUMP("l_out", x, D); DUMP("cs0", conv_state, 3 * CONV_CH);
    }
    if (with_logits) {
        { PrepArgs p{}; p.x = x; p.norm_w = output_norm; p.eps = NORM_EPS; p.signs = signs5120; p.n = D;
          p.flags = PREP_NORM | PREP_SIGN | PREP_HADAMARD | PREP_QUANT; p.xq = xq; p.xs = xs; p.xsum = xsum; launch_prep(p, st); }
        { MvArgs a{}; a.nseg = 1; a.seg[0] = { output, logits, VOCAB / 32, 0 }; a.xq = xq_c; a.xs = xs_c; a.xsum = xsum_c; launch_matvec(a, D, st); }
        launch_argmax(logits, VOCAB, tok_out, nullptr, st);
        if (!dump_dir.empty()) dump("result_output", logits, VOCAB);
    }
    launch_inc_int(pos, st);
}

void Engine::build_forward(bool with_logits) {
    hipGraph_t graph;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    forward_token(with_logits);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    (with_logits ? graph_gen : graph_prompt) = exec;
}

int Engine::step(int token, bool want_logits) {
    if (n_pos >= context) throw std::runtime_error("context full");
    // the ring must not wrap onto a slot a queued step has not consumed yet
    if (pending >= TOK_RING / 2) { HIP_CHECK(hipStreamSynchronize(stream)); pending = 0; }
    h_ring[n_pos & (TOK_RING - 1)] = token;
    if (fused) {
        std::vector<Seq *> sv { &seq0 };
        std::vector<std::vector<int>> tv { { token } };
        std::vector<int> am;
        forward(sv, tv, want_logits, want_logits ? &am : nullptr);
        n_pos++;
        return want_logits ? am[0] : -1;
    } else if (use_graph) {
        hipGraphExec_t & ex = want_logits ? graph_gen : graph_prompt;
        if (!ex) build_forward(want_logits);
        HIP_CHECK(hipGraphLaunch(ex, stream));
    } else {
        forward_token(want_logits);
    }
    n_pos++; pending++;
    if (!want_logits) return -1;
    pending = 0;
    int out;
    HIP_CHECK(hipMemcpyAsync(&out, tok_out, 4, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return out;
}

void Engine::load_mtp(const std::string & path, int nthreads) {
    mtp.st.open(path);
    const std::string pre = "mtp.layers.0.";
    std::vector<std::string> names = { "mtp.fc.weight", pre + "self_attn.q_proj.weight", pre + "self_attn.k_proj.weight", pre + "self_attn.v_proj.weight", pre + "self_attn.o_proj.weight",
                                       pre + "mlp.gate_proj.weight", pre + "mlp.up_proj.weight", pre + "mlp.down_proj.weight" };
    mtp.cache.open_or_build(mtp.st, path, names, nthreads);
    auto q8 = [&](const std::string & n, int64_t N, int64_t K) { const Q8Entry & e = mtp.cache.entry(n); if (e.N != N || e.K != K) throw std::runtime_error("mtp shape mismatch: " + n); return upload<uint8_t>(mtp.cache.data(e), e.bytes); };
    // Qwen3-Next style norms store weight - 1
    auto f32 = [&](const std::string & n) { auto v = mtp.st.f32(n); for (auto & x : v) x += 1.0f; return upload<float>(v.data(), v.size() * 4); };
    mtp.w.fc = q8("mtp.fc.weight", D, 2 * D);
    mtp.w.q = q8(pre + "self_attn.q_proj.weight", Q_OUT, D); mtp.w.k = q8(pre + "self_attn.k_proj.weight", KV_OUT, D); mtp.w.v = q8(pre + "self_attn.v_proj.weight", KV_OUT, D);
    mtp.w.o = q8(pre + "self_attn.o_proj.weight", D, ATTN_OUT);
    mtp.w.gate = q8(pre + "mlp.gate_proj.weight", FF, D); mtp.w.up = q8(pre + "mlp.up_proj.weight", FF, D); mtp.w.down = q8(pre + "mlp.down_proj.weight", D, FF);
    mtp.w.in_norm = f32(pre + "input_layernorm.weight"); mtp.w.post_norm = f32(pre + "post_attention_layernorm.weight");
    mtp.w.q_norm = f32(pre + "self_attn.q_norm.weight"); mtp.w.k_norm = f32(pre + "self_attn.k_norm.weight");
    mtp.w.enorm = f32("mtp.pre_fc_norm_embedding.weight"); mtp.w.hnorm = f32("mtp.pre_fc_norm_hidden.weight");
    mtp.w.out_norm = f32("mtp.norm.weight");
    { auto v = mtp.st.f32("mtp.norm.weight"); for (size_t i = 0; i < v.size(); i++) v[i] = (v[i] + 1.0f) * h_signs5120[i]; mtp.w.out_norm_s = upload<float>(v.data(), v.size() * 4); }
    mtp.h_out = (float *) dmalloc((size_t) RMAX * D * 4);
    mtp.loaded = true;
    fprintf(stderr, "mtp head loaded (%.2f GB on device total)\n", device_bytes / 1e9);
}

void Engine::mtp_forward(Seq & s, const float * h_in, const std::vector<int> & toks, const std::vector<int> & poss, std::vector<int> & am) {
    MtpParams M{};
    M.P = fwd; M.w = mtp.w; M.h_in = h_in; M.h_out = mtp.h_out;
    const int n = (int) toks.size();
    if (n > RMAX) throw std::runtime_error("mtp: too many rows");
    M.P.nseq = 1; M.P.seqs[0] = { 0, n, 0, 0 }; M.P.attn_key_min = 1;
    for (int i = 0; i < n; i++) M.P.rows[i] = { toks[i], s.slot, poss[i], 0 };
    M.P.nrows = n; M.P.with_logits = 1; M.P.hcap = nullptr; M.P.hfinal = nullptr;
    M.P.prof = nullptr; M.P.debug_layers = 0; M.P.debug_stop = getenv("HALO_MTP_STOP") ? atoi(getenv("HALO_MTP_STOP")) : 0;
    if (getenv("HALO_MTP_DUMP")) { HIP_CHECK(hipMemsetAsync(bar, 0, 4, stream)); HIP_CHECK(hipMemsetAsync(work, 0, 4096 * 4, stream)); launch_mtp(M, stream); HIP_CHECK(hipStreamSynchronize(stream));
        std::string dir = getenv("HALO_MTP_DUMP");
        auto wr = [&](const char * nm, const void * d, size_t bytes) { std::vector<char> h(bytes); HIP_CHECK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost)); FILE * f = fopen((dir + "/" + nm + ".bin").c_str(), "wb"); fwrite(h.data(), 1, bytes, f); fclose(f); };
        wr("hin", h_in, (size_t) n * D * 4); wr("tmp", tmp, (size_t) RMAX * D * 4); wr("x", x, (size_t) RMAX * D * 4); wr("xq", xq, (size_t) RMAX * FF); wr("xs", xs, (size_t) RMAX * NB_FF * 4); wr("xsum", xsum, (size_t) RMAX * NB_FF * 4);
        wr("hout", mtp.h_out, (size_t) RMAX * D * 4); wr("qfull", qfull, (size_t) RMAX * Q_OUT * 4); wr("kbuf", kbuf, (size_t) RMAX * KV_OUT * 4);
        std::vector<int> a2(n); HIP_CHECK(hipMemcpy(a2.data(), argmax_out, n * 4, hipMemcpyDeviceToHost)); fprintf(stderr, "mtp dump: rows %d argmax0 %d\n", n, a2[0]); exit(0); }
    HIP_CHECK(hipMemsetAsync(bar, 0, 4, stream));
    HIP_CHECK(hipMemsetAsync(work, 0, 4096 * 4, stream));
    launch_mtp(M, stream);
    am.resize(n);
    HIP_CHECK(hipMemcpyAsync(am.data(), argmax_out, n * 4, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

void Engine::load_dflash(const std::string & path, int nthreads, unsigned want) {
    if (!(want & (DRAFT_Q8 | DRAFT_Q4))) throw std::runtime_error("load_dflash: no drafter weight coordinate requested");
    // The drafter is the only windowed attention in this engine, so its loader is where the unit
    // schedule's environment override belongs. `set_attn_deal` overrides it, so one process can
    // still interleave the arms. docs/drafter-window-units.md.
    if (const char * d = getenv("HALO_ATTN_DEAL")) fwd.attn_window_deal = atoi(d) ? 1 : 0;
    df.st.open(path);
    std::vector<std::string> names = { "fc.weight", "candidate_selector.hidden_projection.weight" };
    for (int l = 0; l < DF_LAYERS; l++) {
        const std::string pre = "layers." + std::to_string(l) + ".";
        for (const char * n : { "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight" })
            names.push_back(pre + n);
    }
    auto f32 = [&](const std::string & n) { auto v = df.st.f32(n); return upload<float>(v.data(), v.size() * 4); };
    auto bf16 = [&](const std::string & n) { const StTensor & t = df.st.t(n); if (t.dtype != "BF16") throw std::runtime_error("dflash: expected bf16 " + n); return upload<unsigned short>(df.st.data(t), t.end - t.begin); };
    // Everything that is not a weight tile is shared by both coordinates and uploaded once.
    df.w.hidden_norm = f32("hidden_norm.weight"); df.w.norm = f32("norm.weight");
    { auto v = df.st.f32("norm.weight"); for (size_t i = 0; i < v.size(); i++) v[i] *= h_signs5120[i]; df.w.norm_s = upload<float>(v.data(), v.size() * 4); }
    df.w.pred_cb = bf16("candidate_selector.predecessor_codebook"); df.w.succ_cb = bf16("candidate_selector.successor_codebook");
    for (int l = 0; l < DF_LAYERS; l++) {
        const std::string pre = "layers." + std::to_string(l) + ".";
        DflashLayerW & L = df.w.layer[l];
        L.in_norm = f32(pre + "input_layernorm.weight"); L.post_norm = f32(pre + "post_attention_layernorm.weight");
        L.q_norm = f32(pre + "self_attn.q_norm.weight"); L.k_norm = f32(pre + "self_attn.k_norm.weight");
        L.abase = f32(pre + "attention_conv.base_kernel"); L.mbase = f32(pre + "mlp_conv.base_kernel");
    }
    df.w4 = df.w;
    // One weight-tile image per requested coordinate, out of its own cache file.
    for (int bits : { 8, 4 }) {
        if (!(want & (bits == 8 ? DRAFT_Q8 : DRAFT_Q4))) continue;
        Q8Cache & cache = bits == 8 ? df.cache : df.cache4;
        DflashW & w = bits == 8 ? df.w : df.w4;
        cache.open_or_build(df.st, path, names, nthreads, bits);
        auto tiles = [&](const std::string & n, int64_t N, int64_t K) { const Q8Entry & e = cache.entry(n); if (e.N != N || e.K != K) throw std::runtime_error("dflash shape mismatch: " + n); return upload<uint8_t>(cache.data(e), e.bytes); };
        w.fc = tiles("fc.weight", D, NCAP * D);
        w.hproj = tiles("candidate_selector.hidden_projection.weight", DF_RANK, D);
        for (int l = 0; l < DF_LAYERS; l++) {
            const std::string pre = "layers." + std::to_string(l) + ".";
            DflashLayerW & L = w.layer[l];
            L.q = tiles(pre + "self_attn.q_proj.weight", DF_Q, D); L.k = tiles(pre + "self_attn.k_proj.weight", DF_KV, D); L.v = tiles(pre + "self_attn.v_proj.weight", DF_KV, D);
            L.o = tiles(pre + "self_attn.o_proj.weight", D, DF_Q);
            L.gate = tiles(pre + "mlp.gate_proj.weight", FF, D); L.up = tiles(pre + "mlp.up_proj.weight", FF, D); L.down = tiles(pre + "mlp.down_proj.weight", D, FF);
            L.akp = tiles(pre + "attention_conv.kernel_projection.weight", DF_KPROJ, D); L.mkp = tiles(pre + "mlp_conv.kernel_projection.weight", DF_KPROJ, D);
        }
        if (bits == 4) df.q4_loaded = true;
    }
    df.use_q4 = !(want & DRAFT_Q8);
    df.th = (float *) dmalloc((size_t) RMAX * D * 4); df.xn = (float *) dmalloc((size_t) RMAX * D * 4); df.dyn = (float *) dmalloc((size_t) RMAX * DF_KPROJ * 4);
    df.cbuf = (float *) dmalloc((size_t) RMAX * D * 4); df.abuf = (float *) dmalloc((size_t) RMAX * DF_Q * 4); df.tmpo = (float *) dmalloc((size_t) RMAX * D * 4);
    df.hproj_out = (float *) dmalloc((size_t) RMAX * DF_RANK * 4);
    df.cand_v = (float *) dmalloc((size_t) RMAX * DF_TOPCAP * 4); df.cand_i = (int *) dmalloc((size_t) RMAX * DF_TOPCAP * 4); df.cand_n = (int *) dmalloc(RMAX * 4); df.thr = (float *) dmalloc(RMAX * 4);
    df.topv = (float *) dmalloc(RMAX * DF_TOPK * 4); df.topi = (int *) dmalloc(RMAX * DF_TOPK * 4); df.draft_out = (int *) dmalloc(DF_BLOCK * 4);
    df.loaded = true;
    fprintf(stderr, "dflash2 drafter loaded, weights %s (%.2f GB on device total)\n",
            (want & DRAFT_Q8) && (want & DRAFT_Q4) ? "q8+q4" : df.use_q4 ? "q4" : "q8", device_bytes / 1e9);
}

void Engine::dflash_step(Seq & s, int n_ctx, int ctx_pos0, int anchor, int start, std::vector<int> * drafts, int cap_row0) {
    DflashParams M{};
    M.P = fwd; M.w = df.use_q4 ? df.w4 : df.w; M.q4 = df.use_q4 ? 1 : 0;
    M.n_ctx = n_ctx;
    for (int i = 0; i < n_ctx; i++) M.ctx_rows[i] = { 0, s.slot, ctx_pos0 + i, 0 };
    M.hcap = hcap + (size_t) cap_row0 * NCAP * D;
    M.th = df.th; M.xn = df.xn; M.dyn = df.dyn; M.cbuf = df.cbuf; M.abuf = df.abuf; M.tmpo = df.tmpo; M.hproj_out = df.hproj_out;
    M.cand_v = df.cand_v; M.cand_i = df.cand_i; M.cand_n = df.cand_n; M.thr = df.thr; M.topv = df.topv; M.topi = df.topi; M.draft_out = df.draft_out;
    M.anchor = anchor; M.do_block = drafts ? 1 : 0;
    if (drafts && start + DF_BLOCK > context) throw std::runtime_error("context full");
    M.P.nseq = 1; M.P.seqs[0] = { 0, DF_BLOCK, 0, 0 }; M.P.nrows = DF_BLOCK;
    for (int i = 0; i < DF_BLOCK; i++) M.P.rows[i] = { i == 0 ? anchor : 248070, s.slot, start + i, 0 };
    M.P.attn_key_min = 0; M.P.attn_noncausal = 1; M.P.attn_window = DF_WINDOW;
    // `attn_window_deal` rides in on the copy of `fwd` above: the window decides the unit list as
    // well as the mask, and `set_attn_deal(0)` / HALO_ATTN_DEAL=0 restores the schedule that dealt
    // every chunk from position 0 and masked it to zero. That control is an arm of one process,
    // not a second binary, because this box moves 10% between panels and 0.6% inside one.
    // docs/drafter-window-units.md.
    M.P.with_logits = 1; M.P.hcap = nullptr; M.P.hfinal = nullptr;
    // The drafter's own segment timeline. `prof` is only allocated under HALO_PROFILE, and the
    // kernel stamps nothing when the pointer is null, so the deployed drafted step is untouched.
    static const bool df_prof = getenv("HALO_PROFILE_DRAFT") && atoi(getenv("HALO_PROFILE_DRAFT"));
    M.P.prof = (prof && df_prof && drafts) ? prof : nullptr;
    M.P.debug_layers = getenv("HALO_DF_LAYERS") ? atoi(getenv("HALO_DF_LAYERS")) : 0; M.P.debug_stop = getenv("HALO_DF_STOP") ? atoi(getenv("HALO_DF_STOP")) : 0;
    HIP_CHECK(hipMemsetAsync(bar, 0, 4, stream));
    HIP_CHECK(hipMemsetAsync(work, 0, 4096 * 4, stream));
    launch_dflash(M, stream);
    if (drafts && getenv("HALO_DF_DUMP")) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::string dir = getenv("HALO_DF_DUMP");
        auto wr = [&](const char * nm, const void * d, size_t bytes) { std::vector<char> h(bytes); HIP_CHECK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost)); FILE * f = fopen((dir + "/" + nm + ".bin").c_str(), "wb"); fwrite(h.data(), 1, bytes, f); fclose(f); };
        wr("hcap", hcap + (size_t) cap_row0 * NCAP * D, (size_t) RMAX * NCAP * D * 4); wr("th", df.th, (size_t) RMAX * D * 4); wr("x", x, (size_t) RMAX * D * 4); wr("tmp", tmp, (size_t) RMAX * D * 4);
        wr("hproj", df.hproj_out, (size_t) RMAX * DF_RANK * 4); wr("topv", df.topv, RMAX * DF_TOPK * 4); wr("topi", df.topi, RMAX * DF_TOPK * 4); wr("draft", df.draft_out, DF_BLOCK * 4);
        wr("tmpo", df.tmpo, (size_t) RMAX * D * 4); wr("kc0", kcache + (size_t) (s.slot * KV_SLOTS + NATTN + 1) * ((size_t) NKV * context * HD), (size_t) NKV * context * HD * 2); wr("vc0", vcache + (size_t) (s.slot * KV_SLOTS + NATTN + 1) * ((size_t) NKV * context * HD), (size_t) NKV * context * HD * 2);
        wr("partials", partials, sizeof(AttnPartial) * RMAX * DF_NH * AMAX_CHUNKS); wr("xq", xq, (size_t) RMAX * NCAP * D); wr("xs", xs, (size_t) RMAX * (NCAP * D / BLOCK) * 4); wr("qrot", qrot, (size_t) RMAX * ATTN_OUT * 4); wr("kbuf", kbuf, (size_t) RMAX * KV_OUT * 4); wr("vbuf", vbuf, (size_t) RMAX * KV_OUT * 4);
        wr("logits", logits, (size_t) RMAX * VOCAB * 4); wr("xn", df.xn, (size_t) RMAX * D * 4); wr("dyn", df.dyn, (size_t) RMAX * DF_KPROJ * 4); wr("cbuf", df.cbuf, (size_t) RMAX * D * 4);
        FILE * f = fopen((dir + "/meta.txt").c_str(), "w"); fprintf(f, "%d %d %d %d\n", n_ctx, ctx_pos0, anchor, start); fclose(f);
        fprintf(stderr, "dflash dump written\n"); exit(0);
    }
    if (drafts) {
        drafts->resize(DF_BLOCK - 1);
        HIP_CHECK(hipMemcpyAsync(drafts->data(), df.draft_out, (DF_BLOCK - 1) * 4, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        if (M.P.prof) {
            std::vector<unsigned long long> t(1024);
            HIP_CHECK(hipMemcpy(t.data(), prof, 1024 * 8, hipMemcpyDeviceToHost));
            const std::vector<std::string> names = dflash_segment_names(n_ctx);
            for (size_t i = 0; i < names.size(); i++) {
                if (t[i + 1] == 0 || t[i] == 0 || t[i + 1] < t[i]) break;
                const double us = (double) (t[i + 1] - t[i]) / 100.0;  // s_memrealtime ticks at 100 MHz
                df_prof_us[names[i]] += us; df_prof_cnt[names[i]]++; df_prof_total_us += us;
            }
            df_prof_steps++;
        }
    }
}

void Engine::dflash_ingest(const std::vector<Seq *> & seqs) {
    if (seqs.empty()) return;
    if ((int) seqs.size() > RMAX) throw std::runtime_error("dflash_ingest takes at most RMAX rows");
    DflashParams M{};
    M.P = fwd; M.w = df.use_q4 ? df.w4 : df.w; M.q4 = df.use_q4 ? 1 : 0;
    M.n_ctx = (int) seqs.size();
    for (size_t i = 0; i < seqs.size(); i++) M.ctx_rows[i] = { 0, seqs[i]->slot, seqs[i]->len - 1, 0 };
    M.hcap = hcap;
    M.th = df.th; M.xn = df.xn; M.dyn = df.dyn; M.cbuf = df.cbuf; M.abuf = df.abuf; M.tmpo = df.tmpo; M.hproj_out = df.hproj_out;
    M.cand_v = df.cand_v; M.cand_i = df.cand_i; M.cand_n = df.cand_n; M.thr = df.thr; M.topv = df.topv; M.topi = df.topi; M.draft_out = df.draft_out;
    M.anchor = 0; M.do_block = 0;
    M.P.with_logits = 1; M.P.hcap = nullptr; M.P.hfinal = nullptr; M.P.prof = nullptr;
    M.P.debug_layers = 0; M.P.debug_stop = 0;
    HIP_CHECK(hipMemsetAsync(bar, 0, 4, stream));
    HIP_CHECK(hipMemsetAsync(work, 0, 4096 * 4, stream));
    launch_dflash(M, stream);
}

// The order `k_dflash`'s grid syncs close its phases. Kept beside the kernel's shape rather than
// inside it: a segment is named by what produced the values the sync publishes.
std::vector<std::string> Engine::dflash_segment_names(int n_ctx) {
    std::vector<std::string> n;
    if (n_ctx > 0) {
        n.insert(n.end(), { "ingest-prep", "ingest-mv-fc", "ingest-norm" });
        for (int l = 0; l < DF_LAYERS; l++) n.insert(n.end(), { "ingest-mv-kv", "ingest-attn-pre" });
    }
    n.insert(n.end(), { "embed", "prep-embed", "keep-embed" });
    const int nl = getenv("HALO_DF_LAYERS") && atoi(getenv("HALO_DF_LAYERS")) > 0 ? atoi(getenv("HALO_DF_LAYERS")) : DF_LAYERS;
    for (int l = 0; l < nl; l++)
        n.insert(n.end(), { "prep-in-norm", "mv-akp", "dconv-a", "prep-conv-a", "mv-qkv", "attn-pre", "attn",
                            "prep-attn-combine", "mv-o", "dconv-a-out", "prep-post-norm", "mv-mkp", "dconv-m",
                            "prep-conv-m", "mv-gate-up", "prep-silu", "mv-down", "dconv-m-out" });
    n.insert(n.end(), { "final-norm", "mv-hproj", "final-norm-s", "mv-lm-head",
                        "topk-slice-max", "topk-threshold", "topk-collect", "topk-select", "selector" });
    return n;
}

void Engine::print_draft_profile() {
    if (df_prof_steps == 0) { fprintf(stderr, "draft profile: set HALO_PROFILE=1 HALO_PROFILE_DRAFT=1 and run a drafted generation\n"); return; }
    const double per_step = df_prof_total_us / (double) df_prof_steps / 1000.0;
    fprintf(stderr, "drafter phase breakdown over %ld steps: %.3f ms per step inside k_dflash\n", df_prof_steps, per_step);
    std::vector<std::pair<std::string, double>> v(df_prof_us.begin(), df_prof_us.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second > b.second; });
    for (auto & [k, us] : v)
        fprintf(stderr, "  %-18s %8.3f ms/step  %5.1f%%  x%ld per step  avg %.1f us\n", k.c_str(),
                us / (double) df_prof_steps / 1000.0, 100 * us / df_prof_total_us,
                df_prof_cnt[k] / std::max(1L, df_prof_steps), us / (double) df_prof_cnt[k]);
}

// ---- snapshot of one sequence slot's state (GDN state, conv ring, pending block cache) into the last slot
static size_t slot_gdn_bytes() { return (size_t) 48 * gdn_region_floats() * 4; }
static size_t slot_ring_bytes() { return (size_t) 48 * GDN_RING_FLOATS * 4; }
static size_t slot_blk_bytes() { return (size_t) 2 * 48 * RMAX * BLK_TOKEN_FLOATS * 4; }

static void copy_slot_state(Engine & e, int dst, int src) {
    // The whole region moves, so the pending updates in its upper half move with it.
    gdn_defer_note_copy(dst, src);
    HIP_CHECK(hipMemcpyAsync(e.gdn_state + (size_t) dst * 48 * gdn_region_floats(), e.gdn_state + (size_t) src * 48 * gdn_region_floats(), slot_gdn_bytes(), hipMemcpyDeviceToDevice, e.stream));
    HIP_CHECK(hipMemcpyAsync(e.conv_ring + (size_t) dst * 48 * GDN_RING_FLOATS, e.conv_ring + (size_t) src * 48 * GDN_RING_FLOATS, slot_ring_bytes(), hipMemcpyDeviceToDevice, e.stream));
    HIP_CHECK(hipMemcpyAsync(e.blk_cache + (size_t) dst * 2 * 48 * RMAX * BLK_TOKEN_FLOATS, e.blk_cache + (size_t) src * 2 * 48 * RMAX * BLK_TOKEN_FLOATS, slot_blk_bytes(), hipMemcpyDeviceToDevice, e.stream));
}

// K/V of one sequence slot for positions [0, n): cache slots [0, NATTN] hold NKV heads of HD with a
// head pitch of context*HD; the drafter slots hold DF_NKV heads of DF_HD with a pitch of
// context*DF_HD. Both groups cover NKV*HD halves per position, so a snapshot buffer packs every
// cache slot as KV_SLOTS blocks of n*NKV*HD halves with the same two head shapes at pitch n*hd.
static_assert(NKV * HD == DF_NKV * DF_HD, "target and drafter cache slots must have equal width");
static constexpr size_t SNAP_KV_HALVES_PER_TOKEN = (size_t) KV_SLOTS * NKV * HD;

// The target K cache interleaves the leaves of 32 consecutive keys, so a prefix of `n` tokens is a
// whole number of tiles; rounding the copy up keeps it one contiguous run per head, and every
// buffer on both sides is sized in tiles for the same reason.
static void copy_slot_kv(Engine & e, __half * dst, size_t dst_tokens, const __half * src, size_t src_tokens, size_t n_tokens) {
    const size_t n = k_tiled_tokens(n_tokens);
    const size_t slot_halves = (size_t) NKV * HD;
    auto group = [&](int first_slot, int count, int heads, int hd) {
        HIP_CHECK(hipMemcpy2DAsync(dst + (size_t) first_slot * slot_halves * dst_tokens, dst_tokens * hd * 2,
                                   src + (size_t) first_slot * slot_halves * src_tokens, src_tokens * hd * 2,
                                   n * hd * 2, (size_t) count * heads, hipMemcpyDeviceToDevice, e.stream));
    };
    group(0, NATTN + 1, NKV, HD);
    group(NATTN + 1, KV_SLOTS - (NATTN + 1), DF_NKV, DF_HD);
}

static void snapshot_free_kv(Engine & e, Engine::Snapshot & sn) {
    if (sn.k) HIP_CHECK(hipFree(sn.k));
    if (sn.v) HIP_CHECK(hipFree(sn.v));
    e.snap_kv_used -= sn.kv_tokens * SNAP_KV_HALVES_PER_TOKEN * 2 * 2;
    sn.k = sn.v = nullptr; sn.kv_tokens = 0; sn.valid = false;
}

void Engine::snapshot_save(const Seq & s, const std::vector<int> & tokens) {
    if (!snap_kv_budget) return;
    for (int i = 0; i < SNAP_AREAS; i++) if (snaps[i].valid && snaps[i].tokens == tokens) { snaps[i].last_use = ++snap_clock; return; }
    const size_t n = k_tiled_tokens(tokens.size());
    const size_t kv_bytes = n * SNAP_KV_HALVES_PER_TOKEN * 2 * 2;
    if (kv_bytes > snap_kv_budget) return; // a prefix this long is recomputed instead of cached
    auto lru = [&](bool only_valid) {
        int area = -1;
        for (int i = 0; i < SNAP_AREAS; i++) {
            if (only_valid && !snaps[i].valid) continue;
            if (area < 0 || (!snaps[i].valid && snaps[area].valid) || (snaps[i].valid == snaps[area].valid && snaps[i].last_use < snaps[area].last_use)) area = i;
        }
        return area;
    };
    int area = lru(false);
    // reuse the area's buffer when it is large enough, otherwise release it and any other least
    // recently used snapshot until the budget holds the new one
    if (snaps[area].kv_tokens < n) {
        snapshot_free_kv(*this, snaps[area]);
        while (snap_kv_used + kv_bytes > snap_kv_budget) { const int victim = lru(true); if (victim < 0) break; snapshot_free_kv(*this, snaps[victim]); }
        snaps[area].k = (__half *) dmalloc(kv_bytes / 2);
        snaps[area].v = (__half *) dmalloc(kv_bytes / 2);
        snaps[area].kv_tokens = n;
        snap_kv_used += kv_bytes;
    }
    copy_slot_state(*this, nslots + area, s.slot);
    const size_t slot_halves = (size_t) KV_SLOTS * NKV * context * HD;
    copy_slot_kv(*this, snaps[area].k, snaps[area].kv_tokens, kcache + (size_t) s.slot * slot_halves, (size_t) context, n);
    copy_slot_kv(*this, snaps[area].v, snaps[area].kv_tokens, vcache + (size_t) s.slot * slot_halves, (size_t) context, n);
    snaps[area].tokens = tokens; snaps[area].seq = s; snaps[area].valid = true; snaps[area].last_use = ++snap_clock;
}

void Engine::snapshot_restore(Seq & s, int area) {
    copy_slot_state(*this, s.slot, nslots + area);
    const Snapshot & sn = snaps[area];
    const size_t slot_halves = (size_t) KV_SLOTS * NKV * context * HD;
    copy_slot_kv(*this, kcache + (size_t) s.slot * slot_halves, (size_t) context, sn.k, sn.kv_tokens, sn.tokens.size());
    copy_slot_kv(*this, vcache + (size_t) s.slot * slot_halves, (size_t) context, sn.v, sn.kv_tokens, sn.tokens.size());
    const int slot = s.slot; s = sn.seq; s.slot = slot;
    snaps[area].last_use = ++snap_clock;
}

int Engine::snapshot_find(const std::vector<int> & prompt) {
    int best = -1;
    for (int i = 0; i < SNAP_AREAS; i++) {
        const Snapshot & sn = snaps[i];
        if (!sn.valid || sn.tokens.empty() || sn.tokens.size() >= prompt.size()) continue;
        if (best >= 0 && sn.tokens.size() <= snaps[best].tokens.size()) continue;
        if (std::equal(sn.tokens.begin(), sn.tokens.end(), prompt.begin())) best = i;
    }
    return best;
}

// Prompt tokens [from, end) in 8-row passes; the last pass produces logits (next = argmax). With the
// DFlash2 drafter loaded, every pass's captured features are ingested into its cache, and the last
// pass drafts a block when `drafts` is given.
// A batch below the wide threshold is slower through the batched schedule than through the
// persistent kernel, and the wide sequence routes need their projection module. Either keeps
// prompt ingestion on the eight-row passes.
//
// DFlash2 rides along: `forward_batch` now captures the features it reads for every row of a wide
// pass, and `prefill` hands them to the drafter RMAX rows at a time, so a drafted prompt is
// ingested at the same width as an undrafted one. The MTP head still needs the per-row final
// hidden of a wide pass, which nothing writes yet, so it keeps the narrow route.
int Engine::prefill_width() const {
    if (!prefill_batch_mode || mtp.loaded) return RMAX;
    if (batch_capacity < 32 || !batch_ffn) return RMAX;
    if (prefill_batch_mode >= 17 && !batch_sequence) return RMAX;
    if (df.loaded && batch_capacity > CAPMAX) return RMAX;
    return batch_capacity;
}

void Engine::prefill(Seq & s, const std::vector<int> & toks, size_t from, int & next, std::vector<int> * drafts) {
    std::vector<Seq *> sv { &s };
    std::vector<int> am;
    fwd.hcap = df.loaded ? hcap : nullptr;
    const int n = (int) toks.size();
    const int width = prefill_width();
    for (size_t i = from; i < toks.size(); i += width) {
        const int k = (int) std::min((size_t) width, toks.size() - i);
        std::vector<std::vector<int>> tv { std::vector<int>(toks.begin() + i, toks.begin() + i + k) };
        const bool last = i + k == toks.size();
        if (width > RMAX) {
            const int saved = batch_mode;
            batch_mode = prefill_batch_mode;
            // Ingestion reads one logit row out of this pass, `am.back()` below, so the head is
            // asked for one. At 256 rows the other 255 were 21 ms of projection and 127 MB of
            // VOCAB floats that nothing loaded.
            try { forward_batch(sv, tv, last, last ? &am : nullptr, nullptr, LogitRows::Tail); }
            catch (...) { batch_mode = saved; throw; }
            batch_mode = saved;
            if (last) { next = am.back(); publish_last_logits(k); }
        } else {
            forward(sv, tv, last, last ? &am : nullptr);
            if (last) next = am[k - 1];
        }
        // The drafter ingests RMAX rows per launch (`DflashParams::ctx_rows`), so a wide pass hands
        // it the pass's captured rows in RMAX-row chunks. The block is drafted from the last chunk
        // of the last pass, which is where the anchor token is. At width RMAX this is one chunk and
        // exactly the call the eight-row route always made.
        for (int off = 0; df.loaded && off < k; off += RMAX) {
            const int c = std::min(RMAX, k - off);
            const bool tail = off + c == k;
            dflash_step(s, c, (int) i + off, next, n, last && tail ? drafts : nullptr, off);
        }
    }
    fwd.hcap = nullptr;
}

// A wide pass leaves its logits in the head module, so sampled decode's `get_logits(fwd.nrows - 1)`
// would otherwise read whatever the last persistent-kernel pass left in `logits`. Copy the row the
// caller is about to sample into row 0 of the engine's own buffer and say so. A pass narrow enough
// to skip the wide head already wrote `logits` itself.
void Engine::publish_last_logits(int rows) {
    if (!batch_head || rows <= RMAX || rows > head_batch_capacity(batch_head)) return;
    HIP_CHECK(hipMemcpyAsync(logits, head_logits(batch_head) + (size_t) (rows - 1) * VOCAB,
        (size_t) VOCAB * sizeof(float), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    fwd.nrows = 1;
}

// top-k / top-p / temperature sampling on the host from full logits
int sample_from_logits(const std::vector<float> & logits, float temp, int top_k, float top_p, std::mt19937 & rng) {
    const int n = (int) logits.size();
    if (temp <= 0.0f) return (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    const int k = top_k > 0 ? std::min(top_k, n) : n;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(k);
    std::vector<double> pr(k);
    const double mx = logits[idx[0]];
    double sum = 0;
    for (int i = 0; i < k; i++) { pr[i] = std::exp((logits[idx[i]] - mx) / temp); sum += pr[i]; }
    double cum = 0; int cut = k;
    for (int i = 0; i < k; i++) { cum += pr[i] / sum; if (cum >= top_p) { cut = i + 1; break; } }
    std::discrete_distribution<int> d(pr.begin(), pr.begin() + cut);
    return idx[d(rng)];
}

void Engine::generate(const std::vector<int> & prompt, const GenParams & p, const std::function<bool(int)> & on_token, SpecStats & st) {
    auto tnow = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    Seq & s = seq0;
    st.prompt_tokens = (long) prompt.size();
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if ((int) prompt.size() + p.max_tokens + DF_BLOCK >= context) throw std::runtime_error("prompt and max_tokens exceed the context of " + std::to_string(context));
    const bool spec = p.speculative && df.loaded && p.temp <= 0.0f;
    // prefix cache: resume from the longest snapshot this prompt strictly extends
    size_t from = 0;
    const int area = snapshot_find(prompt);
    if (area >= 0) { snapshot_restore(s, area); from = snaps[area].tokens.size(); }
    else reset_seq(s);
    st.cached_tokens = (long) from;
    const double t0 = tnow();
    int next = -1;
    std::vector<int> drafts;
    // snapshots are taken at pass boundaries before the prompt's last rows, so they never depend on
    // generated tokens: the requested points (the system message) and the prompt itself
    std::vector<size_t> points;
    for (size_t pt : p.snapshot_points) { const size_t at = pt / RMAX * RMAX; if (at > from && at < prompt.size()) points.push_back(at); }
    // a prompt that shares a long prefix with a snapshot but then diverges (another conversation with
    // the same tools and system prompt) gets a snapshot at the divergence, so the next such prompt resumes there
    {
        size_t best_lcp = 0;
        for (const Snapshot & sn : snaps) {
            if (!sn.valid) continue;
            size_t k = 0; const size_t n = std::min(sn.tokens.size(), prompt.size());
            while (k < n && sn.tokens[k] == prompt[k]) k++;
            if (k < sn.tokens.size() && k > best_lcp) best_lcp = k;
        }
        const size_t at = best_lcp / RMAX * RMAX;
        if (best_lcp >= 256 && at > from && at < prompt.size()) points.push_back(at);
    }
    { const size_t at = prompt.size() > RMAX ? (prompt.size() - 1) / RMAX * RMAX : 0; if (at > from) points.push_back(at); }
    std::sort(points.begin(), points.end()); points.erase(std::unique(points.begin(), points.end()), points.end());
    for (size_t at : points) {
        int dummy = -1;
        std::vector<int> partial(prompt.begin(), prompt.begin() + at);
        prefill(s, partial, from, dummy, nullptr);
        snapshot_save(s, partial);
        from = at;
    }
    prefill(s, prompt, from, next, spec ? &drafts : nullptr);
    st.t_prefill = tnow() - t0;
    int produced = 0;
    std::mt19937 rng(p.seed);
    std::vector<float> lg;
    std::vector<Seq *> sv { &s };
    auto sample = [&](int argmax) { if (p.temp <= 0.0f) return argmax; get_logits(lg, fwd.nrows - 1); return sample_from_logits(lg, p.temp, p.top_k, p.top_p, rng); };
    auto stop = [&]() { return produced >= p.max_tokens || (p.aborted && p.aborted()); };
    if (!spec) {
        next = sample(next);
        for (;;) {
            if (!on_token(next)) break;
            produced++;
            if (stop()) break;
            std::vector<std::vector<int>> tv { { next } };
            std::vector<int> am;
            forward(sv, tv, true, &am);
            next = sample(am[0]);
        }
        return;
    }
    if (!on_token(next)) return;
    produced++;
    std::vector<int> am;
    while (!stop()) {
        std::vector<int> block { next };
        block.insert(block.end(), drafts.begin(), drafts.end());
        std::vector<std::vector<int>> tv { block };
        const double tv0 = tnow();
        fwd.hcap = hcap;
        forward(sv, tv, true, &am);
        fwd.hcap = nullptr;
        st.t_verify += tnow() - tv0;
        int a = 0;
        while (a < (int) drafts.size() && am[a] == drafts[a]) a++;
        st.steps++; st.drafted += (long) drafts.size(); st.accepted += a;
        bool go = true;
        for (int i = 0; i < a && go; i++) { go = on_token(drafts[i]); produced++; if (stop()) go = false; }
        if (go) { go = on_token(am[a]); produced++; if (stop()) go = false; }
        const int start_old = s.len - (int) block.size();
        rollback(s, a + 1);
        if (!go) break;
        next = am[a];
        const double td0 = tnow();
        dflash_step(s, a + 1, start_old, next, s.len, &drafts);
        st.t_draft += tnow() - td0;
    }
}

void Engine::print_profile() {
    if (!prof) { fprintf(stderr, "profile: set HALO_PROFILE=1\n"); return; }
    std::vector<unsigned long long> t(1024);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(t.data(), prof, 1024 * 8, hipMemcpyDeviceToHost));
    // phase sequence mirrors k_forward
    std::vector<std::string> names = { "embed", "prep_embed" };
    for (int l = 0; l < NLAYER; l++) {
        if (layers[l].recurrent) { names.insert(names.end(), { "prep_norm", "mv_qkv_z", "gdn_pre", "gdn", "prep_gdn", "mv_ssm_out" }); }
        else { names.insert(names.end(), { "prep_norm", "mv_qkv", "attn_pre", "attn", "prep_attn", "mv_o" }); }
        names.insert(names.end(), { "prep_norm", "mv_gate_up", "prep_silu", "mv_down" });
    }
    names.insert(names.end(), { "prep_norm", "mv_lm_head", "argmax" });
    std::map<std::string, double> sum; std::map<std::string, int> cnt;
    double total = 0;
    for (size_t i = 0; i < names.size(); i++) {
        if (t[i + 1] == 0 || t[i] == 0) break;
        double us = (double) (t[i + 1] - t[i]) / 100.0; // s_memrealtime is 100 MHz
        sum[names[i]] += us; cnt[names[i]]++; total += us;
    }
    fprintf(stderr, "phase breakdown (us): total %.0f\n", total);
    std::vector<std::pair<std::string, double>> v(sum.begin(), sum.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second > b.second; });
    for (auto & [k, us] : v) fprintf(stderr, "  %-12s %8.0f us  %5.1f%%  x%d  avg %.1f us\n", k.c_str(), us, 100 * us / total, cnt[k], us / cnt[k]);
}

void Engine::get_logits(std::vector<float> & out, int row) {
    out.resize(VOCAB);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(out.data(), logits + (size_t) row * VOCAB, (size_t) VOCAB * 4, hipMemcpyDeviceToHost));
}

std::vector<float> Engine::read(const float * dev, size_t n) {
    std::vector<float> out(n);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(out.data(), dev, n * 4, hipMemcpyDeviceToHost));
    return out;
}

} // namespace halo
