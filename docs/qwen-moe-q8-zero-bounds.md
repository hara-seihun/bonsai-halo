# Installed Q8 nonexpert zero-weight bound

The [installed UD-Q4_K_M inventory](qwen-moe.md#actual-weight-stream-budget) charges 1,492,910,080 Q8_0 nonembedding bytes per single-token full-output request. The [unprofiled device panel](qwen-moe-q8-unprofiled.md) measures 8.333 ms/token of Q8 kernels at depth 1024; it does not measure DRAM transactions. To decide whether an exact sparse Q8 weight reader merits implementation, `tools/qwen-moe/q8_zero_bounds.py` scans every installed nonembedding Q8_0 payload, without decoding a full float image. A block of 32 int8 codes and one FP16 scale decodes to all-real-zero when the finite scale is signed zero or all 32 codes are zero. This is a weight-map test, not an FP32 scheduling identity.

**Result:** 28,416 of 43,909,120 Q8 blocks are exactly zero (0.06472%); *every one* belongs to one of 444 complete zero output rows. The nonzero blocks occur in 7 attention QKV tensors only: layer 0 7,168, layer 1 13,184, layer 2 7,296, layers 9/10 128/256 and layers 21/22 128/256. The other 243 of the 250 scanned Q8 tensors contain none. Exactly 665,600 output rows were scanned. A zero-block bitmap costs 5,488,640 bytes for at most 966,144 bytes removed, so that representation grows the one-read image by 4,522,496 bytes before any sparse index or dispatch. The more favorable full-row representation needs only 83,200 bitmap bytes, leaving **882,944 net removable bytes per complete one-read step**, or **0.03362%** of the 2,626,187,904-byte model stream. This grants free repacking, bitmap lookup, sparse dispatch, and all row output handling. A variable-length list could reduce the bitmap price further, but even zero metadata caps the prize at 966,144 bytes (0.03679% of that stream). Because each full zero row has 64 Q8 blocks, the two observed zero counts coincide exactly.

The bound is **not** a bound on native time, physical DRAM bytes, or all possible Q8 representations: an algorithm may avoid arithmetic on nonzero weights by a different complete-map construction. It excludes the embedding image (one input-dependent row read, not the 540 MB tensor). Nor does deleting a real-zero term automatically preserve FP32 signed-zero or FMA bits; exact scheduling still requires native acceptance. A sparse installed-Q8 zero-reader is not a plausible response to the measured 2.164 ms/token conditional Q8 gap. Measure ordinary unprofiled Q8 consumer traffic, occupancy, and operand preparation instead of porting a row-skip kernel.

CPU replay (no GPU or resident-service change):

```sh
cd .
python3 tools/qwen-moe/q8_zero_bounds.py
```

The [full per-tensor source/inventory/header/model-hashed receipt](../../data/qwen-moe/q8-zero-bounds/receipt.json) has SHA-256 `178a02c59fe4f104daa5385db4bb312df2e752d99d63c5364d7ff30ac9650637`. The pinned full-model SHA-256 comes from the verified acquisition receipt; this CPU scan checks the local header and hashes each scanned Q8 tensor payload, not all 22 GB again. The decoder assumption is exactly Q8_0's `d*q`; finite scales exclude NaNs. The reader makes no claim about original BF16 zeros or other GGUF quantization families.
