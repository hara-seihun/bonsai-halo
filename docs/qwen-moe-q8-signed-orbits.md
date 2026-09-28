# No proportional nonzero Q8_0 dots at shared input coordinates

The [exact-block census](qwen-moe-q8-block-reuse.md) ruled out identical nonzero Q8_0 code vectors at the same K block. Could negating or rescaling a previously computed integer dot serve another output row? `tools/qwen-moe/q8_signed_orbits.py` scans all 250 installed nonembedding, nonexpert Q8_0 tensors, comprising 43,909,120 blocks and 1,492,910,080 payload bytes. It compares output rows **within the same tensor and input-block coordinate**, the condition under which they see the same 32 activation codes. Every tensor's payload and receipt are hashed, with an aggregate binding to the inventory, GGUF header, source and tensor receipts. The model's separate acquisition receipt verifies its complete upstream SHA-256.

For a nonzero integer code vector `v`, divide by `gcd(|v_i|)` and choose the sign so its first nonzero element is positive. Two vectors yield proportional real-linear dot maps on unrestricted 32-coordinate inputs **if and only if** these primitive signed vectors are equal: unit-coordinate probes give necessity, and integer primitive vectors have no nontrivial rational scaling. This tests *all* rational/real proportional code directions, not just byte equality or sign reversal. Zero-code blocks are omitted because their dot is identically zero. A secondary sign-only census excludes vectors containing `-128` from reversible sign pairs, since their negation cannot be represented in int8; the primitive census includes them.

**Result:** 43,882,897 blocks have nonzero codes. **Zero** of these has a nonprimitive code vector, and **zero** repeats a same-K primitive signed vector across an output row in its tensor. The largest nonzero orbit has size one. In particular there are zero reusable nonzero integer dots by either sign reversal or any nonzero scalar rescaling in this family. The other 26,223 blocks have all-zero *codes*; this differs from the 28,416 zero-*weight* blocks in the prior scale-aware census because a zero scale can make nonzero codes decode to zero. These populations and their opportunities overlap; they must not be added.

The result closes exact cross-row proportional **Q8 integer-code** dot reuse for the installed complete nonexpert bank, improving the earlier equality-only result. It does not exclude approximate low-rank sharing, reorganized output tiling, shared activation preparation, correlations across distinct K inputs, or reuse on restricted reachable activations. Even if proportional vectors existed, reusing a dot and rescaling its FP32 output would not automatically reproduce the native rounding chain. This negative is a CPU structural result, not a physical-traffic or whole-model TPS measurement. The next useful native question is ordinary Q8 operand staging, occupancy and actual physical traffic; a dot dictionary (including signed or scaled entries) cannot explain the measured 8.333 ms/token Q8 device phase.

CPU reproduction, without reserving the GPU or changing the selected executable:

```sh
python3 tools/qwen-moe/q8_signed_orbits.py --first 0 --count 50
python3 tools/qwen-moe/q8_signed_orbits.py --first 50 --count 50
python3 tools/qwen-moe/q8_signed_orbits.py --first 100 --count 50
python3 tools/qwen-moe/q8_signed_orbits.py --first 150 --count 50
python3 tools/qwen-moe/q8_signed_orbits.py --first 200 --count 50
python3 tools/qwen-moe/q8_signed_orbits.py --summarize
```

[Aggregate and per-tensor source/payload-hashed receipts](../../data/qwen-moe/q8-signed-orbits/receipt.json) have aggregate SHA-256 `95c299a315b01a106919c2bc481d7b2c0fc9c21c8cbf5b69b8ad6b17a7e5e8fa`. The model header and inventory hashes are in that receipt; the GGUF image and resident service were unchanged.
