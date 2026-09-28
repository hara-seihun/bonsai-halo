# Qwen3.6 MoE speculative draft

The official Qwen3.6-35B-A3B checkpoint has one Multi-Token Prediction block beyond its 40-layer text trunk. The pinned Unsloth UD-Q4_K_M inference image omits that block. Bonsai keeps every byte of that target image unchanged and loads a separate draft GGUF through llama.cpp's `draft-mtp` path. Speculation must not change the target's greedy answer; the target verifies each drafted token.

## Weights

`tools/qwen-moe/mtp.py` acquires `LibertAIDAI/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF` at revision `5d65e0414ad42cbf5aededa15f509d7c9e90a193`, file `mtp-Qwen3.6-35B-A3B-NVFP4.gguf`. The 3,735,545,952-byte GGUF has SHA-256 `c976a2de32fadc7a5632f2dcbb01563b9ae660bcb1ea42b55188e0ddc1057a17`. The publisher converted it from Qwen's official BF16 checkpoint as a standalone draft, rather than quantizing the MTP block to NVFP4. Its 23 tensors consist of a BF16 embedding and output head, normalization, and the expert, attention and `nextn` weights for layer 40. The draft head has one extra block and advertises `qwen35moe.nextn_predict_layers=1`. `mtp.py` checks its tensor set and all shared model and tokenizer metadata against the target, except draft-independent padding-token and chat-template fields. The different padding-token IDs are 248055 in the target and 248044 in the draft; the token inventory, merges and special inference token IDs match.

The target remains `../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, SHA-256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`. The draft's output head is independent of the target's Q6_K output head, so draft proposals may differ. Agreement here means the verified generated text matches the same target without speculation for the same greedy request. Verify that experimentally. The separate draft's provenance is the publisher's official-base claim and the pinned file hash; matching architecture and tokenizer metadata do not prove that its weights came from the target's exact official revision. If acceptance is poor, compare the MTP tensors against official shards 25 and 26 before attributing it to runtime speed.

```sh
cd .
python3 tools/qwen-moe/mtp.py acquire
python3 tools/qwen-moe/mtp.py verify
python3 tools/qwen-moe/mtp_quantize.py build
```

Acquisition writes `.gguf.partial`, downloads six checked 64 MiB HTTP ranges concurrently, and renames the file only after its full hash matches the pinned LFS object. `verify` writes `mtp/receipt.json`. Failed transfers retain the partial file. A separate transfer worker may write those same ranges and publish the complete, hashed file; do not let both processes download to the same partial file concurrently. The full head and its receipt live under `../../data/qwen-moe/mtp/`, not Git. `mtp_quantize.py build` converts only that standalone head with `llama-quantize Q4_K_M`; it leaves the target alone. The resulting 1,260,898,400-byte file has SHA-256 `6ec218ee63c6c2ad94981cbf7dbcec8f019a7744952602507c5861ad1362c79f`. `mtp_quantize.py verify` checks both source and result and writes `mtp/quantization.json`.

## Measured comparison

The server path uses the newly built native llama.cpp `llama-server`, but the resident Bonsai Halo service stays the default. Reserve the single shared Radeon through `tools/run-batch-compare`. `mtp_benchmark.py run` does that for one panel, stops the test server on exit, and writes the server response, load log, wrapper log and receipt under `mtp/benchmarks/`. The target model's existing `benchmark.py prepare` receipt must identify its current inode, size, modification time and hash. The draft panel also needs `mtp.py verify`. The wrapper admits 30 GiB with 4 GiB left for the host and holds a 48-second runtime bound. Do not overlap GPU panels with another worker. The server uses 2048 context, batch 512, ubatch 256, eight CPU threads, 99 GPU layers, flash attention and one request at a time.

```sh
python3 tools/qwen-moe/benchmark.py prepare
python3 tools/qwen-moe/mtp_benchmark.py run --mode plain --tokens 32
python3 tools/qwen-moe/mtp_benchmark.py run --mode mtp --tokens 32 --draft-n 2 --draft-p 0
python3 tools/qwen-moe/mtp_benchmark.py run --mode mtp --draft-model ../../data/qwen-moe/mtp/mtp-Qwen3.6-35B-A3B-Q4_K_M.gguf --tokens 32 --draft-n 2 --draft-p 0
python3 tools/qwen-moe/mtp_benchmark.py compare ../../data/qwen-moe/mtp/benchmarks/PLAIN-DIRECTORY ../../data/qwen-moe/mtp/benchmarks/MTP-DIRECTORY
```

The response's `timings.predicted_per_second` prices completion throughput, and `draft_n_accepted / draft_n` reports verified acceptance. The compare action checks identical prompts and target hash, prints exact greedy text agreement, the first divergent token ID when present and the measured speed ratio. A higher acceptance percentage alone does not establish a speedup. Keep failed panels and their logs too. These measurements do not replace the separate synthetic decode baseline in [the performance report](qwen-moe-performance.md).

## September 23 result and failure

The 32-token compass response matched byte-for-byte on the integrated native build `de04deb26` with a BF16 draft, 18/25 draft tokens accepted. It ran at 49.33 tokens/s plain and 51.54 tokens/s speculative. A longer 96-token list request did **not** preserve the same greedy output. This is an acceptance failure, not a finished runtime speedup. Plain output repeated identically on a second run. It chose ` Start` at generated token index 71, while BF16-MTP and Q4-MTP chose ` Numbers`; plain log probabilities for the two were -0.761588 and -0.801719. The BF16 list pair ran at 51.98 tokens/s plain and 51.06 tokens/s speculative. The Q4 list pair ran at 49.40 tokens/s plain and 73.90 tokens/s speculative, with 59/70 proposals accepted. The list panels used the same target file, temp 0, seed 1, `n-max=2`, and no draft probability threshold. Q4 cuts the draft file from 3.48 GiB to 1.17 GiB and increases measured proposal throughput, but verified output divergence prevents enabling it as the default.

Using the earlier native `1a07bfa5f` libraries with the same new server launcher did not remove divergence. There, plain chose ` Format` and Q4-MTP chose ` Order` at index 83; their plain log probabilities were -1.325012 and -1.351818. The old-library plain and Q4-MTP rates were 43.82 and 70.38 tokens/s, respectively. This control rules out the new single-token Q8 MMVQ width policy as the sole cause. Both decisions had small top-two logit margins and the draft token's returned probability was 0.0 with no top-logprob list when accepted, so the response alone cannot distinguish a verify-indexing bug from batch-width arithmetic changing a near tie. The [target-only batch-logit replay](qwen-moe-batch-logits.md) now narrows this fork. It generates the exact 96-token plain server continuation, then teacher-forces those IDs in target batches. At generated row 71, a four-token target batch with one initial singleton changes plain ID 4980 to MTP's ID 33565 without any draft model or different input prefix. Width-one replay preserves every full-vocabulary float. This proves batch arithmetic can produce the observed fork, though it does not establish the MTP verifier's actual alignment or exclude an independent indexing error. MTP remains opt-in until a selective plain-map verification and state rollback produce the same full greedy stream and a measured net gain. Keep both server receipts and the target-only replay in custody; do not report speculative throughput as an adopted win before that condition holds.
