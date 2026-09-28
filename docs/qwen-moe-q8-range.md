# Exact narrow-code opportunity in Qwen's installed Q8 bank

**Measured negative, September 23.** The unprofiled native Q8 phase takes 8.333 ms/token at occupied depth 1024. Can a direct packed *lossless* narrow-code reader materially cut its weight stream without changing the installed Q8_0 arithmetic? `tools/qwen-moe/q8_range.py` examines all 43,909,120 blocks in all 250 nonembedding, nonexpert Q8_0 tensors of the pinned installed GGUF. Each block is two scale bytes and 32 signed code bytes. Its per-tensor payload is SHA-256 hashed; the installed inventory header and acquisition receipt bind these results to the verified full model. It counts a block as signed-w-bit eligible only if **every** code lies in [-2^(w-1), 2^(w-1)-1].

| Width of all 32 codes | Eligible blocks | Fraction of Q8 blocks |
| --- | ---: | ---: |
| 4, 5, 6 or 7 bits | 26,223 | 0.05972% |
| 8 bits | 43,909,120 | 100% |

The 26,223 narrow blocks fit *all four* narrow widths; none fit only 5, 6 or 7. Even giving the image a free per-block choice and perfect random access, narrowing the 26,223 blocks to four-bit codes can remove just **419,568 bytes**, or 0.01598% of the modeled complete 2,626,187,904-byte one-read token stream. With a one-bit/block tag and one 32-bit row offset for the mixed-width stream (the 250 individual Q8 tensors are each below 4 GiB), the explicit four/eight-bit image instead **grows 7,731,472 bytes**: 419,568 saved codes, 5,488,640 tag bytes, 2,662,400 offset bytes. The original two-byte scale is retained verbatim. This is an image-size calculation, **not** a device timing or decoding implementation. It does not change the model's weight values, but repacking and numerical replay would still be required before a native exact-map claim.

A stronger code family permits exceptions to the narrow base width. For w=7, each block keeps 32 low-seven-bit codes and carries the support and one high bit for every code outside [-64,63]. On this bank there are **7.0675 exceptions per block on average**. A literal 32-bit support bitmap plus exception high bits never saves bytes: its 28-byte low-code region plus four-byte bitmap already equals the installed 32-byte code region. Even an *ideal indexed support* with the exception count and width supplied for free, and a `ceil(log2 binomial(32,e))`-bit support rank, saves only **31,780,438.75 bytes** in aggregate, or **1.2101% of complete one-read weights** (2.129% of Q8 bank bytes). This is a generous restricted-family upper bound on savings, before any count/width metadata, support decoding, bit extraction, addresses, activation preparation, occupancy or block alignment. For fixed w=4/5/6 the same free-count bounds are 12.58/17.63/25.48 MB. These are not entropy bounds for other codecs or bounds on a new direct consumer. The exact-domain statement concerns decoded signed codes and retained FP16 scale bits, not bit-identical output after a new GPU reduction schedule.

This rejects **direct whole-block narrow Q8 recoding**, and the straightforward paid narrow-plus-bitmap escape path; it does not reject a learned lossy representation, a nontrivial entropy-coded code stream or a reorganized Q8 matrix consumer. The cheaper independent next question is to measure ordinary Q8 operand staging, register occupancy and unprofiled device time with the same complete-model workload. A source-level 4-bit proposal cannot explain the 8.333 ms phase when 99.9403% of installed blocks require eight bits as a single fixed signed width. This byte ceiling is conditional on one uncached read and must not be called a speedup or added to the overlapping zero-row shortcut.

CPU reproduction from Bonsai root (each tensor shard is independently resumable):

```sh
python3 tools/qwen-moe/q8_range.py --first 0 --count 250
python3 tools/qwen-moe/q8_range.py --summarize
```

[Aggregate receipt and 250 hashed tensor shards](../../data/qwen-moe/q8-range/receipt.json) have receipt SHA-256 `31db7843ef58054a6d9f99665869f6dad183ed4ab35b47a37b821cc4fee11a5b`. The three CPU-run logs are beside that directory at `q8-range-run-{0,1,2}.log`. The acquisition receipt, not this tool, verified full model SHA-256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`. No GPU, model image, executable or resident service changed.
