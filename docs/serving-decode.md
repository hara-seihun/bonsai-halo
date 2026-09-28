# Where a drafted generation step goes, and why its rows are not free

Every phase map in this lane is prefill-shaped or batch-shaped. The resident server generates one
stream with the DFlash2 drafter, and nothing had measured that step. This is it, with the two
consequences that follow for anyone trying to make it faster.

## The step

`--bench --dflash`, 33-token prompt, 128 generated tokens, on the canonical build, under
`tools/run-batch-compare --engine`:

```
128 tokens in 2.75 s: 46.55 tok/s; 44 steps, 308 drafted, 84 accepted (27.3%),
2.91 tokens/step; per step: draft 10.8 ms, verify 51.7 ms
```

| | ms | share |
|---|---:|---:|
| verify: one eight-row pass through `k_forward_rows` | 51.7 | 82.7% |
| draft: one `k_dflash` launch, ingest plus a seven-token block | 10.8 | 17.3% |
| step | 62.5 | |

The drafter is not the problem. Five sixths of the step is one eight-row pass of the target model,
and [its phase map](ffn-slice-width.md#what-that-costs-in-the-shape-the-machine-actually-runs) says
79.4% of *that* is matrix work whose cost does not depend on the row count at all.

Acceptance is workload dependent: 2.91 tokens per step on this prose prompt (p ≈ 0.67 per position),
against the 3.6-4.5 the README records for code and reasoning (p ≈ 0.85).

## Two shapes, two different machines

Measured in the same runs: a single-token generation step is **33.19 ms** and an eight-row verify
pass is **52.75 ms**. One row costs 33.19; seven more cost 19.6, i.e. 2.8 ms each. That asymmetry is
the whole economics of speculation here, and it comes from two different kernels:

- One row takes the dot4 body (`ph_matvec`, `TT = 1`), which moves the 5.9 GB weight stream at about
  197 GB/s of the 242 GB/s this box measures. It is close to bandwidth-bound.
- Five to eight rows take the WMMA body (`ph_matvec_w`), which moves the same bytes at about
  141 GB/s while issuing **twice** the matrix instructions it uses.

## Why the free columns do not help generation

`mvw_rows` computes sixteen activation columns and a verify pass has eight rows to put in them, so a
ninth through sixteenth row would ride nearly free — 1.36 ms each instead of 2.8 — if anything could
produce them. Nothing can, and the arithmetic says so before the engineering does.

**A longer chain is worthless.** With per-position acceptance `p`, a chain of `d` drafts returns
`1 + sum_{i=1..d} p^i` tokens. At the measured `p = 0.67`, drafts 8..15 add 0.11 tokens over drafts
1..7; at `p = 0.85` they add 1.58. Meanwhile eight more rows cost 10.9 ms of row-scaled work
(`gdn`, `attn`, the preps), and a second drafter block to fill them costs more. At `p = 0.67` a
15-draft step is a **loss** (3.02 tokens in 75.6 ms, 40 tok/s, against 2.91 in 62.5 ms, 46.6); at
`p = 0.85` it is 84 against 76 tok/s, a 10% gain that only appears on the easiest workloads.

**A two-chain tree is also a loss at these rates.** Branching at position 1 on the drafter's second
candidate lifts the accepted count to `(p1 + p2)(1 + sum_{i=1..6} p^i)`: 2.91 to 3.25 tokens at
`p = 0.67`, 4.77 to 5.13 at `p = 0.85`. The verify pass grows by the same 10.9 ms either way, so
throughput falls from 46.6 to 43.0 and from 76.3 to 67.9 tok/s. Tree verification pays when extra
rows are nearly free; on this engine a row still costs 1.36 ms outside the matrix phases, and that
is more than the marginal candidate is worth.

Both estimates use measured step costs and a two-parameter acceptance model, not a built tree. What
they establish is the size of the prize, and it is negative or small — enough that nobody should
spend a turn on branch-aware GDN state or tree attention masks before the row-scaled 20.6% of the
pass gets cheaper.

## What would move it

In the order their measured size suggests:

- **The 41.9 ms of matvec in the verify pass**, which runs at 141 GB/s and half-empty matrix
  instructions. Neither more rows (there are none) nor fewer instructions per byte is available
  without a different instruction or a different activation width. `wmma_i32_16x16x16_iu4` is twice
  the rate of the iu8 form on this device and is what the A4 modules already use; applying it here
  is a numerical change and would need its own quality evidence, but it is the only lever of this
  size that exists at eight rows.
- **The 10.9 ms of row-scaled work.** `gdn` 4.23 ms replays the accepted prefix every pass, `attn`
  3.56 ms rescans the KV cache per slice, and the preps are 3.1 ms. Cutting these does not just help
  generation; it is also what makes tree verification stop losing.
- ~~**The drafter's 10.8 ms**~~, which was close to its own traffic: about 1.9 GB of Q8 weights plus
  the 278 MB vocabulary image at roughly its bandwidth. [Four-bit drafter weights](drafter-q4.md)
  took it to 7.65 ms and the step to about 57, for +10.4% on drafted generation; the estimate of
  "perhaps 3 ms in it" was low by a third. What is left is 1.23 GB, of which 278 MB is the target's
  ternary LM head run at full vocabulary for seven rows.

Prompt ingestion, the other half of serving, is no longer in this list:
[the sixteen-row FFN slice](ffn-slice-width.md) took it from 151.9 to 413.0 tok/s with the drafter
loaded.

Raw samples: [`batch-comparison/ffn-slice-width/baseline-deployed.txt`](../../../data/bonsai2/batch-comparison/ffn-slice-width/baseline-deployed.txt).
