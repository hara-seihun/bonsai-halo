# Wide prompt ingestion, and what the wide route's gain is actually made of

Every wide-batch result this engine has published came out of `tools/batch_compare`. The engine's
own prompt ingestion — chat, `--bench`, `serve` — never used any of it. `Engine::prefill` fed the
prompt to the persistent kernel eight rows at a time, which is the 155 tok/s in the README's status
table, while the same executable ran 128-row passes at 660 tok/s a few function calls away.

Two things were in the way, and only one of them was real:

- The wide routes change the FFN's numerical map (A8 or A4). That is a serving contract change.
- Nothing else. The wide schedule itself — wide sequence projections, resident GDN, wide input
  prep, wide head — was never separable from that map, because every batch mode that selects the
  wide schedule also forces an A8 or A4 FFN module.

`k_proj_opt` was never the only way to fill a wide pass. `forward_batch` already had a branch that
runs the deployed FFN in eight-row slices inside the batched schedule (`mode == 4`), and the wide
sequence block is guarded independently of it. The combination was reachable and had never been
run. Mode 20, `wide-deployed`, selects it.

## What the 4.64x is made of

One process, mode order reshuffled each round, 384-token document in 128-row passes with the head
on all 128 rows of the last pass. Same executable, same clock, no control needed because the arms
share the process. Source `f1ccef4`, `tools/batch_compare` SHA-256 prefix `7bc8ba8c67a5`.

| mode | prefill tok/s | per round | spread | vs deployed | vs wide-deployed |
|---|---:|---|---:|---:|---:|
| 0 deployed | 156.4 | 156.6, 156.1, 157.4, 154.7 | 1.7% | 1.00x | |
| **20 wide-deployed** | **265.4** | 265.6, 265.2, 265.9, 264.2 | 0.6% | **1.70x** | 1.00x |
| 19 wide-commit-a4 | 726.0 | 725.3, 726.8, 726.7, 715.4 | 1.6% | 4.64x | **2.74x** |

**The wide schedule is worth 1.70x at the deployed FFN's own arithmetic. The A4 FFN map is worth a
further 2.74x on top of it.**

That split is the useful part. Anyone who wanted the wide schedule's gain has been paying for an
approximate FFN to get it, and anyone who rejected the approximate FFN has been giving up a 1.70x
that has nothing to do with it.

An earlier five-round panel on the pre-rebase source (`ce4ba22`, before the decay-gate change)
read 154.6 / 260.2 / 660.1 for the same three modes, a 1.68x and 2.54x split, with 10-13% spreads
on a box that was not holding its clock. Both panels agree on the decomposition; this one is the
one to quote. Raw `exact-wide-prefill5/run.json` is retained beside it.

[Panel](../tools/batch-compare/results/exact-wide-prefill.md), raw
`../../data/bonsai2/batch-comparison/exact-wide-prefill-r2/run.json`.

## How close mode 20 stays to the deployed path

Mode 20 keeps the deployed FFN, and resident GDN, the wide input prep and the wide head each
reproduce their deployed predecessor's bits. What is left is the wide sequence projection, which
changes FP32 reduction order — the same reassociation `docs/wide-sequence.md` records. So mode 20
is not bit-identical, and it is not a precision reduction either: it is the deployed numerical map
summed in a different order.

Against mode 0, same executable, same inputs:

| comparison | result |
|---|---|
| 32 streams x 4 greedy steps | **32/32 streams, 128/128 tokens identical** |
| teacher-forced prefill, 32 rows | top-1 32/32, mean KL 0.0006, max KL 0.0030, top-5 overlap 4.9/5 |
| teacher-forced prefill, 128 rows | top-1 128/128, mean KL 0.0003, max KL 0.0030 |
| teacher-forced decode, 32 rows | top-1 32/32, mean KL 0.0002, max KL 0.0007 |

For scale, the published A4 route (mode 19) matches 25/32 prefill top tokens at mean KL 0.0704, and
the A8 route (mode 18) misses one of 128 continuation tokens. Mode 20's mean KL is roughly 120x
smaller than mode 19's and no measured token moved anywhere.

That is good evidence and it is not a deployment verdict. `batch_compare` says so itself: a
teacher-forced sample of this size can show a mode is wrong, not that it is good enough to be a
default. Mode 20 ships as an explicit route.

[Quality panel](../tools/batch-compare/results/exact-wide-quality.md), raw
`../../data/bonsai2/batch-comparison/exact-wide-quality/run.json`.

## The engine now has the route

`--prefill-ffn` arms wide prompt ingestion for ordinary generation and serving. It changes prompt
ingestion only: the route ends at the last prompt token, and every generated token still comes from
the persistent kernel exactly as before.

```sh
./bonsai-halo -p "$(cat long_prompt.txt)" --raw -n 200 --prefill-ffn wide-deployed
./bonsai-halo -p "..." --prefill-ffn wide-commit-a4      # the A4 map, 3.9x, approximate
./bonsai-halo serve --port 8471 --prefill-ffn wide-deployed
```

Measured on the product path, not the comparison driver: a 2663-token raw prompt (the first 7000
bytes of `PLAN.md`), `-n 16 --bench --context 4096`, each arm a separate process under
`tools/run-batch-compare --engine`. These three arms ran on the pre-rebase source `ce4ba22`, so
read the ratios rather than the absolute rates: the in-process panel above puts the same
wide-deployed ratio at 1.697x on the current build, and the A4 arm gained further from the
decay-gate change that landed between the two.

| route | prompt tok/s | generation tok/s | extra device |
|---|---:|---:|---:|
| default, eight-row passes | 146.6 | 27.70 | |
| `--prefill-ffn wide-deployed` | **249.6** (1.70x) | 27.68 | +2.08 GB |
| `--prefill-ffn wide-commit-a4` | **566.5** (3.86x) | 27.67 | +10.10 GB |

Generation is a measured null in all three arms, which is what the change should do. The 18.2 s
that prompt cost became 10.7 s, or 4.7 s with the A4 map. Generation reads 27.7 tok/s rather than
the README's 34.5 because the sequence is 2663 tokens deep, not because of this change; all three
arms carry the same context.

The A4 arm's +10.10 GB is 8.03 GB of repacked weight images (PAIR, DENSE5, SCALES) plus the shared
workspace and projection modules. Mode 20 reads no weight image at all, so it pays only the
+2.08 GB of workspace, sequence projection and head modules.

## What it deliberately does not do

- **A drafter used to turn it off.** `forward_batch` threw whenever DFlash2 or the MTP head was
  loaded, so `prefill_width()` returned to eight rows and the resident `bonsai-halo.service`, which
  runs with `--dflash`, could not use this route at all. [DFlash2 rides it now](wide-drafter.md):
  a drafted 2663-token prompt ingests at 253.3 tok/s against 144.8, same tokens, same drafts, same
  acceptance. The MTP head still falls back to eight rows, and the drafter's own verify pass is
  still eight rows wide.
- **It does not touch generation.** Single-stream and batched generation take the same code they
  took before, measured null above.
- **It is not a default.** Arming it costs device memory and changes logit bits, and the quality
  sample above is too small to retire the current contract on its own.
- **Below 32 rows it is inert by construction.** `forward_batch` dispatches short passes to the
  deployed or sliced path, so a short prompt runs exactly what it ran before.

## Reproduce

```sh
make -j8 bonsai-halo tools/batch_compare
# the decomposition
tools/run-batch-compare --tag exact-wide-prefill5 --modes 0,20,19 --prefill-rows 128 \
  --rounds 5 --only prefill
# mode 20 against the deployed path
tools/run-batch-compare --tag exact-wide-quality --modes 0,20 --only quality,multistep \
  --quality-rows 32,128 --multistep-streams 32 --multistep-steps 4
# the product path
head -c 7000 PLAN.md > /tmp/p.txt
tools/run-batch-compare --engine -p "$(cat /tmp/p.txt)" --raw -n 16 --bench --context 4096
tools/run-batch-compare --engine -p "$(cat /tmp/p.txt)" --raw -n 16 --bench --context 4096 \
  --prefill-ffn wide-deployed
```

Mode 20 needs no weight image, so `--modes 0,20` prepares the shared workspace alone and a panel
over those two modes costs about 12 s end to end.
