# Wide sequence input prep

Every layer of a wide prompt pass starts by normalising the residual, applying the Hadamard,
quantising it into the projection's operand, and projecting the 96 alpha/beta gate scalars. All of
that is independent across token rows, and until now it ran inside the persistent kernel once per
eight-row slice: **1024 cooperative launches per 128-row pass**.

`kernels/prep_batch.hip` runs the same units for the whole batch in two ordinary launches per
layer. The arithmetic is unchanged, so the pass reproduces every logit bit.

## Full-model result

Three rounds per configuration, mode 19 (A4 FFN on the wide committed sequence path), same
executable on both sides, `HALO_WIDE_PREP` flipped between them. Prefill processes a 384-token
document and charges the output head on every row of the final pass.

| workload | per-slice prep | wide prep | gain |
|---|---:|---:|---:|
| A4 prefill, 128 rows/pass | 604.7 | 661.5 | 9.4% |
| A4 prefill, 32 rows/pass | 441.4 | 467.5 | 5.9% |
| A4 generation, 32 streams | 260.6 | 268.7 | 3.1% |

Prefill samples were 604.7/597.9/615.5 against 656.8/661.5/671.1 at 128 rows, and
442.2/441.4/441.1 against 466.5/467.5/469.3 at 32. Decode was 260.3/260.6/261.0 against
268.7/268.6; its third round was cut by the measurement call's timeout and the two recorded
samples differ by 0.1 tok/s.

Reports: [prefill control](../tools/batch-compare/results/wide-prep-a4-prefill-off.md),
[prefill selected](../tools/batch-compare/results/wide-prep-a4-prefill-on.md),
[decode control](../tools/batch-compare/results/wide-prep-a4-decode-off.md),
[decode selected](../tools/batch-compare/results/wide-prep-a4-decode-on.md).

## Where the time went

One traced 128-row A4 pass with logits, from `tools/batch_profile`:

| phase | per-slice prep | wide prep |
|---|---:|---:|
| `sequence-input-prep` launches | 1024 | 64 |
| `sequence-input-prep` | 19.872 ms | 3.574 ms |
| device span | 221.88 ms | 204.69 ms |

Of the 19.872 ms, only 10.47 ms was inside the kernels: the HIP event brackets and the kernels'
own barrier stamps differ by 9.4 ms, or about 9 microseconds of dispatch per cooperative launch.
Splitting the recurrent from the attention layers located the rest. A recurrent slice spent
11.95 microseconds inside the kernel against an attention slice's 5.06, and the difference is the
alpha/beta projection: roughly 5.3 ms of the pass against 5.2 ms for the quantiser itself.

The wide route attacks all three. Recurrent layers now take 66.0 microseconds each and attention
layers 25.4, so the alpha/beta projection is about 1.95 ms and the norm and quantiser together
about 1.63 ms.

## Implementation

`prep_meta` and `prep_chunk` are ordinary launches; neither needs a cooperative grid.

- Blocks `[0, rows)` of `prep_meta` own one token row and write its RMS norm scalar. The deployed
  code recomputed that scalar inside every (row, chunk) unit, five times per row over all 5120
  elements, because each unit needed it and nothing carried it between them.
- The remaining blocks own an (alpha/beta group, token group) pair. The deployed unit was one
  projection row against the eight rows of a slice, so each of the 96 projection rows re-read
  every activation row: 245 MB of reads per layer at 128 rows. A block now holds four token rows
  and the norm weight in registers and walks 16 projection rows against them, which is about 50 MB.
- `prep_chunk` owns one (row, chunk) unit and calls the deployed `prep_chunk_r`, reading the norm
  scalar `prep_meta` produced.

## Why the bits do not move

The alpha/beta accumulation keeps the same per-thread chain,
`fmaf(w0, x.x, fmaf(w1, x.y, fmaf(w2, x.z, fmaf(w3, x.w, acc))))`, over the same five `i` steps in
the same order, with the same `bf16 * norm_w` weights. Only the workgroup that owns a given
(token row, projection row) pair changes, and a dot product's value does not depend on which
workgroup computes it.

`block_sum_256_multi` reduces four of those chains at once. It runs the same `warp_sum` on each,
then sums the same eight wave partials in the same order; it only replaces four pairs of barriers
with one.

The norm scalar keeps the 256-thread strided square-accumulation loop and the same
`block_sum_256` tree, so `rsqrtf(ss / D + eps)` sees the identical `ss`. Computing it once per row
instead of once per chunk cannot change it: the five copies were already bit-identical.

`prep_chunk_r` is called unmodified, including its warp Hadamard, its `warp_max` amax and its
round-to-nearest-even codes into the same wide operand.

## Acceptance

[Full-vocabulary comparison](../tools/batch-compare/results/wide-prep-identity-a4.json):
71,516,160 finite logits at 32, 40, 88 and 128 rows, zero differing bits, no non-finite values,
and 128 continuation tokens from a four-step 32-stream greedy run identical on both sides. Row
counts 40 and 88 exercise partial token and alpha/beta groups.

The traced 128-row pass also reproduces the control's residual FNV-64 over the whole
128 x 5120 hidden state, `7446865760224376151`, after all 64 layers.

## Operation

The route is the default for a wide sequence pass on the direct integer operand, which is every
batched pass of 32 rows or more in modes 17 through 19. Serving never prepares a batch, so its
path, defaults and numerical contract are untouched.

```sh
make -j8 bonsai-halo tools/batch_compare tools/batch_profile
tools/run-batch-compare --tag wide-prep-a4-prefill-on --modes 19 --only prefill \
  --prefill-rows 32,128 --rounds 3
HALO_WIDE_PREP=0 tools/run-batch-compare --tag wide-prep-a4-prefill-off --modes 19 --only prefill \
  --prefill-rows 32,128 --rounds 3
```

`HALO_WIDE_PREP=0` selects the per-slice control. `HALO_SEQUENCE_LAYOUT=staged` also keeps the
deployed path, because the staged control captures the row-major quantiser output that this route
never writes.
