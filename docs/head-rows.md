# The head's row selection

The vocabulary head produced `VOCAB` floats for every row of a prompt pass and ingestion read one of
them. `Engine::prefill` takes `am.back()` and `publish_last_logits` copies that row; nothing loads
the rest. The head is the only phase whose entire cost is output width, so at 256 rows that was
21 ms of projection and 127 MB of buffer for one row of useful work.

It now computes the rows a caller says it will read. Prompt ingestion asks for the tail row, a
generation step asks for every row because there every row is a different sequence's next token, and
a request for `all_logits` is serviced as every row whatever it asked for.

## What it costs now

Mode 19, one process, `--head-rows 0,1` walked as a case axis and shuffled with every other case, 3
rounds, medians. Nine phases the selection cannot reach are the control.

| phase | 128 rows, all | 128 rows, tail | 256 rows, all | 256 rows, tail |
|---|---:|---:|---:|---:|
| `embed` | 0.288 | 0.284 | 0.536 | 0.547 |
| `sequence-input-prep` | 3.517 | 3.493 | 5.779 | 5.764 |
| `sequence-input-projection` | 42.177 | 42.467 | 76.043 | 75.877 |
| `gdn-resident-core` | 15.935 | 15.948 | 31.472 | 31.449 |
| `sequence-output-projection` | 20.620 | 20.699 | 32.442 | 32.427 |
| `ffn` | 70.693 | 71.071 | 142.209 | 142.834 |
| `sequence-core` | 15.164 | 15.567 | 31.691 | 31.398 |
| `head-prep` | 0.207 | 0.206 | 0.404 | 0.406 |
| **`head-projection`** | **10.598** | **1.534** | **21.147** | **1.535** |
| between launches | 1.696 | 1.719 | 2.571 | 2.538 |
| **device span** | **181.240** | **173.298** | **344.327** | **324.834** |
| native wall, untraced | 179.639 | 170.153 | 338.811 | 323.504 |

The nine untouched phases sum to 170.297 against 171.454 at 128 rows, a 1.0068 ratio, and to 323.147
against 323.240 at 256 rows, 1.0003. Dividing the spans by those ratios gives **-5.03% at 128 rows
and -5.69% at 256**. Residual FNV-64 is `7446865760224376151` in all twelve 128-row samples and
`18439807992172728808` in all twelve 256-row samples, across both arms and both traced and untraced
cells, so the change reaches nothing but the head.

**A one-row selection costs the same at any pass width**: 1.534 ms at 128 rows and 1.535 at 256,
against a floor of 278 MB of output image at the 242 GB/s roof, 1.15 ms. The head's cost used to be
the one term that grew with the pass, which is what made a wider pass expensive; it is now flat.
`head_tile` at one token group reads the image exactly once and spends about 0.7 ms filling fifteen
matrix columns nothing stores, so the remaining 0.4 ms above the traffic floor is that waste and it
is not worth a second kernel.

## Installed acceptance

Canonical `f982391`, the installed `bonsai-halo`
`8918a909ee88c8258991eb8b5c3c351611df65ad193f3e5cde5443f68cb8d50c`, profiled with
`tools/batch_profile` `e9e94f4f24d96a5d5b9ba5874a1ed069a460c970d1b9f105200307273460e110`, three
rounds, arms shuffled together. Canonical has since taken the long-context attention work, which is
why `sequence-core` is a third of what it was in the panel above; the head's own row is the same
result on a different engine.

| phase | all | tail |
|---|---:|---:|
| `ffn` | 78.323 | 77.844 |
| `sequence-input-projection` | 46.870 | 46.647 |
| `sequence-output-projection` | 22.019 | 22.198 |
| `gdn-resident-core` | 17.421 | 17.223 |
| `sequence-core` | 4.758 | 4.705 |
| `sequence-input-prep` | 4.102 | 4.056 |
| **`head-projection`** | **11.598** | **1.654** |
| device span | 186.954 | 175.978 |

The eight untouched phases sum to 174.010 against 173.198, so the normalised span is **-5.43%**.
Residual FNV-64 is `7446865760224376151` in every sample of both arms, the same value the measured
build produced, and `batch_compare` on the installed build hashes the tail row to
`5109363277453385011` in both arms - the number the measured build produced. Single-stream `--bench`
is 32.85 tok/s on the installed binary and is a null by construction: a 23-token prompt takes the
eight-row route, which has no wide head, and generation never touches this module.

**Generation does not move, and it cannot.** A generation step asks for every row in both arms, so
the two arms run the same code with the same launch geometry; the token-group width is chosen from
the selected rows and 32 selected rows pick the same width the pass did. Measured anyway on the
installed build, 32 streams, both arms in one process: 299.1/298.6 aggregate tok/s against
299.2/297.9, **-0.10%**.

The full-model prompt panel on the installed build was taken while several engineers were compiling
and reads 484 to 731 tok/s across six measurements in both arms. It is recorded in
`batch-comparison/head-rows-installed/` and it measures the host, not the change; the traced table
above is its own control and is what the acceptance rests on.

## The bits

The row ingestion reads is byte-for-byte the same row. `batch_compare` hashes the head's tail row
straight out of its buffer after the timed region, 248,320 floats, and both arms produce FNV-64
`5109363277453385011` on a 384-token document in 128-row passes.

That is what the kernel's structure predicts. A logit is a dot product over the whole K axis, split
across eight waves at five 128-blocks each and drained through one `fmaf` chain per wave in wave
order. Which workgroup owns a token only sets `first`, and a lane's operand offset is built from its
own token column, so the same bytes enter the same matrix instruction in the same order whatever
else the pass is computing. The token-group width is a scheduling choice that was already measured
bit-identical at 1, 2, 4 and 8 groups.

## Measuring it: two processes cannot

A cross-process pair of the same change, three rounds each, 384-token document, 128-row passes:

| arm | prompt tok/s | 32-stream decode tok/s |
|---|---:|---:|
| tail | 599.9, 839.7, **827.1** | 301.0, 300.8, **300.8** |
| all | 527.5, 670.5, **706.3** | 288.6, 284.6, **288.6** |

That reads as +23% on a change the phase table prices at one 9 ms phase out of a 507 ms prefill,
about +1.8%. The decode column is the same code in both arms - a generation step asks for every row
in both - and it differs by 4.2%, so the two processes did not run at the same clock.

**And the decode column cannot normalise the prefill column.** [The clock
measurements](clock-power.md) put a 128-row prefill pass at 89%/68%/14% elasticity across the clock
range and a 32-stream decode step at 25%, so the same drift costs prefill several times what it
costs decode; dividing by 1.042 still leaves +18%. The instrument that works is the traced phase
table in one process, which is why the table above is the result and this pair is only evidence
about the box. `--head-rows both` walks the arms as a case axis inside one warmed process.

## What the selection is

`forward_batch` takes `LogitRows::Tail` or `LogitRows::All`. `run_head_batch` takes the first row of
a half-open selection and drives all three of its kernels from it: `head_sums` over the columns that
are read, `head_tile` storing only inside the selection, and both argmax kernels over the selected
rows. Workgroup zero starts at the selected column, pushed down only far enough that the last token
group stays inside the padded operand, so a selection never costs more than one token group of
arithmetic beyond what it asked for.

Rows the head did not produce read back as -1 rather than as whatever the buffer held from an
earlier pass, so a caller that reaches past its selection sees that it did.

The selection is contiguous because both shapes this engine serves are: one tail row for a prompt
pass, every row for a step whose rows are one token each of their own sequence. A batched drafter
verifying `k` tokens for each of `n` sequences would want `n` tails at stride `k`, which is a base,
stride and count in the same kernel and no new buffer.

## A width that was pinned by the first wide pass

`run_head_batch` chose its token-group width once and kept it for the life of the process:
`if (!h->tt) h->tt = npad <= 16 ? 1 : 2`. A process whose first pass above `RMAX` carried 9 to 16
rows pinned one group, and every later pass then read the 278 MB output image once per sixteen token
columns - eight reads for a 128-row pass instead of four. The width is chosen per call now, from the
selected rows rather than from the pass, which is also what lets a one-row selection take one group
inside a 256-row pass.

## What this does not fix

The head's logit buffer is still `capacity * VOCAB` floats, 254 MB at 256 rows and 508 at 512,
because `logits` is still indexed by batch row and a selection can name any row. Compacting the
output to selection index `j` would let the buffer be sized by the rows a caller can read rather than
by the pass width, and that is the remaining half of "512-row passes are another 1.2% for another
254 MB of logit buffer". It is a serving-policy decision as much as a kernel change: the identity
panels ask for every row of a 128-row pass and would still need the room.

The 8-row deployed prefill route keeps its own head inside the persistent kernel and still computes
eight rows where it reads one. That costs one pass over the output image either way, so there is
nothing to collect there.

Raw samples: `batch-comparison/head-rows/phase-m19.json`, `batch-comparison/head-rows-inproc/`,
`batch-comparison/head-rows-{tail,all}/` (the cross-process pair above).
