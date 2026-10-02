# Historical Bonsai measurement receipts

This is a bounded publication of existing September 2026 measurements, **not fresh measurements**
and not an assertion that the recovered host can reproduce the GPU runs. The research owner owns
these receipts. No campaign was resumed. Interpret each panel with its report and recorded source,
executable hashes, numerical mode, shape and controls; panels from different builds are not interchangeable.

## Panels and reports

- [Full-model TPS](../docs/full-tps-20260921.md): three rounds of 1/8/32-stream generation and
  768-token prefill, plus ten drafted single-stream cases in aggressive and FP32-state modes.
  `full-tps-20260921-2255/` holds the summary, batch samples, executable hashes and both single-stream logs.
- [128-stream generation](../docs/generation-128.md): the successful managed-allocation panel,
  64/128 streams. Failed admission attempts, model files and prompt bodies are outside this bundle.
- [Decode-stream curve](../docs/decode-streams.md): `curve-a`/`curve-b` include the bridge at
  48 streams and the later 64-stream result; the phase maps, FFN image band and fetch band expose
  the measurement's shape dependence. Output sequence SHA-256 digests and recorded residual hashes remain as acceptance witnesses.
- [Single-stream shape and issue costs](../docs/deployed-matvec-groups.md): `single-stream/`
  contains the grid and prefetch/poll phase maps and original executable hashes.
- [Drafted-step census](../docs/drafted-step.md) and
  [ternary-coordinate ceiling](../docs/ternary-coordinate-ceiling.md): the coordinate panel,
  ISA census and grid arms. Read the correction in the canonical drafted-step report; these are
  original samples, not a claim that an expanded weight image wins.

## Public transformation

The original custody root is represented as `../../data/bonsai2/batch-comparison`, following the
public snapshot's neutral path convention. Files keep their original relative names.

- JSON generated-text, prompt-body strings and output token-ID sequences are removed. Prompt
  counts, shape/configuration values, timing and numerical samples, digests and source/executable/input
  hashes remain. Each removed token sequence is replaced by its SHA-256 over the compact JSON
  integer array and its length, so output equality can still be compared without reconstructing text.
  No prompt bodies, output text, token sequences or model/dataset arrays are included.
- Absolute home paths become neutral project-relative paths or `../../data/...`. Local usernames,
  named hosts, runtime user-directory identities and UUID job/thread identifiers are neutralized.
  Hardware names such as Radeon 8060S and gfx1151 describe the experiment and are retained.
- JSON is parsed and emitted with indentation, so byte hashes change even where no field is removed.
  Log files retain their timing/count lines; only identity/path substitutions apply.
- No executable binaries, weight arrays, environment credentials, internal orchestration boards,
  full private source history or unrelated research is included.

[manifest.json](manifest.json) records original and public SHA-256 hashes, byte sizes and every
removed JSON string or token-sequence field. The original hashes below identify the custody inputs;
they are **not** hashes of these scrubbed publication files. JSON numeric/boolean/null values outside
the explicitly removed token sequences were compared at their original key/index and preserved.
This checks the publication transformation only; it is not a rerun of inference or a new performance
verification.

## Original custody hashes

| Receipt | Original SHA-256 | Original bytes |
| --- | --- | ---: |
| [`full-tps-20260921-2255/summary.json`](full-tps-20260921-2255/summary.json) | `0b8cdcdedd4c6ec7bb2d2fc288612ff956562315f4a3b49cb7615a6be79a332f` | 1611 |
| [`full-tps-20260921-2255/batch/run.json`](full-tps-20260921-2255/batch/run.json) | `74a07a21c2d3bcc5e66109621f25ba88a8ee8e0120a19f925db28b91322639f8` | 103034 |
| [`full-tps-20260921-2255/build.txt`](full-tps-20260921-2255/build.txt) | `76c8e949b3d574663d3cb7c1a841613117c5a97864eaf59089b969ddb3a3d404` | 212 |
| [`full-tps-20260921-2255/single-dflash.log`](full-tps-20260921-2255/single-dflash.log) | `d2892339491d3a22afd2c9709c64fa34789c3c1346979ccc59df98a7f503da52` | 11306 |
| [`full-tps-20260921-2255/single-dflash-f32.log`](full-tps-20260921-2255/single-dflash-f32.log) | `c2726cefdbd4627a21f05f613873e9db258c179bb2be0b64f08dd808bb517551` | 11308 |
| [`tps128-20260921-2306/summary.json`](tps128-20260921-2306/summary.json) | `f8f2f4ea0a95b93f58f0277a8b05b2204af42e49f3745453c2889aa50548c28a` | 2174 |
| [`tps128-20260921-2306/managed/run.json`](tps128-20260921-2306/managed/run.json) | `89994d35fa95a99af0d2f1430c75d161892b993fb0d56dbb15c9cdb3de0d31a4` | 265607 |
| [`tps128-20260921-2306/build.txt`](tps128-20260921-2306/build.txt) | `dc8d0bb1c671b1a57a3cb9cc115ee5edc587d7a44c5f8f1ab7c7e039665cec8b` | 598 |
| [`decode-streams/curve-a/run.json`](decode-streams/curve-a/run.json) | `eb5a6ed68a53352ecdf7377f6202378f370c775309ef6ff83b9191d9cec9609a` | 59694 |
| [`decode-streams/curve-b/run.json`](decode-streams/curve-b/run.json) | `d1ecc7559b094d7f27171aae5eadcf940a9737c547db18e8461b3a4c5e19a650` | 63332 |
| [`decode-streams/phase-s32.json`](decode-streams/phase-s32.json) | `0d9c73a87bb6125fa943a6080fbf0e041927370558fe0c3fb90ed250d949952b` | 79795 |
| [`decode-streams/phase-s48.json`](decode-streams/phase-s48.json) | `9b492c9f88b647a2281c1ff62a3b6848fd83c8c58c616cd4aa0241de359f31d9` | 80715 |
| [`decode-streams/ffn-image-band.json`](decode-streams/ffn-image-band.json) | `39882b7853f9792f438cb61747899a6a33ac684e6fb9d85f08a0ef5599b510fa` | 319020 |
| [`decode-streams/fetch-band.json`](decode-streams/fetch-band.json) | `50ab3f5930ceb3bd8fcaac65908e07acaf71588b90b564a1b6f0f45b23d81291` | 3740 |
| [`single-stream/build.sha256`](single-stream/build.sha256) | `4a306b429b727fc58eda4dee0d0b6ae9b9caec817cc7055f3cef593748adb8d2` | 156 |
| [`single-stream/grid-60-48.txt`](single-stream/grid-60-48.txt) | `468c83284450bb35f15c23be6c9f37e4123392d0396781fb46eded7bdad1e302` | 5145 |
| [`single-stream/prefetch-poll-phasemaps.txt`](single-stream/prefetch-poll-phasemaps.txt) | `a47b710fe2be4f4bc0f7fcdc2ee7b3f34b84bd4d16c1a2818b01c5601f7e0013` | 7080 |
| [`drafted-step/coordinate-panel.txt`](drafted-step/coordinate-panel.txt) | `82dab1fee4ec74318bb68b9a280c56dcdb36a7ce88eaceb9334238916bdc031b` | 6100 |
| [`drafted-step/isa-census.txt`](drafted-step/isa-census.txt) | `eb5e64cb329be29a51787d738b3f1dcf99e82111c37851622f6ce0bd8bc2ff7a` | 3059 |
| [`ternary-coordinate-ceiling/grid-arms-aba.txt`](ternary-coordinate-ceiling/grid-arms-aba.txt) | `0792c62fde3f4bb0afc549dd1f974c551e1abdc76e31ab62d2fc97e55bbc1f8a` | 4579 |
| [`ternary-coordinate-ceiling/grid-arms-first.txt`](ternary-coordinate-ceiling/grid-arms-first.txt) | `3cf943e4a7c43b14f2b4495ecc5210ee2d463e9b8b0b2b01cdea6ae2d6e914cd` | 3570 |
