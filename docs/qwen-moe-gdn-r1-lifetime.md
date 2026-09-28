# The rejected Qwen GDN swap changes which old R1 history enters the convolution

The guarded installed direct-state writer gains 18.66% on its accepted 32-stream generated path, but the rejected broad 31+1 writer changes the swapped row's full head and serialized state. [The saved-state comparison](qwen-moe-gdn-state-diff.md) located R layer 1, row 1; [the plane comparison](qwen-moe-gdn-conv-planes.md) found that its **two old-history positions change while its newly projected input agrees bit-for-bit**. We now identify the source of those old positions against a **full saved pre-swap state**, not just the two post-swap images.

The existing swap probe seeds 32 eight-token streams, takes a stable 32-row generated step, then swaps sequence IDs 0 and 1 in another 32-row step. `gdn_r1_lifetime.cpp` is the same probe with the state snapshot moved from step 2 to step 1; it ran against the accepted guarded artifact `c5a70cfbe7f9f34571f036b05f37dd9668401c9e`. Its step-1 **2,113,550,957-byte state FNV-64 is `b0793935f384e9ad`**, exactly the step-1 FNV already recorded by both rejected gathered and direct runs. Thus both rejected arms enter the swap with identical serialized state. The snapshot is saved in `r1-pre-swap.state` (SHA-256 `bd88bae1621dd6cdd40fb3153ebce8e5698276444d57ec27219cb99909ed3e29`).

`gdn_r1_lifetime.py` compares **FP32 bit patterns**, not closeness, at every one of the 30 recurrent R layers. Each post-swap old-history plane of saved logical row 1 is matched independently against the corresponding later history plane in all **32 physical pre-swap rows**. A full match is unique at every recurrent layer. Results:

| R layer | Gathered post row 1, old positions 0/1 | Rejected direct post row 1, old positions 0/1 |
| --- | --- | --- |
| 0 | both **pre row 1**, 8,192/8,192 words each | both **pre row 1**, 8,192/8,192 |
| **1** | both **pre row 0**, 8,192/8,192 | both **pre row 1**, 8,192/8,192 |
| 2 and remaining 27 recurrent layers | both **pre row 0**, 8,192/8,192 | both **pre row 0**, 8,192/8,192 |

At layer 1, matching the *other* pre row yields only **206/8,192** words, all common zeros. Neither changed direct old plane has a full match with any other row or time coordinate of the gathered post R1 image. The complete [receipt](../../data/qwen-moe/gdn-fused/r1-source-receipt.json) records 32 match counts per arm, old position and layer; the source script, input hashes and exact-state proof are sufficient to replay without another model run.

**Result:** the rejected state path does not merely round layer-1 recurrence differently. It presents **a different pre-step convolution history row** to the first divergent old-history observation, while the newly projected layer-1 QKV is equal. Its `R1 row1` old history is a verbatim pre-row-1 copy where gathered execution has a verbatim pre-row-0 copy. This is an exact observation of saved inputs, not proof of the first divergent GPU graph operation or which physical source the intended sequence map ought to read. The source `build_conv_state` calls `build_rs` for R separately from the S-state direct writer; the selected `build_rs` gathers `s_copy_main` then relocates extra rows, whereas the direct S cache view has its own ordering. The layer-specific mismatch points at **graph ordering/cache lifetime across the layer-0 S writer and R1 history gather**, or at the R1 gather itself, rather than a global old-R row misindex or a QKV projection error. Do not repair this with S-state snapshots or an FP32 tolerance.

**Next native experiment:** in an unchanged no-callback graph, save R1 row 1's old-history operand into disjoint side-band storage *immediately after its `build_rs` gather* and separately after R1's convolution-state copy, with the existing pre-swap state replayed into both arms. Record the physical `s_copy_main` index for logical row 1 and the R1 cache row bits before the 31-row call. If the index is the same but the gather differs, inspect graph dependencies between the preceding S0 direct cache write and the R1 gather; if the index changes, repair the row map owner. Preserve full-head and complete-state acceptance. The selected guarded writer remains installed and neither runtime nor service default changed in this experiment.

## Reproduction and custody

```sh
# Probe built against the retained accepted artifact; only step-1 state is saved.
g++ -O2 -std=c++17 -I../bonsai-hip/include \
  -I../bonsai-hip/ggml/include \
  tools/qwen-moe/gdn_r1_lifetime.cpp \
  -L../../data/qwen-moe/runtime/c5a70cfbe7f9f34571f036b05f37dd9668401c9e/bin \
  -Wl,-rpath,../../data/qwen-moe/runtime/c5a70cfbe7f9f34571f036b05f37dd9668401c9e/bin \
  -lllama -o ../../data/qwen-moe/gdn-fused/gdn-r1-lifetime-probe
# GPU: tools/run-batch-compare --runtime-max 40s --memory-gib 43 \
#       --host-reserve-gib 7 --exec PROBE MODEL DATA/r1-pre-swap 2
python3 tools/qwen-moe/gdn_r1_lifetime.py \
  ../../data/qwen-moe/gdn-fused/r1-pre-swap.state \
  ../../data/qwen-moe/gdn-fused/swap-control.state \
  ../../data/qwen-moe/gdn-fused/swap-default.state \
  --output ../../data/qwen-moe/gdn-fused/r1-source-receipt.json
```

Probe executable SHA-256 `c1e97541f7d5d436e9eb16f6ed0d133e22694a04442ab96abec46595e9fddf05`, model SHA-256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`, gathered post-state SHA-256 `ff6eacf57327d4708ad9a2c7ab558e90165b7f8499ffa009ebea5662cb1c1215`, rejected direct post-state SHA-256 `292d89b9f24098668d2a793d44ac25795e4c67b0cd938469a8a7a6109b6aa0a9`. The [raw wrapper log](../../data/qwen-moe/gdn-fused/r1-pre-swap-wrapper.log) retains the command, pre-state hash and service reconciliation. The GPU lock is free and `systemctl --user is-active bonsai-halo.service` reports active. This is a failure localization, not an accepted speedup or a whole-model equivalence claim.
