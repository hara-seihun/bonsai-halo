# Shared low-bit router input changes actual Qwen MoE routes

**Measured negative, September 24.** A single signed, group-32 activation code cannot be passed to the installed F32 router as a cheap substitute for its actual post-attention producer without changing the selected experts. On both disjoint real-text 64-token captures over **all forty layers**, Q4 changes the ideal-real top-eight set on **896/2,560 held layer-token decisions** (every held token has 8–25 changed layers). Even Q8 changes **61/2,560** (37/64 tokens have at least one changed layer). This is not an FP32/native acceptance panel or a language-loss claim. It rules out *this unchanged-weight, shared max-scale input-code family* as an exact routing shortcut on the observed inputs. The existing grouped-expert gate/up activation preparation is not a new discovery.

| Actual-producer split | Code | Unchanged routes / 2,560 | Sufficient route certificates / 2,560 | Local routed-sum RMS, **only unchanged routes** | Max absolute selected score drift, unchanged routes |
| --- | --- | ---: | ---: | ---: | ---: |
| train | signed Q4 | 1,575 | 0 | .036949 | .056160 |
| held | signed Q4 | 1,664 | 0 | .044607 | .063259 |
| train | signed Q8 | 2,511 | 151 | .002169 | .007073 |
| held | signed Q8 | 2,499 | 189 | .001991 | .004053 |

Each code uses the maximum absolute value of each 32-input group divided by 7 or 127, **rounded to FP16 before quantization and reconstruction**. The signed integers occupy 4 or 8 bits each; 64 FP16 scales add 128 bytes. This charges **1,152 / 2,176 bytes per 2,048-wide producer**, versus 8,192 for F32. Even granting zero-cost direct packed router consumption and charging a fresh F32 read in every layer, the maximum saved *activation* traffic is **281,600 / 240,640 bytes per generated token**, just **.010723% / .009163%** of the conditional 2,626,187,904-byte complete-model one-read weight stream. The installed router **weight image does not shrink**, and a real packed dot needs new operations or unpacking; no native kernel or faster inference follows from the payload ratio. The captured producer is reused by other consumers, so eliminating its F32 storage altogether is not demonstrated.

## Complete observed map and certificate

For each captured layer-token vector `x`, the experiment reads the actual 256×2,048 F32 router image, evaluates `W x` in FP64, then selects the top eight with stable ID tie ordering. Every reference top-eight *set* agrees with the native callback IDs on all 5,120 rows. The largest absolute difference between the host's ideal-real selected-softmax coefficient and captured native FP32 score is 1.77e-6. Real selected-softmax weights after code reconstruction determine score drift. Where the set agrees, the eight *captured* native down vectors and the ideal-real score before/after coding give the table's complete local weighted-sum RMS over those rows; where a route changes, the capture has no unselected expert outputs and **does not assign zero error or manufacture a full sum**. The source scores used for this local comparison are ideal-real on both arms, not the original native FP32 sum.

For a fixed `x` and a chosen code `q`, let `d=q-x`. The exact-real logit displacement of expert `e` is `W_e d`, bounded by `u_e=Σ_j |W_ej| |d_j|`. If `min_selected(l_e-u_e) > max_unselected(l_f+u_f)`, the top-eight set provably survives this quantizer's error without evaluating its new dot products. The checker confirms every certificate implies an unchanged computed route. It certifies **none** of the Q4 held rows and 189 of 2,560 Q8 held rows, so a conservative guard built from this bound could not usually bypass the F32 reference. More generally, every perturbation `||d||∞≤r` is certified by `r < gap / (max_selected ||W_e||₁ + max_unselected ||W_f||₁)`; the minimum observed held certificate radius is 5.623e-8. These are sufficient certificates, **not** necessity bounds on arbitrary codes or a claim that any Q8-induced route change is semantically disastrous. The smallest actual ideal-real eighth-to-ninth held margin is 4.699e-6.

The numerical and execution boundaries matter: host FP64 matrix products and exponentials define this comparison; native fused top-k first computes a rounded 256-way softmax, and width-dependent behavior is documented separately. The activation code is neither the selected native Q8_1 gate/up quantizer nor a complete paid expert-weight image. A change of selected experts has no complete-model quality measurement here. Earlier ternary/sub-bit complete-image experiments showed why a good local response cannot replace held language loss. This measured route-instability and its tiny maximum traffic prize argue against porting this simple router-input code. If a learned route-aware carrier is proposed, it must include all forty layer decisions, paid encoding/packed-consumer cost and disjoint complete-model loss. The independent engine question remains unprofiled ordinary Q8/expert phase timing with complete heads, not another synthetic scalar clip.

## Reproduce and custody

The [four source/model/capture-hashed CPU shards and aggregate](../../data/qwen-moe/router-input-precision/README.md) retain each layer's counts, errors, score checks and changed IDs. From Bonsai's root, with NumPy:

```sh
D=../../data/qwen-moe/router-input-precision
for i in 0 10 20 30; do
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 tools/qwen-moe/router_input_precision.py \
    --first "$i" --last "$((i+10))" --output "$D/part-$i.json"
done
python3 tools/qwen-moe/router_input_precision.py --aggregate --output "$D/summary.json"
```

The four bounded shards can run independently. No GPU was used; the model, selected executable, numerical serving defaults and resident service did not change.
