# K2-Horizon-7B: CUDA quantized-KV-cache corruption (deterministic, value-dependent)

**Status:** reproducible, kernel-math ruled out, exact faulty op not yet localized.
**Reported against:** a tree whose flash-attention CUDA code is byte-identical to
`ggml-org/llama.cpp` `master` (see "Provenance" below) plus the MBZUAI-IFM
K2-Horizon model support. **No beellama code is involved.**

## TL;DR

Running K2-Horizon-7B on CUDA with a **quantized K cache** (q8_0) and V cache in
{q8_0, q5_0, q5_1, q4_1, q3_1} produces a deterministically wrong model: the
perplexity of one 8192-token window is **6.5622** where the same window measures
**5.79-5.82** with an f16 KV cache, with the CPU backend, or on the CPU with the
*same* quantized cache.  The flash-attention kernels themselves are **correct**
(verified per-call against a float64 reference on the exact kernel inputs for
all 576 calls of the repro); the f16-cached forward pass is **bit-identical**
between CUDA and CPU; the quantized-cached forward pass on CUDA **diverges from
the CPU from ~layer 2 onward**.  A mainline model with the identical attention
kernel shape (Llama-3.1-8B-Instruct, GQA 4, head_dim 128) does **not** trigger
it.  We therefore believe the fault is in llama.cpp's CUDA quantized-KV-cache
path (set_rows quantize and/or the FA dequant and/or their interaction with the
mma flash-attention path) and that it is **triggered by K2-Horizon's KV value
distribution**, not by the K2 architecture code itself.

## Minimal repro

```bash
llama-perplexity --ctx-size 8192 -f chunk2_exact.txt -fa on --chunks 1 \
    --hf-repo NANI-Nithin/K2-Horizon-7B-GGUF:Q5_K_M \
    --cache-type-k q8_0 --cache-type-v q5_0
# [1]6.5622   (BROKEN - deterministic, bit-stable across reruns)

# ... --cache-type-k f16 --cache-type-v f16
# [1]5.8171   (clean)
# ... --cache-type-k q8_0 --cache-type-v q4_0
# [1]5.8141   (clean)
```

`repro-k2-quantized-kv.sh` in this commit runs the full matrix.
`scripts/rebuild_chunk2_corpus.py` rebuilds the corpus file (chunk 2 of
`wiki.test.raw` under K2's tokenizer: ids 8193..16385 detokenized, BOS
substituted at chunk start, plus ids 16385..24577 as padding so the file
exceeds 2*n_ctx tokens).  The trigger is **value-dependent**: with other corpus
windows or other models the same quant pair is clean, so keep the exact window.

## Matrix (RTX 3090, sm86, CUDA 13.1, fa on, ctx 8192, 1 chunk)

| config                                      | PPL    | verdict |
| ------------------------------------------- | ------ | ------- |
| K2, CUDA, q8_0 K / q5_0 V                   | 6.5622 | broken  |
| K2, CUDA, q8_0 K / q8_0 V                   | 5.7922 | clean   |
| K2, CUDA, q8_0 K / q4_0 V                   | 5.8141 | clean   |
| K2, CUDA, q6_0 K / q5_0 V                   | —      | clean (per the original sweep, build r2/r3) |
| K2, CUDA, f16 K / f16 V                     | 5.8171 | clean   |
| K2, CPU, q8_0 K / q5_0 V                    | 5.7936 | clean   |
| K2, CPU, f16 K / f16 V                      | 5.7914 | clean   |
| Llama-3.1-8B-Instruct Q8_0, CUDA, q8_0/q5_0 | 4.3456 | clean (f16: 4.3416) |

The sweep-wide damage table (Q5_K_M weights, old sweep log) had V in
{q8_0, q5_0, q5_1, q4_1, q3_1} broken and {q4_0, q3_0} clean, clean for any
q6_0/f16 K, and clean for every `--kv-tail-tokens 128` run.  Damage grows with
the weight quant (Q8_0 weights ≫ Q5_K_M ≫ Q6_K ≈ clean).

## What has been ruled out (with receipts)

1. **Beellama-specific code.** The repro tree (`k2-merge-upstream`, commit
   946e93ffe = MBZUAI-IFM's `model/K2Horizon` merge base 40b80ab96 merged with
   `upstream/master`) has `ggml/src/ggml-cuda/fattn{,-mma-f16,-common}.cu*`
   **byte-identical to upstream/master** (verified by diff).  `git diff
   upstream/master HEAD -- <those files>` is empty.  The bug therefore lives in
   code that is identical to what ggml-org ships.

2. **The flash-attention kernel math.** For every one of the 576 flash-attention
   calls of the broken run (16 ubatches x 36 layers), the kernel inputs (Q, the
   dequantized K/V cache rows) and the kernel output were dumped to disk and the
   output compared against a float64 numpy reference of attention over exactly
   those inputs.  Max deviation over all calls: ~0.1 (f16-accumulation noise).
   The kernel configuration is
   `<DKQ=128, DV=128, ncols1=16, ncols2=4>` (ncols=64, nthreads=128, nwarps=4,
   cols_per_warp=16, **np=1**, nbatch_fa=64, nstages=2, Q_in_reg=1,
   stream-k with blocks = ntiles_dst, no fixup).  The earlier suspicion that
   np=4 (the cross-warp meta exchange) was involved was wrong: the actual
   instantiation has np=1, so the entire online-softmax state lives in one warp.

3. **The dequantization of the quantized cache.** The FA's f16 staging buffers
   (the `to_fp16` dequant of the q8_0 K cache and the q5_0 V cache) were dumped
   and compared against a numpy reference dequant of the raw cache bytes:
   max |diff| = f16 ulp.  (Note for re-verification: ggml's q5_0 packs
   values j and j+16 into one byte - `value j = ((qs[j]&0xF) | ((qh>>j&1)<<4)) - 16) * d`,
   `value j+16 = ((qs[j]>>4) | ((qh>>(j+16)&1)<<4)) - 16) * d`.)

4. **The CUDA graph machinery and stream-k.** `GGML_CUDA_DISABLE_GRAPHS=1`
   gives bit-identical output; the broken shape runs with blocks = ntiles_dst
   (no stream-k fixup path taken), and the anomaly does not need cache reuse
   (single-chunk repro) nor the 2048-token sweep batch size.

5. **The model math (float path).** The f16-cached forward pass is
   **bit-identical** between the CUDA and CPU backends: the per-layer K/V cache
   dumps of layers 0..11 (post-eval, via `KVCACHE_DUMP`) are byte-equal
   (median |diff| = 0.0000).  Every matmul/norm/rope in the K2 graph therefore
   produces the same values on both backends - the fault must sit in the
   quantized-cache write/read path.

6. **A wrong block format / scrambled layout in the caches.** Both the q8_0 K
   and q5_0 V caches are on-grid, self-consistent, and dequant-match between
   CUDA and CPU (K: cosim 0.9999; V: cosim 0.99927).  The FA kernels consume
   exactly those values (verified in 2) and compute correct attention on them.

## What the divergence looks like

Dumping all 12 first-layer K/V caches of the broken config on **both** backends
(same binary family, same tokens) and comparing dequantized values:

| layer | median abs diff (K) | max abs diff (K) |
| ----- | ------------------- | ---------------- |
| 0     | 0.0067              | 1.46             |
| 1     | 0.0302              | 1.59             |
| 2     | 0.0535              | 11.8             |
| 3     | 0.167               | 10.8             |
| 4-11  | 0.30-0.43           | 6.5-8.7          |

(For comparison, the f16-cached layers are byte-equal at every layer.)
So the *quantized* K/V values themselves already differ between CUDA and CPU at
layer 0/1 by about the quantizer's rounding granularity, and the difference is
then **amplified chaotically by the model** until the caches are unrecognizable
by layer 3.  On the CPU the same +-1..2-LSB quantization differences exist
relative to the f16 cache, yet the CPU's own quantized run stays on the f16
reference (PPL 5.7936 vs 5.7914) - i.e. that amount of quantization noise is
normally harmless, which is what makes the CUDA-side amplification suspicious.

## Open observations (for whoever picks this up)

* The per-layer K/V dumps of the CPU's own quantized run differ from the CPU's
  f16 run far more than q8_0/q5_0 rounding can explain (median |diff| ~1.4-1.7),
  while per-(token,head) row norms match exactly (corr 1.000000) and both runs
  produce the correct perplexity, and the FA's Q.K products reconstructed from
  the CPU run's own FA outputs match the quantized cache exactly.  The two
  value sets are therefore related by some (run-dependent, equivalence-
  preserving) orthogonal-looking map.  We could not identify it from the dumps
  (not a rope-angle shift, not a rope type swap, not any of the six axis
  permutations, not a per-row dim permutation).  This is unexplained but is
  *not* the bug (both backends are self-consistent and correct on their own
  values); it is documented here because any localization attempt should be
  aware of it.
* The `GGML_CUDA_FA_NO_STREAMK` path (forcing parallel blocks + the
  `flash_attn_combine_results` fixup) yields NaN output for this shape on
  sm86 - a separate latent issue, also present on the r2/r3 beellama builds
  (documented in the original sweep notes).

## Provenance

* Repro tree: branch `k2-merge-upstream` = MBZUAI-IFM `model/K2Horizon` merge
  base `40b80ab96` ("Merge beellama-llamacpp-base into model/K2Horizon" -
  the model/K2Horizon state with the shared llama.cpp/beellama updates, before
  the later KVarN/mma plumbing) merged with `ggml-org/llama.cpp` `master` at
  `b23701f77` ("cuda : fix CUB argsort corruption", Sep 19).  After resolving
  the five textual conflicts (K2 tokenizer hashes, `LLM_ARCH_K2_HORIZON`,
  `LLAMA_VOCAB_PRE_TYPE_K2_HORIZON`, tensor names) and adapting
  `src/models/k2-horizon.cpp` to the upstream per-layer `n_ff_exp` API (included
  as the first commit of this branch), the tree builds and the K2 model runs.
* The original sweep that found this (and the bug-1 divergent-barrier fix,
  ported from ggml-org #27870) ran on beellama `beellama-staging-v0.4.7-r3`;
  the anomaly survives identically on the r2 and r3 binaries and on this
  upstream-identical tree, i.e. it is unaffected by the barrier fix.

## Suggested next steps for MBZUAI-IFM

1. Reproduce with `repro-k2-quantized-kv.sh` (10 min including build).
2. Bisect the CUDA quantized-KV path with the layer-dump harness: the divergence
   between the CUDA and CPU *quantized* caches already appears at layer 0-1 as
   +-1..2-LSB differences in the quantized bytes (both on-grid, both correct
   dequant) - the question is which of (a) the CUDA `set_rows` quantize kernels
   (per-type: q8_0/q5_0/q5_1/q4_1/q3_1 broken, q4_0/q3_0 clean) and (b) the
   FA-side f16 dequant + mma path, turns those few-LSB differences into the
   catastrophic layer-2+ divergence seen only on CUDA.
3. A cheap differential: run the broken quant pair on CUDA with the
   non-mma flash-attention path (tile kernel) to see whether the corruption
   follows the cache or the kernel family.
