# K2-Horizon chunk-2 anomaly — isolated debug findings (do not merge)

## Status

- **Fixed and shipped** (tag `beellama-staging-v0.4.7-r3`, commit `bacba2fd0`):
  the `fattn-mma-f16.cuh` divergent-barrier fix ported from upstream
  ggml-org/llama.cpp#27870. This cured the catastrophic per-chunk KLD/PPL
  blowups at chunks 9/16/22/33 etc. on quantized-weight runs
  (e.g. Q5_K_M|f16/f16 chunk-9 own PPL 5.11 -> 10.68 pre-fix, clean post-fix).

- **Still broken**: a second, smaller, deterministic anomaly confined to the
  sweep's chunk 2 (one 8192-token window), own-chunk KLD up to 0.24
  (Q8_0 weights), q8_0-K + V in {q8_0,q5_0,q5_1,q4_1,q3_1}; clean for
  V in {q4_0,q6_0,q3_0}, for any q6_0/f16 K, and for every
  `--kv-tail-tokens 128` run. This branch isolates it.

## Minimal repro (RTX 3090, sm86, CUDA 13.1 build, fa on, ctx 8192)

```
# exact chunk-2 window (BOS-substituted) of NANI-Nithin/K2-Horizon-7B-GGUF:Q5_K_M
# over wiki.test.raw — see "how to rebuild the corpus" below
llama-perplexity --ctx-size 8192 -f chunk2_exact.txt -fa on --chunks 1 \
    --hf-repo NANI-Nithin/K2-Horizon-7B-GGUF:Q5_K_M \
    --cache-type-k q8_0 --cache-type-v q5_0
# BROKEN:  [1]6.5622   (CPU and f16-cache runs of the same window: 5.82)

llama-perplexity ... --cache-type-k f16 --cache-type-v f16
# CLEAN:   [1]5.8171
```

The anomaly does **not** need cache reuse, CUDA graphs, stream-k, or the
2048-token sweep batch size: it reproduces as the first and only chunk with
batch 512, with `GGML_CUDA_DISABLE_GRAPHS=1` (bit-identical output), and
with the exact window detokenized into a standalone file. Values are
bit-identical across reruns and across the r2 and r3 binaries.

## What has been ruled out (all verified on-disk)

1. **KV cache writes** — layer-0 K and V storage bytes are identical between
   the broken run and a clean single-chunk run of the same window (dumped via
   `KVCACHE_DUMP=<dir>` instrumentation in tools/perplexity).
2. **Dequantization** — the f16 side buffers produced by
   `launch_fattn`'s `to_fp16` dequant (q8_0 K, q5_0 V) match a CPU reference
   dequant of the dumped cache bytes exactly (max abs diff = f16 ulp).
3. **The ported barrier fix, CUDA graphs, PDL (sm90+ only, dead code on
   sm86)** — no effect on this anomaly.
4. **Kernel selection** — the FA launch config is identical for the broken
   and clean runs: mma-f16 case `<DKQ=128, DV=128, ncols1=16, ncols2=4>`
   (np=4), blocks=(256,1,1) = ntiles_dst, single wave, **no stream-k
   fixup, no parallel combine** — plain single-pass tile processing.
5. **PDL/stream overlap** — `ggml_cuda_pdl_sync/lc` are no-ops below sm90.
6. **Value-independence of the trigger** — the f16-cache run uses the same
   kernel, same shapes, same window, and is clean; q8_0-K+q4_0-V (both
   dequantized, nearly identical kernel timing) is also clean. The failure
   needs the specific *values* that the q5_0/q8_0/q5_1/q4_1/q3_1 dequant
   produces for this content. Not a scale error, not a row permutation,
   not a truncated KV range (all tested against a numpy reference).

## Localization

- FA output dump (`GGML_CUDA_DUMP_FA=<dir>`): 302 of the last ubatch's 512
  queries diverge by up to 5.4 (clean runs differ by <1e-2), spread across
  all 32 heads (~145 rows/head), no tile/row permutation structure.
- The kernel inputs (Q, dequantized K/V) are verified correct; the output is
  garbage -> **the fault is inside `flash_attn_ext_f16`'s tile loop /
  online-softmax state for np=4 on sm86 for these inputs** — a value-dependent
  kernel bug, deterministic under stable scheduling.
- Forcing the non-stream-k path (`GGML_CUDA_FA_NO_STREAMK=1`) yields NaN —
  a separate latent bug in the parallel-blocks/`flash_attn_combine_results`
  path for this shape (ncols2=4, gqa 4) that also deserves attention.

## Instrumentation on this branch (all marked "XXX debug (do not merge)")

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `GGML_CUDA_DISABLE_GRAPHS=1` bypass.
- `ggml/src/ggml-cuda/fattn-common.cuh` (`launch_fattn`):
  `GGML_CUDA_DUMP_FA=<dir>` — dumps dequantized K/V f16, the FA output, Q
  and prints the launch config for the first two full-KV calls.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: `GGML_CUDA_FA_NO_STREAMK=1`
  forces the parallel-blocks path.
- `src/llama-kv-cache.{h,cpp}`: `get_v_storage()` accessor.
- `tools/perplexity/perplexity.cpp`: `KVCACHE_DUMP=<dir>` (+ optional
  `KVCACHE_DUMP_LAYERS=0,1`) dumps raw K/V cache storage after each chunk.

## How to rebuild the exact corpus

The ppl tool substitutes BOS over `tokens[start]` at pos 0 of every chunk, so
the effective chunk-2 window the model sees is
`[BOS, tokens[8193 .. 16384]]` of `[BOS] + tokenize(wiki.test.raw)`.
Detokenize ids 8193..16384 (byte-level BPE: standard GPT-2 bytes-to-unicode
reverse, then concatenate; the vocab comes from the GGUF
`tokenizer.ggml.tokens`) and append ids 16385..24577 as padding so the file
exceeds 2*n_ctx tokens. Round-trip-verify with `llama-tokenize --ids`.

## Next steps

1. Symbol-level debugging of `flash_attn_ext_f16` (np=4, sm86) fed with the
   dumped kf16_0/vf16_0/q_0 — replayable in isolation in a small harness.
2. Check whether upstream `ggml-org/llama.cpp` master reproduces (the tile
   loop is shared code; if yes, report there, else it is beellama-specific).
3. The `GGML_CUDA_FA_NO_STREAMK` NaN path on sm86 for ncols2=4.
