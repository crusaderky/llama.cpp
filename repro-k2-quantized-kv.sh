#!/usr/bin/env bash
# Reproducer: K2-Horizon-7B + CUDA + quantized KV cache => corrupted model output.
#
# Branch/commit context:
#   * Tree = ggml-org/llama.cpp master (flash-attention CUDA code byte-identical to
#     upstream/master) + the MBZUAI-IFM K2-Horizon model support. No beellama code.
#   * The flash-attention kernels themselves are verified correct per-call against a
#     float64 numpy reference of attention over the exact kernel inputs (see the
#     accompanying MD); the divergence is in the quantized-KV-cache path on CUDA.
#
# Hardware used: RTX 3090 (sm86), CUDA 13.1.  Expected results:
#
#   --cache-type-k q8_0 --cache-type-v q5_0   PPL = 6.5622   (BROKEN; deterministic)
#   --cache-type-k f16 --cache-type-v f16     PPL = 5.8171   (clean)
#   --cache-type-k q8_0 --cache-type-v q4_0   PPL = 5.8141   (clean)
#   --cache-type-k q8_0 --cache-type-v q8_0   PPL = 5.7922   (clean, this quant pair)
#   CPU backend, any cache quant              PPL = 5.79     (clean)
#   Llama-3.1-8B-Instruct Q8_0 (same GQA4/D=128 kernel shape), q8_0/q5_0
#                                             PPL = 4.3456 ≈ f16 4.3416 (clean)
#
# The broken value is bit-stable across reruns and independent of CUDA graphs.

set -euo pipefail

REPO=${REPO:-$(pwd)}            # this llama.cpp checkout
CORPUS=${CORPUS:-/tmp/chunk2_exact.txt}
BUILD=${BUILD:-$REPO/build-merge}
HF_HOME=${HF_HOME:-$REPO/.tmp-hf}
export HF_HOME

# ---------------------------------------------------------------------------
# 1. Build llama-perplexity with CUDA (arch 86 here; any sm>=75 GPU works).
# ---------------------------------------------------------------------------
if [ ! -x "$BUILD/bin/llama-perplexity" ]; then
    cmake -S "$REPO" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_BACKEND_DL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
    cmake --build "$BUILD" --target llama-perplexity
fi

# ---------------------------------------------------------------------------
# 2. Corpus: any >2*n_ctx-token text works, but the value-dependence of the
#    trigger needs this exact chunk-2 window of wiki.test.raw under K2's
#    tokenizer (ids 8193..16385 detokenized, BOS substituted at chunk start,
#    plus ids 16385..24577 as padding so the file exceeds 2*n_ctx tokens).
#    Rebuild it with scripts/rebuild_chunk2_corpus.py if /tmp is empty.
# ---------------------------------------------------------------------------
if [ ! -s "$CORPUS" ]; then
    echo "corpus $CORPUS missing - run scripts/rebuild_chunk2_corpus.py first" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. The runs.
# ---------------------------------------------------------------------------
run() {  # label, extra args...
    local label=$1; shift
    local ppl
    ppl=$("$BUILD/bin/llama-perplexity" \
        --ctx-size 8192 -f "$CORPUS" -fa on --chunks 1 \
        --hf-repo NANI-Nithin/K2-Horizon-7B-GGUF:Q5_K_M "$@" 2>&1 \
        | grep 'Final estimate' | grep -o 'PPL = [0-9.]*')
    printf '%-40s %s\n' "$label" "${ppl:-<failed>}"
}

echo "== K2-Horizon-7B (Q5_K_M), CUDA, fa on, ctx 8192, 1 chunk =="
run "K q8_0 / V q5_0  (BROKEN)"     --cache-type-k q8_0 --cache-type-v q5_0
run "K f16  / V f16   (clean)"      --cache-type-k f16 --cache-type-v f16
run "K q8_0 / V q4_0  (clean)"      --cache-type-k q8_0 --cache-type-v q4_0
run "K q8_0 / V q8_0  (clean)"      --cache-type-k q8_0 --cache-type-v q8_0

echo
echo "== sanity: broken value is deterministic =="
run "K q8_0 / V q5_0  (rerun)"      --cache-type-k q8_0 --cache-type-v q5_0
run "K q8_0 / V q5_0  (no graphs)"  --cache-type-k q8_0 --cache-type-v q5_0 --no-cnv
