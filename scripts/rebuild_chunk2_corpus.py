#!/usr/bin/env python3
"""Rebuild the chunk-2 corpus for the K2-Horizon quantized-KV reproducer.

The perplexity tool splits a file into chunks of n_ctx tokens and substitutes
BOS over tokens[start] of every chunk, so the effective window of chunk k is
[BOS, tokens[k*n_ctx+1 .. (k+1)*n_ctx]].  The trigger window for the K2
anomaly is the sweep's chunk 2 over wiki.test.raw, i.e. ids 8193..16385 of
[BOS] + tokenize(wiki.test.raw).  This script detokenizes those ids with the
K2-Horizon GGUF's tokenizer (byte-level BPE) and appends ids 16385..24577 as
padding so the file exceeds 2*n_ctx tokens, then round-trip-verifies.

Requires: the K2-Horizon-7B GGUF (any quant - only the tokenizer is used) and
the raw corpus (wiki.test.raw, 281810 file tokens under K2's tokenizer).
"""
import json
import subprocess
import sys
from pathlib import Path

GGUF = Path(sys.argv[1])          # path to a K2-Horizon-7B GGUF
RAW = Path(sys.argv[2])           # wiki.test.raw (the original sweep corpus)
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/chunk2_exact.txt")

LLAMA_TOKENIZE = sys.argv[4] if len(sys.argv) > 4 else "llama-tokenize"


def tokenize_ids(path, model):
    out = subprocess.run(
        [LLAMA_TOKENIZE, "--ids", "-m", str(model), "-f", str(path)],
        check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def detokenize(ids, model):
    # llama-tokenize has no ids->text mode; decode via the GGUF vocab with the
    # byte-level-BPE unicode map.  Token pieces are stored pre-unicode-escaped.
    import re
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(str(model))
    tensors = {t.name: t for t in reader.tensors}
    data = tensors["tokenizer.ggml.tokens"].data
    n = data.size // 4  # u32 strides
    toks = data.reshape(n)[: len(data) // 4] if False else None
    # simpler: read via the field API
    field = [f for f in reader.fields.values() if f.name == "tokenizer.ggml.tokens"][0]
    parts = b"".join(bytes(field.parts[i]) for i in field.data)
    offs = field.data[2:-1]
    vocab = []
    for i in range(len(offs) + 1):
        start = 0 if i == 0 else offs[i - 1]
        end = offs[i] if i < len(offs) else len(parts)
        vocab.append(parts[start:end])

    def to_utf8(piece):
        # reverse GPT-2 bytes-to-unicode
        bs = list(range(33, 127)) + list(range(161, 173))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        table = {c: b for c, b in zip(cs, bs)}
        out = bytearray()
        s = piece.decode("utf-8", errors="replace")
        for ch in s:
            out.append(table[ord(ch)] if ord(ch) in table else ord(ch))
        return out.decode("utf-8", errors="replace")

    text = b"".join(vocab[i] for i in ids)
    return to_utf8(text)


if not OUT.exists():
    ids = tokenize_ids(RAW, GGUF)          # ids[0] = BOS
    window = ids[8193:16385]               # the sweep's chunk-2 window
    padding = ids[16385:24577]
    OUT.write_text(detokenize(window + padding, GGUF), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

# round-trip verification: re-tokenizing the file must give [BOS] + window ids
fresh = tokenize_ids(OUT, GGUF)
window = tokenize_ids(RAW, GGUF)[8193:16385]
if fresh[:1 + len(window)] != [fresh[0]] + window and fresh[1:1 + len(window)] != window:
    raise SystemExit("round-trip verification FAILED - the window does not "
                     "re-tokenize to ids[8193:16385]")
print("round-trip OK:", len(fresh), "tokens")
