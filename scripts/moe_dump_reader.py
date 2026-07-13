#!/usr/bin/env python3
"""Reader for the binary MoE routing dump produced by llama-server --moe-dump-file.

The dump captures, per token and per MoE layer, the full routing state of the
*target* model during inference:

  - the full gate probability distribution over ALL experts   (probs)
  - (optional) the raw router logits                          (logits)
  - (optional) the selection-driving probs, biased/masked     (sel_probs)
  - the selected expert ids                                   (expert_ids)
  - the selected experts' gate weights   g_j(x)               (gate_weights)
  - the L2 norm of each selected expert's output  |f_j(x)|_2  (out_l2)
  - (optional) the L2 norm of the shared-expert output        (shexp_l2)

plus, per token, a prefill/decode tag, sequence position, sequence id and an
incrementing (batch, token) index, and — for speculative / MTP / DFlash decode —
an accept/reject flag (emitted as separate records, joined on (seq_id, pos)).

The REAP saliency criterion (https://arxiv.org/abs/2510.13999) is
    S_j = mean over tokens where j is selected of ( g_j(x) * |f_j(x)|_2 )
i.e. exactly gate_weights * out_l2 averaged per expert; see reap_saliency().

The stream is flushed once per batch by the writer, so it can be read live while
inference runs: pass follow=True (or --follow on the CLI) to tail it.

Format: see common/moe-dump.cpp for the authoritative layout. Little-endian.
"""

import argparse
import struct
import sys
import time

import numpy as np

MAGIC = b"MOEDUMP\x01"

REC_UBATCH = 0x01
REC_LAYER = 0x02
REC_ACCEPT = 0x03

PHASE_PREFILL = 0
PHASE_DECODE = 1
PHASE_OTHER = 2
PHASE_NAME = {0: "prefill", 1: "decode", 2: "other"}


class _Stream:
    """Sequential byte reader that can optionally block waiting for more data."""

    def __init__(self, f, follow, poll=0.05):
        self.f = f
        self.follow = follow
        self.poll = poll

    def read(self, n):
        """Return exactly n bytes, or None at a clean record boundary EOF.

        In follow mode, blocks (polling) until n bytes are available. This makes
        partial records at the tail of a still-growing file safe to handle.
        """
        buf = b""
        while len(buf) < n:
            chunk = self.f.read(n - len(buf))
            if chunk:
                buf += chunk
                continue
            # no bytes available right now
            if len(buf) == 0 and not self.follow:
                return None  # clean EOF between records
            if not self.follow:
                # truncated record at EOF of a non-growing file
                return None
            time.sleep(self.poll)
        return buf


def _u8(b, o):
    return b[o], o + 1


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0], o + 2


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0], o + 4


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0], o + 8


def _i32(b, o):
    return struct.unpack_from("<i", b, o)[0], o + 4


def _parse_ubatch(payload):
    o = 0
    batch_id, o = _u32(payload, o)
    n, o = _u32(payload, o)
    token_base, o = _u64(payload, o)
    # per token: i32 pos, i32 seq_id, u8 phase  (9 bytes each)
    rec = np.frombuffer(payload, dtype=np.uint8, count=n * 9, offset=o)
    rec = rec.reshape(n, 9)
    pos = rec[:, 0:4].copy().view(np.int32).reshape(n)
    seq = rec[:, 4:8].copy().view(np.int32).reshape(n)
    phase = rec[:, 8].copy()
    return {
        "type": "ubatch",
        "batch_id": batch_id,
        "n_tokens": n,
        "token_base": token_base,
        "pos": pos,
        "seq_id": seq,
        "phase": phase,
    }


def _parse_layer(payload):
    o = 0
    batch_id, o = _u32(payload, o)
    layer, o = _u16(payload, o)
    nt, o = _u32(payload, o)
    ne, o = _u16(payload, o)
    neu, o = _u16(payload, o)
    rflags, o = _u8(payload, o)
    have_logits = bool(rflags & 1)
    have_sel = bool(rflags & 2)
    have_shexp = bool(rflags & 4)

    def take_f32(count):
        nonlocal o
        a = np.frombuffer(payload, dtype="<f4", count=count, offset=o)
        o += count * 4
        return a

    def take_i32(count):
        nonlocal o
        a = np.frombuffer(payload, dtype="<i4", count=count, offset=o)
        o += count * 4
        return a

    probs = take_f32(nt * ne).reshape(nt, ne)
    logits = take_f32(nt * ne).reshape(nt, ne) if have_logits else None
    sel = take_f32(nt * ne).reshape(nt, ne) if have_sel else None
    ids = take_i32(nt * neu).reshape(nt, neu)
    weights = take_f32(nt * neu).reshape(nt, neu)
    out_l2 = take_f32(nt * neu).reshape(nt, neu)
    shexp = take_f32(nt) if have_shexp else None

    return {
        "type": "layer",
        "batch_id": batch_id,
        "layer": layer,
        "n_tokens": nt,
        "n_expert": ne,
        "n_expert_used": neu,
        "probs": probs,
        "logits": logits,
        "sel_probs": sel,
        "expert_ids": ids,
        "gate_weights": weights,
        "out_l2": out_l2,
        "shexp_l2": shexp,
    }


def _parse_accept(payload):
    o = 0
    batch_id, o = _u32(payload, o)
    seq_id, o = _i32(payload, o)
    n, o = _u32(payload, o)
    rec = np.frombuffer(payload, dtype=np.uint8, count=n * 5, offset=o).reshape(n, 5)
    pos = rec[:, 0:4].copy().view(np.int32).reshape(n)
    accepted = rec[:, 4].copy().astype(bool)
    return {
        "type": "accept",
        "batch_id": batch_id,
        "seq_id": seq_id,
        "pos": pos,
        "accepted": accepted,
    }


_PARSERS = {
    REC_UBATCH: _parse_ubatch,
    REC_LAYER: _parse_layer,
    REC_ACCEPT: _parse_accept,
}


def read_header(stream):
    hdr = stream.read(8 + 6 * 4)
    if hdr is None or len(hdr) < 8 + 6 * 4:
        raise EOFError("could not read file header (empty/short file)")
    if hdr[:8] != MAGIC:
        raise ValueError("bad magic; not a MoE dump file")
    version, flags, n_expert, n_expert_used, n_embd, _res = struct.unpack_from("<6I", hdr, 8)
    return {
        "type": "header",
        "version": version,
        "flags": flags,
        "n_expert": n_expert,
        "n_expert_used": n_expert_used,
        "n_embd": n_embd,
    }


def read_records(path, follow=False, emit_header=True):
    """Yield records (dicts) from a dump file. If follow=True, tail it forever."""
    with open(path, "rb") as f:
        stream = _Stream(f, follow)
        header = read_header(stream)
        if emit_header:
            yield header
        while True:
            frame = stream.read(5)
            if frame is None:
                return  # clean EOF (non-follow)
            rtype = frame[0]
            (length,) = struct.unpack_from("<I", frame, 1)
            payload = stream.read(length)
            if payload is None:
                return
            parser = _PARSERS.get(rtype)
            if parser is not None:
                yield parser(payload)
            # unknown record types are silently skipped (payload already consumed)


def reap_saliency(path):
    """Compute REAP saliency S_j = mean( g_j * |f_j|_2 ) per (layer, expert).

    Returns a dict layer -> (saliency[n_expert], counts[n_expert]).
    """
    acc = {}   # layer -> np.float64[n_expert]
    cnt = {}   # layer -> np.int64[n_expert]
    n_expert = None
    for rec in read_records(path, follow=False):
        if rec["type"] == "header":
            n_expert = rec["n_expert"] or None
            continue
        if rec["type"] != "layer":
            continue
        ne = rec["n_expert"]
        il = rec["layer"]
        if il not in acc:
            acc[il] = np.zeros(ne, dtype=np.float64)
            cnt[il] = np.zeros(ne, dtype=np.int64)
        ids = rec["expert_ids"].reshape(-1)
        contrib = (rec["gate_weights"] * rec["out_l2"]).reshape(-1).astype(np.float64)
        np.add.at(acc[il], ids, contrib)
        np.add.at(cnt[il], ids, 1)
    out = {}
    for il in acc:
        with np.errstate(invalid="ignore", divide="ignore"):
            sal = np.where(cnt[il] > 0, acc[il] / np.maximum(cnt[il], 1), 0.0)
        out[il] = (sal, cnt[il])
    return out


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="path to the .moe dump file")
    ap.add_argument("--follow", "-f", action="store_true", help="tail the file in real time")
    ap.add_argument("--reap", action="store_true", help="compute and print REAP saliency per layer/expert")
    ap.add_argument("--limit", type=int, default=0, help="stop after N records (0 = unlimited)")
    args = ap.parse_args()

    if args.reap:
        sal = reap_saliency(args.path)
        for il in sorted(sal):
            s, c = sal[il]
            order = np.argsort(s)
            active = int((c > 0).sum())
            print(f"layer {il:3d}: {active}/{len(s)} experts active")
            print(f"  least salient experts (prune first): {order[:8].tolist()}")
            print(f"  most  salient experts:               {order[::-1][:8].tolist()}")
        return

    n = 0
    for rec in read_records(args.path, follow=args.follow):
        t = rec["type"]
        if t == "header":
            print(f"header: version={rec['version']} n_expert={rec['n_expert']} "
                  f"n_expert_used={rec['n_expert_used']} n_embd={rec['n_embd']}")
        elif t == "ubatch":
            phases = ",".join(sorted({PHASE_NAME.get(int(p), '?') for p in rec["phase"]}))
            print(f"ubatch batch={rec['batch_id']} n_tokens={rec['n_tokens']} "
                  f"token_base={rec['token_base']} phase={{{phases}}} "
                  f"seq={np.unique(rec['seq_id']).tolist()}")
        elif t == "layer":
            print(f"  layer batch={rec['batch_id']} il={rec['layer']:3d} "
                  f"n_tokens={rec['n_tokens']} n_expert={rec['n_expert']} "
                  f"n_used={rec['n_expert_used']} "
                  f"top0_expert(tok0)={int(rec['expert_ids'][0,0])} "
                  f"out_l2(tok0,slot0)={rec['out_l2'][0,0]:.4f}"
                  + ("" if rec['shexp_l2'] is None else f" shexp_l2(tok0)={rec['shexp_l2'][0]:.4f}"))
        elif t == "accept":
            print(f"  accept batch={rec['batch_id']} seq={rec['seq_id']} "
                  f"pos={rec['pos'].tolist()} accepted={rec['accepted'].astype(int).tolist()}")
        n += 1
        if args.limit and n >= args.limit:
            break


if __name__ == "__main__":
    try:
        _cli()
    except KeyboardInterrupt:
        sys.exit(130)
