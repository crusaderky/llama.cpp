# Per-token MoE routing dump

`llama-server --moe-dump-file <path>` records, for **every token and every MoE
layer** of the target model, the full expert-routing state during inference. It
is intended for MoE research: expert pruning/merging (e.g. REAP), layered
expert-cache modeling (disk → host RAM → VRAM), and training expert-prefetch
predictors (e.g. MoE-SpeQ).

The feature is fully opt-in and has **zero overhead when the flag is not set**
(the eval callback is never installed, so a single build works either way — no
recompilation is needed to turn it on or off). When it *is* set, expect a large
file and a significant slowdown: every selected expert's output vector is copied
from the compute backend to host memory and reduced.

## What is recorded

Per token, per MoE layer:

| Field | Source tensor | Notes |
|---|---|---|
| full gate distribution over **all** experts | `ffn_moe_probs` | one float per expert (e.g. all 256), not just the top-k |
| raw router logits (optional) | `ffn_moe_logits` | pre-activation |
| selection-driving probs (optional) | `ffn_moe_probs_biased` / `_masked` | DeepSeek-V3 / DFlash group routing |
| selected expert ids | `ffn_moe_topk` | maps each output norm to its expert # |
| selected experts' gate weights `g_j` | `ffn_moe_weights*` | post-renormalization weight actually used |
| **L2 norm of each selected expert output** `‖f_j‖₂` | `ffn_moe_down` | the raw down-projection output, reduced per expert |
| shared-expert output L2 (optional) | `ffn_shexp` | always-on expert(s), no routing |

Per token: `prefill`/`decode` tag, sequence position, sequence id (slot), an
incrementing batch id, and a global token index.

For speculative / MTP / DFlash decode: an **accept/reject** flag per verified
token (emitted as separate records and joined on `(seq_id, pos)`).

> **REAP metric.** The REAP saliency criterion
> ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)) is
> `S_j = mean over selected tokens of ( g_j(x) · ‖f_j(x)‖₂ )` — the *L2 norm*
> of the expert output (not the mean), weighted by the gate value. Both terms
> are dumped per token, so `S_j` is a one-line reduction offline (see
> `reap_saliency()` in the reader) while the full temporal series is preserved
> for the other use cases.

Non-MoE (dense) layers produce no records; a gap in layer indices marks them.

## Usage

```bash
# capture (add your usual model / offload / speculative flags)
llama-server -m model.gguf --moe-dump-file routing.moe

# inspect live while the server runs
python scripts/moe_dump_reader.py routing.moe --follow

# compute REAP saliency per layer/expert from a finished capture
python scripts/moe_dump_reader.py routing.moe --reap
```

The stream is flushed once per batch, so it can be read/tailed in real time.

Programmatic use:

```python
from scripts.moe_dump_reader import read_records, reap_saliency

for rec in read_records("routing.moe", follow=True):
    if rec["type"] == "layer":
        probs   = rec["probs"]         # [n_tokens, n_expert]  full gate distribution
        ids     = rec["expert_ids"]    # [n_tokens, n_expert_used]
        weights = rec["gate_weights"]  # [n_tokens, n_expert_used]  g_j
        out_l2  = rec["out_l2"]        # [n_tokens, n_expert_used]  ||f_j||_2
```

## Notes & limitations

- **Target model only.** The draft / MTP head context does not inherit the
  callback, so records are the target model's routing. Accept/reject flags tell
  you which candidate tokens (that all passed through the target model) were
  kept.
- **Per-token phase.** A single server batch can mix prompt tokens from one slot
  with generation tokens from another; the phase tag is therefore per token,
  captured when the token is added to the batch.
- **Sequence id / position.** `pos` and `seq_id` come from the server batch.
  Under the standard in-order ubatch split they align exactly; positions are
  authoritative.
- **Endianness.** The stream is little-endian (matches x86 / ARM hosts).
- The binary layout is documented at the top of `common/moe-dump.cpp`.
