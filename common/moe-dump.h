#pragma once

#include <cstdint>
#include <memory>
#include <string>

// MoE routing dump
// ----------------
// Captures, per token and per layer, the full MoE routing state of the *target*
// model during inference, by observing the compute graph through a
// ggml_backend_sched_eval_callback (the same mechanism used by tools/imatrix and
// common/debug.cpp). It is fully opt-in: when no dump object is created the
// callback is never installed and the graph runs unchanged (zero overhead).
//
// For every MoE layer of every token it records:
//   - the full gate probability distribution over ALL experts (ffn_moe_probs)
//   - (optional) the raw router logits          (ffn_moe_logits)
//   - (optional) the selection-driving probs    (ffn_moe_probs_biased/_masked)
//   - the selected expert ids                    (ffn_moe_topk)
//   - the selected experts' gate weights         (ffn_moe_weights*)
//   - the L2 norm of each selected expert's output vector (from ffn_moe_down);
//     this is exactly the |f_j(x)|_2 term of the REAP saliency criterion
//     S_j = mean_x( g_j(x) * |f_j(x)|_2 ).
//   - (optional) the L2 norm of the shared-expert output (ffn_shexp)
//
// Per token it also records a prefill/decode tag, position, sequence id and an
// incrementing (batch, token) index, plus (for speculative / MTP / DFlash decode)
// an accepted/rejected flag emitted separately by the driver.
//
// Output is a self-describing little-endian binary stream (see moe-dump.cpp for
// the exact layout, and scripts/moe_dump_reader.py for a reader). The stream is
// flushed at the end of every batch so it can be read/tailed in real time.

struct common_params;
struct ggml_tensor;

enum common_moe_dump_phase : uint8_t {
    COMMON_MOE_DUMP_PHASE_PREFILL = 0,
    COMMON_MOE_DUMP_PHASE_DECODE  = 1,
    COMMON_MOE_DUMP_PHASE_OTHER   = 2,
};

// ggml_backend_sched_eval_callback. user_data must be a common_moe_dump *.
bool common_moe_dump_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

struct common_moe_dump {
    struct impl;
    std::unique_ptr<impl> pimpl;

    // Opens `path` for writing (binary, truncated) and installs this object as
    // params.cb_eval / params.cb_eval_user_data, and sets params.warmup = false.
    // Throws std::runtime_error if the file cannot be opened.
    common_moe_dump(common_params & params, const std::string & path);
    ~common_moe_dump();

    common_moe_dump(const common_moe_dump &)             = delete;
    common_moe_dump & operator=(const common_moe_dump &) = delete;

    // Called by the driver right before llama_decode() of the target model.
    // Opens a new logical batch and provides per-token metadata for the tokens
    // of the batch (in batch order); the callback consumes this sequentially as
    // the graph is split into ubatches. Any of the arrays may be null.
    //   pos       : per-token sequence position       (length n_tokens)
    //   seq_id    : per-token sequence id (slot id)    (length n_tokens)
    //   is_decode : 1 => generation/decode token, 0 => prefill token (length n_tokens)
    void begin_batch(const int32_t * pos,
                     const int32_t * seq_id,
                     const uint8_t * is_decode,
                     int32_t         n_tokens);

    // Called by the driver right after llama_decode() returns. Flushes to disk.
    void end_batch();

    // Emit speculative/MTP/DFlash accept-reject flags for verified decode tokens
    // (point: last layer, decode only). Records are keyed by (seq_id, pos) and
    // tagged with the most recent batch id, so the reader can join them onto the
    // routing records. accepted[i] corresponds to sequence position pos[i].
    void mark_accept(int32_t         seq_id,
                     const int32_t * pos,
                     const uint8_t * accepted,
                     int32_t         n);
};
