#include "moe-dump.h"

#include "common.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Binary stream layout (all little-endian; host is assumed LE)
//
//   File header (written once, before the first record):
//     char[8]  magic       = "MOEDUMP\x01"
//     u32      version     = 1
//     u32      flags       (informational; per-record flags are authoritative)
//     u32      n_expert        (best-effort, 0 if unknown when header flushed)
//     u32      n_expert_used   (best-effort)
//     u32      n_embd          (best-effort)
//     u32      reserved
//
//   Every record is framed as: [u8 type][u32 payload_len][payload...]
//   so a reader can stream/tail and skip unknown record types.
//
//   type 0x01  UBATCH header (one per ubatch = one graph forward pass):
//     u32 batch_id
//     u32 n_tokens
//     u64 token_base           (global running token index of first token)
//     n_tokens * { i32 pos; i32 seq_id; u8 phase }   phase: 0=prefill 1=decode 2=other
//
//   type 0x02  LAYER block (one per MoE layer per ubatch; rows follow ubatch order):
//     u32 batch_id
//     u16 layer
//     u32 n_tokens
//     u16 n_expert
//     u16 n_expert_used
//     u8  rflags               (bit0: logits present, bit1: sel_probs present,
//                               bit2: shexp present)
//     f32 probs   [n_tokens * n_expert]        full gate distribution (all experts)
//    [f32 logits  [n_tokens * n_expert]]       if bit0  raw router logits
//    [f32 sel     [n_tokens * n_expert]]       if bit1  selection-driving probs
//                                                       (biased / group-masked)
//     i32 ids     [n_tokens * n_expert_used]   selected expert ids
//     f32 weights [n_tokens * n_expert_used]   selected experts' gate weights (g_j)
//     f32 out_l2  [n_tokens * n_expert_used]   L2 norm of each selected expert output (|f_j|_2)
//    [f32 shexp   [n_tokens]]                  if bit2  L2 norm of shared-expert output
//
//   type 0x03  ACCEPT (speculative/MTP/DFlash accept-reject, decode only):
//     u32 batch_id             (most recent batch id at emit time)
//     i32 seq_id
//     u32 n
//     n * { i32 pos; u8 accepted }
// ============================================================================

static constexpr uint8_t REC_UBATCH = 0x01;
static constexpr uint8_t REC_LAYER  = 0x02;
static constexpr uint8_t REC_ACCEPT = 0x03;

// which optional full-width tensor a value comes from
enum sel_src { SEL_NONE = 0, SEL_BIASED = 1, SEL_MASKED = 2 };

// read one element of a ggml tensor as float, honoring type and strides
static float md_get_f32(const uint8_t * data, ggml_type type, const size_t * nb,
                        int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    switch (type) {
        case GGML_TYPE_F32:  return *(const float    *) &data[i];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
        case GGML_TYPE_I64:  return (float) *(const int64_t *) &data[i];
        case GGML_TYPE_I32:  return (float) *(const int32_t *) &data[i];
        case GGML_TYPE_I16:  return (float) *(const int16_t *) &data[i];
        case GGML_TYPE_I8:   return (float) *(const int8_t  *) &data[i];
        default: GGML_ABORT("moe-dump: unsupported tensor type %s", ggml_type_name(type));
    }
}

// tensor names we want to observe (called in the cheap `ask` phase)
static bool md_want(const char * name) {
    if (name[0] == 'f') {
        if (!strncmp(name, "ffn_moe_probs",   13)) return true; // + _biased / _masked
        if (!strncmp(name, "ffn_moe_logits-", 15)) return true; // plain logits only
        if (!strncmp(name, "ffn_moe_topk-",   13)) return true;
        if (!strncmp(name, "ffn_moe_weights", 15)) return true; // any weights variant
        if (!strncmp(name, "ffn_moe_down-",   13)) return true;
        if (!strncmp(name, "ffn_shexp-",      10)) return true;
    }
    return false;
}

// split "name-<il>" into base + layer index; il = -1 if no numeric suffix
static void md_parse_name(const char * name, std::string & base, int & il) {
    const char * dash = strrchr(name, '-');
    if (dash && dash[1]) {
        bool digits = true;
        for (const char * p = dash + 1; *p; ++p) {
            if (!isdigit((unsigned char) *p)) { digits = false; break; }
        }
        if (digits) {
            base.assign(name, dash - name);
            il = atoi(dash + 1);
            return;
        }
    }
    base = name;
    il   = -1;
}

struct common_moe_dump::impl {
    FILE * f = nullptr;
    bool   header_written = false;

    // best-effort dims for the file header
    uint32_t n_expert = 0, n_expert_used = 0, n_embd = 0;

    // global counters
    uint64_t token_global = 0;
    uint32_t batch_id     = 0;
    uint32_t batch_seq    = 0; // incremented every begin_batch

    std::vector<uint8_t> scratch; // GPU->host staging

    // ---- current batch (one target-model llama_decode) ----
    bool                  batch_open = false;
    std::vector<int32_t>  b_pos;
    std::vector<int32_t>  b_seq;
    std::vector<uint8_t>  b_phase;
    size_t                seq_cursor = 0; // batch tokens consumed by prior ubatches

    // ---- current ubatch ----
    bool                  ub_open           = false;
    bool                  ub_header_written = false;
    uint32_t              ub_n_tokens       = 0;
    uint64_t              ub_token_base     = 0;
    std::vector<int32_t>  ub_pos;
    std::vector<int32_t>  ub_seq;
    std::vector<uint8_t>  ub_phase;

    // ---- current open layer ----
    int      cur_il          = -1;
    uint32_t l_n_tokens      = 0;
    uint32_t l_n_expert      = 0;
    uint32_t l_n_expert_used = 0;
    bool     have_probs = false, have_logits = false, have_sel = false;
    bool     have_topk = false, have_weights = false, have_down = false, have_shexp = false;
    int      sel_priority = SEL_NONE;
    std::vector<float>   l_probs;
    std::vector<float>   l_logits;
    std::vector<float>   l_sel;
    std::vector<int32_t> l_ids;
    std::vector<float>   l_weights;
    std::vector<float>   l_out_l2;
    std::vector<float>   l_shexp;

    // ---------- small write helpers ----------
    template <class T> void wr(const T & v) { fwrite(&v, sizeof(T), 1, f); }
    void wrn(const void * p, size_t n)      { if (n) fwrite(p, 1, n, f); }

    void ensure_header() {
        if (header_written) return;
        const char magic[8] = { 'M','O','E','D','U','M','P','\x01' };
        fwrite(magic, 1, 8, f);
        wr<uint32_t>(1);          // version
        wr<uint32_t>(0);          // flags
        wr<uint32_t>(n_expert);
        wr<uint32_t>(n_expert_used);
        wr<uint32_t>(n_embd);
        wr<uint32_t>(0);          // reserved
        header_written = true;
    }

    const uint8_t * host(const ggml_tensor * t) {
        if (ggml_backend_buffer_is_host(t->buffer)) {
            return (const uint8_t *) t->data;
        }
        const size_t nb = ggml_nbytes(t);
        scratch.resize(nb);
        ggml_backend_tensor_get(t, scratch.data(), 0, nb);
        return scratch.data();
    }

    // ---------- batch / ubatch lifecycle ----------
    void begin_batch(const int32_t * pos, const int32_t * seq_id, const uint8_t * is_decode, int32_t n) {
        // flush anything left open from a previous (unterminated) batch
        flush_layer();
        b_pos.assign(n, -1);
        b_seq.assign(n, -1);
        b_phase.assign(n, COMMON_MOE_DUMP_PHASE_OTHER);
        for (int32_t i = 0; i < n; ++i) {
            if (pos)       b_pos[i]   = pos[i];
            if (seq_id)    b_seq[i]   = seq_id[i];
            if (is_decode) b_phase[i] = is_decode[i] ? COMMON_MOE_DUMP_PHASE_DECODE
                                                     : COMMON_MOE_DUMP_PHASE_PREFILL;
        }
        seq_cursor = 0;
        batch_id   = ++batch_seq;
        batch_open = true;
        ub_open    = false;
        cur_il     = -1;
    }

    void begin_ubatch(uint32_t n) {
        flush_layer();                 // flush last layer of the previous ubatch
        ub_open           = true;
        ub_header_written = false;
        ub_n_tokens       = n;
        ub_token_base     = token_global;
        token_global     += n;
        cur_il            = -1;
        ub_pos.assign(n, -1);
        ub_seq.assign(n, -1);
        ub_phase.assign(n, COMMON_MOE_DUMP_PHASE_OTHER);
        for (uint32_t i = 0; i < n; ++i) {
            const size_t j = seq_cursor + i;
            if (j < b_pos.size())   ub_pos[i]   = b_pos[j];
            if (j < b_seq.size())   ub_seq[i]   = b_seq[j];
            if (j < b_phase.size()) ub_phase[i] = b_phase[j];
        }
        seq_cursor += n;
    }

    void write_ubatch_header() {
        if (ub_header_written) return;
        ensure_header();
        const uint32_t n   = ub_n_tokens;
        const uint32_t len = 4 + 4 + 8 + n * (4 + 4 + 1);
        wr<uint8_t>(REC_UBATCH);
        wr<uint32_t>(len);
        wr<uint32_t>(batch_id);
        wr<uint32_t>(n);
        wr<uint64_t>(ub_token_base);
        for (uint32_t i = 0; i < n; ++i) {
            wr<int32_t>(ub_pos[i]);
            wr<int32_t>(ub_seq[i]);
            wr<uint8_t>(ub_phase[i]);
        }
        ub_header_written = true;
    }

    void reset_layer() {
        have_probs = have_logits = have_sel = false;
        have_topk = have_weights = have_down = have_shexp = false;
        sel_priority    = SEL_NONE;
        l_n_tokens      = 0;
        l_n_expert      = 0;
        l_n_expert_used = 0;
    }

    // decide whether an incoming layer tensor starts a new ubatch, then open the
    // right layer buffer. is_probs marks the canonical "first tensor of a layer".
    void feed(int il, uint32_t ntok, bool is_probs) {
        bool restart = false;
        if (!ub_open)                                    restart = true;
        else if (il < cur_il)                            restart = true;   // layer index reset => new ubatch
        else if (is_probs && il == cur_il && have_probs) restart = true;   // single-MoE-layer models
        if (restart) begin_ubatch(ntok);
        if (cur_il != il) { flush_layer(); cur_il = il; reset_layer(); }
        if (l_n_tokens == 0) l_n_tokens = ntok;
    }

    void flush_layer() {
        if (cur_il < 0) { return; }
        if (!(have_probs && have_topk && have_down)) { reset_layer(); return; }
        write_ubatch_header();

        const uint32_t nt  = l_n_tokens;
        const uint32_t ne  = l_n_expert;
        const uint32_t neu = l_n_expert_used;

        uint8_t rflags = 0;
        if (have_logits) rflags |= 1;
        if (have_sel)    rflags |= 2;
        if (have_shexp)  rflags |= 4;

        uint32_t len = 4 + 2 + 4 + 2 + 2 + 1;
        len += (uint32_t) nt * ne * 4;                       // probs
        if (have_logits) len += (uint32_t) nt * ne * 4;
        if (have_sel)    len += (uint32_t) nt * ne * 4;
        len += (uint32_t) nt * neu * 4;                      // ids
        len += (uint32_t) nt * neu * 4;                      // weights
        len += (uint32_t) nt * neu * 4;                      // out_l2
        if (have_shexp)  len += (uint32_t) nt * 4;

        wr<uint8_t>(REC_LAYER);
        wr<uint32_t>(len);
        wr<uint32_t>(batch_id);
        wr<uint16_t>((uint16_t) cur_il);
        wr<uint32_t>(nt);
        wr<uint16_t>((uint16_t) ne);
        wr<uint16_t>((uint16_t) neu);
        wr<uint8_t>(rflags);

        wrn(l_probs.data(), (size_t) nt * ne * 4);
        if (have_logits) wrn(l_logits.data(), (size_t) nt * ne * 4);
        if (have_sel)    wrn(l_sel.data(),    (size_t) nt * ne * 4);
        wrn(l_ids.data(),     (size_t) nt * neu * 4);
        wrn(l_weights.data(), (size_t) nt * neu * 4);
        wrn(l_out_l2.data(),  (size_t) nt * neu * 4);
        if (have_shexp)  wrn(l_shexp.data(), (size_t) nt * 4);

        reset_layer();
    }

    // ---------- per-tensor collectors ----------
    // full distribution over all experts: probs / logits.  t: [n_expert, n_tokens]
    void store_full(const ggml_tensor * t, int il, bool is_probs) {
        const uint32_t ne = (uint32_t) t->ne[0];
        const uint32_t nt = (uint32_t) t->ne[1];
        feed(il, nt, is_probs);
        auto & dst = is_probs ? l_probs : l_logits;
        dst.resize((size_t) nt * ne);
        const uint8_t * data = host(t);
        const bool fast = (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float));
        for (uint32_t it = 0; it < nt; ++it) {
            if (fast) {
                const float * row = (const float *) (data + it * t->nb[1]);
                for (uint32_t e = 0; e < ne; ++e) dst[(size_t) it * ne + e] = row[e];
            } else {
                for (uint32_t e = 0; e < ne; ++e)
                    dst[(size_t) it * ne + e] = md_get_f32(data, t->type, t->nb, e, it, 0, 0);
            }
        }
        if (is_probs) { l_n_expert = ne; have_probs = true; if (ne > n_expert) n_expert = ne; }
        else          { l_n_expert = ne; have_logits = true; }
    }

    // selection-driving probs (biased / group-masked). t: [n_expert, n_tokens]
    void store_sel(const ggml_tensor * t, int il, int prio) {
        const uint32_t ne = (uint32_t) t->ne[0];
        const uint32_t nt = (uint32_t) t->ne[1];
        feed(il, nt, false);
        if (prio < sel_priority) return; // keep the strongest (masked > biased)
        l_sel.resize((size_t) nt * ne);
        const uint8_t * data = host(t);
        const bool fast = (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float));
        for (uint32_t it = 0; it < nt; ++it) {
            if (fast) {
                const float * row = (const float *) (data + it * t->nb[1]);
                for (uint32_t e = 0; e < ne; ++e) l_sel[(size_t) it * ne + e] = row[e];
            } else {
                for (uint32_t e = 0; e < ne; ++e)
                    l_sel[(size_t) it * ne + e] = md_get_f32(data, t->type, t->nb, e, it, 0, 0);
            }
        }
        sel_priority = prio;
        have_sel     = true;
    }

    // selected expert ids. t: [n_expert_used, n_tokens], I32
    void store_ids(const ggml_tensor * t, int il) {
        const uint32_t neu = (uint32_t) t->ne[0];
        const uint32_t nt  = (uint32_t) t->ne[1];
        feed(il, nt, false);
        l_ids.resize((size_t) nt * neu);
        const uint8_t * data = host(t);
        for (uint32_t it = 0; it < nt; ++it)
            for (uint32_t s = 0; s < neu; ++s)
                l_ids[(size_t) it * neu + s] = (int32_t) md_get_f32(data, t->type, t->nb, s, it, 0, 0);
        l_n_expert_used = neu;
        have_topk       = true;
        if (neu > n_expert_used) n_expert_used = neu;
    }

    // selected experts' gate weights. Normally [1, n_expert_used, n_tokens], but
    // tolerate a 2-D [n_expert_used, n_tokens] layout. Keep the latest variant seen.
    void store_weights(const ggml_tensor * t, int il) {
        const bool     lead1 = (t->ne[0] == 1);           // leading unit dim?
        const uint32_t neu   = (uint32_t) (lead1 ? t->ne[1] : t->ne[0]);
        const uint32_t nt    = (uint32_t) (lead1 ? t->ne[2] : t->ne[1]);
        feed(il, nt, false);
        l_weights.resize((size_t) nt * neu);
        const uint8_t * data = host(t);
        for (uint32_t it = 0; it < nt; ++it)
            for (uint32_t s = 0; s < neu; ++s)
                l_weights[(size_t) it * neu + s] =
                    lead1 ? md_get_f32(data, t->type, t->nb, 0, s, it, 0)
                          : md_get_f32(data, t->type, t->nb, s, it, 0, 0);
        l_n_expert_used = neu;
        have_weights    = true;
    }

    // per-expert raw output; reduce to L2 norm. t: [n_embd, n_expert_used, n_tokens]
    void store_down(const ggml_tensor * t, int il) {
        const uint32_t nem = (uint32_t) t->ne[0];
        const uint32_t neu = (uint32_t) t->ne[1];
        const uint32_t nt  = (uint32_t) t->ne[2];
        feed(il, nt, false);
        l_out_l2.resize((size_t) nt * neu);
        const uint8_t * data = host(t);
        const bool fast = (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float));
        for (uint32_t it = 0; it < nt; ++it) {
            for (uint32_t s = 0; s < neu; ++s) {
                double acc = 0.0;
                if (fast) {
                    const float * v = (const float *) (data + it * t->nb[2] + s * t->nb[1]);
                    for (uint32_t e = 0; e < nem; ++e) acc += (double) v[e] * v[e];
                } else {
                    for (uint32_t e = 0; e < nem; ++e) {
                        const float v = md_get_f32(data, t->type, t->nb, e, s, it, 0);
                        acc += (double) v * v;
                    }
                }
                l_out_l2[(size_t) it * neu + s] = (float) std::sqrt(acc);
            }
        }
        have_down = true;
        if (nem > n_embd) n_embd = nem;
    }

    // shared-expert raw output; reduce to L2 norm per token. t: [n_embd, n_tokens]
    void store_shexp(const ggml_tensor * t, int il) {
        const uint32_t nem = (uint32_t) t->ne[0];
        const uint32_t nt  = (uint32_t) t->ne[1];
        feed(il, nt, false);
        l_shexp.resize(nt);
        const uint8_t * data = host(t);
        const bool fast = (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float));
        for (uint32_t it = 0; it < nt; ++it) {
            double acc = 0.0;
            if (fast) {
                const float * v = (const float *) (data + it * t->nb[1]);
                for (uint32_t e = 0; e < nem; ++e) acc += (double) v[e] * v[e];
            } else {
                for (uint32_t e = 0; e < nem; ++e) {
                    const float v = md_get_f32(data, t->type, t->nb, e, it, 0, 0);
                    acc += (double) v * v;
                }
            }
            l_shexp[it] = (float) std::sqrt(acc);
        }
        have_shexp = true;
    }

    void observe(const ggml_tensor * t) {
        std::string base;
        int il;
        md_parse_name(t->name, base, il);
        if (il < 0) return;

        if      (base == "ffn_moe_probs")        store_full(t, il, true);
        else if (base == "ffn_moe_logits")       store_full(t, il, false);
        else if (base == "ffn_moe_probs_masked") store_sel(t, il, SEL_MASKED);
        else if (base == "ffn_moe_probs_biased") store_sel(t, il, SEL_BIASED);
        else if (base == "ffn_moe_topk")         store_ids(t, il);
        else if (base.rfind("ffn_moe_weights", 0) == 0) store_weights(t, il);
        else if (base == "ffn_moe_down")         store_down(t, il);
        else if (base == "ffn_shexp")            store_shexp(t, il);
    }

    void end_batch() {
        flush_layer();
        if (f) fflush(f);
        batch_open = false;
        ub_open    = false;
        cur_il     = -1;
    }

    void mark_accept(int32_t seq_id, const int32_t * pos, const uint8_t * accepted, int32_t n) {
        if (n <= 0) return;
        ensure_header();
        const uint32_t len = 4 + 4 + 4 + (uint32_t) n * (4 + 1);
        wr<uint8_t>(REC_ACCEPT);
        wr<uint32_t>(len);
        wr<uint32_t>(batch_id);
        wr<int32_t>(seq_id);
        wr<uint32_t>((uint32_t) n);
        for (int32_t i = 0; i < n; ++i) {
            wr<int32_t>(pos ? pos[i] : -1);
            wr<uint8_t>(accepted ? accepted[i] : 0);
        }
        if (f) fflush(f);
    }
};

bool common_moe_dump_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (common_moe_dump *) user_data;
    auto * p = d->pimpl.get();
    if (!p->batch_open) {
        return false; // not inside a target-model decode we are tracking
    }
    if (ask) {
        return md_want(t->name);
    }
    if (md_want(t->name)) {
        p->observe(t);
    }
    return true; // must return true, else the graph compute is aborted
}

common_moe_dump::common_moe_dump(common_params & params, const std::string & path)
    : pimpl(std::make_unique<impl>()) {
    pimpl->f = fopen(path.c_str(), "wb");
    if (!pimpl->f) {
        throw std::runtime_error("moe-dump: failed to open '" + path + "' for writing");
    }
    setvbuf(pimpl->f, nullptr, _IOFBF, 1 << 20);

    params.cb_eval           = common_moe_dump_cb_eval;
    params.cb_eval_user_data = this;
    params.warmup            = false; // avoid dumping warmup passes

    LOG_INF("%s: dumping per-token MoE routing to '%s'\n", __func__, path.c_str());
}

common_moe_dump::~common_moe_dump() {
    if (pimpl && pimpl->f) {
        pimpl->flush_layer();
        fflush(pimpl->f);
        fclose(pimpl->f);
        pimpl->f = nullptr;
    }
}

void common_moe_dump::begin_batch(const int32_t * pos, const int32_t * seq_id, const uint8_t * is_decode, int32_t n_tokens) {
    pimpl->begin_batch(pos, seq_id, is_decode, n_tokens);
}

void common_moe_dump::end_batch() {
    pimpl->end_batch();
}

void common_moe_dump::mark_accept(int32_t seq_id, const int32_t * pos, const uint8_t * accepted, int32_t n) {
    pimpl->mark_accept(seq_id, pos, accepted, n);
}
