# HANDOVER — pinned mmap weights & MoE upload/compute overlap

Working notes for branch `pin-mmap-host-register`. Written so this investigation can be
resumed without the original chat session.

**Base:** `e9fa0781f` (`master`, = `upstream/master` at the time, `ggml-org/llama.cpp`).
**All file:line references below are valid at this branch's base.** They will drift; grep
the quoted comment text instead if they no longer match. In particular they do **not**
match `beellama-staging`, which has extra code and different offsets.

---

## 0. TL;DR

Two independent optimisations for prefill when weights are spilled to host RAM
(`--cpu-moe` / `--n-cpu-moe` / `-ot ...=CPU`), taken from
`https://github.com/thecodacus/llama.cpp`, branch `fable5/prefetch-experts`:

1. **Pin mmap-backed CPU weights** — cherry-picked into this branch. Reactivates dead code
   so H2D copies from mmap'd weights DMA directly instead of staging through the CUDA
   driver's bounce buffer.
2. **Overlap expert uploads with compute** — *not* picked. Pipelines the per-layer expert
   H2D copies against compute on a second stream.

Change 1 is the more interesting one upstream, mostly because it unblocks someone else's
PR (`ggml-org/llama.cpp#21067`). Change 2 overlaps heavily with that same PR.

---

## 1. Repo state

```
remote  thecodacus   https://github.com/thecodacus/llama.cpp
branch  thecodacus/fable5/prefetch-experts     (tag: fable5-prefetch-experts-b9863-5e7f627)
branch  thecodacus/fable5/host-register        (change 1 alone, same commit)
```

Related, unexamined branches on that fork: `fable5/moe-expert-cache`,
`fable5/moe-cache-diagnostics`, `fable5/moe-cache-laguna` (keeping hot routed experts
resident in VRAM — a different idea, not evaluated here).

`fable5/prefetch-experts` = 4 commits on upstream `4fc4ec554`:

| commit | subject | picked here? |
|---|---|---|
| `20f5994bf` | llama : pin mmap-backed CPU weights for faster H2D uploads | **yes** → `985473a7b` |
| `1163cb349` | ggml : overlap offloaded expert weight uploads with compute | no |
| `5f83fbbe7` | ggml : size prefetch slots per layer and fix fallback use-after-free | no |
| `5e7f6271c` | docs(readme): usage + benchmark instructions | no |

The cherry-pick applied cleanly (one auto-merge in `llama-model-loader.cpp`). Both changed
TUs were syntax-checked against the base; they compile. **Nothing has been run or
benchmarked** — no numbers in this document are ours.

Origin of the lead: a YouTube video (`VytSYCDhWQ0`) describing the two changes as
Claude-generated.

---

## 2. Background — the two H2D paths

When an op is offloaded to the GPU but its weights live in host RAM, the weights are
copied H2D on **every eval**. How fast that copy is depends entirely on whether the source
pages are page-locked:

- **Pageable source** → `cudaMemcpy` cannot DMA. The driver memcpys into its own internal
  pinned bounce buffer first, then DMAs. Roughly half the achievable PCIe bandwidth, and
  it cannot be made properly asynchronous.
- **Pinned source** → direct DMA, and `cudaMemcpyAsync` is genuinely async.

llama.cpp already knows this. `make_cpu_buft_list()` (`src/llama-model.cpp:908-914`) adds
the device's *host* buffer type — CUDA pinned memory via `cudaHostAlloc`,
`ggml/src/ggml-cuda/ggml-cuda.cu:1291` — ahead of the plain CPU buffer type, with the
comment "storing the tensors in a host buffer is useful when the processing of large
batches is offloaded to a GPU device, since it reduces the time spent on data transfers".

**But it is disabled whenever mmap is on.** `src/llama-model-loader.cpp:1186`:

```cpp
// avoid using a host buffer when using mmap
auto * buft_dev = ggml_backend_buft_get_device(buft);
if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    buft = ggml_backend_dev_buffer_type(cpu_dev);   // plain pageable CPU buffer
}
```

and `use_mmap` covers both mmap modes (`src/llama-model-loader.cpp:545`):

```cpp
this->use_mmap = load_mode == LLAMA_LOAD_MODE_MMAP || load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
```

This is inherent, not an oversight: with mmap the tensor data is *already* in the mapped
region. Allocating a pinned host buffer would mean a full extra copy at load and double
the RAM, destroying the point of mmap. The two are mutually exclusive **by construction**
— which is exactly what change 1 works around, by pinning the mapping in place instead of
copying it into pinned memory.

Consequence today: `--load-mode mlock` (no mmap) is the *only* way to get pinned weights,
and it pays a full read-into-anonymous-pinned-buffer at load.

### Why prefill specifically

`ggml_backend_cuda_device_offload_op` (`ggml-cuda.cu:5184`) offloads an op to the GPU once
its batch is ≥ 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`, `ggml-cuda.cu:5357`). So spilled
weights are streamed H2D on every prefill batch, and prefill is H2D-bound. Decode stays
below the threshold, runs on CPU, and is unaffected — expect no decode change from any of
this.

---

## 3. Change 1 — pin mmap-backed CPU weights (`985473a7b`)

### What it does

`ggml_backend_cuda_register_host_buffer()` (`ggml-cuda.cu:4494`) calls
`cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly)`.
It is exposed via `get_proc_address` as `"ggml_backend_register_host_buffer"`
(`ggml-cuda.cu:5328`) — **and nothing in the tree calls it.** Verified by grepping the
whole source: it is reachable only through `get_proc_address` and has no consumer.

The commit adds one:

- `src/llama-mmap.cpp:634` — new `llama_mmap::register_host(first, last, reg_fn, unreg_fn)`,
  which expands `[first, last)` outward to page boundaries and calls `reg_fn`.
- `src/llama-mmap.cpp:622` — `~llama_mmap()` gains a body that calls `unreg_fn` before
  `pimpl` (and therefore the mapping) is destroyed. Correct: members are destroyed after
  the destructor body.
- `src/llama-model-loader.cpp:1665-1690` — at the end of `load_all_data`, walks the backend
  registry for the two proc addresses and registers `[mmap_used.first, mmap_used.second)`
  of each mapping, logging `"pinned %.2f MiB of mapped model memory..."`.

Placement is after all uploads, so load-time H2D does not benefit — only inference.

### Verified correct

- **Page math.** `align_range()` (`src/llama-mmap.cpp:478`) aligns *inward* when unmapping
  (first rounds up, last rounds down), so `unmap_fragment` retains the partial pages at
  both ends. `register_host`'s outward expansion lands exactly on those retained pages. No
  overlap with unmapped memory, no gap.
- **All-on-GPU case.** `mmaps_used` is initialised `{mapping->size(), 0}`
  (`src/llama-model-loader.cpp:1346`), so the `mmap_used.second > mmap_used.first` guard
  correctly skips a mapping with no CPU-resident tensors.
- **Portability of the ggml side.** `hip.h` and `musa.h` both `#define` `cudaHostRegister`,
  `cudaHostRegisterPortable` and `cudaHostRegisterReadOnly`, so ROCm/MUSA build fine.
- **Lifetime.** The mappings outlive the loader (moved into the model), so registering in
  `load_all_data` and unregistering in `~llama_mmap` is sound.

### The blocker: this code was disabled on purpose

`d0a71233f` / **PR #6206** (slaren, Mar 2024):

> Disables pinning the model when using mmap, since it can cause instability on some
> systems. Host pinned memory can still be used enabled by setting the environment
> variable `GGML_CUDA_REGISTER_HOST`, or with `--no-mmap`.

It fixes **issue #6149**, where the process hung at `llama_free_model`. The reporter
bisected it to the `cudaHostUnregister` call in `~llama_model()` and added *"my computer
still dramatically slows down afterward"*, on a 4 GB-VRAM laptop. The commit re-enables
that exact path and puts the unregister in a destructor — the same place it hung.

Note also that `GGML_CUDA_REGISTER_HOST` appears nowhere in `docs/`, `common/` or
`tools/`. It is an undocumented, dead env var.

### Claimed benefit (unverified by us)

Qwen3.6-35B-A3B, RTX 3060, pp2048: **1144 → 1385 t/s** (~21%). One model, one GPU.

---

## 4. Residency semantics — what each mode actually does

This is the part that is easy to get wrong.

| mode | H2D source | resident? | elastic under memory pressure? |
|---|---|---|---|
| `mmap` | pageable mapped pages | on demand | **yes** — pages fault in, get reclaimed under pressure, reload from disk |
| `mmap+mlock` | pageable mapped pages | locked | **no** — `mlock()` means never evict |
| `mlock` (no mmap) | **pinned** (`cudaHostAlloc`) | locked | no |
| `mmap` + this patch | **pinned** | locked | **no** — this is the behaviour change |
| `mmap+mlock` + this patch | **pinned** | locked | no — unchanged |

Key points:

- The self-managing/elastic behaviour belongs to plain `--load-mode mmap`, **not** to
  `mmap+mlock`. `llama_mlock::grow_to()` (`src/llama-mmap.cpp:777`) calls `raw_lock()` →
  `mlock()`, growing the locked region from the mapping's start to the high-water mark of
  CPU-resident tensors as they load. Locked pages are not reclaimed; under pressure the
  kernel evicts something else or OOMs.
- **Caveat that looks like elasticity:** if `RLIMIT_MEMLOCK` is too low, `raw_lock` fails,
  warns, sets `failed_already = true`, and loading continues **unlocked**. You then
  silently get `mmap` behaviour under a mode named `mlock`. (This is what
  `pixi-llm-recipes/scripts/install-memlock.sh` exists to prevent.)
- `cudaHostRegister` does not merely annotate pages — it faults them in and locks them. So
  applying the patch to plain `mmap` **removes** the elasticity that is the whole reason to
  choose that mode. Applying it to `mmap+mlock` changes nothing about residency.
- **Untested hypothesis worth measuring:** NVIDIA pins via its own kernel path
  (`get_user_pages`), which is believed *not* to be charged against `RLIMIT_MEMLOCK`, unlike
  `mlock()`. If true, registration buys locked residency without the ulimit raise.

### Correction to an earlier claim

An earlier draft of this analysis criticised the patch for pinning `[min, max]`, which
spans the GPU-resident tensors interleaved between CPU-resident ones (GGUF is layer-ordered,
so with `-ncmoe 26` the span swallows layers 0-25 whole, attention included).

**This critique is wrong, or at least not a new sin:** `grow_to` locks from **offset 0** to
the high-water mark, so `mlock` already over-locks the same holes and then some. Do not put
it on an upstream gap list.

---

## 5. Upstream assessment of change 1

### Is the idea worth a PR?

Yes. There is currently *no* way to have mmap and fast H2D simultaneously, and the pinned
host-buffer path is unconditionally disabled under mmap. But the commit as it stands is a
starting point, not a submittable PR.

### Recommended framing (better than a new flag)

**Make `--load-mode mmap+mlock` imply pinning.** Reasons:

- `mlock` already declares "I have the RAM, keep it resident", so the residency objection
  is void by construction — the strongest argument against re-enabling this evaporates.
- No new user-facing surface: no flag, no env var, no docs.
- Automatically a no-op where it doesn't apply. Only the CUDA family exposes
  `ggml_backend_register_host_buffer`; SYCL has it commented out
  (`ggml/src/ggml-sycl/ggml-sycl.cpp:6414`), Vulkan/Metal/others don't have it at all.
  Metal/Vulkan/CPU users get nothing and pay nothing.
- **It makes `mmap+mlock` strictly dominate `mlock`**: same pinned H2D performance, but
  faster load (no read into an anonymous pinned buffer) and a page cache shared across
  processes. That is a clean story for a PR.

Rejected alternative: a new `--load-mode mmap+pin` value, or a revived env var. Upstream
just consolidated all loading flags into `--load-mode`; swapping a dead env var for a live
one will not fly.

### What is still missing

1. **An answer to #6149.** Reproduce it or argue why it no longer applies. Non-negotiable —
   this code has a revert in its history.
2. **Windows.** `register_host()` is `#ifdef _POSIX_MAPPED_FILES` and returns 0 elsewhere.
   mmap is the default on Windows too and `cudaHostRegister` works on `MapViewOfFile`
   memory.
3. **Cost reporting.** Registering tens of GB is a single blocking call over the whole span
   *after* load otherwise appears done — a distinct multi-second stall, unlike `mlock`,
   whose cost is incremental and hides inside the load. Failure currently goes to
   `LLAMA_LOG_DEBUG` inside ggml. Report load-time delta and RSS delta alongside t/s.
4. **Small bug.** The discovery loop
   `for (...; i < ggml_backend_dev_count() && !reg_fn; i++)` assigns `reg_fn` and
   `unreg_fn` independently but tests only `reg_fn`. A backend exposing one and not the
   other exits the loop and then silently no-ops in `register_host`'s null guard.
5. **Benchmarks beyond one model on one RTX 3060.**
6. **AI disclosure is mandatory.** `CONTRIBUTING.md:9-25` — AI-generated code is allowed,
   undisclosed AI usage *"may result in your account being permanently banned"*, you must
   be able to explain every line, and you may **not** use AI to write the PR description
   itself.

### The strongest argument: it unblocks #21067

See §7. `ggml-org/llama.cpp#21067` currently states that `--no-mmap` is required for its
prefetching to do anything. Change 1 removes that requirement. Commenting on #21067 is
probably a better home for this than a standalone PR — and `CONTRIBUTING.md:22` explicitly
requires checking for an existing PR addressing the same change and working with its author
rather than opening a duplicate.

---

## 6. Change 2 — overlap expert uploads with compute (NOT cherry-picked)

`1163cb349` + `5f83fbbe7`, 173 lines, entirely inside `ggml/src/ggml-backend.cpp`.

### What mainline does today

The selective-expert-copy path (added in #15346) lives at
`ggml/src/ggml-backend.cpp:1585-1670`. Per MoE layer it:

1. waits on the split backend — `ggml_backend_event_wait`, or a **full
   `ggml_backend_synchronize(split_backend)` when `events == NULL`** (line 1578);
2. reads the routing ids back from the GPU and syncs again (line 1597 and the
   `tensor_get_async` + `synchronize` pair just after) — the router must have *finished*
   before the CPU learns which experts to copy;
3. copies only the used experts, grouped into runs.

So the copies cannot overlap anything, and with `events == NULL` there is a full device
sync per expert tensor — which validates the commit message's "3× per MoE layer" claim
(gate/up/down). `prev_ids_tensor` (line 1554) is function-scoped, so the *readback* itself
happens once per distinct ids tensor, not three times; the syncs are the 3×.

### What the change does

Adds a second backend instance on the same device (`ggml_backend_dev_init(dev, NULL)` — its
own stream) plus N staging slots with paired ready/free events. When the batch is large
enough that selective copying is pointless, it uploads the **whole** expert tensor on that
stream, so tensor N+1's upload overlaps tensor N's compute. Default 3 slots = one MoE
layer of lookahead. Requires `props.caps.async && props.caps.events`. Gated on
`GGML_SCHED_PREFETCH_EXPERTS` (=1 for default slots, higher sets the count).

### The guard — the genuinely good idea

```cpp
const ggml_tensor * ids = node->src[2];
const int64_t n_expert = input->ne[2];
if (ids->ne[0]*ids->ne[1] >= 2*n_expert && ...)
```

`ids` is `selected_experts` from `build_moe_ffn` —
`ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)`, shape
`[n_expert_used, n_tokens]` (`src/llama-graph.cpp:1926`). So `ne[0]` = top-k experts per
token, `ne[1]` = tokens in the ubatch, and the product is the total (token, expert)
selections in the ubatch. The condition is therefore:

```
n_tokens >= 2 * n_expert / n_expert_used
```

(32 tokens for a 128-expert / 8-active model.) Balls-in-bins: at 2× coverage with uniform
routing ≈ `(1 − 1/E)^(2E)` ≈ e⁻² ≈ 13.5% of experts go untouched, and real routers are
peakier than uniform so somewhat more. Above the threshold selective copy saves a small
minority of the bytes while costing a hard device sync per layer — so switch to full-tensor
pipelined uploads.

Crucially it is a pure **shape** test: no values are read, no sync is needed to evaluate it.

### Why it wasn't picked

- It temporarily rebinds `input_cpy->buffer` / `->data` to a staging slot behind the graph
  allocator's back and restores them after kernel launch. That already produced one
  use-after-free (fixed in `5f83fbbe7`). This is the kind of pattern slaren rejects on
  sight.
- Spawning a second full backend instance per sched is a second likely objection.
- It substantially overlaps #21067, which does the same thing properly.

### Claimed benefit (unverified by us)

Qwen3.6-35B-A3B, RTX 3060, `-ncmoe 26`, ub 2048, pp2048: mainline **1143** → **1880 t/s**
(1383 → 1663 from `1163cb349`, → 1880 after `5f83fbbe7`). Output claimed token-identical.

### Note for this project

With `load-mode = mlock` (no mmap — what `pixi-llm-recipes/models.ini` currently uses) the
CPU weights are *already* in the CUDA pinned host buffer. **Change 2 therefore works
standalone there; it does not need change 1.** Change 1 only matters for mmap users.

---

## 7. Comparison with `ggml-org/llama.cpp#21067`

**"ggml: allow prefetching tensor overrides"**, am17an, open **draft** since 2026-03-27.
Adds `-pw` / `--prefetch-weights`; prefetches the *next layer's* CPU-override weights on a
copy stream, overlapping the current layer's compute. CUDA-only implementation, described
by its author as a PoC.

| | codacus `GGML_SCHED_PREFETCH_EXPERTS` | #21067 `-pw` |
|---|---|---|
| Scope | MoE expert tensors (`MUL_MAT_ID` inputs) only | Any `-ot` CPU override — dense **and** MoE |
| Lookahead | ~3 tensors ≈ one MoE layer, within the current graph | The next layer |
| Trigger | Automatic shape heuristic + env var | User flag, always on when set |
| Footprint | `ggml-backend.cpp` only, no API change | New backend interface across ~15 backends, `llama.h`, cparams, llama-bench |
| mmap | Wants change 1 to overlap under mmap | Documented as requiring `--no-mmap` |
| Allocation | Rebinds tensor ptrs to staging slots (hacky) | Goes through the allocator properly |

### Does change 1 alone fix #21067's `--no-mmap` requirement? — Yes

There is **no hard mmap guard anywhere in the PR's code.** The prefetch does:

```cpp
ggml_backend_tensor_set_async(copy_backend, input_cpy, input->data, 0, ggml_nbytes(input));
```

reading straight from `input->data`, which *is* the mmap pointer when mmap is on.
`cudaMemcpyAsync` from unregistered pageable memory is effectively synchronous — the driver
stages it — so the copy stream can never run ahead and the overlap collapses. Register
those pages and it becomes a real DMA on the copy stream. The `--no-mmap` note in the PR
body is a performance consequence, not a code constraint.

### Porting the guard onto #21067

Conceptually trivial — everything needed is static (`n_tokens` from the ubatch,
`n_expert`/`n_expert_used` from hparams, or read off the next split's `MUL_MAT_ID` node as
codacus does). Two things to get right:

1. **Per-input, not global.** #21067's prefetch loop takes *any* host `WEIGHTS` buffer in
   `next->n_inputs`. The guard must suppress only tensors feeding a `MUL_MAT_ID`, leaving
   dense weights always prefetched. The PR draws no such distinction.
2. **The skip flag must follow.** #21067 tracks a single `next_weights_prefetched` bool and
   later skips the normal copy for prefetched inputs. If the guard suppresses experts but
   not dense weights in the same split, that bool has to become per-input, or suppressed
   tensors get skipped by both paths. Suppressed tensors then fall through to mainline's
   selective-copy branch, which composes correctly.

### Dense spills are #21067's clean case

For dense `ffn_*` weights there is no selective copy to defeat, so the MoE ambiguity
disappears. Mainline today: `ggml_backend_cuda_cpy_tensor_async` (`ggml-cuda.cu:2423`)
returns false on its first check because the source backend isn't CUDA, so the sched falls
to `ggml_backend_synchronize(input_backend)` → full `ggml_backend_synchronize(split_backend)`
(when `events == NULL`) → blocking `ggml_backend_tensor_copy`
(`ggml/src/ggml-backend.cpp:1671-1678`). Fully serialised, per weight tensor, per layer.
Pure dead time; prefetch deletes it.

am17an's own dense benchmark is Qwen3.5-27B Q4_K_M with `-ot "ffn_(gate|up|down).*=CPU"`,
showing gains at every ubatch tested — the same shape as this project's Qwen3.6-27B
(`models.ini`, `ot = blk\.[0-9]+\.ffn_.*=CPU`).

Caveats on magnitude, not direction:

- **Prefill only.** Gain is bounded by `min(C_n, T_{n+1})`. At decode `n_tokens = 1` is
  below the offload threshold of 32, so the ffn runs on CPU and isn't offloaded at all.
- **Costs VRAM** for double-buffered next-split weights, traded against KV cache. The
  Qwen3.6-27B preset runs 256k ctx with `kvarn5`/`kvarn5` and `kv-tail-tokens = 1024`, so
  this is where the real uncertainty lies for this host.
- Needs `--no-mmap` today, or change 1.
- Riskiest part of the PR is the graph-allocator work (split fusing, keepalive nodes), not
  the copy logic.

---

## 8. Next steps

Untested by anyone as far as we can tell, and the experiment that would decide whether to
push change 1 upstream:

1. Build this branch with CUDA, run with `--load-mode mmap+mlock` and
   `GGML_CUDA_REGISTER_HOST=1`. Measure pp t/s, load time, and RSS vs `--load-mode mlock`.
   If `mmap+mlock`+pin matches `mlock` on pp and beats it on load time and RSS, that is the
   PR.
2. Apply #21067 on top and repeat **with mmap** — confirming the `--no-mmap` requirement is
   lifted is the highest-value single result here.
3. Test whether `cudaHostRegister` is charged against `RLIMIT_MEMLOCK` (drop `ulimit -l`,
   see whether registration still succeeds).
4. Try to reproduce #6149's hang at model free on modern CUDA.
5. Separately: cherry-pick `1163cb349` + `5f83fbbe7` onto a second branch and test with
   `GGML_SCHED_PREFETCH_EXPERTS=1` under the existing `load-mode = mlock`. No dependency on
   change 1.

---

## 9. Reproduction

```bash
git remote add thecodacus https://github.com/thecodacus/llama.cpp
git fetch thecodacus
git log --oneline thecodacus/fable5/prefetch-experts -5

# this branch
git switch -c pin-mmap-host-register master
git cherry-pick 20f5994bf

# the other two, when wanted
git cherry-pick 1163cb349 5f83fbbe7
```

Upstream references: PR #6206 (the disable), issue #6149 (the hang), PR #15346 (selective
expert copy), PR #21067 (prefetch tensor overrides), PR #6083 (offload large batches to GPU
— where `register_host_buffer` lost its caller).

---

## 10. Provenance

Both cherry-picked/analysed upstream changes are, per their author, Claude-generated. This
document was written by Claude Opus 5 from a source reading of this branch; the performance
figures are quoted from the commit messages and PR body and have **not** been reproduced
here. Any upstream submission must disclose AI usage per `CONTRIBUTING.md:9-25`, and the PR
description must be written by a human.
