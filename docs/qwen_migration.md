# Qwen migration — swapping the frozen LLM

The execution record for [`model_swap_qwen.md`](model_swap_qwen.md), which is the plan. That
file says what should happen and why; this one says what actually happened, with the numbers.
Append a section per step.

The goal throughout: replace the frozen LLM and change nothing else, so the graph effect under
Qwen3-14B is comparable to the **+0.473** measured under Vicuna-7B.

---

## Step 1 — the container and the model on Komondor

**Done 2026-09-03.** Both models now sit side by side; neither run can disturb the other.

### The question that decided the design

The plan expected a painful dependency cascade: Qwen3 needs `transformers >= 4.51`, a modern
transformers was assumed to force a newer torch, and a newer torch would force new
`torch_scatter` / `torch_sparse` wheels — compiled extensions that fail at import rather than at
runtime.

That turned out to be false, and checking it is what made the step small. Read straight off
PyPI metadata, `transformers==4.51.3` declares **no torch pin at all** — only
`huggingface-hub>=0.30`, `tokenizers>=0.21,<0.22`, `numpy>=1.17`, `safetensors>=0.4.3`,
`python>=3.9`. So the base image, torch, CUDA, `numpy<2` and both compiled extensions could all
stay exactly as the Vicuna image has them, and the only moving part is the transformers stack.

### Decisions

| decision | reasoning |
|---|---|
| base image unchanged (`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`) | nothing forced a bump, and both PyG wheel pins stay byte-identical |
| `transformers==4.51.3` | smallest jump from 4.39.3 that supports Qwen3 natively; below 4.51.0 the config raises `KeyError: 'qwen3'`. Pinned exactly, as 4.39.3 was, because the prefix injection depends on `Cache` internals that move between versions — step 3 re-derives against this pin |
| `peft==0.15.2`, `accelerate==1.6.0` | contemporaries of 4.51.3 by release date (0.15.2 landed one day after, 1.6.0 two weeks before), not the newest available |
| `sentencepiece` and `protobuf` **kept** | the plan said drop them as Llama-only. They are what lets the *Vicuna* tokenizer load in this image, and the frozen reference lists of step 2 are defined by that tokenizer — dropping them would lock any re-verification of the freeze out of the container |
| new `.def`, new `.sif`, old ones untouched | the Vicuna runs have to stay reproducible |

### Two dead ends, recorded so they are not retried

**Bootstrapping from `adapter-gpu.sif`.** When the PyG index proved unreachable (below), the
obvious fix was `Bootstrap: localimage` from the built Vicuna image — inheriting the compiled
extensions rather than reinstalling them. It fails:

```
FATAL: conveyor failed to get: failed to retrieve path for .../adapter-gpu.sif:
       lstat /project/c_gnn42/csanad_thesis: permission denied
```

`--fakeroot` remaps the user identity, which then cannot traverse the intermediate `/project`
directory. This is the same Lustre-vs-fakeroot constraint that already forces builds onto
node-local `/tmp` (`adapter.md`, Stage 4d). It is specific to a `localimage` bootstrap — the
docker bootstrap reads its `.def` from that same tree without complaint.

**Letting pip fall back to PyPI for `torch_scatter`.** The build failed once at
`data.pyg.org` with `Name or service not known` while PyPI worked fine. The tempting fix — drop
the `+pt22cu121` local version so pip resolves `torch_scatter==2.1.2` from PyPI — is wrong:
PyPI carries only the source distribution, which would compile **CPU-only kernels** with no GPU
present at build time. Silent degradation, not an error. The exact pins exist precisely so a
missing wheel fails immediately. The index was reachable on a later attempt and the pins stood.

### Result

Built on a login node, to `/tmp` and copied (Lustre cannot host a fakeroot build):

```bash
module load singularity
cd /project/c_gnn42/csanad_thesis/baseline
singularity build --fakeroot /tmp/adapter-qwen-gpu.sif hpc/komondor/containers/adapter-qwen-gpu.def
cp /tmp/adapter-qwen-gpu.sif adapter-qwen-gpu.sif && rm /tmp/adapter-qwen-gpu.sif
```

`%test` output, on the real target:

```
torch 2.2.0 | numpy 1.26.3 | pyg 2.8.0 | transformers 4.51.3 | wandb 0.28.0
sampler backend: pyg_lib False | torch_sparse True
torch_scatter 2.1.2+pt22cu121 | torch_sparse 0.6.18+pt22cu121
qwen3 forward ok | layers 4 | kv shape (1, 2, 8, 16) | GQA 2 : 1 | full attention on every layer
```

The `%test` was extended beyond a version assert. It builds a live `Qwen3ForCausalLM` at
`num_attention_heads=4, num_key_value_heads=2` and asserts a per-layer cache with `(1, 2, 8, 16)`
keys — the **first time grouped-query attention is exercised anywhere in this project**. Both
existing verify scripts use `kv == q`, so the cache layout could be wrong in exactly the new
dimension and they would still pass (`model_swap_qwen.md` §7). It also pins
`sliding_window=None` explicitly, because `Qwen3Config` *defaults* it to 4096 while the real
model ships `null`, and full attention on every layer is what makes the prefix mechanism work
at all.

| artifact | |
|---|---|
| `adapter-gpu.sif` | 3.6 G, Aug 2 17:17 — **unchanged**, size and mtime |
| `adapter-qwen-gpu.sif` | 3.6 G, Sep 3 22:33 |
| `.hf_cache` | 13 G → **41 G** (Qwen3-14B added ~28 GiB ≈ 30 GB, against 29.6 GB expected) |

Model downloaded through the new container, so the cache is written by the same
`huggingface_hub` 0.36.2 the jobs will read it with.

### Architecture, verified against the plan's §4

Read off the downloaded config under `HF_HUB_OFFLINE=1`, which doubles as proof the cache loads
without network — what the compute nodes will face. All eleven fields match:

```
model_type qwen3 | num_hidden_layers 40 | num_attention_heads 40 | num_key_value_heads 8
head_dim 128 | hidden_size 5120 | vocab_size 151936 | max_position_embeddings 40960
sliding_window None | use_sliding_window False
im_end id 151645 | eos 151645 | pad 151643
```

Three consequences:

- **`T = 32` is confirmed with real numbers.** `40 × 2 × 8 × 32 × 128 = 2,621,440` — identical to
  Vicuna's prefix volume at `T = 10`. The GQA ratio is **5:1**, the addressability limitation
  §3 says to carry into the write-up.
- **`eos_token_id` is `<|im_end|>`** (151645), so `encode()`'s existing `tokenizer.eos_token_id`
  append already terminates an assistant turn correctly. No change needed there.
- **`pad_token_id` is 151643, not `None`**, so the "Llama ships without one" fallback in
  `encode()` will not fire — as §5 predicted. Pad and EOS are distinct ids.

### A local environment for steps 3 and 7

`DyTAG-qwen`, cloned from `DyTAG` and pinned to transformers 4.51.3 / peft 0.15.2 /
accelerate 1.6.0 on torch 2.2.1. `verify_generation.py` and `verify_lora_prefix.py` cannot be
re-derived under the 4.39.3 in `DyTAG`, and this avoids needing Komondor to iterate on them.

It caught one thing before it reached the container: the first `%test` draft emitted
`Sliding Window Attention is enabled but not implemented for sdpa`. That was the synthetic
config's default, not the model — but unexamined it would have put a warning contradicting the
plan's central architectural claim into every build log.

### Known limitations

- **Peak VRAM is unmeasured.** The GPU load test was skipped by choice rather than hold an
  allocation. Step 8's pilot run measures it instead, so an OOM would surface as a failed
  queued job rather than a two-minute check. Qwen3-14B is 29.6 GB bf16 on a 40 GB A100 and the
  plan's estimate is ~31.7 GB peak *only after* step 6 slices the `lm_head`.
- **`enable_thinking=False` is unverified in practice.** The chat template needs no GPU, so this
  is CPU work still owed before step 5 writes the ChatML wrapper.
- The local env proves Python-level compatibility only; the `+pt22cu121` extensions are Linux
  CUDA builds, verified by the Komondor `%test` rather than locally.

---

## Step 2 — freezing the reference lists

**Done 2026-09-03.** The plan calls this *"the step that makes 'the data is the same' true
rather than assumed."*

### What was actually wrong

`text.capped_example` decided a node's bibliography at call time: keep the 15 references closest
to the centroid, then drop the farthest one at a time until the **tokenizer's** count fits the
budget. So the bibliography is a function of the tokenizer. Qwen3's BPE is more efficient on
English than Vicuna's SentencePiece — more references would survive the same 4,096-token budget,
and the two arms would silently read different evidence.

**Nothing new is decided here, and that is worth being clear about.** The frozen node sets were
selected in August by running this same computation, so these lists were already implicit in
every run since. They were simply never written down: reproducibility came from *recomputation*
— same tokenizer, same code, same reserve ⇒ same answer. That held fine until now. Swapping the
LLM is precisely what breaks the assumption.

Checked before building anything: no reference lists were stored anywhere. Both 08-15
`generations.csv` files are 945 × `[node_id, mag_id, original_abstract, generated_abstract]`,
and `generate.py:215` discards the list on the spot (`prompt, target, _ = capped_example(...)`).
`data/frozen/` held only node-id lists. The only persisted bibliographies in the repo were
`prompt_sensitivity_gpt/0*/test_set.csv` — 48 nodes, 0.5% of the 9,501.

### Decisions

| decision | reasoning |
|---|---|
| `capped_example(..., frozen=True)` **raises** on a missing node | the plan's "load it when it exists" wording admits a silent fallback. A node quietly recomputed under Qwen's tokenizer would be shown a different bibliography and would still produce a fluent abstract — no error, nothing downstream can tell |
| `frozen=False` kept as the producer path | pinned explicitly in `prepare_data_capped.py`, which walks candidates not yet in any frozen set and is what decides membership. It is not re-run |
| tokenizer pinned **in the script**, not read from `config.BASE_MODEL` | `BASE_MODEL` becomes Qwen at step 4; re-running the freeze afterwards must still reproduce the same file |
| new `freeze_prompt_refs.py` rather than a mode on `text.py`'s `main()` | that `main()` builds the *legacy* non-capped training sets; the two jobs share nothing |
| reserve **320**, not 256 | see `model_swap_qwen.md` §1 — the measurement that settled it is the era-1 gate below |

### How to run

```bash
cd src/adapter
python freeze_prompt_refs.py            # ~20 min on a laptop CPU
```

Writes `data/frozen/prompt_refs_capped.json` (`node_id → [ref_node_id, ...]`, `edges.csv`
order), then runs its own checks. Versioned deliberately, like the node lists: a result cannot
be reproduced without it.

### Results

9,501 nodes — 945 eval + 7,750 train + 806 valid. File is 1,040,720 bytes.

| check | result |
|---|---|
| coverage | keys are exactly the three frozen sets; every node keeps ≥1 reference; `edges.csv` order preserved |
| **era-1 gate** | **48/48** prompts *and* reference lists match the stored build, string-for-string |
| regression | 450 nodes (150/set) byte-identical between `frozen=True` and `frozen=False` |
| `verify_capped_prompts.py` | 900 sampled nodes ok; largest prompt 3,766 tokens = the budget exactly |
| determinism | two full runs produced md5 `6044dbffd30bb8c5131f1f8e8ca9554c` both times |

References shown across all 9,501: **mean 12.87, median 14, min 5, max 15**. Consistent with the
13.25 mean the `prompt_sensitivity_gpt` README reports for its 48-node stratified sample.

The era-1 gate is the load-bearing number. It is the direct confirmation that reserve 320 was
right: the same check is 45/48 at 256.

### Known limitations

- **The hard gate covers 48 of 9,501 nodes — 0.5%.** The other 9,453 have no stored prompt to
  diff against. The only guarantee for them is that they came from this code path, this
  tokenizer and this reserve. Do not claim a verification that was not done.
- **`verify_capped_prompts.py`'s check 1 is now vacuous.** It compares `np.load(...)` against
  `np.load(..., mmap_mode="r")` — train.py's embedding source versus generate.py's — asserting
  they build identical prompts. On the frozen path `embeddings` is never consulted, so the
  assertion passes without testing anything. The divergence it guarded against is now impossible
  by construction, so this is not a correctness problem, but a test that silently stops testing
  is exactly the failure class this project keeps hitting. Fix it in step 7's rewrite of that
  script.
- `tokenizer` and `embeddings` remain in `capped_example`'s signature though unused when frozen,
  so the two paths stay interchangeable and no caller had to change.
