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

---

## Step 3 — prefix injection under transformers 4.51.3

**Done 2026-09-03.** The plan's highest-risk item, and the instruction to *re-derive the patch,
not port it* was the right call: the bug still exists, but it is **a different bug**.

### What the plan expected, and what is actually true

The plan's hypothesis was that modern transformers had fixed the problem — that
`_get_initial_cache_position` derives the offset from `past_key_values.get_seq_length()` and is
already correct, so `patch_for_prefix` should be **deleted**. Measured with a tiny random Qwen3
on CPU, that is half right and half wrong:

- **The forward pass was never broken.** Calling the model directly with a pre-filled cache
  agrees with the un-prefixed run to **2.7e-07** max logit error, whether `cache_position` is
  passed explicitly or omitted entirely. This is why `train.py` needs no patch at all.
- **`generate()` is still wrong** — 19 of 20 tokens differ unprefixed. The negative control
  holds, so the patch stays.

### The actual 4.51.3 failure

Read off `generation/utils.py`:

```python
cache_position = arange(len(input_ids))         # [0 .. L-1]
past_length    = past_key_values.get_seq_length()
cache_position = cache_position[past_length:]   # [P .. L-1]
```

It assumes the cache holds a **prefix of `input_ids`** — the ordinary prefill-then-decode case.
Our cache holds no `input_ids` at all: it holds P graph positions, and all L body tokens are
unprocessed. The damage lands in `_cache_dependant_input_preparation`, which with a
`cache_position` of length L−P against `input_ids` of length L takes its default branch,
`input_ids[:, cache_position]` — **silently dropping the first P body tokens**. The model then
writes a fluent continuation of a truncated prompt.

4.39.3 was the opposite: `arange(cur_len)` with *no* slice, so the P positions were simply
uncounted and every mask row shifted by P. Porting the old `+P` correction into 4.51.3 would
have been wrong in both halves — the offset is already partly accounted for, and the truncation
it cannot see would have remained. The old patch hooked `prepare_inputs_for_generation`; the new
one replaces `_get_initial_cache_position`, which is the function that actually decides.

### The fix

```python
cache_position = arange(L) + P
```

Every body token is new; they simply start at position P. This repairs the input slicing for
free: with `cache_position[-1] == P+L-1 >= L`, `_cache_dependant_input_preparation` takes its
Exception 3 branch, `input_ids[:, -len(cache_position):]`, which is all L tokens. On later steps
`_update_model_kwargs_for_generation` carries the last position forward and adds one, so decode
needs no further help.

The patch also **asserts** that the cache width equals the `num_prefix_tokens` it was installed
for. That is a live risk here, not a hypothetical: `NUM_PREFIX_TOKENS` moves from 10 to 32 at
step 4. A consequence is that a patched model raises on a plain `generate()` — deliberate, and
safe because `generate.py` patches only when `--mode != baseline` and each mode is its own
process, so the control arm always runs unpatched.

### Cache API changes

Smaller than the plan feared. In 4.51.3 `DynamicCache` still holds `key_cache` / `value_cache`
lists — it is **not** yet composed of per-layer objects (that lands in a later release), and it
still iterates as `(key, value)` tuples, so `real_prefix()`'s stacking into the adapter's
`(1, layers, 2, kv_heads, tokens, head_dim)` layout is unchanged. Only one rename was needed:
`get_max_length()` → `get_max_cache_shape()` in `PrefixCache`. Qwen3's modeling calls exactly
two cache methods, `get_seq_length` (line 538) and `get_max_cache_shape` (line 654).

### Results — `python verify_generation.py`, exit 0

```
prefix layout (1, 4, 2, 2, 10, 16) — 2:1 GQA exercised
unpatched: 19/20 tokens differ from the reference (as expected)
patched:   20/20 tokens match the reference exactly
width guard: a prefix that disagrees with NUM_PREFIX_TOKENS raises
forward pass: max logit error 2.7e-07 against the plain run
PrefixCache:  max logit error 2.7e-07, read-only across repeated passes
```

Two gaps in the old script were closed while it was open:

- **GQA is now exercised.** The config is `num_attention_heads=4, num_key_value_heads=2`. Every
  earlier version used `kv == q`, so the cache layout could have been wrong in exactly the
  dimension Qwen3 introduces and the script would still have passed (`model_swap_qwen.md` §7).
- **`PrefixCache` is now covered.** Everything else in the file exercises `DynamicCache`;
  training uses `PrefixCache`, a different class with a different contract. Its defining
  property — storing nothing, so gradient checkpointing can run a layer twice — is asserted by
  running the same instance twice and requiring identical logits.

### Known limitations

- Verified on a 4-layer random Qwen3 on CPU, which is the method's design: the weights are
  irrelevant because the test compares two runs of the *same* model. It exercises the real
  `forward`, mask construction, RoPE and cache code, but not 40 layers, bf16, or CUDA.
- `2:1` GQA is exercised, not Qwen3-14B's `5:1`. The layout is head-count-agnostic, but the
  ratio itself is untested.
- The version assertion is now `4.51.3`. Since 4.39.3 and 4.51.3 fail differently, an upgrade
  must re-derive rather than assume — the negative control is what says which world you are in.

---

## Step 4 — `config.py` and `model.py`

**Done 2026-09-03.** The constants move to Qwen3-14B's geometry. `python model.py` is the gate.

### The rename is the change

`NUM_HEADS` was a name that lied. It has always been the **key/value** head count in the cache
layout, and only equalled the query-head count because Llama-2 has no GQA. Qwen3-14B serves 40
query heads from 8 KV heads, so the two now differ by 5x, and a constant whose name lies is
exactly how a silent bug gets planted here. Renaming it to `NUM_KV_HEADS` *is* the GQA change —
one constant, one import and two uses in `model.py`, and nothing else in the tree referenced it.
The `PrefixAdapter` parameter and attribute were renamed with it.

### `NUM_PREFIX_TOKENS: 10 → 32`

Holding T at 10 would not have held the method fixed. The prefix is a KV tensor, so GQA narrows
the graph's channel into the LLM by 3.2x as a side effect — and a fallen gap could then not be
attributed to the LLM rather than to the narrower prefix. T = 32 holds the prefix **volume**
constant, which is the quantity that has to be constant. Confirmed by measurement, not
arithmetic — `python model.py` reports `2,621,440` values per node, identical to Vicuna's.

What it cannot restore is **addressability**: 8 KV heads serve 40 query heads, so the prefix can
no longer say different things to different heads inside a group. There is no workaround for a
GQA model. Recorded as a limitation, not fixed.

### Everything else

| constant | from | to | why |
|---|---|---|---|
| `BASE_MODEL` | `lmsys/vicuna-7b-v1.5` | `Qwen/Qwen3-14B` | the variable under test |
| `NUM_LAYERS` | 32 | 40 | |
| `HEAD_DIM` | 128 | 128 | read off `config.head_dim`, which Qwen3 states explicitly, rather than derived from `hidden_size // num_attention_heads` — those agree here by coincidence |
| `BATCH_SIZE` × `GRAD_ACCUM` | 2 × 8 | **1 × 16** | effective batch stays 16, the invariant; halved because the weights go 13.5 GB → 29.6 GB on a 40 GB A100 |
| `MAX_LENGTH` | 4096 | **4096** | deliberately *not* raised to Qwen's 40,960 — see below |
| `GENERATION_RESERVE` | 320 | 320 | unchanged; §1 |

`MAX_LENGTH` staying at Vicuna's context is the one that looks like an oversight and is not. The
centroid cap is what gives the GNN information the prompt does not have — the GNN reads every
reference, the prompt shows ~13. If every reference fit the prompt, the graph's unique signal
would collapse to 2-hop structure alone. That reasoning now sits in the constant's comment so it
survives contact with the next reader.

### Results — `python model.py`, all existing assertions pass

```
trainable parameters: 4,197,376          (encoder, unchanged — it does not depend on the LLM)
adapter parameters:  12,494,672
prefix: (8, 40, 2, 8, 32, 128), 2,621,440 values per node
seed features do not affect the output (mask works)
gate=0: prefix is node-independent and equals base (prefix-tuning fallback)
gate=1: the graph embedding moves the prefix
gradients reach every parameter of both modules
```

The adapter grew 12,108,096 → 12,494,672 (+3.2%), from the wider `codes` matrix that 40 layers
needs, partly offset by a narrower `decoder` (8 KV heads instead of 32). `base` is 2,621,440 in
both. Trainable total: **16.7M**.

**Prompt budget, measured.** With T = 32 the budget tightens to 4096 − 32 − 320 = **3,744**, but
Qwen's BPE is more efficient than Vicuna's SentencePiece on the same frozen bibliographies:
`verify_capped_prompts.py` over 900 sampled nodes gives a largest prompt of **3,496 tokens, 248
under budget**. The 22 extra prefix tokens are paid for several times over. This answers ahead of
time what `model_swap_qwen.md` §7 defers to step 7 — though on a sample, not all 9,501 nodes.

The stale `config.py` LoRA comment was corrected while there: 80 modules over 40 layers,
**10.5M** parameters, and under GQA the two targets are *different sizes* — `q_proj` is
5120 × 5120 but `v_proj` only 5120 × 1024. Measured off a real `Qwen3ForCausalLM`, not derived.

### Known limitations

- **`BASE_MODEL` has a blast radius outside `src/adapter/`.** Two experiment scripts read it:
  `experiments/prompt_sensitivity_gpt/build_test_set.py` uses it to *measure prompt lengths* when
  rebuilding the era test sets, so re-running it now would build prompts under the Qwen tokenizer
  and no longer reproduce the stored `test_set.csv`; `experiments/rubric_sensitivity/analyse_arms.py`
  loads it but only passes it to `capped_example`, which ignores it on the frozen path, so that
  one is harmless. Neither was touched — `model_swap_qwen.md` §8 excludes `experiments/` — but
  `build_test_set.py` is a live trap and should get the same pinned-tokenizer treatment
  `freeze_prompt_refs.py` has.
- The prompt-budget headroom is a 900-node sample, not the full 9,501. *(Closed in step 5:
  measured over all 9,501.)*
- Parameter counts and prefix volume are verified; nothing here has been run on a GPU.

---

## Step 5 — the ChatML wrapper

**Done 2026-09-03.** Keep the text, swap the delimiters.

### The split

`build_prompt` was doing two jobs: assembling the instruction and reference block — the
*experimental content* — and wrapping it in Vicuna's `SYSTEM … USER: … ASSISTANT:` framing — a
*calling convention*, like the tokenizer. Only the second may move. They are now separate
functions, which is what `experiments/prompt_sensitivity_gpt/prompts.py` already does:

```python
user_message(neighbours)            # the experiment. Unchanged, byte for byte.
build_prompt(tokenizer, neighbours) # apply_chat_template(..., enable_thinking=False)
build_prompt_vicuna(neighbours)     # the historical format the freeze was computed under
```

**Verified against the GPT run, not just asserted.** `text.user_message(ns)` is byte-identical
to `prompts.user_message(1, ns)[1]`, and `text.SYSTEM` to that call's system message — so the
adapter now splits into exactly the `(system, user)` pair the era-1 GPT prompt-sensitivity run
used. `SYSTEM` and `INSTRUCTION` are unchanged since commit `ee55816`.

### Why the wrapper had to change at all

The adapter arm is *trained*, so it would learn any format and learn to stop. The baseline arm
is frozen and untrained. A frozen Qwen3 given non-ChatML input is off-distribution precisely on
instruction following, and its stop behaviour is unreliable — it emits `<|im_end|>` because it
was tuned to end an assistant turn. The degradation would land entirely on the control and
inflate the measured gap: the same artifact class Stage 6c already caught, where era 2's
reference-count line took the baseline's third-person rate from 15.4% to 58.1%.

### What the real template does

```
<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

`enable_thinking=False` does **not** remove the thinking block — it emits an *empty,
already-closed* one, which is how Qwen3 is told to answer directly. Seeing `<think>` in the
prompt is correct; seeing reasoning in the output would not be. Frozen in
`verify_capped_prompts.py` as `CHAT_WRAPPER`, checked on every run before the GPU hours, so a
`tokenizer_config` update cannot move every prompt in both arms without a word.

Both of the plan's §5 predictions about `encode()` held: Qwen3 has **no BOS** (`bos_token is
None`, and `add_special_tokens` adds nothing), so the "carries BOS" comment was wrong and is
corrected; and `pad_token` is set (151643), so the "Llama ships without one" fallback never
fires. `eos_token_id` is `<|im_end|>` (151645), so the existing target-termination is already
right for ChatML.

### The bug the regression check caught

Re-running the freeze gates after the split failed on train node 32196. The cause was **not**
the wrapper: `NUM_PREFIX_TOKENS` moved 10 → 32 in step 4, and the recompute budget is
`MAX_LENGTH − NUM_PREFIX_TOKENS − reserve`, so 22 fewer tokens dropped a reference.

```
stored (from the freeze): 14 refs
recompute at T=10 : 14 refs   matches stored: True
recompute at T=32 : 13 refs   matches stored: False
```

So the freeze is defined by **four** things, not three, and all are now pinned: the Vicuna
tokenizer, the Vicuna wrapper, `GENERATION_RESERVE = 320`, and `FREEZE_PREFIX_TOKENS = 10`.
This does not make the frozen lists wrong — it makes them *historical*, which is the point. The
lists were sized for a narrower prefix than the run now uses, which is safe but not free, so it
was measured rather than assumed (below).

Two gates changed meaning as a result, and the change is deliberate:

- **`check_era1`** now reads reference lists out of the file and rebuilds the prompt with
  `build_prompt_vicuna`, so it checks *the artifact that ships* rather than the code that made
  it. Still **48/48**. It needs no tokenizer at all.
- **`check_regression`** compares reference lists, not prompt strings. The two branches now
  build deliberately different formats, so a byte-for-byte prompt comparison would fail by
  design. `check_era1` is what still pins a full string.

### Results

```
wrapper  ChatML, thinking disabled, 154+247 chars of prompt text intact
train  7750 nodes ok | valid 806 nodes ok | eval 945 nodes ok
largest prompt 3,627 tokens, 117 under budget
era-1 gate: 48/48 prompts and reference lists match
regression: 450 nodes match on recompute
prefix injection survives generate()
```

**The full-set budget check is the important number.** ChatML's delimiters plus the `<think>`
block cost ~130 tokens more than the Vicuna wrapper, and T = 32 tightens the budget to
4096 − 32 − 320 = **3,744**. Over **all 9,501 nodes** the largest prompt is **3,627 — 117 under
budget**, so generation should produce no `ERROR:` rows. Qwen's BPE still pays for both the
wrapper and the 22 extra prefix tokens. This is no longer a sample: step 4's limitation is
closed.

### Known limitations

- 117 tokens of headroom is thin. It holds for the frozen bibliographies at reserve 320, and
  `generate.py` aborts rather than silently truncates if it is ever breached, but there is no
  slack for a longer instruction or a bigger `T`.
- `prepare_data_capped.py` is now internally inconsistent: its recompute path is pinned to the
  historical budget while its own overflow assertion uses the live `NUM_PREFIX_TOKENS`. It is
  never re-run — `model_swap_qwen.md` §6 forbids it — so this is recorded, not repaired.
- The chat template is frozen against the tokenizer as downloaded on 2026-09-03. The freeze is
  the guard, not a guarantee it will never change upstream.

---

## Step 6 — computing the loss on the supervised positions only

**Done 2026-09-04.** The first step in this migration whose motive is memory rather than correctness. Qwen3-14B is
**29.6 GB** in bf16 on a **40 GB** A100, and the thing that does not fit alongside it is not the
weights or the activations — it is the logits, because Qwen's vocabulary is **151,936** tokens,
4.75× Vicuna's 32,000.

### The waste, measured

`train.py` used to ask the LLM for its own loss (`llm(..., labels=labels).loss`), which runs
`lm_head` over every position in the sequence. But labels are `-100` across the entire prompt —
only the target abstract carries loss. Over 300 random train nodes under the Qwen tokenizer:

| | mean | max |
|---|---|---|
| sequence length (prompt + target) | 3,075 | 3,793 |
| positions carrying a label | **219** | 831 |

**7.1% of the logits were used. The other 92.9% were computed, materialised, upcast to fp32 for
the cross-entropy, and discarded.**

| at the mean sequence | full | sliced |
|---|---|---|
| logits, bf16 | 0.87 GB | 0.06 GB |
| fp32 copy for the loss | 1.74 GB | 0.12 GB |
| **total** | **~2.6 GB** | **~0.19 GB** |

So ~2.4 GB at the mean and ~3.2 GB at the longest sequence, on a card with roughly 10 GB free
after the weights. The plan's estimate (§6: 35.2 → 31.7 GB peak) was in the right place.

### The change

`step_loss` now calls `llm.model(...)` for hidden states and applies the head itself:

```python
keep = labels[:, 1:] != -100
logits = llm.lm_head(hidden[:, :-1][keep]).float()
return F.cross_entropy(logits, labels[:, 1:][keep])
```

Three details make this the *same* number rather than a similar one, and each is a place a
silent bug would live:

- **the shift.** Position `i` predicts token `i+1`; `labels=` does this internally, so doing it
  here by hand is the off-by-one risk.
- **the fp32 upcast.** `ForCausalLMLoss` upcasts before the cross-entropy. Skipping it would
  compute a slightly different loss in bf16 and nothing would say so.
- **the denominator.** `reduction="mean"` over the selected rows is a mean over the batch's
  target tokens — the same denominator, because the selection *is* the set of non-ignored
  positions.

`validate` needed no change: it calls `step_loss`.

`llm.model`, not `llm`, is also what keeps this compatible with Stage 6d — `peft` injects LoRA
in place, so the injected modules are still inside `llm.model`.

### Verification — `verify_sliced_loss.py` (new)

The claim "mathematically identical" is checked against the `labels=` path itself, on a 4-layer
random Qwen3 in **float32** (bf16 rounding would swamp the tolerance the check exists to
enforce). Three checks, because the failure is silent in the usual way — a wrong shift or
denominator produces a plausible loss that falls during training:

```
no prefix:   5.542445 vs 5.542445  (error 4.8e-07), head ran on 40/64 positions
with prefix: 5.547050 vs 5.547050  (error 4.8e-07)
padding:     5.551479 batched vs 5.551479 token-weighted  (error 1.2e-07, 32 tokens of padding)
```

1. **equivalence, no prefix** — isolates the slicing from the injection.
2. **equivalence through `PrefixCache`** — the exact call `step_loss` makes.
3. **padding neutrality** — a right-padded batch of two unequal sequences (40 and 72 tokens,
   32 of padding) gives the token-weighted mean of the same two computed singly. This is the
   Stage-4 check re-run, and it is not redundant: the selection now indexes the label tensor
   directly, where a padded row would be a wrong answer rather than an error.

All three land at ~5e-07 against a 1e-4 tolerance. `verify_generation.py` still passes
unchanged.

The script is wired into `run_train_adapter.sh` next to `verify_capped_prompts.py`, so it runs
on CPU before the GPU hours.

### Known limitations

- **Peak VRAM is still an estimate.** The saving above is arithmetic from measured token counts,
  not a reading off `torch.cuda.max_memory_allocated`. Step 8's pilot run is what measures it —
  and it is the same gap left open by skipping the step-1 GPU load test.
- The tolerance is checked at `hidden_size=64, vocab=256`. It is a bookkeeping check, not a
  numerical-stability one; nothing here says the two paths agree to 1e-4 at Qwen3-14B's width,
  only that they compute the same quantity.
- Backward through the boolean selection scatters into a zero tensor of the full sequence shape
  (~35 MB at batch 1). Accounted for, not eliminated.
