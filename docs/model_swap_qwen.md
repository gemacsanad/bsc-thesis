# Stage 7 (planned) — swapping Vicuna-7B for Qwen3-14B

**Status:** planned, nothing implemented. Written 2026-09-02 to be executed as an agentic
coding session. Amended 2026-09-03: the reserve decision in §1 was reversed after measurement
— `GENERATION_RESERVE` stays at 320.

**The one-sentence goal:** replace the frozen LLM and change nothing else, so the measured
graph effect under Qwen3-14B is directly comparable to the +0.473 measured under Vicuna-7B.

---

## 0. Context for an agent starting cold

Read this section before touching anything. The rest of the plan assumes it.

### What the system does

```
citation neighbourhood → GraphSAGE → prefix adapter → 10 soft KV tokens
                                   → frozen Vicuna-7B → generated abstract
```

A paper's abstract is generated from the titles and abstracts of the papers it cites. A
GraphSAGE encoder reads the reference neighbourhood, an adapter turns that into a key/value
prefix injected into **every layer** of a frozen LLM, and the prefix is the only trainable
channel into the LLM. The LLM's own weights never move.

The experiment is a **difference between two arms** on the same nodes with the same prompts:

- **baseline** — the frozen LLM, no prefix
- **adapter** — the same frozen LLM with the graph prefix

The result that matters is the *gap* between them, not either absolute score. Anything that
changes one arm without changing the other is a confound, and this project has already been
burned by exactly that (see §7).

Start with the root `README.md`, then `docs/adapter.md` Stage 1 for why the evaluation set is
what it is. `docs/README.md` is the index.

### Repo conventions that bite

- **Adapter scripts use flat imports.** Run them from inside `src/adapter/`, not from the repo
  root. `cd src/adapter && python train.py`.
- **`data/frozen/` is versioned on purpose.** `eval_nodes_capped.json`,
  `splits_eligible_capped.json` and `training_sets_capped.json` decide which nodes every
  experiment runs on. A result cannot be reproduced without them. Everything else under
  `data/` is large and regenerable.
- **GPU work runs on Komondor** (`/project/c_gnn42/csanad_thesis/baseline`, single A100 40 GB,
  Singularity, SLURM). CPU data prep runs locally in the `DyTAG` conda env — and should
  actually be run, not just written.
- **Compute nodes are offline.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in
  every job script. Models must be pre-downloaded into `$BASE_DIR/.hf_cache` from a login node.
- **Secrets** live in a gitignored `.env` at the repo root.
- `CLAUDE.md` governs: surgical changes only, no speculative abstractions, ask instead of
  assuming, and every piece of work ends with a documentation update.

### The culture: every failure in this layer is silent

This is the single most important thing to internalise. Nothing in the prefix path throws when
it is wrong — it produces fluent, plausible abstracts and a falling loss. `docs/README.md`
lists this as one of the three findings that constrain everything downstream:

> **Prefix injection is silent when wrong** — a wrong `cache_position` produces fluent text and
> a falling loss, not an error.

That is why `verify_generation.py`, `verify_capped_prompts.py` and `verify_lora_prefix.py`
exist and run *before* the GPU hours in every job script. Every step below has a verification
criterion for the same reason. **Do not skip a verify step because the code "looks right".**

### What this migration is not

It is not an opportunity to improve the prompt, lift the context window, tune hyperparameters,
or refactor the adapter. Four prompt eras have already been measured and are recorded in
`docs/adapter.md` Stage 6b–6c. The variable under test here is the LLM and nothing else.

---

## 1. The reference arm — what we are comparing against

**Not re-run.** These runs exist and stand as the comparison.

| run | mean `llm_accuracy` |
|---|---|
| `experiments/2026-08-15-14207605_vicuna-baseline` | **2.815** |
| `experiments/2026-08-15-14207605_adapter` | **3.288** |
| gap | **+0.473 ± 0.098** (95% CI) |

945 nodes each, from `data/frozen/eval_nodes_capped.json`.

`experiments/rubric_sensitivity/runs.py` already designates this pair as *the* comparison, and
its docstring explains why a cross-era pair cannot be used — the four prompt rewrites would end
up inside the measured graph effect. The same logic applies to this swap.

### The exact settings the Qwen run must reproduce

| setting | era-1 value | note |
|---|---|---|
| instruction | era 1 | **already what is in the working tree** — `text.py`'s `INSTRUCTION` |
| node sets | capped | 945 eval, 7,750 train |
| `GNN_LAYERS` | 2 | |
| LoRA | **off** | LoRA is Stage 6d (08-24), after this run |
| decoding | greedy, bf16, batch 1 | |
| `REPETITION_PENALTY` | 1.00 | matches current `config.py` |
| `MAX_LENGTH` | 4096 | |
| `NUM_PREFIX_TOKENS` | 10 | **see §3 — this one must change** |
| `GENERATION_RESERVE` | **320** | ⚠ *not* the era-1 value — see below |
| `MAX_NEW_TOKENS` | **320** | = the reserve |

> **⚠ The reserve, and why this plan no longer pins it back.** `config.py` records
> `GENERATION_RESERVE` as *"Raised from 256 on 2026-08-19"* — after the 08-15 run and after the
> frozen sets were written (their mtime is Aug 16). So 320 does **not** reproduce the era-1 code
> path, and the first draft of this plan pinned it back to 256 for that reason.
>
> **Decided 2026-09-03: keep 320.** The two things the first draft asked for cannot both hold.
> Rebuilding the 48 stored era-1 prompts in
> `experiments/prompt_sensitivity_gpt/01_era1/test_set.csv` through the working-tree `text.py`:
>
> | reserve | prompts matching the stored era-1 build |
> |---|---|
> | 256 | 45 / 48 |
> | 320 | **48 / 48** |
>
> That folder's README says so outright — those prompts were built with *"10 prefix tokens and a
> 320-token reserve."* A smaller reserve is a larger budget, so each of the three disagreements
> keeps exactly one more reference, and the 320 list is a strict subset of the 256 list in all
> three:
>
> | node | out-degree | refs @320 | refs @256 | target tokens |
> |---|---|---|---|---|
> | 68842 | 83 | 13 | 14 | 285 |
> | 5164 | 64 | 14 | 15 | 275 |
> | 59631 | 27 | 14 | 15 | 242 |
>
> **The cost.** On roughly 6% of nodes the Qwen run shows one fewer of the farthest-from-centroid
> references than the 08-15 Vicuna arm saw. That lands on **both Qwen arms identically** — they
> read the same frozen prompt — so the within-Qwen gap, which is the number this experiment
> reports, is untouched. Only the cross-model comparison absorbs it. Carried into §7 as item 4.
>
> **The gain.** A hard 48/48 string-for-string gate on the freeze (§5 step 2), and a reserve
> consistent with every run since 08-19 — the four prompt eras and the whole
> `prompt_sensitivity_gpt` battery.

Verification that the working tree still builds era-1 prompts already exists:
`experiments/prompt_sensitivity_gpt/README.md` records *"The 48 era-1 prompts were rebuilt the
same way through the working-tree `text.py`: 0 mismatches."* Re-measured 2026-09-03, still 0:
the working tree's `SYSTEM` and `INSTRUCTION` are byte-identical to `prompts.ERA1`.

---

## 2. Invariants — do not touch these

If a diff touches anything in this list, it is wrong.

**Graph**
`src/adapter/graph.py`, `GraphSAGEEncoder` in `model.py`, `GNN_LAYERS = 2`, `num_neighbors()`,
the learned seed `[MASK]`, `IN/HIDDEN/OUT_CHANNELS`, `DROPOUT`.

**Nodes**
`data/frozen/eval_nodes_capped.json`, `training_sets_capped.json`,
`splits_eligible_capped.json`. Same ids, same order. Do **not** re-run
`prepare_data_capped.py`.

**Text**
`SYSTEM`, `INSTRUCTION` and the reference-block format in `build_prompt`. The *strings* do not
change. Only the chat delimiters around them do (§4).

**Data**
`data/processed/joint/{nodes.csv,edges.csv,node_embeddings.npy}`, `neighbour_cap.py`'s centroid
distance, `PROMPT_MAX_REFS = 15`, `MAX_LENGTH = 4096`.

> Qwen3-14B natively supports 40,960 tokens. **Do not raise `MAX_LENGTH`.** The centroid cap is
> what gives the GNN information the prompt does not have — the GNN reads every reference, the
> prompt shows ~13. If every reference fits the prompt, the graph's unique signal collapses to
> 2-hop structure alone. `docs/adapter.md` Stage 6c already measured that accuracy rises with
> bibliography size *in both arms in parallel* — i.e. that is prompt information, not graph
> information. Lifting the context is a separate experiment and is expected to **shrink** the
> gap.

**Training**
`LEARNING_RATE`, `WEIGHT_DECAY`, `WARMUP_STEPS`, `GRAD_CLIP`, `SEED = 42`, `EPOCHS`,
`PATIENCE`, and the effective batch size of 16.

**Measurement**
`src/evaluation/evaluate_results.py`, judge `gpt-5.4-mini` at temperature 0.

---

## 3. The one constant that must move

### `NUM_PREFIX_TOKENS: 10 → 32`

The prefix is a KV tensor, so its size is fixed by the **model's** attention geometry, not by
the method. Vicuna-7B (Llama-2) has 32 KV heads and no GQA. Qwen3-14B has 40 query heads but
only **8** KV heads.

```
Vicuna-7B,  T=10:  32 layers × 2 × 32 kv × 10 tok × 128 = 2,621,440 values/node
Qwen3-14B,  T=10:  40 layers × 2 ×  8 kv × 10 tok × 128 =   819,200   ← 3.2× narrower
Qwen3-14B,  T=32:  40 layers × 2 ×  8 kv × 32 tok × 128 = 2,621,440   ← exactly equal
```

**Holding `T = 10` does not hold the method fixed.** It silently narrows the graph's channel
into the LLM by 3.2× as a side effect of GQA. If the gap then falls, there is no way to tell
whether that was the LLM or the narrower prefix. `T = 32` holds the graph pathway's capacity
constant, which is the quantity that actually needs to be constant.

Cost: 22 more tokens charged against the 4,096 context. It affects the **adapter arm only** —
the baseline has no prefix — so the within-model gap is unaffected either way.

**Limitation to record in the write-up, not fixable:** 8 KV heads serve 40 query heads (5:1).
The prefix regains its *volume* but loses per-query-head *addressability* — it can no longer say
different things to different heads inside a group. No workaround exists for a GQA model.

**Optional second arm.** If the strictly-literal "nothing but the LLM changed" run is wanted,
`T = 10` is a second training run, and the two together separate the LLM effect from the prefix
width. Run `T = 32` first.

---

## 4. Target model — verified facts

`Qwen/Qwen3-14B`, Apache 2.0. From its `config.json`:

| field | value |
|---|---|
| `model_type` | `qwen3` |
| `architectures` | `Qwen3ForCausalLM` |
| `num_hidden_layers` | **40** |
| `num_attention_heads` | 40 |
| `num_key_value_heads` | **8** |
| `head_dim` | **128** (explicit; do not derive it) |
| `hidden_size` | 5120 |
| `intermediate_size` | 17408 |
| `vocab_size` | **151936** |
| `max_position_embeddings` | 40960 |
| `sliding_window` | `null`, `use_sliding_window: false` |
| `rope_theta` | 1000000 |
| `transformers_version` | 4.51.0 |

Two properties made this the choice and both must hold for the prefix mechanism to work at all:

1. **Full attention on every layer**, with a real per-layer KV cache. `PrefixAdapter` emits one
   (key, value) pair *per layer*; `prefix_to_cache` writes them into a `DynamicCache` layer by
   layer. Models with hybrid linear attention (Qwen3.5/3.6 — 3:1 Gated DeltaNet) or interleaved
   sliding windows (OLMo 3 — 3:1 SWA at 4,096) break this: the prefix would silently vanish
   from most layers. Qwen3 has neither.
2. **No sliding window**, so prefix positions never fall outside an attention window as
   generation advances.

**Thinking mode.** Qwen3 is a hybrid thinking model. `enable_thinking=False` is mandatory — a
`<think>` block would poison both the training targets and the generations.

---

## 5. Steps

Each step has a verification criterion. Do not move on without it.

### Step 1 — container

New file `hpc/komondor/containers/adapter-qwen-gpu.def`. **Keep the existing `.def` and the
built `adapter-gpu.sif`** — the Vicuna runs must stay reproducible.

- `transformers>=4.51.0` (Qwen3 raises `KeyError: 'qwen3'` below it).
- torch 2.2.0/cu121 is old for that transformers. If the base image is bumped,
  `torch_scatter` and `torch_sparse` are **compiled extensions that must match the torch build
  exactly** — pull new wheels from the matching `https://data.pyg.org/whl/` index. A mismatch
  fails at import, not at runtime, which is the good case.
- Drop `sentencepiece` and `protobuf` (Llama-only; Qwen3 uses a BPE tokenizer).
- Update the `%test` assert from `transformers.__version__ == '4.39.3'` to the new pin.

**Verify:** `%test` block passes at build time, including
`torch_geometric.typing.WITH_TORCH_SPARSE or WITH_PYG_LIB` and `numpy < 2`.

**Also:** download the model on a **login node** — compute nodes are offline.

```bash
export HF_HOME=/project/c_gnn42/csanad_thesis/baseline/.hf_cache
huggingface-cli download Qwen/Qwen3-14B
```

~29.6 GB of safetensors. Check the project quota first.

---

### Step 2 — freeze the reference lists  ← everything depends on this

**This is the step that makes "the data is the same" true rather than assumed.**

Today `text.capped_example` recomputes which references a prompt shows at runtime, by dropping
the farthest-from-centroid reference until the **tokenizer's** count fits the budget. A
different tokenizer therefore shows a different bibliography. Qwen's BPE is more efficient on
English than Llama's SentencePiece, so more references would survive the same 4,096 budget —
and the two models would silently be reading different evidence.

Fix: compute the kept-reference list **once, under the Vicuna tokenizer**, store it, and have
both models read it.

1. Leave `GENERATION_RESERVE = 320` as the working tree has it. Do **not** pin it back to 256
   — see §1 for the measurement that decided this.
2. Run the current Vicuna path over all of `eval_nodes_capped.json` and
   `training_sets_capped.json`, emitting `data/frozen/prompt_refs_capped.json`
   (`node_id → [ref_node_id, ...]`, in `edges.csv` order, as `fit_refs` returns them).
3. Change `capped_example` to **load** that file instead of recomputing, when it exists.

**Verify (hard):** `experiments/prompt_sensitivity_gpt/01_era1/test_set.csv` stores
`ref_node_ids` and the finished `prompt` for 48 era-1 nodes. The freeze must match **all 48**,
in the reference lists *and* string-for-string in the prompts. This is the same method that
folder used to validate era 1 and era 2 (0 mismatches both times). It is a 48/48 gate only at
reserve 320; at 256 it is 45/48 (§1), which is what settled that choice.

**Verify (soft):** for the remaining 897 eval nodes there is no stored prompt to diff against.
The guarantee is only that they came from the same code path at the same pinned reserve. **Say
so in the write-up** — do not claim a verification that was not done.

**Verify (regression):** with the frozen file in place, the Vicuna path must still build
byte-identical prompts. This is what protects the existing runs from the change.

---

### Step 3 — `prefix_generation.py` and `PrefixCache`

The highest-risk item in the whole migration.

**Re-derive the `cache_position` patch. Do not port it.**

`patch_for_prefix` adds `+P` to `cache_position` on every generation step, to work around a
specific transformers 4.39.3 bug — `generate()` sets `cache_position = arange(cur_len)` from
`input_ids` alone, not counting the P positions already in the cache. In modern transformers,
`_get_initial_cache_position` derives the offset from `past_key_values.get_seq_length()`, which
for a pre-filled cache is **already correct**. Applying the patch there would corrupt positions
by +32 in the other direction.

`verify_generation.py` is the instrument that tells you which world you are in. Its **negative
control** asserts that *unpatched generation must disagree with the reference*. If that assert
fails on the new version, the bug is gone: **delete `patch_for_prefix`**, do not "fix" the test.

`PrefixCache` (in `train.py`) also needs rewriting for the modern `Cache` API — `get_max_length()`
is gone in favour of `get_max_cache_shape()`, and `DynamicCache` is now composed of per-layer
objects. **Preserve its defining property:** it is read-only and stores nothing, because
gradient checkpointing runs each layer twice and a cache that appended would grow by the
sequence length on the second pass.

**Verify:** `python verify_generation.py` — negative control still fails unpatched (or the patch
is deleted and the check passes clean), patched generation matches the reference token for
token, and the prefilled forward pass agrees within `1e-4` max logit error.

---

### Step 4 — `config.py` and `model.py`

`config.py`:

```python
BASE_MODEL = "Qwen/Qwen3-14B"

NUM_LAYERS = 40             # was 32
NUM_KV_HEADS = 8            # was NUM_HEADS = 32   ← the GQA change
HEAD_DIM = 128              # unchanged; read from config.head_dim, not hidden//heads

NUM_PREFIX_TOKENS = 32      # was 10   (§3)
GENERATION_RESERVE = 320    # unchanged (§1)
MAX_NEW_TOKENS = 320        # = the reserve

BATCH_SIZE = 1              # was 2    (§6)
GRAD_ACCUM = 16             # was 8    — effective batch stays 16
```

> **On the rename.** `NUM_HEADS` is currently *misnamed*: it is used as the KV-head count in the
> cache layout, and only equals the query-head count because Llama-2 has no GQA. Renaming it to
> `NUM_KV_HEADS` **is** the GQA change — one constant plus two uses in `model.py` and two in the
> verify scripts. Do the rename rather than reassigning `NUM_HEADS = 8`; a constant whose name
> lies is exactly how a silent bug gets planted here.

`model.py`:
- `NUM_HEADS → NUM_KV_HEADS` in `PrefixAdapter.__init__` and in the `__main__` shape assert.
- Fix the now-wrong comment on `delta.transpose(3, 4)` — it says *"matches how Llama reshapes a
  4096-d projection"*. Qwen3-14B's `v_proj` output is 8 × 128 = 1024, not 5120.
- `HEAD_DIM` must come from `config.head_dim`. It happens to equal `hidden_size //
  num_attention_heads` for both Qwen3-8B and 14B, but that is a coincidence, not a rule.

Expected prefix cost per node: **2,621,440 values** — identical to Vicuna's. Read the actual
parameter counts off `python model.py` rather than trusting arithmetic here.

**Verify:** `cd src/adapter && python model.py`. The existing smoke test already asserts output
shapes, that seed features do not leak through the `[MASK]`, that `gate = 0` gives a
node-independent prefix equal to `base`, that `gate = 1` moves it, and that gradients reach every
parameter of both modules.

---

### Step 5 — `text.py`, the chat template

**Keep the text. Swap the delimiters.**

`SYSTEM` and `INSTRUCTION` do not change — they are the experimental content. What changes is
the wrapper, which is a calling convention like the tokenizer:

```
SYSTEM ... USER: {user} ASSISTANT:        →   <|im_start|>system ... <|im_end|>
                                              <|im_start|>user {user}<|im_end|>
                                              <|im_start|>assistant
```

Build it with `tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
enable_thinking=False)` rather than hand-rolling the string, then **freeze the resulting string
in a test** so it cannot drift under a tokenizer update.

Precedent: `experiments/prompt_sensitivity_gpt/` did exactly this when it ran the four eras
under a GPT model — it kept *"the Vicuna `SYSTEM … USER: … ASSISTANT:` framing, split into a
system and a user message."* Text preserved, mechanism adapted.

> **Why not keep the raw Vicuna delimiters.** The adapter arm is *trained*, so it learns the
> format and learns to stop. The baseline arm is *frozen and untrained*. A frozen Qwen3 given a
> non-ChatML prompt is off-distribution precisely on instruction following, and its stop
> behaviour is unreliable — it emits `<|im_end|>` because it was tuned to end an assistant turn.
> The degradation would land entirely on the control, inflating the measured gap. That is the
> same artifact class `docs/adapter.md` Stage 6c already caught: era 2's reference-count line
> took the baseline's third-person rate from 15.4% to 58.1%, and +0.343 is still reported as
> *"an upper bound"* because of it.

Two assumptions in `encode()` to re-check:
- The comment says the prompt *"carries BOS"*. Qwen3's chat template does **not** prepend one.
- `tokenizer.pad_token is None` (the "Llama ships without one" fallback) will **not** fire —
  Qwen3 has a pad token. Confirm the target's appended EOS is `<|im_end|>` (151645), which is
  what should terminate an assistant turn, and that `generate()` stops on it.

**Verify:** a frozen-string test asserting the exact finished prompt for a known node; and that
`SYSTEM` / `INSTRUCTION` are byte-identical to the era-1 values under `git diff`.

---

### Step 6 — memory: slice the `lm_head`

Qwen3-14B is **29.6 GB** in bf16 on a **40 GB** A100. The headroom problem is not the weights,
it is Qwen's 151,936-token vocabulary — 4.75× Vicuna's.

| | with `labels=` | with sliced `lm_head` |
|---|---|---|
| weights, bf16 | 29.6 GB | 29.6 GB |
| activations, grad-checkpointed | ~1.6 GB | ~1.6 GB |
| logits + fp32 CE copy | **~3.7 GB** | **~0.3 GB** |
| encoder + adapter + AdamW | ~0.27 GB | ~0.27 GB |
| **peak, batch 1** | **~35.2 GB** | **~31.7 GB** |

Labels are `-100` over the whole prompt, so only ~250 of a ~3,400-token sequence carry loss.
Change `step_loss` (and `validate`) to call `llm.model(...)` for hidden states and run `lm_head`
on **only the supervised positions**, instead of `llm(..., labels=labels).loss`. Mathematically
identical, ~3.4 GB cheaper, and it is what turns "nearly fits" into "fits".

With LoRA injected in place by `peft.inject_adapter_in_model`, `llm.model` still holds the
injected modules, so this is compatible with Stage 6d later.

**Verify:** the sliced loss matches the `labels=` path to `< 1e-4` on a small random model; and
the Stage-4 padding-neutrality check still holds (batched loss vs token-weighted singles).

---

### Step 7 — the verify scripts

**There is a real gap here.** Both `verify_generation.py` and `verify_lora_prefix.py` build
their test config with `num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4`.
Because `kv == q`, **GQA is currently untested** — the cache layout could be wrong in exactly
the new dimension and both scripts would still pass.

- Switch both to `Qwen3Config(..., num_attention_heads=4, num_key_value_heads=2)` so the
  grouped layout is actually exercised.
- Update `verify_generation.py`'s `assert transformers.__version__ == "4.39.3"` to the new pin.
- `real_prefix()` stacks the cache into the adapter's layout; confirm it still produces
  `(1, layers, 2, kv_heads, tokens, head_dim)` under the modern `DynamicCache`.

**Verify:** both scripts pass with `kv=2, q=4`, and `verify_capped_prompts.py` passes against
the frozen reference lists from step 2 — with the assertion inverted for Qwen: rather than
"does it fit", assert **`len(qwen_prompt) + 32 + reserve ≤ 4096` for all 945 eval and 7,750
train nodes, with headroom**. Qwen's tokenizer is more efficient, so this should pass
comfortably, but assert it rather than assume it.

---

### Step 8 — pilot run

**Not optional.** The current `~4.3 h/epoch` figure in `docs/adapter.md` is an unvalidated FLOP
estimate for a 7B at batch 2. Qwen3-14B at batch 1 is roughly 2.3–2.5× that — call it ~10–11 h
per epoch, so `EPOCHS = 3` is ~30–33 h against a 48 h walltime, with no margin, on a number
nobody has ever measured.

One epoch on a subset, and read off:

- measured `train/tokens_per_sec` → size the real job from it
- peak VRAM (`nvidia-smi` / `torch.cuda.max_memory_allocated`) — must sit under 40 GB
- `gate/mean_abs` rising, and `prefix/graph_fraction > 0`

Those last two matter more than the loss. The gate starts at exactly zero, so the entire graph
pathway receives zero gradient at initialisation and has to learn its way open from a random
delta (`docs/adapter.md` Stage 3b addendum). **A flatlining gate against a falling train loss
means the LLM is learning the task without the graph** — stop and diagnose rather than spending
30 more hours.

---

### Step 9 — full training run

`run_train_adapter.sh` with:

```
LORA=0                 # era 1 had no LoRA
--gnn-layers 2
CKPT_DIR=$BASE_DIR/checkpoints/qwen14b      # its own directory — see below
```

Point `SIF` at the new image. Give the run its **own checkpoint directory**: the resume logic
picks the newest `epoch*.pt` it finds and `best.pt` is a fixed name, so sharing a directory with
the Vicuna checkpoints would resume from and overwrite the wrong run. `train.py` refuses a
resume across a change of `--sets` or `--gnn-layers`, but it has no idea the base model changed.

**Verify:** `best.pt` written; early stopping behaves; the W&B config records
`model: Qwen/Qwen3-14B` and `num_prefix_tokens: 32`.

---

### Step 10 — generation

`run_generate_eval.sh` with `RUN_BASELINE=1` and `RUN_GATE0=1`, both arms in **one job on one
GPU**, as the script already does — that is what keeps everything but the prefix fixed between
them.

`--max-new-tokens 320` — the reserve the sets are frozen under, which is what `generate.py`
refuses to exceed. This is **not** era 1's value (256); §7 item 1 is where that difference is
accounted for.

`gate0` is not redundant here even without LoRA — it is a cheap cross-check that the
frozen-Qwen baseline and the gate-zero adapter agree, which they should.

**Verify:** 945/945 rows in every arm, **0 `ERROR:` rows** (under the capped sets an overflow
means the prompt path drifted, and the run should abort rather than record it).

---

### Step 11 — scoring

```bash
python src/evaluation/evaluate_results.py \
    --input  experiments/<run>/generations.csv \
    --output experiments/<run>/scored.csv
```

Score **both** ways:

- **R1**, the project's current rubric, unchanged — this is the number directly comparable to
  2.815 / 3.288 / +0.473.
- **R2** from `experiments/rubric_sensitivity/` — the rubric that folder adopted going forward
  (problem and task only; methodology excluded because it is not recoverable from a
  bibliography). Under R2 the Vicuna era-1 pair scored 3.17 / 4.01, gap **+0.833**, on its
  150-paper subset.

R1 and R2 cannot be mixed in one table. Report both gaps side by side.

---

## 6. Do NOT do these

An agent trying to be helpful can wreck this experiment in ways that produce no error:

- ❌ Change `SYSTEM` or `INSTRUCTION`, or "improve" the prompt in any way
- ❌ Raise `MAX_LENGTH` to Qwen's 40,960 (§2)
- ❌ Re-run `prepare_data_capped.py` or regenerate any `data/frozen/*.json` node list
- ❌ Enable `--lora` on this run
- ❌ Change `GNN_LAYERS`, the encoder, or anything in `graph.py`
- ❌ Pin `GENERATION_RESERVE` back to 256 — decided against in §1, with measurements
- ❌ Tune learning rate, warmup, seed, or the effective batch size
- ❌ Switch dtype, add flash-attention, or add quantisation
- ❌ Delete or overwrite `adapter-gpu.sif`, the Vicuna checkpoints, or any existing
  `experiments/` directory
- ❌ Reassign `NUM_HEADS = 8` instead of renaming it to `NUM_KV_HEADS`
- ❌ Weaken a `verify_*.py` assertion to make it pass

---

## 7. What stays non-comparable

Carry all four into the write-up. None is fatal; all are measurable.

1. **Qwen writes to a 320-token budget where era-1 Vicuna had 256, with a more efficient
   tokenizer on top.** Both differences push the same way: Qwen is markedly less likely to
   truncate, and truncation is a marker worth **−0.86** accuracy where it fires
   (`docs/adapter.md` Stage 6c). Era-1 Vicuna had 238/945 outputs ending mid-sentence. This is
   the largest of the four and it was taken deliberately (§1). **Report word-count distributions
   and truncation rates for both arms** so the reader can see the size of it.
2. **Prefix addressability under GQA** (§3) — volume restored, per-head addressing not.
3. **Per-token losses are not comparable across models**, because the tokenizers differ. Only
   the generated text and its judged scores are.
4. **~6% of nodes show one fewer reference than the 08-15 arm saw** — the consequence of
   freezing at reserve 320 (§1). Identical across both Qwen arms, so it does not reach the gap.

### And one threat that is not about comparability

Qwen3 is trained on far more scientific text than Llama-2 was, and ogbn-arxiv is pre-2020 arXiv
CS — certainly in the corpus. **If the LLM recalls the target abstract, both arms score high for
the same wrong reason and the gap collapses.** Low overlap with the gold is the experiment
working as designed, not a failure: `experiments/prompt_sensitivity_gpt/` measured a stronger
model scoring **+2.15** while its gold bigram overlap stayed at **0.027** — Vicuna's own figure —
which is what "better inference, not retrieval" looks like.

Re-run that folder's battery on the Qwen baseline arm, ideally before step 9:

- gold-only-name leak count (an acronym/CamelCase/alphanumeric identifier in the target's
  abstract and in **none** of the shown references — it cannot come from the prompt)
- bigram overlap with the gold (expect ≈ 0.03; era-1 Vicuna was 0.027)
- near-verbatim copies of a single cited reference (>0.5 overlap; era-1 Vicuna was **10.4%**,
  and those copies were 19–24% of everything scoring ≥6)

Plus the direct test that folder flags as never run: hand the model the reference list and ask
it to name the paper that cites all of them. ~48 calls.

If Qwen names them, the eval set is contaminated for this model and the swap stops here.

---

## 8. Files touched

| file | change |
|---|---|
| `hpc/komondor/containers/adapter-qwen-gpu.def` | **new** — keep the old one |
| `src/adapter/config.py` | model, architecture, prefix tokens, batch (reserve unchanged) |
| `src/adapter/model.py` | `NUM_KV_HEADS` rename, stale comment |
| `src/adapter/text.py` | ChatML wrapper; `SYSTEM`/`INSTRUCTION` unchanged |
| `src/adapter/neighbour_cap.py` | possibly untouched — the freeze happens in `text.py` |
| `src/adapter/prefix_generation.py` | re-derive the patch; modern `Cache` API |
| `src/adapter/train.py` | `PrefixCache` rewrite; sliced `lm_head` in `step_loss`/`validate` |
| `src/adapter/verify_generation.py` | `Qwen3Config`, `kv=2 q=4`, version pin |
| `src/adapter/verify_capped_prompts.py` | assert Qwen prompts fit with headroom |
| `src/adapter/verify_lora_prefix.py` | `kv=2 q=4` (not used this run, but keep it honest) |
| `data/frozen/prompt_refs_capped.json` | **new, versioned** — the frozen bibliographies |
| `hpc/komondor/jobs/run_train_adapter.sh` | new `.sif`, `LORA=0`, own `CKPT_DIR` |
| `hpc/komondor/jobs/run_generate_eval.sh` | new `.sif`, `RUN_BASELINE=1`, `RUN_GATE0=1` |
| `docs/adapter.md` | append **Stage 7** when the work is done (`CLAUDE.md` §5) |

Not touched: `graph.py`, `prepare_data.py`, `prepare_data_capped.py`, `lora.py`,
`src/evaluation/*`, `src/data/*`, `src/filters/*`, everything under `data/frozen/` except the
new file, and every existing `experiments/` directory.

---

## 9. Open questions

- **Will the gap survive?** It has been falling as the no-graph control was repaired
  (+0.473 → +0.431 → +0.343 across eras 1–3), and a stronger LLM repairs the control further on
  its own. Against that, `rubric_sensitivity` finding 2 says both arms are currently pinned near
  the floor — 76% of baseline outputs score ≤3 under R1 — so a squashed scale may be hiding a
  difference that exists. R3 also showed the adapter's relative advantage *growing* with question
  specificity (+4.7% on domain → +19.7% on scope). The swap is partly a bet on which of those
  dominates, and it is worth doing to find out.
- **Is `T = 32` enough, or does GQA's 5:1 sharing cost more than volume restores?** The `T = 10`
  arm answers it, at the price of a second training run.
- **`EPOCHS = 3`** was set for a 7B at batch 2 with 5.7M trainable parameters. It may be wrong
  for this run in either direction; step 8's measurement is what should decide.
