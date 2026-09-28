"""
Stage 9, step 1 — prove the neighbour tokens mean the same thing in training and generation.

Training feeds `[neighbour tokens ++ prompt]` to the model in one pass as `inputs_embeds`
(train.step_loss). Generation cannot: `generate()` in transformers 4.51.3 only forwards
`inputs_embeds` when `cache_position[0] == 0`, and with a prefix in the cache it never is.
So generation prefills the neighbour tokens into the cache first
(prefix_generation.prefill_neighbours) and then decodes on the `input_ids` path.

Two different code paths that must produce the same model state. Splitting the pass at the
neighbour/prompt boundary is exact in theory — attention is causal, so a neighbour token's
keys and values cannot depend on a prompt token that follows it — but "exact in theory" is
how this stage got the `cache_position` bug in the first place. Every way this can be wrong
produces fluent text rather than an exception, so it is checked before the GPU hours.

Method, same as verify_generation.py: compare two runs of the *same* model, so the weights
are irrelevant and a 4-layer random Qwen3 on CPU exercises the same forward,
`_update_causal_mask`, RoPE and `DynamicCache.update` as Qwen3-14B. Grouped-query attention
is on (4 query heads, 2 KV heads), because the prefix is sized by the KV count and a bug in
exactly that dimension is the one Qwen3 introduced.

The negative controls matter as much as the checks. Dropping the prefill, or leaving the
neighbour tokens out of the position arithmetic, must *fail* — otherwise the comparison is
vacuous.

Usage:
    cd src/adapter
    python verify_neighbour_tokens.py     # ~15 s on a laptop CPU
"""

import numpy as np
import torch
import transformers
from torch_geometric.data import Data
from transformers import Qwen3Config, Qwen3ForCausalLM

from graph import centroid_distances, select_neighbour_rows
from model import NeighbourTokenAdapter
from neighbour_cap import centroid_distances as centroid_distances_numpy
from neighbour_cap import rank_refs
from prefix_generation import prefill_neighbours, prefix_to_cache
from train import PrefixCache

assert transformers.__version__ == "4.51.3", f"expected 4.51.3, got {transformers.__version__}"

PREFIX_LEN, NUM_NEIGHBOURS, BODY_LEN = 5, 3, 12


def tiny_model():
    """A 4-layer random Qwen3 with GQA, small enough to run on a CPU in seconds."""
    config = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
    )
    torch.manual_seed(0)
    llm = Qwen3ForCausalLM(config).eval()
    return llm, config


def random_prefix(config, tokens):
    """(1, layers, 2, kv_heads, tokens, head_dim) — the shape PrefixAdapter.forward returns.

    Random rather than a real token span's KV, unlike verify_generation.py. That script had
    to prove the prefix lands at the *right position*, which needs a cache whose correct
    effect is already known. This one proves two code paths agree on the same cache, and a
    random prefix is the stricter choice: it shares no structure with anything the model
    would produce itself, so an accidental agreement cannot come from the weights.
    """
    return torch.randn(1, config.num_hidden_layers, 2, config.num_key_value_heads,
                       tokens, config.head_dim) * 0.1


def training_pass(llm, prefix, neighbours, input_ids):
    """The layout train.step_loss builds: one forward over [neighbours ++ prompt]."""
    p, k, length = prefix.size(4), neighbours.size(1), input_ids.size(1)
    embeds = torch.cat([neighbours, llm.model.embed_tokens(input_ids)], dim=1)
    with torch.no_grad():
        out = llm.model(
            inputs_embeds=embeds,
            past_key_values=PrefixCache(prefix),
            attention_mask=torch.ones(1, p + k + length, dtype=torch.long),
            cache_position=torch.arange(p, p + k + length),
            use_cache=False,
        ).last_hidden_state
    return out[:, k:]                      # the prompt positions only


def generation_pass(llm, prefix, neighbours, input_ids, prefill=True):
    """The layout generate.py builds: prefill the neighbours, then decode on input_ids."""
    p, length = prefix.size(4), input_ids.size(1)
    cache = prefix_to_cache(prefix)
    k = prefill_neighbours(llm, cache, neighbours) if prefill else 0
    with torch.no_grad():
        return llm.model(
            input_ids=input_ids,
            past_key_values=cache,
            attention_mask=torch.ones(1, p + k + length, dtype=torch.long),
            cache_position=torch.arange(p + k, p + k + length),
            use_cache=True,
        ).last_hidden_state


def check_selection():
    """select_neighbour_rows must take the seed's references, centroid-ordered and capped."""
    # A hand-built graph: node 0 cites 1..6, and 7 cites 0 (a later paper, which must never
    # be aggregated — the edge direction is the thing that could silently be backwards).
    torch.manual_seed(1)
    x = torch.randn(8, 16)
    cites = [(0, r) for r in range(1, 7)] + [(7, 0)]
    citing = torch.tensor([c for c, _ in cites])
    cited = torch.tensor([r for _, r in cites])
    data = Data(x=x, edge_index=torch.stack([cited, citing]), num_nodes=8)
    data.batch_size = 1
    data.n_id = torch.arange(8)

    rows = select_neighbour_rows(data, max_tokens=6)
    assert set(rows.tolist()) == {1, 2, 3, 4, 5, 6}, \
        f"1-hop rows are {rows.tolist()}, expected the seed's six references"
    assert 7 not in rows.tolist(), "a citing paper was aggregated — edge direction is reversed"
    print(f"  1-hop rows are exactly the seed's references: {sorted(rows.tolist())}")

    # The torch centroid score must equal neighbour_cap's numpy one, which is what the text
    # prompt trims by. If they drifted, the graph arm and the text arm would silently be
    # shown different references and the comparison between them would be meaningless.
    refs = list(range(1, 7))
    mine = centroid_distances(x[refs]).numpy()
    theirs = centroid_distances_numpy(np.asarray(x[refs], dtype=np.float32))
    assert np.abs(mine - theirs).max() < 1e-5, \
        f"torch and numpy centroid distances disagree by {np.abs(mine - theirs).max():.2e}"
    print(f"  centroid score matches neighbour_cap to "
          f"{np.abs(mine - theirs).max():.1e}")

    # And the cap keeps the same references neighbour_cap.rank_refs would keep, in the
    # same order — the guarantee that makes the graph and text arms comparable.
    capped = select_neighbour_rows(data, max_tokens=3).tolist()
    wanted = rank_refs(refs, x.numpy(), max_refs=3)
    assert capped == wanted, f"cap kept {capped}, neighbour_cap would keep {wanted}"
    print(f"  cap to 3 keeps neighbour_cap's choice in its order: {capped}")


def check_gate(llm):
    """gate=0 must leave every neighbour token identical — the ngate0 ablation."""
    torch.manual_seed(2)
    ntoken = NeighbourTokenAdapter(in_channels=32, hidden_size=llm.config.hidden_size,
                                   max_tokens=8).eval()
    ntoken.match_embeddings(llm.model.embed_tokens.weight)
    h = torch.randn(NUM_NEIGHBOURS, 32)

    with torch.no_grad():
        closed = ntoken(h)
    assert torch.allclose(closed, closed[:1].expand_as(closed), atol=1e-5), \
        "gate is zero but the neighbours still move their tokens"

    # The scale has to put the tokens where real embeddings live, or the LLM reads them
    # out of distribution and the arm fails for a reason that looks like the idea failing.
    real = llm.model.embed_tokens.weight.norm(dim=1).mean()
    assert torch.isclose(closed.norm(dim=1).mean(), real, rtol=0.25), \
        f"token norm {closed.norm(dim=1).mean():.3f} against embedding norm {real:.3f}"
    print(f"  gate=0: tokens identical, norm {closed.norm(dim=1).mean():.3f} "
          f"vs embedding norm {real:.3f}")

    with torch.no_grad():
        ntoken.gate.fill_(1.0)
        opened = ntoken(h)
    assert not torch.allclose(opened, opened[:1].expand_as(opened), atol=1e-5), \
        "opening the gate left every neighbour token identical"
    print("  gate=1: each neighbour token differs")
    return ntoken


def main():
    llm, config = tiny_model()
    torch.manual_seed(3)

    print("Neighbour selection")
    check_selection()

    print("\nNeighbour token adapter")
    check_gate(llm)

    print("\nTraining pass vs generation pass")
    prefix = random_prefix(config, PREFIX_LEN)
    neighbours = torch.randn(1, NUM_NEIGHBOURS, config.hidden_size) * 0.05
    input_ids = torch.randint(0, config.vocab_size, (1, BODY_LEN))

    train_hidden = training_pass(llm, prefix, neighbours, input_ids)
    gen_hidden = generation_pass(llm, prefix, neighbours, input_ids)
    assert train_hidden.shape == gen_hidden.shape, \
        f"{tuple(train_hidden.shape)} vs {tuple(gen_hidden.shape)}"
    drift = (train_hidden - gen_hidden).abs().max().item()
    assert drift < 1e-4, (
        f"training and generation disagree by {drift:.2e} — the two-pass prefill is not "
        f"the one-pass forward, so generation would decode against a state training never "
        f"produced")
    print(f"  identical to {drift:.1e} over {BODY_LEN} prompt positions")

    # Negative control 1: without the prefill the neighbour tokens are simply absent, and
    # the prompt sits k positions earlier. If this passed, the check above would be vacuous.
    skipped = generation_pass(llm, prefix, neighbours, input_ids, prefill=False)
    assert (train_hidden - skipped).abs().max().item() > 1e-3, \
        "dropping the neighbour prefill changed nothing — the tokens are not being read"
    print(f"  dropping the prefill moves the output by "
          f"{(train_hidden - skipped).abs().max().item():.2e} (control)")

    # Negative control 2: the same two passes with the neighbour tokens replaced. The state
    # has to depend on their *content*, not merely on k positions existing.
    other = torch.randn_like(neighbours) * 0.05
    assert (train_hidden - training_pass(llm, prefix, other, input_ids)).abs().max() > 1e-3, \
        "changing the neighbour tokens changed nothing — their content is not reaching the LLM"
    print("  changing their content moves the output (control)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
