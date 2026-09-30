"""
Stage 10 — prove the ranking loss measures what it claims, on a tiny random model.

Same convention as verify_generation.py and verify_neighbour_tokens.py: a 4-layer random
Qwen3 with GQA on, so the head layout is exercised and the whole thing runs on a CPU in
seconds. Weights are irrelevant to every claim below — each one is a statement about
plumbing, ordering or gradient flow, not about quality.

What would go wrong without it: a ranking loss that silently scores the wrong rows, or
averages over padding, or never reaches the encoder, produces a falling number and a
plausible training curve. It would look exactly like the run working.

Usage:
    python verify_rank.py
"""

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from model import PrefixAdapter
from rank import MIN_CANDIDATE_WORDS, NegativePool, rank_loss
from train import PrefixCache

PAD = 0


def tiny():
    """A 4-layer random Qwen3 with 4 query heads over 2 KV heads — GQA on, as in the real one."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(config).eval()
    return model, config


def make_prefix(config, tokens=8, batch=1, requires_grad=False):
    prefix = torch.randn(batch, config.num_hidden_layers, 2, config.num_key_value_heads,
                         tokens, config.head_dim) * 0.02
    return prefix.requires_grad_(requires_grad)


class FakeCorpus:
    """The two attributes NegativePool reads, over a handful of synthetic abstracts.

    `refs` is a pandas Series because that is what Corpus builds and what `_hard` indexes.
    """

    def __init__(self, abstracts, refs):
        import pandas as pd
        self.abstract = abstracts
        self.refs = pd.Series(refs, dtype=object)


class FakeTokenizer:
    """Whitespace tokenisation with ids 1..N. Ids are irrelevant to every claim below —
    each one is about which SLICE of a candidate is returned, not about what it means."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return type("Encoding", (), {
            "input_ids": [1 + (hash(w) % 200) for w in str(text).split()]})()


def check_pool():
    """The windowing added in run 8, and the stub filter beside it.

    What would go wrong without this: a window that is always the opening reproduces step 7
    while the config says otherwise, and a window that runs off the end of a short abstract
    returns padding — which every candidate predicts equally well, dragging the whole
    comparison toward chance. Both produce a plausible falling curve.
    """
    words = {n: " ".join(f"w{n}x{i}" for i in range(120)) for n in range(6)}
    words[6] = "far too short to rank against"          # under MIN_CANDIDATE_WORDS
    corpus = FakeCorpus(words, {n: [] for n in range(7)})
    pool = NegativePool(FakeTokenizer(), corpus, list(range(7)), length=64, seed=0)

    assert 6 not in pool.index and pool.dropped == 1, (
        f"the {MIN_CANDIDATE_WORDS}-word floor did not drop the stub: {pool.dropped} dropped")
    print(f"stubs dropped from the pool: {pool.dropped}")

    opening = pool.window(0, 16, random_offset=False)
    assert torch.equal(opening, pool.ids[0, :16]), \
        "the deterministic window is not the abstract opening — validation would move per call"
    again = pool.window(0, 16, random_offset=False)
    assert torch.equal(opening, again), "the deterministic window is not deterministic"
    print("validation window is the opening, and is stable across calls")

    seen = {tuple(pool.window(0, 16, random_offset=True).tolist()) for _ in range(40)}
    assert len(seen) > 1, \
        "the random window never moved — training would reproduce step 7's objective"
    print(f"random window takes {len(seen)} distinct positions out of 40 draws")

    # A window sliced past the end would be pad, and pad is a free win for every candidate.
    for _ in range(100):
        w = pool.window(0, 16, random_offset=True)
        assert (w != FakeTokenizer.pad_token_id).all(), \
            "a random window ran off the end of the abstract and returned padding"
    print("no random window runs into padding")

    negs = pool.sample(0, 4, tokens=16, random_offset=False)
    assert negs.shape == (4, 16), f"sample returned {tuple(negs.shape)}, expected (4, 16)"
    # Deterministic windows on both sides, so this compares candidates and not offsets.
    own = pool.window(0, 16, random_offset=False)
    assert not any(torch.equal(negs[i], own) for i in range(4)), \
        "a node was sampled as its own negative"
    print(f"sample returns {tuple(negs.shape)} and never the node itself")


def main():
    llm, config = tiny()
    torch.manual_seed(1)
    M, T = 5, 12

    gold = torch.randint(1, 256, (T,))
    negatives = torch.randint(1, 256, (M, T))
    prefix = make_prefix(config)

    loss, correct = rank_loss(llm, prefix, gold, negatives, PrefixCache, torch.float32, PAD)
    print(f"loss {loss.item():.4f}  gold won: {correct}")
    assert loss.ndim == 0, "the loss must be a scalar"
    # log(M+1) is the loss of a model that cannot tell the candidates apart. A random model
    # scores near it; anything far below on random weights would mean the gold row is being
    # read differently from the negatives.
    chance = torch.log(torch.tensor(float(M + 1)))
    assert abs(loss.item() - chance) < 2.0, (
        f"loss {loss.item():.3f} is implausibly far from chance {chance:.3f} on a random "
        f"model — the gold row is probably not being scored the same way as the negatives")
    print(f"near chance log(M+1) = {chance:.4f}, as a random model should be")

    # 1. The gold must win when it IS the argmax by construction: duplicate it into every
    #    negative slot and the scores must tie, so the loss is exactly log(M+1).
    tied, _ = rank_loss(llm, prefix, gold, gold.unsqueeze(0).expand(M, -1).contiguous(),
                        PrefixCache, torch.float32, PAD)
    assert torch.allclose(tied, chance, atol=1e-4), (
        f"identical candidates scored {tied.item():.5f}, expected exactly log(M+1) = "
        f"{chance:.5f} — the candidates are not being scored independently")
    print(f"identical candidates tie at exactly log(M+1): {tied.item():.5f}")

    # 2. Padding must not vote. Appending pad to every candidate changes how many positions
    #    exist but not how many are scored, so the loss must not move.
    pad = torch.full((4,), PAD, dtype=torch.long)
    g2 = torch.cat([gold, pad])
    n2 = torch.cat([negatives, pad.unsqueeze(0).expand(M, -1)], dim=1)
    padded, _ = rank_loss(llm, prefix, g2, n2, PrefixCache, torch.float32, PAD)
    assert torch.allclose(padded, loss, atol=1e-4), (
        f"padding moved the loss {loss.item():.5f} -> {padded.item():.5f}; pad tokens are "
        f"being scored, which favours whichever candidate is shortest")
    print(f"padding does not move the loss: {loss.item():.5f} -> {padded.item():.5f}")

    # 3. Row order must matter in the right direction: the loss is defined with the gold at
    #    index 0, so swapping the gold for a negative must change the number. A loss that
    #    ignored the label would pass every test above and train on nothing.
    swapped, _ = rank_loss(llm, prefix, negatives[0], torch.cat([gold.unsqueeze(0),
                                                                negatives[1:]]),
                           PrefixCache, torch.float32, PAD)
    assert not torch.allclose(swapped, loss, atol=1e-6), \
        "swapping the gold with a negative did not change the loss — the label is ignored"
    print(f"the gold's identity matters: {loss.item():.5f} vs swapped {swapped.item():.5f}")

    # 4. Gradients must reach the prefix. This is the whole point: the ranking loss exists to
    #    train the graph channel, and a detached prefix would train nothing while still
    #    producing a falling curve.
    prefix = make_prefix(config, requires_grad=True)
    loss, _ = rank_loss(llm, prefix, gold, negatives, PrefixCache, torch.float32, PAD)
    loss.backward()
    assert prefix.grad is not None and prefix.grad.abs().sum() > 0, \
        "no gradient reached the prefix"
    print(f"gradient reaches the prefix: |grad| = {prefix.grad.abs().sum():.3e}")

    # 5. And all the way through a real PrefixAdapter to the graph vector, which is the path
    #    that actually matters in train.py.
    adapter = PrefixAdapter(in_channels=16, trunk_dim=32, code_dim=8,
                            num_layers=config.num_hidden_layers,
                            num_kv_heads=config.num_key_value_heads,
                            head_dim=config.head_dim, num_prefix_tokens=8)
    adapter.calibrate_gate(torch.randn(4, 16), fraction=0.2)
    g = torch.randn(1, 16, requires_grad=True)
    loss, _ = rank_loss(llm, adapter(g), gold, negatives, PrefixCache, torch.float32, PAD)
    loss.backward()
    assert g.grad is not None and g.grad.abs().sum() > 0, \
        "no gradient reached the graph vector through the adapter"
    print(f"gradient reaches the graph vector through PrefixAdapter: "
          f"|grad| = {g.grad.abs().sum():.3e}")

    # 6. A calibrated gate must actually open the channel — gate=0 was step 6's initialisation
    #    and it is what this run is changing. If calibration silently left it shut, the run
    #    would reproduce the failure it is meant to fix.
    shut = PrefixAdapter(in_channels=16, trunk_dim=32, code_dim=8,
                         num_layers=config.num_hidden_layers,
                         num_kv_heads=config.num_key_value_heads,
                         head_dim=config.head_dim, num_prefix_tokens=8)
    vectors = torch.randn(8, 16)
    assert torch.equal(shut(vectors), shut(torch.randn(8, 16))), \
        "gate starts non-zero; the gate=0 ablation would no longer be the same weights"
    value = shut.calibrate_gate(vectors, fraction=0.25)
    assert value != 0, "calibrate_gate left the gate at zero"
    with torch.no_grad():
        moved = (shut(vectors) - shut.base).flatten(1).norm(dim=1).mean() / shut.base.norm()
    assert abs(moved - 0.25) < 0.02, \
        f"asked for a graph fraction of 0.25 and got {moved:.3f}"
    print(f"calibrate_gate(0.25) -> gate {value:.5f}, measured graph fraction {moved:.3f}")

    check_pool()
    print("\nall ranking-loss checks passed")


if __name__ == "__main__":
    main()
