"""
Stage 10 — the contrastive ranking loss, and the negatives it ranks against.

Why this exists
---------------
Step 6 produced the lowest training loss of any run in the project and its worst
conditioning at the same time. That is not a contradiction. The LM objective is
cross-entropy over ~230 abstract tokens, and that quantity is dominated by generic
scientific English: "We propose a novel approach to the problem of…" is high-probability
under *any* conditioning. Getting the specific paper right buys a few nats on a handful of
content words; writing fluent boilerplate buys more, on every token. So the model learned
the register and ignored the graph, and the loss went down the whole time.

Nothing in that objective ever asked the prefix to be about *this* paper. This one asks for
nothing else:

    under this node's own prefix, is the right abstract more likely than M wrong ones?

It is the identifiability metric the evaluation reports, turned into a differentiable loss.
No voyage embedding appears anywhere — the candidates are gold abstracts from the training
set — so the only ceiling is the LLM's own, and no third-party model enters the objective.

Where the candidates are scored, and why not after the prompt
--------------------------------------------------------------
Candidates are scored directly behind the KV prefix:

    positions 0 .. P-1      the prefix
    positions P .. P+T-1    one candidate's first T tokens

and NOT behind the 241-token prompt that `train.step_loss` uses. That is a compute decision
with a real cost, so it is stated rather than buried.

Scoring behind the prompt would mean recomputing those 241 tokens once per candidate:
9 x 273 = 2,457 tokens against a 471-token baseline, **+460%**. The obvious fix — cache the
prompt's keys and values once and score candidates against the cache — is not available
here. Gradient checkpointing forces `use_cache=False` (it runs each layer twice, and a
`DynamicCache` would append on both passes), which is the same constraint that
`train.PrefixCache` exists to work around. Scoring behind the prefix alone costs
(M+1) x T = 288 tokens, **+61%**, and reuses `PrefixCache` unchanged at batch M+1.

What it costs: every candidate sits at a different absolute distance from the prefix than it
does during LM training, and the model is not in its "assistant is answering" mode. Both are
*shared by all candidates*, so the softmax over them is still a valid relative ranking, and
the ranking is driven by the prefix — which is the thing being trained. `--rank-context
prompt` runs the expensive variant for anyone who wants to check that claim rather than
accept it.

The first candidate token is not scored: at position P it is predicted by the prefix alone,
whose hidden state lives in the cache and is not returned. So T tokens of candidate yield
T-1 tokens of signal.
"""

import torch
import torch.nn.functional as F

from config import RANK_HARD_FRACTION, RANK_STORED_TOKENS, RANK_TOKENS

MIN_CANDIDATE_WORDS = 30    # below this an "abstract" is a stub, and a stub is a free win for
                            # whichever candidate it is: it says almost nothing to be wrong
                            # about. Run 8 drops them from the pool rather than ranking against
                            # them. One node in 384,756 has an empty abstract.


def candidate_ids(tokenizer, corpus, node_ids, length=RANK_STORED_TOKENS):
    """-> ((len(node_ids), length) of tokens, right-padded; the real length of each row)

    Enough of each abstract is kept that a *window* can be sliced out of it later, rather than
    only the opening. Run 8 samples the window from a random offset during training: the
    opening states the problem and the middle states the method, and methodology is the stated
    goal. See NegativePool.window.

    This is not a correction of the old behaviour. The gold abstracts are 98.6% distinct by
    word 8 (7,645 of 7,750), so the opening was never formulaic and the old window was sound —
    it was aimed at the wrong part of the abstract, not at nothing.

    Only the first ~2,000 characters are tokenised, which covers 256 tokens with room to spare
    and keeps startup at seconds for 8,245 nodes.
    """
    out = torch.full((len(node_ids), length), tokenizer.pad_token_id, dtype=torch.long)
    lengths = torch.zeros(len(node_ids), dtype=torch.long)
    for i, node_id in enumerate(node_ids):
        text = str(corpus.abstract[node_id])[:2000]
        ids = tokenizer(text, add_special_tokens=False).input_ids[:length]
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths[i] = len(ids)
    return out, lengths


class NegativePool:
    """Tokenised abstract openings for every training node, plus the sampler over them.

    Built once at startup (7,750 short tokenisations, seconds) rather than per step: the same
    nodes are drawn thousands of times over a run, and re-tokenising them each time would be
    pure waste.
    """

    def __init__(self, tokenizer, corpus, node_ids, length=RANK_STORED_TOKENS, seed=0):
        # Stubs are dropped rather than ranked against: an abstract of a dozen words is a free
        # win for whichever candidate it is, because there is almost nothing in it to be wrong
        # about, and it teaches the prefix that shortness beats correctness.
        self.node_ids = [n for n in node_ids
                         if len(str(corpus.abstract[n]).split()) >= MIN_CANDIDATE_WORDS]
        self.index = {n: i for i, n in enumerate(self.node_ids)}
        self.ids, self.lengths = candidate_ids(tokenizer, corpus, self.node_ids, length)
        self.refs = corpus.refs
        self.generator = torch.Generator().manual_seed(seed)
        self.dropped = len(list(node_ids)) - len(self.node_ids)

    def window(self, row, tokens, random_offset):
        """-> (tokens,) sliced out of candidate `row`.

        `random_offset=False` returns the opening and is what validation uses, so `rank_acc`
        stays comparable across epochs and with step 7's numbers. Training uses True.
        """
        span = max(1, int(self.lengths[row]) - tokens)
        start = (int(torch.randint(span, (1,), generator=self.generator))
                 if random_offset and span > 1 else 0)
        return self.ids[row, start:start + tokens]

    def _hard(self, node_id, want):
        """The node's OWN references, which are the negatives that matter.

        A prefix that had only learned its neighbourhood's *topic* would rank the references
        as highly as the target — they are the topic. Forcing them apart is what turns a topic
        vector into an identity vector, and "carries topic but not paper identity" is how every
        run of this project so far has been described.
        """
        if want <= 0 or node_id not in self.refs.index:
            return []
        rows = [self.index[r] for r in self.refs[node_id] if r in self.index]
        if not rows:
            return []
        picked = torch.randperm(len(rows), generator=self.generator)[:want]
        return [rows[i] for i in picked.tolist()]

    def sample(self, node_id, count, tokens=RANK_TOKENS, random_offset=True,
               hard_fraction=RANK_HARD_FRACTION):
        """-> (count, tokens) of wrong abstracts: hard ones first, random filling the rest.

        Each negative gets its own window offset, so the gold is not the only candidate whose
        window moves — a shared offset would make position itself a cue.
        """
        rows = self._hard(node_id, int(round(count * hard_fraction)))
        taken = set(rows)
        self_row = self.index.get(node_id)
        while len(rows) < count:
            # Rejection rather than exclusion-by-construction: the pool is ~8,200 wide and at
            # most ~100 rows are ineligible, so a redraw is cheaper than building a mask.
            row = int(torch.randint(len(self.node_ids), (1,), generator=self.generator))
            if row != self_row and row not in taken:
                rows.append(row)
                taken.add(row)
        return torch.stack([self.window(r, tokens, random_offset) for r in rows])


def rank_loss(llm, prefix, gold, negatives, cache_cls, dtype, pad_id, prompt_ids=None):
    """Cross-entropy over [gold, *negatives], scored behind `prefix`. -> (loss, correct)

    `prefix` is one node's KV prefix, (1, layers, 2, heads, P, head_dim). `gold` is (T,) and
    `negatives` is (M, T); both are abstract openings. `cache_cls` is train.PrefixCache, passed
    in rather than imported so this module stays free of a circular import.

    `prompt_ids` is the expensive variant: (L,) of the shared prompt, placed between the prefix
    and every candidate. None scores candidates directly behind the prefix — see the module
    docstring for what that trades away.

    `correct` is whether the gold outscored every negative, which is the quantity model
    selection reads. LM validation loss cannot be used for that: step 6 had the lowest LM loss
    on record and the worst conditioning on record, so selecting on it prefers exactly the
    failure this loss exists to prevent.
    """
    candidates = torch.cat([gold.unsqueeze(0), negatives], dim=0)   # (M+1, T)
    n, length = candidates.shape
    device = prefix.device
    p = prefix.size(4)

    embeds = llm.model.embed_tokens(candidates)
    scored_from = 0
    if prompt_ids is not None:
        prompt = prompt_ids.unsqueeze(0).expand(n, -1)
        embeds = torch.cat([llm.model.embed_tokens(prompt), embeds], dim=1)
        scored_from = prompt.size(1)

    total = embeds.size(1)
    hidden = llm.model(
        inputs_embeds=embeds.to(dtype),
        # expand, not repeat: the prefix is identical for every candidate of one node, so this
        # is a stride-0 view and costs no memory. Gradients from all M+1 rows sum back into it.
        past_key_values=cache_cls(prefix.expand(n, *prefix.shape[1:])),
        attention_mask=torch.ones(n, p + total, dtype=torch.long, device=device),
        cache_position=torch.arange(p, p + total, device=device),
        use_cache=False,
    ).last_hidden_state

    # position i predicts token i+1, so the candidate's own first token is never scored: at
    # position P it is predicted by the prefix, whose hidden state is in the cache.
    logits = llm.lm_head(hidden[:, scored_from:-1]).float()
    targets = candidates[:, 1:]
    token_lp = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(n, -1)

    # Mean over real tokens, not sum. A summed log-probability rewards whichever candidate has
    # the fewest scored positions, so an abstract shorter than T would win on length alone.
    # Padding is excluded for the same reason: every candidate predicts a pad run equally well,
    # so including it drags the comparison toward chance.
    keep = (targets != pad_id).float()
    scores = (token_lp * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)

    loss = F.cross_entropy(scores.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
    return loss, bool(scores.argmax() == 0)
