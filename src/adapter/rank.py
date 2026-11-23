"""The contrastive ranking loss and the negatives it is computed against."""

import torch
import torch.nn.functional as F

from config import RANK_HARD_FRACTION, RANK_STORED_TOKENS, RANK_TOKENS

MIN_CANDIDATE_WORDS = 30    # shorter abstracts are stubs and are excluded


def candidate_ids(tokenizer, corpus, node_ids, length=RANK_STORED_TOKENS):
    """-> ((len(node_ids), length) of tokens, right-padded; the real length of each row)"""
    out = torch.full((len(node_ids), length), tokenizer.pad_token_id, dtype=torch.long)
    lengths = torch.zeros(len(node_ids), dtype=torch.long)
    for i, node_id in enumerate(node_ids):
        text = str(corpus.abstract[node_id])[:2000]
        ids = tokenizer(text, add_special_tokens=False).input_ids[:length]
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths[i] = len(ids)
    return out, lengths


class NegativePool:
    """Tokenised abstract openings for every training node, plus the sampler over them."""

    def __init__(self, tokenizer, corpus, node_ids, length=RANK_STORED_TOKENS, seed=0):
        self.node_ids = [n for n in node_ids
                         if len(str(corpus.abstract[n]).split()) >= MIN_CANDIDATE_WORDS]
        self.index = {n: i for i, n in enumerate(self.node_ids)}
        self.ids, self.lengths = candidate_ids(tokenizer, corpus, self.node_ids, length)
        self.refs = corpus.refs
        self.generator = torch.Generator().manual_seed(seed)
        self.dropped = len(list(node_ids)) - len(self.node_ids)

    def window(self, row, tokens, random_offset):
        """-> (tokens,) sliced out of candidate `row`."""
        span = max(1, int(self.lengths[row]) - tokens)
        start = (int(torch.randint(span, (1,), generator=self.generator))
                 if random_offset and span > 1 else 0)
        return self.ids[row, start:start + tokens]

    def _hard(self, node_id, want):
        """The node's OWN references, which are the negatives that matter."""
        if want <= 0 or node_id not in self.refs.index:
            return []
        rows = [self.index[r] for r in self.refs[node_id] if r in self.index]
        if not rows:
            return []
        picked = torch.randperm(len(rows), generator=self.generator)[:want]
        return [rows[i] for i in picked.tolist()]

    def sample(self, node_id, count, tokens=RANK_TOKENS, random_offset=True,
               hard_fraction=RANK_HARD_FRACTION):
        """-> (count, tokens) of wrong abstracts: hard ones first, random filling the rest."""
        rows = self._hard(node_id, int(round(count * hard_fraction)))
        taken = set(rows)
        self_row = self.index.get(node_id)
        while len(rows) < count:
            row = int(torch.randint(len(self.node_ids), (1,), generator=self.generator))
            if row != self_row and row not in taken:
                rows.append(row)
                taken.add(row)
        return torch.stack([self.window(r, tokens, random_offset) for r in rows])


def rank_loss(llm, prefix, gold, negatives, cache_cls, dtype, pad_id, prompt_ids=None):
    """Cross-entropy over [gold, *negatives], scored behind `prefix`. -> (loss, correct)"""
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
        past_key_values=cache_cls(prefix.expand(n, *prefix.shape[1:])),
        attention_mask=torch.ones(n, p + total, dtype=torch.long, device=device),
        cache_position=torch.arange(p, p + total, device=device),
        use_cache=False,
    ).last_hidden_state

    logits = llm.lm_head(hidden[:, scored_from:-1]).float()
    targets = candidates[:, 1:]
    token_lp = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(n, -1)

    keep = (targets != pad_id).float()
    scores = (token_lp * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)

    loss = F.cross_entropy(scores.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
    return loss, bool(scores.argmax() == 0)
