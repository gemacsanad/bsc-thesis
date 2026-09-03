"""Which generation runs the rubrics are scored over.

The comparison pair is era 1 (2026-08-15). Both arms come from the same generation job, the
same prompt and the same 945 capped eval nodes, and differ only in whether the graph prefix is
present — which is the whole point, and the reason a cross-era pair could not be used. The
2026-08-25 3-layer run the rubrics were originally to be scored on is era 4, and its baseline
was never judged; pairing it with the era-1 baseline would have put the four prompt rewrites
inside the measured graph effect.

The negative control is two era-4 adapter runs that differ only in repetition penalty
(1.15 vs 1.00, means 3.143 and 3.204 under R1). They are not identical, but the difference
between them has nothing to do with the graph. A rubric that widens the baseline-vs-adapter
gap *and* this one is not measuring the graph, it is measuring its own leniency spread.
"""

import os

EXPERIMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS = {
    "baseline": "2026-08-15-14207605_vicuna-baseline",   # frozen Vicuna-7B, no prefix
    "adapter":  "2026-08-15-14207605_adapter",           # GraphSAGE + prefix adapter
    "neg_a":    "2026-08-23-14233259_adapter-gnn2",      # era 4, repetition penalty 1.15
    "neg_b":    "2026-08-23-14233391_adapter-gnn2",      # era 4, repetition penalty 1.00
}

# The comparison, and the two guards against reading a rubric's own spread as a graph effect.
MAIN = ("baseline", "adapter")
NEGATIVE_CONTROL = ("neg_a", "neg_b")
RETEST = "adapter"        # off by default; score it twice under one rubric for a noise floor


def generations(key):
    return os.path.join(EXPERIMENTS, RUNS[key], "evaluated_result.csv")
