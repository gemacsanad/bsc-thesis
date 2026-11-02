# Style decisions

Preferences stated by the author. They override the defaults in `CLAUDE.md` §7.

| topic | decision | set on |
|---|---|---|
| spelling | British English (en_GB): -ise, -our, -tre, "modelling", "neighbour" | 2026-10-14 |
| voice | first person singular, "I" | 2026-10-14 |
| thesis title | kept exactly as assigned, including the US spelling "Neighborhood"; the body text uses en_GB "neighbourhood" | 2026-10-14 |
| citations in the sentence | state the information and put `~\cite{}` at the claim. Name the authors ("Gilmer et al.~\cite{…}") only where the paper really leads the sentence or the name adds value: presenting that paper's own contribution, or contrasting works | 2026-10-15 |
| background scope | keep the background minimal: leave out variants the project does not use (e.g. GraphSAGE's LSTM / pooling aggregators) and details that add complexity without being needed there (GAT multi-head attention was removed from Ch. 2) | 2026-10-24 |
| common knowledge | do not define basics the reader knows (e.g. what a graph is); introduce only the notation that later equations use | 2026-10-15 |
| empirical findings | scope a paper's results to its experiments, in past tense ("In the original experiments…", "Experiments with T5 on SuperGLUE showed…"); no wording that reads as a general guarantee ("strictly more expressive", "matches fine-tuning") | 2026-10-27 |
| order of explanation | open with a concrete, familiar case; say where a mechanism acts (which layer, which vectors) before any metaphor ("virtual tokens"); lead with the mechanism, not the authors | 2026-10-27 |
| precision of equations | state the scope of an equation (e.g. "for one attention head of layer $\ell$, with the causal mask omitted") and qualify dependencies exactly (e.g. prefix affects token keys/values only from the next layer on) | 2026-10-27 |
| numbers from papers | leave out parameter percentages and benchmark numbers unless the thesis uses them; keep the mechanism | 2026-10-27 |
| algorithms and comparisons | explain the intuition before details (e.g. why a connected subgraph, before prizes); write "seeks" rather than "selects the optimal" when an implementation only approximates an optimum; give each distinct comparison its own short paragraph; do not transfer another paper's result to this thesis's task | 2026-10-30 |
| scope of processing steps | describe only steps the final model (run 11) uses; earlier or reverted designs are not told as history (e.g. no reverted undirected graph, no whitening, no early relevance labels) | 2026-10-30 |
| tools and methods not used | do not mention methods the project did not use (e.g. no word2vec / OGB features); name a tool (e.g. DuckDB) only if really necessary, otherwise describe it generically ("SQL queries") | 2026-10-30 |
| data wording | say what the data records, not what exists: "no outgoing citations recorded in ogbn-arxiv", not "no references"; never "full reference list / bibliography"; "self-loop" (paper → itself), not "self-citation"; define populations explicitly (which papers are targets, which only supply neighbourhood information); avoid intermediate dataset states and counts that the argument does not need | 2026-10-30 |
| paper populations | "ogbn-arxiv papers" (the 169,343 original papers; the only prediction targets) and "added papers" (contributed by Semantic Scholar; neighbourhood information only), instead of the glossary's "core" / "S2-only"; introduced in §4.2, proposed 2026-10-30, to be confirmed by the author | 2026-10-30 |
| cross-references | `\autoref{fig:...}` ("Figure 2.1"), with a panel letter appended directly: `\autoref{fig:transformer}b`; first used 2026-10-27, not yet confirmed by the author | 2026-10-27 |

Still open: tense convention, notation; confirm the cross-reference style.
