# Info tank — Chapter 2: Background and related work

The facts Chapter 2 is written from. Each one carries its source. Plan:
`plans/02-background.md`; reading list: `plans/reading_list.md`.

**How this file was made.** Every paper fact below was read from the paper itself: the arXiv
PDF, converted to text with `pdftotext`. Locations (§, Eq., Def., Thm., Table) refer to that
arXiv version. Text in "quotes" is verbatim; everything else is paraphrase. Project facts come
from `thesis_context/` (file named) or from the code (file and line named).

**Status:** §2.1 (graph neural networks) filled 2026-10-15. §2.2–2.6 still empty.

---

## 2.1 Graph neural networks

### Bibliographic records (venue as printed in the paper)

| key (proposed) | record | arXiv |
|---|---|---|
| `gilmer2017neural` | J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl. *Neural Message Passing for Quantum Chemistry.* Proc. 34th ICML, PMLR 70, 2017. | 1704.01212 (v2) |
| `hamilton2017inductive` | W. L. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning on Large Graphs.* NIPS 2017 (31st Conference on Neural Information Processing Systems). | 1706.02216 (v4) |
| `velickovic2018graph` | P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. *Graph Attention Networks.* ICLR 2018. | 1710.10903 (v3) |
| `brody2022attentive` | S. Brody, U. Alon, E. Yahav. *How Attentive are Graph Attention Networks?* ICLR 2022. | 2105.14491 (v3) |
| `li2018deeper` | Q. Li, Z. Han, X.-M. Wu. *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning.* AAAI 2018 (copyright line: AAAI 2018). | 1801.07606 (v1) |
| `zhu2020beyond` | J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, D. Koutra. *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs.* NeurIPS 2020. | 2006.11468 (v2) |
| `mcpherson2001birds` | M. McPherson, L. Smith-Lovin, J. M. Cook. *Birds of a Feather: Homophily in Social Networks.* Annual Review of Sociology 27(1), 415–444, 2001. doi:10.1146/annurev.soc.27.1.415 | — (record from Crossref) |

Still needed for the bib: the page ranges of the conference papers. DBLP did not answer from
this machine; take them from DBLP when the entries are added.

---

### 2.1.1 Graphs with text on the nodes; homophily

**Homophily: the principle (McPherson et al. 2001, abstract, via Crossref).**
- Opens with "Similarity breeds connection. This principle—the homophily principle—structures
  network ties of every type…".
- It is a sociology paper about social networks. Cite it only for the term and its origin,
  not for anything about citation graphs or GNNs.

**Homophily in GNNs (Zhu et al. 2020).**
- §1: "Homophily is a key principle of many real-world networks, whereby linked nodes often
  belong to the same class or have similar features ('birds of a feather flock together')".
  Their example: "papers tend to cite papers from the same research area" (they cite Newman,
  *Networks*, 2018, for this).
- §1: "GNNs model the homophily principle by propagating features and aggregating them within
  various graph neighborhoods".
- Heterophily, §1: "linked nodes are likely from different classes or have dissimilar
  features".
- **Def. 1, edge homophily ratio:** the fraction of edges that connect nodes with the same
  class label,
  $h = \frac{|\{(u,v) : (u,v) \in \mathcal{E} \wedge y_u = y_v\}|}{|\mathcal{E}|}$.
- **Def. 2:** strong homophily means $h \to 1$; strong heterophily means $h \to 0$.
- Abstract: under heterophily, "Many popular GNNs fail to generalize to this setting, and are
  even outperformed by models that ignore the graph structure (e.g., multilayer perceptrons)."
- **Table 5, $h$ of the citation benchmarks:** Cora 0.81, Pubmed 0.8, Citeseer 0.74,
  Cora Full 0.57. For contrast, the heterophilous web/actor graphs have 0.11–0.3.
- **Their three designs:** ego- and neighbour-embedding separation; higher-order
  neighbourhoods; combining intermediate representations. §2 credits GraphSAGE with modelling
  "the ego-features distinctly from the neighbor-features".
- **Caution:** $h$ is defined on class labels. The thesis measures similarity in embedding
  space instead (the cosine of the reference mean to the target). That is related to
  homophily but is not Zhu's $h$. Say "homophily in embedding space", or define the measure.

**Text-attributed graph: notation the thesis needs** (no single source defines it; this is a
proposal).
- A graph $G=(V,E)$; each node $v$ has a text $t_v$ (here: the abstract) and a feature vector
  $\mathbf{x}_v$ (here: its 1024-d Voyage embedding of $t_v$);
- $\mathcal{N}(v)$ is the neighbourhood.
- Record the final notation in `plans/style.md`.

**Project facts this subsection prepares for:**
- The reference mean has cosine 0.823 to the target's own embedding on the 945 eval papers
  (0.816 on the 806 validation papers). It retrieves the target at 64.3% top-1 among the 945
  (`02_data.md` §3).
- Interpretation in `09` §5, labelled as one: "on homophilous graphs like citations, the
  average of neighbours is already a strong estimator of a node's own representation, which
  limits what a GNN can add".
- The edge direction in the project is cited → citing. A paper aggregates only the works it
  cites, because citing papers would leak the future (`02_data.md` §4).
- A homophily-rate figure (category agreement, embedding cosines) is only *proposed* in
  `02_data.md`; it has not been made. Do not quote numbers for it.

---

### 2.1.2 Message passing; over-smoothing

**The framework (Gilmer et al. 2017, §2).**
- They reformulate existing models "into a single common framework we call Message Passing
  Neural Networks (MPNNs)" (abstract).
- Set-up: undirected graphs with node features $x_v$ and edge features $e_{vw}$. "It is
  trivial to extend the formalism to directed multigraphs."
- The forward pass has a **message passing phase** of $T$ time steps and a **readout phase**:
  - Eq. (1): $m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})$
  - Eq. (2): $h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$
  - Eq. (3): $\hat{y} = R(\{h_v^T \mid v \in G\})$
- "The message functions $M_t$, vertex update functions $U_t$, and readout function $R$ are
  all learned differentiable functions."
- $R$ "must be invariant to permutations of the node states".
- The models §2 recasts as MPNNs:
  - Duvenaud et al. 2015;
  - Li et al. 2016 (GG-NN);
  - Battaglia et al. 2016 (Interaction Networks);
  - Kearnes et al. 2016;
  - Schütt et al. 2017 (DTNN);
  - Bruna et al. 2013, Defferrard et al. 2016 and Kipf & Welling 2016 (the "Laplacian based
    methods").
- Their own application is molecules (QM9). Only the framework is relevant here.
- **For the thesis:**
  - the readout $R$ is graph-level; the thesis task is node-level and has no readout.
  - The thesis's per-node output is $h_v^T$ followed by the linear head.
  - The sum in Eq. (1) is the "aggregate"; Brody et al. write the same idea as
    $h_i' = f_\theta(h_i, \mathrm{AGGREGATE}(\{h_j \mid j \in \mathcal{N}_i\}))$ (Brody Eq. 1),
    which may be the simpler form to show.

**Depth = receptive field.**
- GraphSAGE §3.1: "at each iteration, or search depth, nodes aggregate information from their
  local neighbors, and as this process iterates, nodes incrementally gain more and more
  information from further reaches of the graph."
- So $K$ layers reach the $K$-hop neighbourhood. This is a direct reading of the recursion;
  no paper states it as a theorem.

**Over-smoothing (Li, Han & Wu 2018).**
- Abstract: "the graph convolution of the GCN model is actually a special form of Laplacian
  smoothing, which is the key reason why GCNs work, but it also brings potential concerns of
  over-smoothing with many convolutional layers."
- §"Why GCNs work":
  - GCN propagation is Eq. (8), $Y = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X$, with
    $\tilde{A} = A + I$.
  - Laplacian smoothing is Eq. (9), $\hat{y}_i = (1-\gamma)x_i + \gamma\sum_j
    \frac{\tilde{a}_{ij}}{d_i}x_j$.
  - The GCN convolution is the symmetric version of it with $\gamma = 1$.
  - "Since vertices in the same cluster tend to be densely connected, the smoothing makes
    their features similar, which makes the subsequent classification task much easier."
- §1: "If a GCN is deep with many convolutional layers, the output features may be
  over-smoothed and vertices from different clusters may become indistinguishable."
- **Thm. 1:** for a graph with no bipartite components, repeated smoothing makes the features
  converge to a linear combination of the indicator vectors of the connected components. For
  the symmetric version this is scaled by $D^{-1/2}$. The paper reads this as follows: within
  a connected component, all vertices converge to the same values (proportional to the square
  root of the degree, for the symmetric version).
- **Illustration (Fig. 2):** untrained GCNs with 1–5 layers on Zachary's karate club (34
  vertices). Two layers separate the classes; 3–5 layers mix them.
- **Caution, for honesty:**
  - Li et al.'s experiments report that 2-layer GCNs performed best. They attribute the drop
    with more layers to overfitting ("probably due to overfitting", §Experiments), not
    explicitly to over-smoothing.
  - The over-smoothing result is about *many* layers. The thesis's grid compares only 1 vs 2
    layers.
  - So do not write "2 layers were worse *because of* over-smoothing". Write it as a possible
    explanation, next to the fact (below) that 2-hop neighbourhoods were also worse with no
    learning at all.

**Project facts this subsection prepares for (`03_final_model.md` §4.2, `06_results.md` §4,
`09` §2):**
- GNN grid: 2-layer variants are worse than 1-layer ones. Eval cosine:
  - GATv2 cos: 0.833 (1 layer) vs 0.822 (2 layers);
  - SAGE cos: 0.825 vs 0.821.
- Without learning, a uniform 1+2-hop mean retrieves the target at 33.2%, against 64.3% for
  the 1-hop mean (`09` §2).
- Undirected edges were tried first and reverted. Hub papers (in-degree up to 16,513) inflated
  2-hop neighbourhoods to 29,650 nodes at p99 and 94,721 at worst (`02_data.md` §4).
- Binding (`09` §2): do not claim that structure beyond 1 hop helps.

---

### 2.1.3 GraphSAGE

**Problem it solves (Hamilton et al. 2017, abstract):**
- earlier node-embedding approaches "are inherently transductive and do not naturally
  generalize to unseen nodes";
- GraphSAGE is "a general inductive framework that leverages node feature information (e.g.,
  text attributes) to efficiently generate node embeddings for previously unseen data";
- "Instead of training individual embeddings for each node, we learn a function that
  generates embeddings by sampling and aggregating features from a node's local neighborhood."

**Algorithm 1 (forward propagation):**
- line 1: $h_v^0 \leftarrow x_v$;
- line 2: for $k = 1 \ldots K$, for each node $v$:
  - line 4: $h^k_{\mathcal{N}(v)} \leftarrow \mathrm{AGGREGATE}_k(\{h_u^{k-1}, \forall u \in
    \mathcal{N}(v)\})$;
  - line 5: $h_v^k \leftarrow \sigma\big(W^k \cdot \mathrm{CONCAT}(h_v^{k-1},
    h^k_{\mathcal{N}(v)})\big)$;
- line 7: $h_v^k \leftarrow h_v^k / \lVert h_v^k \rVert_2$;
- line 9: $z_v \leftarrow h_v^K$.

**Concatenation as a skip connection (§3.3):**
- The concatenation keeps the node's own representation separate from its neighbours'.
- The paper calls it "a simple form of a 'skip connection'" that "leads to significant gains
  in performance".
- Note: $W \cdot [h_v \Vert h_{\mathcal{N}(v)}] = W_1 h_v + W_2 h_{\mathcal{N}(v)}$, writing
  $W$ as two column blocks. This is the form PyG implements (below), and it is how
  `01_literature.md` describes it ("a separate root (self) weight and a neighbour weight").

**Aggregators (§3.3).** Neighbours "have no natural ordering", so the aggregator should be
symmetric (permutation invariant). Three are examined:
- **Mean:** the element-wise mean of $\{h_u^{k-1}\}$.
  - The "GCN" inductive variant is Eq. (2):
    $h_v^k \leftarrow \sigma(W \cdot \mathrm{MEAN}(\{h_v^{k-1}\} \cup \{h_u^{k-1}, \forall u
    \in \mathcal{N}(v)\}))$, with no concatenation.
  - The paper says this "differs from Kipf et al's exact equation by a minor normalization
    constant".
- **LSTM:** applied to "a random permutation of the node's neighbors"; not symmetric.
- **Pooling,** Eq. (3): $\mathrm{AGGREGATE}^{\mathrm{pool}}_k = \max(\{\sigma(W_{\mathrm{pool}}
  h_{u_i}^k + b), \forall u_i \in \mathcal{N}(v)\})$, element-wise max.

**Sampling (§3.1):**
- "we uniformly sample a fixed-size set of neighbors, instead of using full neighborhood
  sets", to keep the per-batch cost fixed at $O(\prod_{i=1}^K S_i)$;
- "high performance with $K = 2$ and $S_1 \cdot S_2 \le 500$"; the experiments use
  $S_1 = 25$, $S_2 = 10$.

**Findings:**
- §4.3: $K = 2$ gave "a consistent boost in accuracy of around 10–15%, on average, compared
  to $K = 1$"; going beyond 2 gave 0–5% for a 10–100× runtime.
- §4.4: "the LSTM- and pool-based aggregators performed the best".
- One of their benchmarks is a citation graph: the Web of Science citation dataset (§4).
- **Contrast to note:** their $K = 2$ beat $K = 1$; in this project 1 layer beat 2 (2.1.2).
  The tasks differ (node classification vs estimating a node's own embedding), so this is not
  a contradiction. It is worth one sentence.

**Relation to Weisfeiler–Lehman (§3.1):** GraphSAGE is "a continuous approximation to the WL
test". This is optional and probably beyond the budget.

**Project facts:**
- The library is PyG 2.8.0 (`01_literature.md`; version confirmed in the installed package).
- The project uses `SAGEConv(in_dim, out_dim, aggr="mean")` (`src/adapter/model.py:87`).
- PyG's documented `SAGEConv` is $\mathbf{x}'_i = \mathbf{W}_1\mathbf{x}_i + \mathbf{W}_2
  \cdot \mathrm{mean}_{j\in\mathcal{N}(i)}\mathbf{x}_j$, with `root_weight=True` by default
  (PyG source docstring). That is Algorithm 1 line 5 without the nonlinearity or the
  normalisation inside the layer.
- The project uses all references, with no sampling cap (`03_final_model.md` §2.1). This
  differs from GraphSAGE's fixed-size sampling; say so when describing it.
- Earlier runs used a 2-layer SAGE encoder (history, `01_literature.md`). In run 11, SAGE is
  one of the two grid architectures.
- Eval cosine: 1-layer SAGE cos 0.825 vs the plain reference mean 0.823 (`03_final_model.md`
  §4.2).

---

### 2.1.4 GAT and GATv2

**GAT (Veličković et al. 2018, §2.1).**
- Shared linear map $W \in \mathbb{R}^{F' \times F}$, then the attention coefficient,
  Eq. (1): $e_{ij} = a(W\vec{h}_i, W\vec{h}_j)$, "the importance of node j's features to
  node i".
- Masked attention:
  - "we only compute $e_{ij}$ for nodes $j \in \mathcal{N}_i$";
  - in all their experiments $\mathcal{N}_i$ is "exactly the first-order neighbors of $i$
    (including $i$)".
- Eq. (2): $\alpha_{ij} = \mathrm{softmax}_j(e_{ij}) = \exp(e_{ij}) / \sum_{k\in\mathcal{N}_i}
  \exp(e_{ik})$.
- $a$ is a single-layer feedforward network with weight vector $\vec{a} \in
  \mathbb{R}^{2F'}$, and LeakyReLU with negative slope 0.2. Fully expanded, Eq. (3):
  $\alpha_{ij} = \frac{\exp(\mathrm{LeakyReLU}(\vec{a}^T[W\vec{h}_i \Vert W\vec{h}_j]))}
  {\sum_{k\in\mathcal{N}_i}\exp(\mathrm{LeakyReLU}(\vec{a}^T[W\vec{h}_i \Vert W\vec{h}_k]))}$.
- Output, Eq. (4): $\vec{h}_i' = \sigma(\sum_{j\in\mathcal{N}_i}\alpha_{ij}W\vec{h}_j)$.
- **Multi-head:**
  - Eq. (5) concatenates $K$ heads: $\vec{h}_i' = \Vert_{k=1}^{K}\sigma(\sum_j
    \alpha^k_{ij}W^k\vec{h}_j)$, giving $KF'$ features;
  - on the final prediction layer, Eq. (6) averages the heads instead.
  - Motivation: multi-head attention was found "beneficial" "to stabilize the learning
    process of self-attention", "similarly to Vaswani et al. (2017)".
- **Stated advantages (§2.2):**
  - cost $O(|V|FF' + |E|F')$ per head, "on par with" GCN;
  - "(implicitly) assigning different importances to nodes of a same neighborhood";
  - no need for "upfront access to the global graph structure";
  - "The graph is not required to be undirected";
  - "directly applicable to inductive learning";
  - unlike GraphSAGE, "it works with the entirety of the neighborhood" and "does not assume
    any ordering within it".

**GATv2 (Brody, Alon & Yahav 2022).**
- **The problem (abstract):** "GAT computes a very limited kind of attention: the ranking of
  the attention scores is unconditioned on the query node."
- **Def. 3.1, static attention.** A family of scoring functions computes static scoring if
  every $f$ has a "highest scoring" key $j_f$ such that, for every query $i$ and key $j$,
  $f(q_i, k_{j_f}) \ge f(q_i, k_j)$.
- **Def. 3.2, dynamic attention.** For any mapping $\varphi: [m] \to [n]$ there is an $f$
  such that, for every query $i$ and every key $j \ne \varphi(i)$,
  $f(q_i, k_{\varphi(i)}) > f(q_i, k_j)$. In other words, each query can pick its own top key.
- **Thm. 1:** "A GAT layer computes only static attention, for any set of node
  representations".
  - Proof idea: write $a = [a_1 \Vert a_2]$; then Eq. (5) is
    $e(h_i, h_j) = \mathrm{LeakyReLU}(a_1^\top W h_i + a_2^\top W h_j)$.
  - The node maximising $a_2^\top W h_j$ is the top key for every query, because LeakyReLU
    and softmax are monotonic.
  - Consequence (§3.2): "The only effect of $h_i$ is in the 'sharpness' of the produced
    attention distribution."
  - With multiple heads, Thm. 1 holds for each head separately.
- **The fix (§3.3):** "the learned layers $W$ and $a$ are applied consecutively, and thus can
  be collapsed into a single linear layer". GATv2 applies $a$ after the nonlinearity and $W$
  after the concatenation, "effectively applying an MLP to compute the score for each
  query-key pair":
  - GAT, Eq. (6): $e(h_i, h_j) = \mathrm{LeakyReLU}(a^\top \cdot [W h_i \Vert W h_j])$;
  - GATv2, Eq. (7): $e(h_i, h_j) = a^\top \mathrm{LeakyReLU}(W \cdot [h_i \Vert h_j])$;
  - footnote: a bias vector is also added before the nonlinearity; it is omitted from
    Eq. (7).
- **Thm. 2:** "A GATv2 layer computes dynamic attention for any set of node
  representations". The proof (App. A) rests on universal approximation.
- **Cost:**
  - "same time-complexity as GAT's declared complexity: $O(|V|dd' + |E|d')$" (§3.3);
  - App. G writes $W = [W_1 \Vert W_2]$, so $W\cdot[h_i \Vert h_j] = W_1 h_i + W_2 h_j$.
  - Parameters: GAT has $2d' + dd'$; GATv2 has $d' + 2dd'$. In their experiments they
    constrain $W = [W' \Vert W']$ ($d' + dd'$ parameters) "to rule out the increased number
    of parameters over GAT as the source of empirical difference" (§4, App. G.2).
- **Results:**
  - abstract: "GATv2 outperforms GAT across 12 OGB and other benchmarks while we match their
    parametric costs";
  - Table 1a, **ogbn-arxiv** accuracy (10 runs ± std):
    - GCN 71.74±0.29;
    - GraphSAGE 71.49±0.27;
    - GAT 71.59±0.38 (1 head), 71.54±0.30 (8 heads);
    - GATv2 71.78±0.18 (1 head), 71.87±0.25 (8 heads).
    - The GCN and GraphSAGE rows are taken from Hu et al. 2020.
  - **Caution:** the arXiv differences are about 0.3 points, within a couple of standard
    deviations. Do not use this as "GATv2 is much better". It shows that GATv2 matches or
    slightly exceeds GAT on the same dataset the thesis builds on.
  - §4.2: GATv2 degrades more gently under added noise edges on ogbn-arxiv and ogbn-mag
    (Fig. 4). Their hypothesis: dynamic attention helps tell data edges from noise edges.

**How the project's GATv2 relates to the paper (code + `03_final_model.md` §2.1):**
- The layer is `GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=0.0)`
  (`src/adapter/model.py:89`): 4 heads × 256 channels, concatenated to 1024, then a 1024 ×
  1024 linear head and L2 normalisation.
- PyG's documented `GATv2Conv` (PyG source docstring, v2.8.0):
  - $\alpha_{i,j} \propto \exp(\mathbf{a}^\top\mathrm{LeakyReLU}(\Theta_s\mathbf{x}_i +
    \Theta_t\mathbf{x}_j))$, normalised over $j \in \mathcal{N}(i)\cup\{i\}$;
  - output $\mathbf{x}'_i = \sum_{j\in\mathcal{N}(i)\cup\{i\}}\alpha_{i,j}\Theta_t\mathbf{x}_j$.
  - This is Eq. (7) with $W = [\Theta_s \Vert \Theta_t]$ unconstrained. The installed
    defaults are `share_weights=False`, `negative_slope=0.2` and `add_self_loops=True`.
- `03_final_model.md` §2.1 writes the same thing as $e_{ij} = a^\top\mathrm{LeakyReLU}(W_l x_j
  + W_r x_i)$ and $h_i = \sum_j \alpha_{ij} W_l x_j$, with $x_i$ the `[MASK]` vector.
- **Discrepancy to settle before writing Ch. 5:**
  - with PyG's default `add_self_loops=True`, the sum also runs over $j = i$. The target then
    attends to its own `[MASK]` row as well as to its references.
  - `03_final_model.md` does not mention this. From reading the code (not from running it):
    - `src/adapter/graph.py:72` removes self-loops from the stored `edge_index`;
    - `GATv2Conv` is built without `add_self_loops=False` (`model.py:89`), so it adds them
      back at run time.
    - So the target most likely attends to its own `[MASK]` row. Likewise SAGE's root term
      $W_1 x_i$ sees the `[MASK]`.
    - Confirm with the author before writing the equation in Ch. 2 or Ch. 5.
  - It is not a leak either way: the `[MASK]` vector is shared by all targets
    (`03_final_model.md` §2.1, `02_data.md` §7.1).
- **Grid results (eval cosine / eval top-1 of 945, `03_final_model.md` §4.2):**
  - GATv2 1-layer cos (the winner): 0.833 / 70.8%;
  - SAGE 1-layer cos: 0.825 / 68.4%;
  - plain reference mean: 0.823 (eval top-1 not reported in that table).
  - All eight variants are within 0.02 cosine. "Attention (GATv2) beats mean aggregation
    (SAGE) slightly" (`06_results.md` §4).
- **What GATv2 adds over the mean, conceptually:** a learned weight per reference instead of
  $1/|\mathcal{N}(i)|$ (`03_final_model.md` §2.1).
  - Because the query $x_i$ is the shared `[MASK]` vector, the query is the same for every
    target.
  - *My inference, not in any source:* the "dynamic" part of GATv2 (query-dependent ranking)
    therefore has little to work with here. The weights depend on the references themselves,
    through $\Theta_t x_j$ inside the nonlinearity together with the constant $\Theta_s x_i$.
  - Worth checking with the author before it goes anywhere near the text.

---

### Open points from §2.1

1. Self-loops in the project's GATv2: the code suggests that the target attends to its own
   `[MASK]` row; the author should confirm (see 2.1.4).
2. Over-smoothing: phrase the 1-vs-2-layer result as consistent with over-smoothing, not
   caused by it (see 2.1.2).
3. Homophily wording: "homophily in embedding space" vs Zhu's label-based $h$ (see 2.1.1).
4. Bib page ranges from DBLP when the entries are added.
