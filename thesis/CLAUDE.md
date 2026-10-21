# CLAUDE.md — writing the BSc thesis

You are helping the author write their BSc thesis (BME VIK, *szakdolgozat*, in English):
**"Neighborhood-Based Node Text Generation with Graph Neural Networks and Large Language
Models."** The research is finished or nearly finished; your job is to help turn it into a
thesis, **one small, approved piece at a time**. The author controls the content and the
style. You do not write ahead.

This file overrides the repository-level `../CLAUDE.md` where they conflict. In particular,
the "Always document" rule there does **not** apply here: the thesis is the document. Do not
write `documentation.md` files in this folder.

---

## 1. Hard rules

1. **Do not modify the template.** Never edit:
   - `include/` (all of it),
   - `huplain.bst`, `magyar.ldf`, `lua/`, `Makefile`, `manual_build.bat`,
   - the package list, the preamble, the caption setup or the page layout.

   No new `\usepackage` and no new `\newcommand` in any template file. If something seems to
   need a new package or macro, say so and ask. The author has said this emphatically.
2. **What you may edit:**
   - the variables at the top of `thesis.tex` (author, advisors, title, department), but only
     with values the author gives you;
   - the `\include{content/...}` lines in `thesis.tex`, to add the author's own chapters;
   - files in `content/`;
   - `bib/mybib.bib`;
   - files in `plans/` (see §2).

   Removing the template's guideline page, its assignment placeholder page and its sample
   chapters (`content/latex-tools`, `content/thesis-format`, `content/template-usage`) needs
   the author's explicit go-ahead first.
3. **Write only what was asked.**
   - The unit of writing is whatever the author names: a paragraph, a subsection, a list of
     bullet points.
   - Never draft other parts of the section "while you are at it".
   - Never fill in neighbouring sections.
   - Never rewrite text the author did not point at.
4. **Ask, don't assume.**
   - When a real choice appears, stop and ask, usually with AskUserQuestion. Examples: what
     to include, how deep to go, what a figure should show, which term to use.
   - The author has to defend every sentence at the defence; a choice you made silently is
     one they cannot explain.
   - Answering a question is not permission to write. Wait for an explicit "write it".
5. **Never invent facts, numbers or citations.**
   - Every number comes from `thesis_context/` or from a file it cites. Give its source in
     the chat when you use it.
   - Every bibliography entry must come from a verifiable record (arXiv, DBLP, ACL Anthology,
     the publisher's page), never from memory.
   - If a fact is missing, write a visible placeholder and tell the author. The template loads
     `xcolor`, so this placeholder works:
     `\textcolor{red}{[TODO: what is missing]}`
6. **No training, no GPU work, no heavy computation** on this machine, ever. New figures or
   analyses are separate tasks: describe what is needed and let the author decide where it
   happens.
7. **Git:**
   - commit only when asked;
   - the commit message is one line, imperative, with **no** `Co-Authored-By` or any other
     trailer.
8. **Secrets:** never read or print `../.env_local`.

---

## 2. The workflow: plan, then write in small parts

The author works section by section, in two modes. Always know which mode you are in; if
unclear, ask.

### Planning mode — no prose

When the author wants to plan a chapter or section:

1. Read the relevant `thesis_context/` files first (map in §4), plus any section of the thesis
   already written that the new one depends on.
2. Propose a **skeleton**:
   - the subsections, in order;
   - for each paragraph, one line saying what it argues;
   - the evidence it uses (number + source file);
   - the figures and tables, with their file paths if they exist;
   - the citations needed (keys, or "needs a bib entry");
   - the terms it introduces.
3. List the **open decisions** as questions (scope, order, depth, what to leave out). Do not
   resolve them yourself.
4. Iterate until the author approves. Then save the approved plan to
   `plans/<chapter-file-name>.md`, overwriting the earlier version of that plan. The plans
   are what keeps sessions consistent; read them at the start of every writing session.

### Writing mode — one unit at a time

When the author says to write a part:

1. Re-read that part of the approved plan, `plans/style.md`, and the text immediately before
   the insertion point, so the new text continues it.
2. Write **only that unit** into the right `content/*.tex` file.
3. In the chat, show:
   - the text you wrote;
   - the facts used, each with its source;
   - anything you were unsure of (a claim, a term, a number).
4. Compile (§6) and report errors and new warnings: undefined references or citations, and
   overfull boxes in the new text.
5. **Stop.** Wait for feedback. Apply revisions only to what the author points at.
6. When the author states a style preference ("don't use X", "call it Y", "shorter
   sentences"), add it to `plans/style.md` immediately, so it holds for every later part.

### Session start checklist

Do these silently, then say in one or two lines where things stand:

- read `plans/style.md` and the plan of the section in progress;
- run `git status` on this folder;
- open the last-written `.tex` file.

### AI-use declaration

The thesis must contain a declaration of generative-AI use: a table in the appendix with the
type of use, the sections affected, a brief summary of the prompts, and the share of use.
`include/AI_nyilatkozat_tablazat_EN.tex` is the table.

Do **not** keep an AI-use log during writing. The author will handle the declaration once the
thesis is finished.

---

## 3. What a BME VIK BSc thesis must contain and look like

### Required structure

From the faculty guideline in `include/guideline.tex` and the template documentation in
`content/template-usage.tex` and `content/thesis-format.tex`, in this order:

1. **Assignment** (*feladatkiírás*). In the printed copy this is the signed sheet from the
   department. It is **not** part of the electronic version; `include/project.tex` is only a
   placeholder.
2. **Title page** (`include/titlepage.tex`, filled from the variables in `thesis.tex`).
3. **Table of contents.**
4. **Student declaration** (`include/declaration_EN.tex`). Mandatory: without it the thesis
   cannot be reviewed or defended.
5. **Abstracts:**
   - Hungarian (*Kivonat*) **and** English, both in `content/abstract.tex`;
   - both are required even for an English thesis;
   - about one page each, with the same content.
6. **Introduction:** the interpretation of the task, the goal, the motivation, and a short
   outline of the thesis chapters (the introduction ends with the outline).
7. **Numbered chapters** covering:
   - the detailed analysis and refinement of the assigned task;
   - prior work (literature review, similar solutions) and the conclusions drawn from it;
   - the detailed description of the design, with the alternatives considered and **why**
     each choice was made;
   - the evaluation and critical analysis of the result, and further development.
8. **Acknowledgements** (optional).
9. **Bibliography:**
   - detailed and accurate;
   - every entry is cited in the text (BibTeX enforces this);
   - titles in their original language.
10. **Appendices.** The AI-use declaration table goes here; the template already includes it
    from `content/appendices.tex`.

The thesis must address **every subtask in the assignment** (listed in
`thesis_context/00_index.md`). If a subtask was not completed, the thesis must say so plainly
and explain why: "no GitHub publication yet", "not truly end-to-end".
`thesis_context/00_index.md` lists where the project deviates from the assignment wording.
These deviations must be stated honestly, not hidden.

The reviewer's question is whether the student can work independently as an engineer, apply
engineering methods, interpret the task, and **evaluate and critically analyse** their own
solution. Design justification and honest evaluation weigh more than volume.

### Length

- The template's guidance for a BSc thesis is **at least 30 pages, about 45 recommended** for
  the main text.
- The department or advisor may expect otherwise; ask the author if it matters.
- The PDF and any attachment may each be at most **15 MB**, so include figures as PDF, not
  PNG.

### Typographic rules

The template already implements these; do not fight them.

- A4, 11 pt, Latin Modern (XeLaTeX with fontspec), single spacing. English mode sets no
  paragraph indent and a 0.5em gap between paragraphs, so **paragraphs are separated by a
  blank line**.
- Chapters numbered decimally, at most 3 levels deep (`\section`, `\subsection`,
  `\subsubsection`). No `\paragraph`-level numbering.
- Figures, tables and equations are numbered per chapter (2.4, (3.2)).
- Figures and tables are centred. The faculty guideline puts each caption **below** the
  figure or table it labels, so use that for tables too unless the author decides otherwise.
- Figure captions are limited to 0.75 `\textwidth` (template setting).
- Every figure and table is referenced in the text **before** it appears.
- Body text is justified and headings are left-aligned (template default).
- Citations are numbered (`plainnat` with the `numbers` option):
  - write `text~\cite{key}`;
  - cite at the claim, not at the end of the paragraph;
  - web sources need the owner and the access date, in `note` or `howpublished`.

---

## 4. Sources of content

`thesis_context/` holds the factual record of the project. It is a copy of
`../docs/thesis_context/`, which is the original. If the two ever differ, say so; do not
pick one silently.

| file | use for |
|---|---|
| `00_index.md` | the assignment and its 6 subtasks, the deviations from it, the **glossary (use its terms exactly)**, what is not done yet, ground rules for numbers |
| `01_literature.md` | related-work chapter; every method with its paper and URL (source for bib entries) |
| `02_data.md` | data chapter: ogbn-arxiv + Semantic Scholar graph, splits, Voyage embeddings, preprocessing, the S2 coverage confound |
| `03_final_model.md` | method chapter: the run-11 architecture, the three training phases, design decisions with reasons |
| `04_development_history.md` | why the design is what it is: earlier runs, the end-to-end failures. Use as evidence for design decisions, not as a diary |
| `05_evaluation_methods.md` | evaluation chapter: the judge, TF-IDF identification, ROUGE/BERTScore, the detailed evaluator, validity studies |
| `06_results.md` | results: the ablation table, paired differences, the GNN stage. **Not final**; the evaluation is still being extended |
| `07_other_datasets.md` | future work only; nothing was trained on these |
| `08_implementation_reproducibility.md` | implementation chapter or appendix: stack, HPC, verification scripts, reproduction |
| `09_limitations_and_discussion.md` | discussion and conclusion: what the evidence supports and what it **does not** |

Deeper detail is in `../docs/` (index: `../docs/README.md`) and in `../experiments/*/`. Read
them only when a context file points there and the detail is actually needed.

### Rules for numbers and claims

These come from `00_index.md` and `09_limitations_and_discussion.md`.

- Every number states its **arm** (text baseline, joint, reader←GNN, reader←refs-mean, oracle,
  zero) and its **instrument** (old judge rubric, detailed evaluator, TF-IDF identification).
  Coherence scores from the two judges are not comparable.
- The **oracle is a ceiling**, never a result. The **zero input is a floor**.
- Do not claim:
  - that methodology or contribution is recovered;
  - that the soft prompt beats text;
  - that the GNN gain is large (+0.11 on a 1–10 scale);
  - that multi-hop structure helps;
  - that the joint phase moved the GNN.

  The list in `09` §2 is binding.
- Runs before run 11 are history. Cite them only as evidence for a design decision.
- Results that are still pending (oracle judge, detailed evaluator on all arms, judge
  validation) get a placeholder, not an estimate.
- Interpretations are written as interpretations ("this suggests…"), separate from findings.

### Bibliography

`bib/mybib.bib` still holds the template's sample entries. Add real entries as they are first
cited:

- take the details from the record behind the URL in `01_literature.md`; prefer DBLP or the
  ACL Anthology BibTeX, and the arXiv page for preprints;
- clean each entry. With `plainnat`, an entry that has both `doi` and a DOI `url` prints
  twice, so drop one;
- use keys of the form `firstauthorYEARword`, for example `hu2022lora`;
- removing the template's sample entries is part of removing the sample chapters: ask first.

---

## 5. Figures

### Training curves

These already exist, made for print with the same font, sizes and colour-blind-safe palette
as the thesis:

- GNN grid comparison, all 8 variants overlaid:
  `../diagrams/figures/gnn_comparison/gnn_*.pdf`.
  The source notebook is `../diagrams/gnn_comparison.ipynb`, with an `INCLUDE` toggle per
  variant.
- Per run: `../diagrams/data/run11-<run>/figures/<short>_*.pdf`, where `<run>` is
  `reader-14541449`, `joint-14541449` or `gnn-{gatv2,sage}-l{1,2}-{cos,cosnce}-14541449`.
- Each notebook's markdown cells hold a **suggested caption** for each figure.
- `../diagrams/documentation.md` explains what each metric means, and its known artefacts
  (for example, the throughput spikes after each validation).

### Including a figure

The figures are drawn at their printed size: 112.5 × 70 mm by default, which equals the
caption width. Include them **without `width=`**, so the fonts stay at 8–9 pt:

```latex
\begin{figure}[!ht]
  \centering
  \includegraphics{../diagrams/figures/gnn_comparison/gnn_cosine_validation.pdf}
  \caption{Validation cosine of the eight GNN variants \ldots}
  \label{fig:gnn-cosine-validation}
\end{figure}
```

- Reference figures from `../diagrams/` directly, so there is one copy of each. If the author
  wants copies under `figures/` instead, ask.
- Do not regenerate or restyle figures from the thesis session. If a figure needs to change,
  or a new one is needed, describe exactly what should change and let the author decide.
  Examples of new figures: results bar charts, an architecture diagram, data statistics.
- Every figure needs a caption that stands on its own: what is plotted, which run, and what
  to notice. Keep in-figure text out; there are no titles inside the plots.

---

## 6. LaTeX conventions

### Files and labels

- **One file per chapter** in `content/`, named by number and topic (`content/02-background.tex`),
  and added with `\include{content/...}` in `thesis.tex`, in place of the sample chapters
  once the author approves removing them.
- **Labels** by prefix: `chap:`, `sec:`, `fig:`, `tab:`, `eq:`, `lst:`. Use lower-case words
  joined with hyphens.

### Cross-references and macros

- **Cross-references:** `\autoref{fig:...}` gives "Figure 2.3", because the English mode sets
  these names. You can also write `Figure~\ref{...}`. Pick one and keep it; record the choice
  in `plans/style.md`. Use `\eqref{eq:...}` for equations. The Hungarian `\refstruc` and
  `\az` macros do not apply in English text.
- **Abbreviation macros** from the template: `\eg`, `\ie`, `\etal`, `\vs`, `\wrt`, `\cf`,
  `\etc`. They handle the spacing after the period.

### Tables

- Use `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`); no vertical lines.
- Right-align numbers, keep the same number of decimals within a column, and put units and
  the instrument in the header.

### Math

- `amsmath` is loaded; `\argmax` is defined.
- Number only the equations you reference.
- Define every symbol at its first use; keep notation consistent across chapters and record
  it in `plans/style.md`.

### Code and other environments

- `listings` is configured, but code belongs in the thesis only when it explains something
  prose cannot.
- `definition` and `theorem` environments exist; use them sparingly.

### Language and typography

- **Non-ASCII characters:** XeLaTeX handles UTF-8 directly, so Hungarian names can be written
  as they are, in both `.tex` and `.bib`.
- **Typography:**
  - `--` for ranges (1--10);
  - `---` only if the author wants dashes at all;
  - `~` before `\cite` and `\ref`;
  - `$\times$` for dimensions;
  - `\,` or `\texttt{}` as appropriate;
  - code identifiers in `\texttt{}`, sparingly; the thesis is not the code documentation.
- **Hungarian abstract:** wrap it in the `\selecthungarian` block that already exists in
  `content/abstract.tex`. The author may want to write or check the Hungarian themselves; ask.

### Building

From `thesis/`:

```bash
latexmk -xelatex -interaction=nonstopmode thesis.tex
```

- This is equivalent to the VS Code recipe (xelatex → bibtex → xelatex → xelatex).
- Do not use the `Makefile`: it moves the PDF into a `../pdf/` folder that does not exist.
- Build products (`*.aux`, `*.log` and so on) are git-ignored, but `thesis.pdf` and
  `thesis.thm` are not. Do not commit them unless the author asks.
- After each build, check the log for undefined references or citations and for overfull
  boxes in the new text.

---

## 7. Writing style

These are defaults. **The author's preferences in `plans/style.md` override them.** Ask about
these at the first writing session and record the answers there:

- the spelling variant (the template suggests en_GB);
- the voice ("we", "I", or passive);
- the tense convention.

### Default style

- **Plain, precise and concrete.** One idea per paragraph, with the topic sentence first. Prefer
  a number to an adjective ("18.5% vs 16.5% top-1", not "substantially better").
- **Claims match the evidence.** State effect sizes with their uncertainty (paired SE), say
  what was *not* tested, and keep the limitations visible.
- **Define before use.**
  - Introduce every term from the glossary in `00_index.md` once, then use it unchanged; no
    synonyms for variety. The terms include reader, prefix adapter, gate, reference mean,
    oracle and arm.
  - Acronyms are spelled out at first use in each chapter where they matter.
- **Explain the reasons, not just the actions.** The method chapter answers "why this and not
  the alternative" for each design decision (`03_final_model.md` §6, `04`).
- **No LLM tics.** Avoid:
  - "delve", "crucial", "pivotal", "leverage", "seamlessly", "robust" (unless measured),
    "in the realm of", "it is worth noting";
  - rhetorical questions and rule-of-three lists;
  - summary sentences that repeat the paragraph.
- **No filler.** No paragraph should exist to reach a page count.
- **No bullets in the running text,** unless the author asks for them. A thesis is prose;
  lists are for genuinely enumerable items.
- **Tense:**
  - present tense for what the model is and what the results show;
  - past tense for what was done in a specific experiment.

---

## 8. Starting proposal for the chapter structure

This is **a proposal only**; the author plans the real structure. It maps the faculty's
required elements to the content files.

| chapter | content | context files |
|---|---|---|
| Introduction | the task, motivation (text-attributed graphs, generating a node's text from its neighbourhood), contributions, outline | `00`, `09` §1 |
| Background and related work | GNNs, LLMs, soft prompts and prefix tuning, LoRA, graph + LLM methods, LLM-as-a-judge | `01` |
| Data | the citation graph, the S2 extension, splits, embeddings, preprocessing, the coverage confound | `02` |
| Method | the task formulation, the GNN estimator, the prefix adapter, the reader (Qwen3-14B + LoRA), the three-phase training, design decisions | `03`, `04` (as justification) |
| Evaluation methodology | the arms ladder (zero / refs-mean / GNN / joint / oracle / text), the instruments and their validity | `05` |
| Results | the ablation table, the GNN stage (with the diagrams), the training dynamics, the detailed metrics, qualitative examples | `06`, diagrams |
| Discussion | what the evidence supports and what it does not, limitations, deviations from the assignment | `09`, `00` |
| Conclusion and future work | a summary, end-to-end training, per-reference tokens, other datasets | `09` §4, `07` |
| Appendices | implementation and reproducibility, prompts, hyperparameters, the AI declaration | `08`, `03` |

Things the author has to decide early:

- whether the development history gets its own chapter or is folded into the method chapter
  as justification;
- whether implementation is a chapter or an appendix;
- the page budget per chapter.
