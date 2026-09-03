"""The four prompt eras, verbatim, and the reference block each one wraps them in.

Every era that has been run against Vicuna-7B is reproduced here character for character so
the same string can be sent to a different model. Where each came from:

    era 1  src/adapter/text.py, working tree (= commit 3ae4c3b's wording)
    era 2  src/adapter/text.py at commit 4105279
    era 3  experiments/sentiment_analysis/03_prompt_fix/prompt.md
    era 4  pasted by the author; the string never entered the repository

Two things change between eras, not one: the instruction, and the framing of the reference
block. Era 1 heads the block `Source publications:` and closes with `Generated abstract:` and
never discloses the true out-degree; eras 2-4 state how many works the paper cites, describe
the shown ones as the most representative, and close with `Abstract:`. Keeping both under one
`era` key is deliberate — an era is the whole prompt, not just its first paragraph.

The Vicuna chat wrapper (`SYSTEM ... USER: ... ASSISTANT:`) is kept even though the model
being called is not Vicuna. It is what the 7B runs sent, it is what the archived
gpt-5.4-mini baseline in `experiments/_archive/gpt-5.4-mini/` sent, and dropping it would
add a second difference to a comparison that is supposed to have one.
"""

SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# --- era 1 -------------------------------------------------------------------
# In use until 2026-08-19. Asks for invention ("a new application paper") and fixes the
# genre, which is what the three later rewrites were trying to undo.
ERA1 = (
    "You are a scientific writing assistant. "
    "Based on the source publications listed below, write a concise abstract "
    "for a new application paper that builds on and combines these works. "
    "Output ONLY the abstract text, without any heading or explanation."
)

# --- era 2 -------------------------------------------------------------------
# 2026-08-19. Reconstruction instead of invention, a genre menu, a word budget, and the
# out-degree disclosed in the block below.
ERA2 = (
    "You are a scientific writing assistant. "
    "The publications listed below are cited by one paper, whose own abstract is missing. "
    "Reconstruct that missing abstract: the problem the paper addresses, how it builds on "
    "and departs from the work it cites, what it proposes, and what it concludes. "
    "Infer from the references what kind of paper it is, a new method, an application, "
    "an empirical study, or a survey, rather than assuming. "
    "Write 150-180 words. "
    "Output ONLY the abstract text, without any heading or explanation."
)

# --- era 3 -------------------------------------------------------------------
# 2026-08-20. Genre menu and "departs from" removed, invention banned outright, hedging
# made explicit, authorial voice imposed.
ERA3 = (
    "You are a scientific writing assistant. "
    "The publications listed below are cited by one paper, whose own abstract is missing. "
    "Reconstruct that missing abstract: the problem the paper addresses, how it builds on "
    "the work it cites, what it proposes, and what it concludes. "
    "Do not name a method, model, system, or dataset that does not appear in the cited "
    "works. "
    "Where the sources do not determine the paper's specific contribution, state what they "
    "do determine about its problem and approach, and stop there — a correct general "
    "statement is worth more than a specific guess. "
    "Write in the voice of the paper's own authors: never mention the cited works as a "
    "collection, how many there are, or the authors. "
    "Write 150-180 words. "
    "Output ONLY the abstract text, without any heading or explanation."
)

# --- era 4 -------------------------------------------------------------------
# 2026-08-22, the wording behind the 08-23 runs. Not recoverable from the repository — the
# working tree holds era 1 and no commit ever carried era 3 or era 4 — so it was supplied by
# the author and transcribed here unchanged. Against era 3 it adds that the target is not
# among the listed papers (aimed at the 10.4% near-verbatim copy rate), replaces "what it
# proposes" with a description by mechanism rather than by name (aimed at the 35.2% coined-
# name rate, which a prohibition alone had failed to move), drops the reference-restatement
# licence, and requires plain prose (aimed at the 11.9% LaTeX rate). See docs/adapter.md,
# Stage 6c.
ERA4 = (
    "You are a scientific writing assistant. "
    "The publications listed below are the reference list of one later paper, whose own "
    "abstract is missing. That paper is not among them: it is a separate, more recent work "
    "that cites all of them. "
    "Reconstruct its abstract as continuous prose covering, in order: the problem it "
    "addresses; what the work it cites has already established about that problem; what it "
    "does about what remains open, described by the mechanism it uses the kind of model, "
    "data or analysis rather than by a name; and what it evaluates. "
    "Where the references leave the paper's specific contribution undetermined, state what "
    "they do determine about its problem and approach, and stop there — a correct general "
    "statement is worth more than a specific guess. "
    "Write in the voice of the paper's own authors: never mention the cited works as a "
    "collection, how many there are, or the authors, and do not restate any single one "
    "of them. "
    "Write plain prose of 150-180 words, with no markup, mathematical notation or headings. "
    "Output ONLY the abstract text."
)

ERAS = {1: ERA1, 2: ERA2, 3: ERA3, 4: ERA4}

# Era 1 predates the reference cap being disclosed; eras 2-4 state the true out-degree.
DISCLOSES_COUNT = {1: False, 2: True, 3: True, 4: True}


def _sources(neighbours):
    return "\n\n".join(f"- Title: {t}\n  Abstract: {a}" for t, a in neighbours)


def build_prompt(era, neighbours, total_refs=None):
    """-> the full prompt string for one node under one era.

    `neighbours` is a list of (title, abstract) for the references that survived the
    centroid cap; `total_refs` is the node's true out-degree, which under the cap is
    larger. Mirrors `src/adapter/text.build_prompt` in both of its versions.
    """
    instruction = ERAS[era]
    if instruction is None:
        raise ValueError(
            f"era {era} has no prompt text — fill in ERA{era} in prompts.py before running it")

    shown = len(neighbours)
    total = shown if total_refs is None else total_refs
    sources = _sources(neighbours)

    if not DISCLOSES_COUNT[era]:
        block = f"Source publications:\n{sources or '(no neighbouring sources available)'}"
        tail = "Generated abstract:"
    else:
        if not neighbours:
            block = "No cited publications are available for this paper."
        elif shown == total:
            block = f"This paper cites {total} works, all listed below:\n{sources}"
        else:
            kept = ("the most representative one is" if shown == 1
                    else f"the {shown} most representative are")
            block = f"This paper cites {total} works; {kept} listed below:\n{sources}"
        tail = "Abstract:"

    user = f"{instruction}\n\n{block}\n\n{tail}"
    return f"{SYSTEM} USER: {user} ASSISTANT:"


def user_message(era, neighbours, total_refs=None):
    """-> (system, user) for the OpenAI chat API.

    The same string `build_prompt` returns, split at the Vicuna role markers so the two
    halves land in the roles the API expects. `src/generation/generate_gpt.py` does the
    same for the archived gpt-5.4-mini baseline.
    """
    prompt = build_prompt(era, neighbours, total_refs)
    body = prompt[len(SYSTEM) + len(" USER: "):-len(" ASSISTANT:")]
    return SYSTEM, body
