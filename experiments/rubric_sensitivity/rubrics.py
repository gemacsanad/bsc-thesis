"""Four judge rubrics, to be run over the same two generation arms.

`src/evaluation/evaluate_results.py` holds one judge prompt and every score in the project
comes from it. This folder asks what that choice costs: the same baseline and adapter
outputs, scored four ways, so the measured graph effect can be read as a property of the
rubric as well as of the model.

All four return the same JSON keys, so `score.py` writes the same columns whichever is used:
`accuracy`, `coherence`, `language`, `reason`. R3 adds four component scores on top; they are
extra, not a replacement, so R3 stays comparable with the others on `accuracy`.

The `coherence` and `language` rubrics are held identical across all four — only the
`accuracy` definition moves. They are not the object of study, and letting them drift would
put three differences into a comparison meant to have one.

Why these four
--------------
`analyse_arms.py` measures what the adapter does differently from the no-graph baseline over
the 945 era-1 nodes. Summarised: it is more *precise* and no more *complete* (gold-term
recall 0.0855 vs 0.0876, but precision against the gold 0.1146 vs 0.0944), its advantage sits
in the upper tail rather than the middle, and it does **not** come from neighbourhood
information the prompt could not show — the gap is uncorrelated with how many references the
centroid cap hid (Spearman -0.05, p=0.11). Qualitatively the baseline's dominant failure is
writing one cited paper's abstract instead of the citing paper's.

R2 and R4 follow from that; R3 is there to say which component carries the effect; R1 is the
control that makes any of it interpretable.
"""

# --- shared, held fixed across all four ---------------------------------------
_COHERENCE = ("- coherence (1-10): Is the argument logically coherent? Does it progress "
              "smoothly from a background problem to a proposed solution, or does it feel "
              "like a disjointed list of ideas?")
_LANGUAGE = ("- language  (1-10): Does the text adhere to scientific publishing norms? Score "
             "on precise domain terminology, appropriate passive/active voice, and absence "
             "of generic conversational fluff.")

_HEAD = ("You are a rigorous scientific peer reviewer. Given an original paper abstract and a "
         "generated abstract for the same paper, evaluate the generated abstract on three "
         "metrics. Return valid JSON with exactly this structure:\n")

_KEYS = """{
  "accuracy":  <int 1-10>,
  "coherence": <int 1-10>,
  "language":  <int 1-10>,
  "reason":    <string, 3-4 sentences explaining the scores overall>
}
"""

# --- R1: the rubric every existing number in the project was produced with ----
# Verbatim from src/evaluation/evaluate_results.py. Present so the other three are measured
# against a known quantity: it must reproduce the +0.473 era-1 gap, or something else moved.
R1 = _HEAD + _KEYS + """
Rubrics:
- accuracy  (1-10): How closely does the generated abstract align with the core scientific \
objective, problem domain, and baseline methodologies of the original? \
Penalise explicit contradictions; do not penalise creative but plausible predictions.
""" + _COHERENCE + "\n" + _LANGUAGE

# --- R2: problem and task only ------------------------------------------------
# Methodology is not recoverable from a bibliography — docs/adapter.md puts wrong_methodology
# at 70% for Vicuna and 95% for gpt-5.4-mini — so a third of R1's accuracy definition scores a
# dimension no arm can achieve. That is variance that cannot discriminate, and it pins both
# arms near the floor (46.7% of baseline scores are <= 2 under R1).
R2 = _HEAD + _KEYS + """
Rubrics:
- accuracy  (1-10): How closely does the generated abstract identify the same research \
problem and the same specific task as the original? Judge exactly two things: (a) the \
research area and the problem being addressed, and (b) the specific task or objective the \
paper takes on within it. \
Do NOT reward or penalise the particular methods, models, architectures, datasets or \
numerical results named in either abstract — those are outside this score and must not \
influence it. \
An abstract that identifies the right problem and the right task scores highly even if it \
names no technique at all and states them in general terms. An abstract that names plausible \
techniques while addressing a different problem or a different task scores low. \
Penalise explicit contradictions; do not penalise omission of detail.
""" + _COHERENCE + "\n" + _LANGUAGE

# --- R3: the same thing, decomposed ------------------------------------------
# Diagnostic rather than headline. If the adapter's advantage is concentrated in scope and
# problem rather than task, that is the finding, and no aggregate score shows it.
R3 = _HEAD + """{
  "accuracy":  <int 1-10>,
  "domain":    <int 1-10>,
  "problem":   <int 1-10>,
  "task":      <int 1-10>,
  "scope":     <int 1-10>,
  "coherence": <int 1-10>,
  "language":  <int 1-10>,
  "reason":    <string, 3-4 sentences explaining the scores overall>
}
""" + """
Rubrics:
- domain    (1-10): Is the research field correctly identified?
- problem   (1-10): Is the underlying problem or motivation the same one the original \
addresses?
- task      (1-10): Is the specific task or objective the same (e.g. detection vs \
segmentation, translation vs summarisation)?
- scope     (1-10): Is the kind and breadth of contribution the same — a single new \
technique, an application, an empirical study, a survey of a field? An abstract proposing one \
new method where the original surveys a field scores low here even if the field is right.
- accuracy  (1-10): Your overall judgement of how closely the generated abstract identifies \
the same research problem and the same specific task as the original, weighing the four \
components above. Do NOT reward or penalise the particular methods, models, architectures, \
datasets or numerical results named in either abstract.
""" + _COHERENCE + "\n" + _LANGUAGE

# --- R4: R2 plus the baseline's dominant failure named outright ---------------
# The generated abstracts are built from a reference list, and the commonest failure is
# describing one of those references instead of the paper citing them. R1 has no language for
# this, so a fluent, accurate description of the wrong paper can score mid-range. This is also
# the rubric with the largest rigging risk — see README.md — because it names the failure mode
# the baseline was observed making.
R4 = _HEAD + _KEYS + """
Rubrics:
- accuracy  (1-10): How closely does the generated abstract identify the same research \
problem and the same specific task as the original? Judge exactly two things: (a) the \
research area and the problem being addressed, and (b) the specific task or objective the \
paper takes on within it. \
Do NOT reward or penalise the particular methods, models, architectures, datasets or \
numerical results named in either abstract — those are outside this score and must not \
influence it. \
An abstract that identifies the right problem and the right task scores highly even if it \
names no technique at all and states them in general terms. \
One failure is specific and overrides the rest: the generated abstract was written from the \
original paper's reference list, and it may describe one of those cited works rather than the \
paper that cites them. If it reads as an account of a single earlier, narrower work — a \
particular named system or study that the original would have cited rather than been — score \
accuracy 3 or below, however fluent and however close the field. Correctly describing a work \
the original cites is not partial credit for describing the original. \
Penalise explicit contradictions; do not penalise omission of detail.
""" + _COHERENCE + "\n" + _LANGUAGE

RUBRICS = {"R1": R1, "R2": R2, "R3": R3, "R4": R4}

ARMS = {"R1": "01_R1_control", "R2": "02_R2_problem_task",
        "R3": "03_R3_decomposed", "R4": "04_R4_target_vs_source"}

# R3's extras, written as their own columns when present.
R3_COMPONENTS = ["domain", "problem", "task", "scope"]
