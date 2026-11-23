# Reviewer brief for a thesis diagram

Give this to a separate reviewer agent together with the spec, the 300 dpi preview, zoomed crops,
and the approved sister figures (render them too). The reviewer does not edit; it scores and
lists concrete fixes.

## Score each criterion 0–10

| criterion | what to check |
|---|---|
| correctness | every element matches the spec's facts (and the paper / code it cites); arrows point in the stated direction; nothing implies a claim the text does not make |
| clarity | a reader who has not read the section can follow it with the caption; the reading order is obvious; one idea per panel |
| minimalism | no word, label or element that the caption could carry instead; no step titles or decorative frames unless needed (the author rejects "too complicated") |
| legibility at print size | judged at the printed size (≤ 113 mm wide): text ≥ 9 px (≈ 6.5 pt), nothing clipped, labels not on lines, arrowheads touching their targets, no overlaps |
| consistency with the sister figures | Latin Modern, 1 px black strokes, block arrowheads size 4, Okabe–Ito tints with the same meaning as before (trainable = vermillion tint, frozen = grey, ...), "(a)/(b)" under panels, no title |

## Verdict

- **Approve** only if overall ≥ 9, no criterion < 8, and no correctness issue.
- Otherwise **revise**: list each fix as "element → what is wrong → what to change", most
  important first. Flag anything that needs the author's decision instead of guessing.
