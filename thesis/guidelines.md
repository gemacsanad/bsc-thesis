# Guidelines removed from the LaTeX template

On 2026-10-14 the template's guidance pages were removed from the build. This file keeps their
substance, translated from Hungarian. Sources:

- `include/guideline.tex`: the faculty's one-page guideline. Still on disk, no longer included.
- `include/project.tex`: the assignment placeholder page. Still on disk, no longer included.
- `content/thesis-format.tex`, `content/template-usage.tex`, `content/latex-tools.tex`: the
  sample chapters. Deleted; recoverable from git.

## 1. Required structure of the thesis (BME VIK)

1. Assignment (*feladatkiírás*).
2. Title page.
3. Table of contents.
4. Student declaration on independent work and on the handling of electronic data.
5. Abstract in Hungarian and in English (1–2 pages; further languages optional).
6. Introduction: interpretation of the task, the goal of the design, the motivation, and a short
   summary of the thesis structure.
7. Refinement and detailed analysis of the assignment.
8. Prior work (literature review, similar solutions) and the conclusions drawn from it.
9. Detailed description of the design, evaluation of the options, and justification of the
   chosen solutions.
10. Evaluation and critical analysis of the result, and possible further development.
11. Acknowledgements (optional).
12. Detailed and accurate bibliography.
13. Appendices.

Items 7–10 are numbered chapters.

**Mandatory:** without the student declaration the thesis cannot be reviewed or defended.

## 2. The assignment page

- The printed copy must contain the **original** assignment sheet, stamped by the department and
  signed by the head of department, in place of the placeholder page.
- The **electronic** version must **not** contain the assignment; the department uploads it
  separately.

## 3. Length

- BSc thesis: **at least 30 pages, about 45 recommended**.
- Expectations vary by department; ask the advisor.

## 4. Language

- The default language is Hungarian; English is allowed.
- The template warns that an English thesis means extra work in phrasing and correctness, for
  the student and for the advisor and reviewer alike: "a hard-to-read text is a burden for
  everyone."

## 5. Page layout and typography

- **Paper and type:** standard A4. The faculty default is 12 pt Times New Roman with 1.5 line
  spacing; small deviations and other fonts are allowed. The thesis-format chapter suggests
  11–12 pt, a serif font and 1.5 spacing.
- **Margins:** 2.5 cm all round with an extra 1 cm for binding, as mirrored margins.
- **Page numbers** on every page except the first four structural elements.
- **Headings:** chapters numbered decimally, at most 3 levels deep (e.g. 2.3.4); headings
  left-aligned, body text justified.
- **Figures and tables:** inserted at the right place, centred, captioned below, and numbered
  per chapter (Figure 2.4, Table 4.2). Each needs a descriptive title.
- **Equations:** numbered per chapter, e.g. (3.2), and written with an equation editor (LaTeX).
- **Drawings:** made with a drawing program.
- Follow basic typographic rules; the template cites Jeney's typography guide.

The LaTeX template already implements the layout (11 pt, its own margins, single spacing). Do
not change it: `CLAUDE.md` forbids editing the preamble and the page layout.

## 6. Bibliography

- **Citation style:** numbered citations are preferred; Harvard (author, year) is also allowed.
- **Order of the list:** in citation order when numbered, alphabetical otherwise.
- **Titles** always in the original language, optionally with a translation in brackets.
- **Every listed publication must be cited** in the text (BibTeX enforces this).
- **Journal articles:** give the exact title, journal, volume, issue and page range. Abbreviate
  journal names only when they are very well known or very long.
- **Web sources:** give the owner and the content of the page before the URL, plus the access
  date, because links break.
- **Finding entries:** search good sources first (ACM DL, DBLP, IEEE Xplore, SpringerLink), then
  Google Scholar, and clean the entries.
- **`plainnat` duplicates:** this style prints both the DOI and the URL, which often
  duplicates. Delete URL fields that only contain the DOI.

## 7. File size

The thesis PDF and any attachment may each be at most **15 MB**.

## 8. Template usage notes (from the sample chapters)

- **Folders:** `include/` holds template parts that are normally not edited; `content/` holds
  your own work, one `.tex` file per chapter, added to `thesis.tex` with `\include`.
- **Labels:** use the prefixes `fig:`, `tab:`, `eq:`, `sec:` and `lst:`.
- **Figures:** prefer vector graphics (PDF). Use PNG/JPEG only when needed, at ≥ 300 dpi.
  Several images can share one figure, with explicit horizontal and vertical spacing.
- **Tables:** follow the `booktabs` style.
- **Notation:** lower case for scalars, bold lower case for column vectors, bold capitals for
  matrices. Define any other notation in its own subsection.
- **Equations:** use `align` rather than `equation` or `eqnarray`.
- **Lists:** items end with a comma and the last with a full stop, unless items are full
  sentences.
- **Code listings:** the `listings` environment is configured. Use it only for code that has to
  be set apart from the text.
- **Environments:** `definition`, `example` and `theorem` exist.
- **Building:** compile several times; after changing the `.bib`, make sure BibTeX actually
  reran. This project builds with `latexmk -xelatex` (see `CLAUDE.md` §6).

## 9. Placeholder text still in the content files

These Hungarian placeholders remain and will be replaced when the sections are written:

- `content/introduction.tex`: "The introduction contains the analysis of the assignment, its
  history, the motivation, previous solutions, and a summary of the student's solution in
  their light. The introduction usually ends with the structure of the thesis, i.e. a short
  description of what each chapter covers."
- `content/acknowledgement.tex`: "Not mandatory and may be deleted. Acknowledging the advisor
  is not required either; advising is their official duty."
- `content/abstract.tex`: the template's own description, in Hungarian and English.
- `content/appendices.tex`: two sample sections (the TeXstudio screenshot and the "42"
  equations) after the AI-use declaration.
