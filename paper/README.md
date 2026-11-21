# sesmesh paper (draft)

LaTeX sources of a full-length research/software paper on sesmesh 1.1.2,
meant for an arXiv preprint (primary **q-bio.BM**, cross-lists **cs.CG** and
**physics.comp-ph**) and the *Journal of Molecular Graphics and Modelling*
(Elsevier; see "arXiv submission" and "JMGM submission" below). Nothing
has been submitted anywhere.

Every number in the text comes from the repository: the validation CSVs
(`docs/validation/*.csv`), the cross-tool comparison
(`docs/validation/compare/*.csv`, `bench-field-zen5.txt`; harness in
`tools/compare/`), the design document, the validation report, the
CHANGELOG, or the analytic-reference program in `scripts/analytic`. Claims
that need a source or a measurement the repository does not have are marked
`\todo{…}`; they print in red where they stand. `\todoquiet{…}` marks an
item in the source only and prints nothing. None is left: `main.tex` sets
`\showtodosfalse`, so a build with one fails (switch to `\showtodostrue`
while drafting).

## Build

Requirements: a TeX Live 2026 installation (pdfLaTeX, BibTeX, `latexmk`,
`pgfplots`, `longtable`, `listings`, `cleveref`, `natbib`) and Python 3
(standard library only).

```bash
cd paper && make
```

`make` runs `scripts/make_data.py` (CSV → `generated/` tables, `\val{}`
numbers and plot data) and then `latexmk -pdf main.tex`. This draft was built
with **latexmk** (TeX Live 2026, pdfTeX 1.40.29); tectonic is not needed.
Plots are drawn by pgfplots from `generated/*.dat` at compile time, so there
is no matplotlib dependency.

Other targets:

| Target | What it does |
|---|---|
| `make data` | Rebuilds `data/analytic.csv` by running `scripts/analytic` (a stand-alone Cargo package that links `crates/sesmesh` and `crates/sesmesh-testkit`; release build, about a minute) |
| `make tables` | Only regenerates `generated/` |
| `make arxiv` | Builds the PDF, then `scripts/make_arxiv.py` writes `arxiv/` and `arxiv.tar.gz` (`main.tex` with `\pdfoutput=1` and a fixed date, `main.bbl`, the sections, and only the generated files and figures the paper uses), checks that the bundle builds, and writes the form text `arxiv-metadata.txt` |
| `make jmgm` | Builds the PDF, then `scripts/make_jmgm.py` writes the JMGM submission files to `jmgm/submission/`: flat LaTeX sources with figure files, highlights, graphical abstract, cover letter, form text (needs Ghostscript; see `jmgm/README.md`) |
| `make clean` | Removes LaTeX build files |

## Layout

| Path | Content |
|---|---|
| `main.tex` | Preamble, `\todo` and `\val` macros, section order |
| `sections/*.tex` | Abstract, introduction, background, method, implementation, validation, performance (incl. x86-64 §6.2), comparison with six programs (§7), discussion, availability + declarations, appendices |
| `references.bib` | Methods references of the README ("How it works") and Spec §12, transcribed from the repository; plus a continuum-electrostatics section checked against Crossref |
| `figures/*.jpg` | Gallery renders from `site/img/gallery/*.webp`, flattened on white and trimmed (`magick … -background white -alpha remove -trim`) |
| `data/analytic.csv` | One/two-sphere references meshed at h = 1, 0.5, 0.25, 0.125 Å (from `make data`) |
| `data/scaling.csv` | Thread-scaling numbers transcribed from Design §11.1 |
| `scripts/make_data.py` | CSV → LaTeX tables, macros, pgfplots data (validation, comparison, x86 SIMD and scaling) |
| `scripts/make_jmgm.py` | Assembles and checks the JMGM submission files (`make jmgm`) |
| `scripts/make_arxiv.py` | Builds and checks the arXiv bundle and form text (`make arxiv`) |
| `scripts/analytic/` | Program that produced `data/analytic.csv` |
| `generated/` | Output of `make_data.py` (committed, so the paper builds without Python), including the journal highlights `highlights.txt` |
| `jmgm/` | JMGM submission: `README.md` (what to upload, what is left), graphical abstract and cover letter sources; `make jmgm` writes `jmgm/submission/` (not committed) |

## arXiv submission

Checked against arXiv's help pages on 2026-11-18. `make arxiv` writes:

- `arxiv.tar.gz`, the file to upload. arXiv compiles `main.tex` with
  pdfLaTeX and uses `main.bbl` (natbib, `unsrtnat`). It holds no `.bib`,
  no unused figures and no output files, as arXiv asks. `\pdfoutput=1` is
  within the first five lines. The date is fixed when the bundle is built,
  because arXiv advises against `\today` (it rebuilds PDFs later).
- `arxiv-metadata.txt`, the text for the form: title, authors, abstract,
  comments and categories. arXiv's metadata fields accept ASCII only, so
  the abstract writes −, Å, → and ² as `-`, `$\mathring{\mathrm{A}}$`,
  `$h \to 0$` and `$O(h^2)$`. TeX math is rendered with MathJax, and the
  script checks the 1920-character limit.

On the submission form:

| Field | Value |
|---|---|
| Primary category | q-bio.BM (the endorsement must cover q-bio) |
| Cross-lists | cs.CG, physics.comp-ph (arXiv: "rarely" more than one or two) |
| Comments | From `arxiv-metadata.txt` (pages, figures, tables, code and Zenodo DOI). Add "Submitted to the Journal of Molecular Graphics and Modelling" once that is true. |
| Licence | CC BY 4.0 ("Creative Commons Attribution license"), chosen on 2026-11-19. It allows the widest reuse with attribution, like the code's MIT or Apache-2.0, and matches the CC BY licence of the journal's open-access route. It cannot be changed later. Elsevier lets authors post preprints anywhere at any time and does not count them as prior publication. |

Then check the PDF that arXiv builds before announcing. arXiv compiles
with TeX Live 2025 by default (2023 is still offered), and its help page
warns that `cleveref` "has not been updated" for 2025. The local build uses
TeX Live 2026 and has no `cleveref` problems. If cross-references come out
wrong on arXiv, select TeX Live 2023. If the preprint is posted before the
journal submission, add its arXiv ID to `jmgm/cover-letter.tex` and answer
the preprint question on the journal's form. After acceptance, add the
journal reference and DOI to the arXiv record.

## JMGM submission

Checked against the journal's Guide for Authors (ScienceDirect, saved on
2026-11-18; single-anonymized review, so the manuscript keeps the author's
name):

| Requirement | Where it is met |
|---|---|
| Abstract at most 250 words, stands alone, references only as author and year | `sections/abstract.tex` (250 words, counting every space-separated token of the plain text, symbols and units included; `make jmgm` checks the limit; "Quan and Stamm (2016)") |
| Keywords: 1 to 7 | Six, after the abstract |
| Highlights, required: 3–5 bullets, at most 85 characters each with spaces, separate editable file with "highlights" in its name | `generated/highlights.txt` (written by `make_data.py` from the same values as the text; it asserts both limits), as a Word file `jmgm/submission/highlights.docx` |
| Graphical abstract, required: at least 1328 px wide, longest side horizontal, readable at 13 cm; TIFF, EPS, PDF or MS Office | `jmgm/submission/graphical-abstract.tiff` (2661 × 1065 px) and `.pdf`, from `jmgm/graphical-abstract.tex` |
| Title page: affiliation with postal address and country, corresponding author and email | `main.tex` (Independent researcher, Nedlands, WA 6009, Australia; corresponding author with email) |
| Declaration of generative AI and AI-assisted technologies in the manuscript preparation process: a new section before the references, Elsevier's statement | §9, after Author contributions |
| Funding: the standard sentence when there is none | §9 |
| CRediT roles; acknowledgements directly before the references | §9 |
| Tables as text, no vertical rules or shading | All tables (booktabs) |
| References in any consistent style, numbered in square brackets, DOIs where available | natbib numbers (54 of 79 entries have a DOI) |
| Research data, option C: deposit the data and give the link at submission | Zenodo, version DOI 10.5281/zenodo.24238837 (the 1.1.2 source holds `docs/validation/` and `paper/data/`) |
| LaTeX sources, not only a PDF ("A PDF is not an acceptable source file"); `elsarticle` encouraged, not required | `jmgm/submission/latex/` (flat, built and checked by `make jmgm`) |
| Figures as separate files, named `Figure_1`, `Figure_2`, … | `jmgm/submission/latex/Figure_*` (photographs per panel; TikZ and pgfplots figures as vector PDFs) |
| Cover letter | `jmgm/cover-letter.tex` → `jmgm/submission/cover-letter.pdf` |

`jmgm/README.md` maps every file to its item type in the submission system
and lists what is left: the declarations tool (its Word file is uploaded
with the submission), the suggested reviewers, and the form choices
(preprint on SSRN, open access). Publishing-agreement and open-access
eligibility depend only on the corresponding author's affiliation as given
in the submission.

## Open items (`\todo`)

None (2026-11-18). The previous abstract item (length/format for the target
venue) is resolved: the abstract meets the JMGM limit.

Resolved by the comparison (previous items 5–8, 9, 11–18): the MSMS and
NanoShaper statements are now checked against Goddard's notes and the
NanoShaper paper (Goddard: "MSMS crashes on most structures larger than
10,000 atoms"; NanoShaper paper: MSMS fails to detect the fullerene cavity)
and backed by our own measurements; the failure-mode, non-manifold and
determinism claims of §1 are measured (§7); C₆₀ is in the synthetic set;
x86-64 timings, SIMD levels and thread scaling are measured (§6.2).
Resolved by 1.1.1 (previous item 6): the refinement fix is released; §3.5
describes it, §7 reports the sesmesh runs repeated with 1.1.1, and the
analytic data of §5.1 are unchanged under 1.1.1 (`make data` gives the same
`data/analytic.csv`).
Resolved on 2026-11-14 (previous items 4, 5): §1 cites Poisson–Boltzmann
solvers (review, three boundary-element and two finite-element papers,
checked against Crossref); §6 names the machine (MacBook Air 15″, Apple M4,
4 + 6 cores, 32 GB, fanless, macOS 27.0) and the compilers (rustc 1.99 for
Phase 9, committed before the stable toolchain was updated to 1.100.0 that
morning; 1.100.0 for all later runs).
Resolved on 2026-11-14 (previous item 4, "Open measurements"): the
1 000-entry sweep (#23, `docs/validation/sweep1000.csv`, §5.4) and the
fuzzing campaign (#22, `docs/validation/fuzz.csv`, §4.5) are done. The
8-performance-core Apple timings were closed by decision on 2026-11-15:
the specification (§5.1) now takes the two measured machines (10-core M4,
16-core Zen 5) as its reference machines; §6 and §8 ("Measurement scope")
say so.
Resolved on 2026-11-15 (previous items 1, 2): the title gives "Independent
researcher, Nedlands, WA, Australia", the ORCID iD and the author's personal
email; Declarations → Affiliations names CSIRO, Pawsey and UWA and states
that the work was done independently of them, on the author's own time,
equipment and expense.
Resolved on 2026-11-15 (previous item 2): Zenodo record for 1.1.2,
version DOI 10.5281/zenodo.24238837 (cited in §9 and the `sesmesh` bibliography entry),
all-versions DOI 10.5281/zenodo.24238836 (`CITATION.cff`, README badge).
Resolved on 2026-11-15 (previous items 3–5): the author finalized the
AI-use statement and the acknowledgements; Appendix B names the release tag
`v1.1.2`, which holds the code described and every data file the paper uses
(nothing under `docs/validation/`, `paper/data/`, `crates/`, `xtask/` or the
test sets changed after it); the paper sources themselves are later.
Resolved on 2026-11-15 (previous items 4–6): competing interests (none),
funding (none; the author bore all costs) and a single-author CRediT
statement are written.
Resolved on 2026-11-15 (previous item 10): the acknowledgements cite the
archive as the RCSB PDB and wwPDB policies ask (Berman et al. 2000; Berman,
Henrick, Nakamura 2003; wwPDB consortium 2019; checked against Crossref).

### Bibliography gaps (not marked in the PDF)

`references.bib` deliberately contains only what the repository states,
except the continuum-electrostatics, Protein Data Bank and structures
sections (complete, checked against Crossref). Before submission:

* DOIs missing for Richards 1977, Connolly 1983 (*Science*), Sanner et al.
  1996, Can et al. 2006 (LSMS), Custódio et al. 2013, Ju et al. 2002, Taubin
  1995, Frisken et al. 2000; page range missing for Schroeder et al. 2015
  (LDAV); Kozlíková et al. 2017 lists only the first author ("and others").
* Web and software references (MSMS manual, ChimeraX guide, wwPDB formats,
  hybrid-36, extended XYZ, PQR, OBJ, rayon, fearless_simd, flate2,
  rust-sasa, Rust issue 86656) have no year or access date (BibTeX warns
  "empty year"); add `urldate`/`note = {accessed …}` and versions.
* PDB entries: the RCSB policy asks that an entry be cited by its PDB ID
  and its primary publication. The 16 entries named in the text are cited
  at first mention (primary citations from the RCSB Data API, checked
  against Crossref; `make_data.py` asserts that the IDs the text takes from
  the data are still the cited ones). The 53-entry corpus of Appendix A and
  the 1 200 sweep entries (`tests/data/sets.toml`) are identified by PDB ID
  only; a journal may want the corpus cited too (supplementary table).

## What reviewers will flag

1. **Refinement outliers in 1.1.0**: the comparison found rare vertices up
   to 0.22 Å off the SES in sesmesh 1.1.0. 1.1.1 fixes them; 1.1.2, the
   version described, writes the same meshes as 1.1.1 (only `io/xyz.rs`
   differs under `crates/`); §7.2 reports both. The sesmesh accuracy, sweep,
   synthetic, determinism and capsid runs were repeated with 1.1.1 on the
   Apple Silicon machine (same sphere files and checker); the comparison
   timings are still 1.1.0, backed by a same-machine 1.1.0/1.1.1 timing
   (`docs/validation/compare/ab-refinement-fix.txt`), not by x86 reruns.
2. **Comparison scope**: one machine type, default build options, settings
   matched to sesmesh's resolution rather than tuned per program;
   NanoShaper's mesh smoothing off; MSMS density mapped from h. EDTSurf and
   sesa cannot take user radii, so only their robustness and speed compare;
   EDTSurf's grid does not reach the requested resolution on large inputs.
   sesx (exact triangulation) has no mesh export; GPU codes are out of scope.
   NanoShaper was built from a GitHub mirror of 0.7 (with the official 0.7.5
   patch level reported by the binary), not from the IIT GitLab (unreachable
   from here).
3. **SIMD**: explicit AVX2/AVX-512 kernels gain only 1.2× over the
   auto-vectorized fallback in the field benchmark (target 3×, #8).
4. **Apple Silicon timings** come from one loaded, fanless 4 + 6-core
   laptop (the x86-64 runs are clean). The specification's original
   8-performance-core reference machine was replaced by decision with the
   two measured machines, so the paper has no 8-core Apple figure.
5. **A parser bug found after 1.1.1**: the fuzzing campaign found an
   integer overflow in the extended-XYZ `Properties` parser (a panic on
   crafted input, no wrong mesh), fixed in 1.1.2, the version the paper
   describes (§4.5); the 1.1.1 measurements hold for 1.1.2 (§5, §7).
   The robustness evidence itself is now at the planned scale: 1 000 + 200
   random entries and ≥ 24.9 CPU-hours per fuzz target.
6. **Inscribed-mesh bias**: area always underestimated (−0.20 % at 0.25 Å
   for proteins, up to −0.64 % for atom-sized spheres); no analytic area.
7. **Sub-grid fragments** are counted in the default output; removal is
   opt-in in 1.x.
8. **Memory figures vary by session** on macOS (3J3Q: 7.16 GB vs 4.36–4.86 GB);
   the comparison reports GiB from GNU time on Linux.
9. **AI-assisted development**: needs a clear disclosure (draft in
   Declarations) that matches the venue's policy.

## Journal adaptation notes

* arXiv: the `article` class here is fine; see "arXiv submission" above.
* J. Appl. Cryst. (IUCr "computer programs"): IUCr template; the program
  summary block maps to §9.
* Bioinformatics / JCIM application note: 2–4 pages; keep §1 (short), a
  condensed §3, Tables 4 and 6, Fig. 5 and §9, move the rest to
  supplementary material.
* Computer Physics Communications: needs the CPC "program summary" (title,
  licence, language, nature of problem, solution method) and usually a
  Mendeley Data/CPC library deposit.
