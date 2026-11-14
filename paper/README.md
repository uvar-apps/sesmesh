# sesmesh paper (draft)

LaTeX sources of a full-length research/software paper on sesmesh 1.1.0,
meant for an arXiv preprint (primary **q-bio.BM**, cross-lists **cs.CG** and
**physics.comp-ph**) and later a journal (J. Appl. Cryst. "computer
programs", Bioinformatics or JCIM application note, or Computer Physics
Communications). Nothing has been submitted anywhere.

Every number in the text comes from the repository: the validation CSVs
(`docs/validation/*.csv`), the cross-tool comparison
(`docs/validation/compare/*.csv`, `bench-field-zen5.txt`; harness in
`tools/compare/`), the design document, the validation report, the
CHANGELOG, or the analytic-reference program in `scripts/analytic`. Claims
that need a source or a measurement the repository does not have are marked
`\todo{…}`; they print in red and are collected on the last page of the PDF
("Open items").

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
| `make arxiv` | Builds the PDF, then packs `main.tex`, `main.bbl`, `references.bib`, `sections/`, `generated/`, `figures/` into `arxiv.tar.gz` (arXiv wants the `.bbl`) |
| `make clean` | Removes LaTeX build files |

## Layout

| Path | Content |
|---|---|
| `main.tex` | Preamble, `\todo` and `\val` macros, section order |
| `sections/*.tex` | Abstract, introduction, background, method, implementation, validation, performance (incl. x86-64 §6.2), comparison with six programs (§7), discussion, availability + declarations, appendices |
| `references.bib` | Methods references of the README ("How it works and what to cite") and Spec §12, transcribed from the repository only |
| `figures/*.jpg` | Gallery renders from `site/img/gallery/*.webp`, flattened on white and trimmed (`magick … -background white -alpha remove -trim`) |
| `data/analytic.csv` | One/two-sphere references meshed at h = 1, 0.5, 0.25, 0.125 Å (from `make data`) |
| `data/scaling.csv` | Thread-scaling numbers transcribed from Design §11.1 |
| `scripts/make_data.py` | CSV → LaTeX tables, macros, pgfplots data (validation, comparison, x86 SIMD and scaling) |
| `scripts/analytic/` | Program that produced `data/analytic.csv` |
| `generated/` | Output of `make_data.py` (committed, so the paper builds without Python) |

## Open items (`\todo`)

Numbers match the red markers in the PDF (draft of 2026-11-14, after the
cross-tool comparison).

| # | Where | Item |
|---|---|---|
| 1 | title | Affiliation (or "Independent researcher"), city, country |
| 2 | title | ORCID iD |
| 3 | abstract | Final abstract length/format for the target venue (Bioinformatics: 150 words, structured; J. Appl. Cryst.: unstructured) |
| 4 | §1 Molecular surfaces | Cite representative Poisson–Boltzmann BEM/FEM solvers that consume SES meshes (none in the repository's reference list) |
| 5 | §6 Machine | Exact chip, memory size, macOS and rustc versions of the Apple Silicon machine |
| 6 | §7.2 | Name the release that contains the refinement fix (branch `claude/intelligent-pasteur-0b9022`) and update §3.5, §5.1 |
| 7 | §8 Open measurements | Update the paragraph as items are completed |
| 8 | §9 Archive | Mint a Zenodo DOI for the described version; cite it in the paper and `CITATION.cff` |
| 9 | Declarations | Review and finalize the AI-use statement; check the venue's policy |
| 10 | Declarations | Confirm "no competing interests" |
| 11 | Declarations | Funding statement |
| 12 | Declarations | CRediT statement if required |
| 13 | Declarations | Acknowledgements |
| 14 | Declarations | Cite the PDB reference recommended by RCSB/wwPDB |
| 15 | Appendix B | Release tag or commit the paper refers to |

Resolved by the comparison (previous items 5–8, 9, 11–18): the MSMS and
NanoShaper statements are now checked against Goddard's notes and the
NanoShaper paper (Goddard: "MSMS crashes on most structures larger than
10,000 atoms"; NanoShaper paper: MSMS fails to detect the fullerene cavity)
and backed by our own measurements; the failure-mode, non-manifold and
determinism claims of §1 are measured (§7); C₆₀ is in the synthetic set;
x86-64 timings, SIMD levels and thread scaling are measured (§6.2).

### Bibliography gaps (not marked in the PDF)

`references.bib` deliberately contains only what the repository states.
Before submission:

* DOIs missing for Richards 1977, Connolly 1983 (*Science*), Sanner et al.
  1996, Can et al. 2006 (LSMS), Custódio et al. 2013, Ju et al. 2002, Taubin
  1995, Frisken et al. 2000; page range missing for Schroeder et al. 2015
  (LDAV); Kozlíková et al. 2017 lists only the first author ("and others").
* Web and software references (MSMS manual, ChimeraX guide, wwPDB formats,
  hybrid-36, extended XYZ, PQR, OBJ, rayon, fearless_simd, flate2,
  rust-sasa, Rust issue 86656) have no year or access date (BibTeX warns
  "empty year"); add `urldate`/`note = {accessed …}` and versions.
* The sesmesh entry should carry the Zenodo DOI (item 19).

## What reviewers will flag

1. **Refinement outliers in the described version**: the comparison found
   rare vertices up to 0.22 Å off the SES in sesmesh 1.1.0 (now fixed on a
   branch, not yet released). The paper says so (§7.2); release the fix and
   decide which version the paper describes (items 6, 15).
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
4. **Apple Silicon timings** still come from one loaded 10-core machine
   (the x86-64 runs are clean); the 8-performance-core reference machine
   was never measured.
5. **Scale of the robustness evidence**: 200 random entries instead of the
   planned 1 000 (#23); fuzzing 10 min per target instead of 24 CPU-hours
   (#22).
6. **Inscribed-mesh bias**: area always underestimated (−0.20 % at 0.25 Å
   for proteins, up to −0.64 % for atom-sized spheres); no analytic area.
7. **Sub-grid fragments** are counted in the default output; removal is
   opt-in in 1.x.
8. **Memory figures vary by session** on macOS (3J3Q: 7.16 GB vs 4.36–4.86 GB);
   the comparison reports GiB from GNU time on Linux.
9. **AI-assisted development**: needs a clear disclosure (draft in
   Declarations) that matches the venue's policy.

## Journal adaptation notes

* arXiv: the `article` class here is fine; upload `arxiv.tar.gz`.
* J. Appl. Cryst. (IUCr "computer programs"): IUCr template; the program
  summary block maps to §9.
* Bioinformatics / JCIM application note: 2–4 pages; keep §1 (short), a
  condensed §3, Tables 4 and 6, Fig. 5 and §9, move the rest to
  supplementary material.
* Computer Physics Communications: needs the CPC "program summary" (title,
  licence, language, nature of problem, solution method) and usually a
  Mendeley Data/CPC library deposit.
