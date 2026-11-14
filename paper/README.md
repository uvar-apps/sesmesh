# sesmesh paper (draft)

LaTeX sources of a full-length research/software paper on sesmesh 1.1.0,
meant for an arXiv preprint (primary **q-bio.BM**, cross-lists **cs.CG** and
**physics.comp-ph**) and later a journal (J. Appl. Cryst. "computer
programs", Bioinformatics or JCIM application note, or Computer Physics
Communications). Nothing has been submitted anywhere.

Every number in the text comes from the repository: the validation CSVs
(`docs/validation/*.csv`), the design document, the validation report, the
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
| `sections/*.tex` | Abstract, introduction, background, method, implementation, validation, performance, comparison (placeholder), discussion, availability + declarations, appendices |
| `references.bib` | Methods references of the README ("How it works and what to cite") and Spec §12, transcribed from the repository only |
| `figures/*.jpg` | Gallery renders from `site/img/gallery/*.webp`, flattened on white and trimmed (`magick … -background white -alpha remove -trim`) |
| `data/analytic.csv` | One/two-sphere references meshed at h = 1, 0.5, 0.25, 0.125 Å (from `make data`) |
| `data/scaling.csv` | Thread-scaling numbers transcribed from Design §11.1 |
| `scripts/make_data.py` | CSV → LaTeX tables, macros, pgfplots data |
| `scripts/analytic/` | Program that produced `data/analytic.csv` |
| `generated/` | Output of `make_data.py` (committed, so the paper builds without Python) |

## Open items (`\todo`)

Numbers match the red markers in the PDF (draft of 2026-11-14).

| # | Where | Item |
|---|---|---|
| 1 | title | Affiliation (or "Independent researcher"), city, country |
| 2 | title | ORCID iD |
| 3 | abstract | Final abstract length/format for the target venue (Bioinformatics: 150 words, structured; J. Appl. Cryst.: unstructured) |
| 4 | §1 Molecular surfaces | Cite representative Poisson–Boltzmann BEM/FEM solvers that consume SES meshes (none in the repository's reference list) |
| 5 | §1 Existing programs | Check Goddard's statement that MSMS fails on many structures > 10 000 atoms (Chimera notes, [R19]) and its scope |
| 6 | §1 Existing programs | Verify the NanoShaper paper's fullerene/MSMS failure claim and the MSMS version |
| 7 | §1 Failure modes (ii) | Source for "meshes that are not closed 2-manifolds" per tool, or drop the comparison and keep only sesmesh's guarantee |
| 8 | §1 Failure modes (iv) | Verify that NanoShaper, EDTSurf, ChimeraX and the GPU codes do not document thread-independent output; rephrase if any does |
| 9 | §5.4 Degenerate geometry | Add a real C60 test (the suite uses an icosahedron + dodecahedron cage) or drop the remark |
| 10 | §6 Machine | Exact chip, memory size, macOS and rustc versions of the measurement machine |
| 11 | §6 End-to-end | Re-measure on a quiet, homogeneous machine with hyperfine (10 runs, 2 warm-ups) and on x86-64 with AVX2/AVX-512 (#8, #9, #10) |
| 12 | §7.1 Programs | Versions, build options, exact command lines of MSMS, NanoShaper, EDTSurf, sesa, ChimeraX; decide on ESES and the Quan–Stamm mesher |
| 13 | §7.3 item 1 | Run the accuracy comparison (#5); fill Table 8 from `tests/data/reference/*.toml` |
| 14 | §7.3 item 2 | Choose an independent mesh checker; run the robustness comparison |
| 15 | §7.3 item 3 | Run the performance comparison on both reference machines (#6) |
| 16 | §7.3 item 4 | Run the determinism comparison (1/2/4/8/16 threads per program) |
| 17 | §7.3 item 5 | Run the normal comparison |
| 18 | §8 Open measurements | Update the paragraph as items are completed |
| 19 | §9 Archive | Mint a Zenodo DOI for the described version; cite it in the paper and `CITATION.cff` |
| 20 | Declarations | Review and finalize the AI-use statement; check the venue's policy |
| 21 | Declarations | Confirm "no competing interests" |
| 22 | Declarations | Funding statement |
| 23 | Declarations | CRediT statement if required |
| 24 | Declarations | Acknowledgements |
| 25 | Declarations | Cite the PDB reference recommended by RCSB/wwPDB |
| 26 | Appendix B | Release tag or commit the paper refers to |

Placeholders that are not `\todo`s: the whole of §7 (comparison) is
framed as a placeholder box, and Table 8 cells are red `?` (`\tbd`).

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

1. **No cross-program comparison** (§7, issues #5 and #6): accuracy is shown
   only against analytic surfaces and sesmesh's own h → 0 extrapolation,
   which measures discretization error, not agreement with MSMS, NanoShaper,
   EDTSurf or sesa. Speed and memory are not compared either.
2. **One measurement machine, under load**: all timings come from a 10-core
   Apple Silicon machine (4 P + 6 E cores) with load average 15–27; no x86
   numbers, no AVX2/AVX-512 speed-up (#8, #9, #10, #13), and the 8-core
   parallel-efficiency target is untested. The provisional performance
   targets were missed by up to 2.2× and then revised to the measurements.
3. **Failure-mode claims about other tools** (items 5–8) rest on two
   secondary sources or have no source yet.
4. **Scale of the robustness evidence**: 200 random entries instead of the
   planned 1 000 (#23); fuzzing 10 min per target instead of 24 CPU-hours
   (#22).
5. **Inscribed-mesh bias**: area always underestimated (−0.20 % at 0.25 Å
   for proteins, up to −0.64 % for atom-sized spheres); no analytic area.
6. **Sub-grid fragments** are counted in the default output (and in
   `accuracy.csv`/`sweep.csv` component counts); removal is opt-in in 1.x.
7. **Memory figures vary by session** on macOS (3J3Q: 7.16 GB vs 4.36–4.86 GB).
8. **AI-assisted development**: needs a clear disclosure (draft in
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
