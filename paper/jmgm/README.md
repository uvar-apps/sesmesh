
# JMGM submission

These are the files for submitting the paper to the *Journal of Molecular
Graphics and Modelling* (Elsevier, Editorial Manager). Everything is here
except the suggested reviewers. The requirements were checked against the
journal's Guide for Authors; `../README.md`, *JMGM submission*, lists them.

```bash
cd paper && make jmgm
```

`make jmgm` first builds the paper (`make pdf`). Then
`scripts/make_jmgm.py` writes `jmgm/submission/` from scratch. That
directory is not committed. The script needs pdfLaTeX and Ghostscript.

It checks the following, and stops if a check fails:

- The flat sources build into a PDF with as many pages as `main.pdf`, with
  no undefined references, citations or values.
- Each figure number matches `main.aux`.
- The abstract has at most 250 words.
- There are 1 to 7 keywords.
- There are 3 to 5 highlights, each at most 85 characters.
- The graphical abstract is at least 1328 px wide.

The sources in this directory, which are committed, are
`graphical-abstract.tex`, `cover-letter.tex` and this README.

## What to upload

| File in `submission/` | Item type in the submission system | Notes |
|---|---|---|
| `cover-letter.pdf` | Cover letter | Built from `cover-letter.tex`. Edit the source, not the PDF. |
| `latex/manuscript.tex`, the other `latex/*.tex`, `latex/manuscript.bbl`, `latex/references.bib`, `latex/*.dat` | LaTeX source files | `manuscript.tex` is the main file. The other `.tex` files are the sections and the generated tables and values; the `.dat` files are the plot data. All files sit in one directory, because the system puts every upload in one directory. |
| `latex/Figure_1a.jpg` … `latex/Figure_8c.jpg` | Figure | The photographs. The source includes them under these names. The letters are the panels (a), (b), (c). |
| `latex/Figure_2.pdf`, `Figure_5.pdf`, `Figure_6.pdf`, `Figure_7.pdf` | Figure | Vector PDFs of the TikZ and pgfplots figures, with fonts embedded. They are cut from the compiled source, which also draws these figures itself. |
| `highlights.docx` | Highlights | 5 bullets. |
| `graphical-abstract.tiff` (or `.pdf`) | Graphical abstract | 2661 × 1065 px, 13 × 5.2 cm. |
| (not built here) | Declaration of interest | The Word file from Elsevier's declarations tool (see below). |
| `latex.zip` | — | The files of `latex/` in one archive, for a system that accepts one. |
| `manuscript-preview.pdf` | — | The manuscript built from `latex/`. Compare it with the PDF the system builds. A PDF is not a source file. |
| `submission-form.txt` | — | Plain text for the form fields: article type, title, author, abstract, keywords, highlights, funding, competing interests, research data, AI declaration. |

Figure 4 is a code listing. It is typeset from the source as text and has
no separate file. Figures 5 and 6 each hold two plots, (a) and (b), in one
file, as the source draws them. Elsevier asks for separate files except for
images meant to be compared. Split the plots only if production asks for it.

The flat sources differ from `main.tex` and `sections/` in three ways:

- The paths are flat.
- The images are renamed `Figure_<n><panel>`.
- The date line no longer starts with "Draft of".

The text is otherwise the same.

## Left to you

1. **Declarations tool.** Elsevier asks that it always be completed, and
   its Word file is uploaded with the other files. The manuscript states
   "The author declares no competing interests", which corresponds to the
   option "I have nothing to declare". The affiliations are already
   disclosed in the manuscript (Declarations, Affiliations).
2. **Suggested reviewers.**
3. **Form choices.** Whether to post a free preprint on SSRN (it has no
   effect on the editorial decision), and the open-access or subscription
   route. Publishing-agreement eligibility follows the corresponding
   author's affiliation as given in the submission.
4. **Preprint.** If the paper is posted as a preprint (for example on arXiv)
   before you submit, add a sentence to `cover-letter.tex` and rebuild.
5. **Check the system's PDF.** After the system builds the manuscript,
   compare its PDF with `manuscript-preview.pdf`. The system uses its own
   TeX installation. This one is TeX Live 2026, and the packages that need
   a recent version are `pgfplots` (compat 1.18) and `orcidlink`.
