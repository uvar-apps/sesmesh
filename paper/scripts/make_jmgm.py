#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Assemble the JMGM submission files in paper/jmgm/submission/.

Run by `make jmgm` after `make pdf` (it needs main.bbl and main.aux). Needs
pdfLaTeX and Ghostscript; Python standard library only. The output directory
is deleted and rebuilt on every run.

Outputs (paper/jmgm/submission/, see paper/jmgm/README.md):
  latex/                  flat LaTeX sources and figure files: manuscript.tex,
                          the sections, generated tables and plot data,
                          manuscript.bbl, references.bib, Figure_<n>[a-c].jpg
                          (photographs, included by the source) and
                          Figure_<n>.pdf (TikZ/pgfplots figures cut from the
                          compiled source, which also draws them)
  latex.zip               the same files in one archive
  manuscript-preview.pdf  the manuscript built from latex/ (a check)
  highlights.docx         generated/highlights.txt as a Word file
  graphical-abstract.pdf, graphical-abstract.tiff
  cover-letter.pdf        from jmgm/cover-letter.tex
  submission-form.txt     title, abstract, keywords and statements as plain
                          text for the submission form
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

PAPER = Path(__file__).resolve().parent.parent
JMGM = PAPER / "jmgm"
OUT = JMGM / "submission"
LATEX = OUT / "latex"
GA_DPI = 520  # 13 cm wide -> 2661 px (Elsevier: at least 1328 px)
ZIP_DATE = (2026, 1, 1, 0, 0, 0)  # fixed, so the same files give the same archive

FIGURE = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)
GRAPHIC = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}")


def run(cmd: list[str], cwd: Path) -> str:
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, errors="replace", check=False)
    if r.returncode != 0:
        sys.exit(f"failed in {cwd}: {' '.join(cmd)}\n{r.stdout[-4000:]}")
    return r.stdout


def pdflatex(tex: str, cwd: Path, outdir: Path | None = None) -> str:
    """One pdfLaTeX run; returns the log file's text."""
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]
    if outdir is not None:
        cmd.append(f"-output-directory={outdir}")
    run([*cmd, tex], cwd)
    log = (outdir or cwd) / (Path(tex).stem + ".log")
    return log.read_text(encoding="latin-1")


def pages(log: str) -> int:
    # TeX wraps log lines at 79 characters, also inside long paths.
    m = re.search(r"Output written on .*?\((\d+) pages?", log.replace("\n", ""))
    assert m, "no PDF written"
    return int(m.group(1))


def search(pattern: str, text: str, flags: int = 0) -> re.Match[str]:
    m = re.search(pattern, text, flags)
    assert m, pattern
    return m


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


# ----------------------------------------------------------------------------
# Plain text for the submission form


def macro_values() -> dict[str, str]:
    text = (PAPER / "generated" / "macros.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"^\\@namedef\{smval@([^}]+)\}\{(.*)\}$", text, re.M))


VALUES = macro_values()


def plain(latex: str) -> str:
    """The LaTeX of the abstract, title and declarations as plain text."""
    s = strip_comments(latex)
    s = re.sub(r"\\val\{([^}]+)\}", lambda m: VALUES[m.group(1)], s)
    s = s.replace("$-$", "\u2212").replace("$+$", "+")
    s = re.sub(r"\\(?:begin|end)\{abstract\}|\\noindent\b", "", s)
    s = re.sub(r"\\sesmesh\b(\{\})?", "sesmesh", s)
    s = re.sub(r"\\(?:emph|textbf|code)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\Ang\b(\{\})?", " \u00c5", s)
    s = s.replace("\\,\\%", " %").replace("\\%", "%").replace("\\,", " ")
    s = s.replace("\\to", "\u2192").replace("^2", "\u00b2").replace("$", "")
    s = s.replace("\\\\", " ").replace("~", " ")
    s = re.sub(r"\s+", " ", s).strip()
    assert not re.search(r"[\\{}]", s), s
    return s


def paragraph(text: str, title: str) -> str:
    """A \\paragraph{title.} of the availability section, as plain text."""
    m = re.search(r"\\paragraph\{" + re.escape(title) + r"\.\}(.*?)(?=\\paragraph|\\section|\Z)",
                  text, re.S)
    assert m, title
    return plain(m.group(1))


def submission_form(main: str) -> None:
    abstract_tex = (PAPER / "sections" / "abstract.tex").read_text(encoding="utf-8")
    body, keywords_tex = abstract_tex.split("\\textbf{Keywords:}")
    abstract = plain(body)
    words = len(abstract.split())
    assert words <= 250, words
    keywords = [k.strip() for k in plain(keywords_tex).rstrip(".").split(";")]
    assert 1 <= len(keywords) <= 7, keywords

    title = plain(search(r"\\title\{(.*?)\}\s*\\author", main, re.S).group(1))
    orcid = search(r"\\orcidlink\{([^}]+)\}", main).group(1)
    affiliation = search(r"\\small (Independent researcher[^\\]*)\\\\", main).group(1)
    email = search(r"mailto:([^}]+)\}", main).group(1)

    avail = (PAPER / "sections" / "availability.tex").read_text(encoding="utf-8")
    dois = re.findall(r"doi:(10\.5281/zenodo\.\d+)", avail)
    assert len(dois) == 2, dois  # version DOI first, then all versions
    repo = search(r"Source code & \\url\{([^}]+)\}", avail).group(1)
    highlights = (PAPER / "generated" / "highlights.txt").read_text(encoding="utf-8")

    data = (
        f"The sesmesh source code, version 1.1.2 (the version the article describes), "
        f"and the validation data used in the article are archived on Zenodo: "
        f"https://doi.org/{dois[0]} (all versions: https://doi.org/{dois[1]}). "
        f"Development repository: {repo}. The structures were obtained from the "
        f"Protein Data Bank (RCSB PDB) and are identified by their PDB IDs."
    )
    sections = [
        ("ARTICLE TYPE", "Research article (the full-length research type the form offers)"),
        ("TITLE", title),
        ("AUTHOR", f"Ugo Varetto (ORCID {orcid})\n{affiliation}\n"
                   f"Corresponding author: {email}"),
        (f"ABSTRACT ({words} words)", abstract),
        (f"KEYWORDS ({len(keywords)})", "; ".join(keywords)),
        ("HIGHLIGHTS (also in highlights.docx)", highlights.rstrip("\n")),
        ("FUNDING", paragraph(avail, "Funding")),
        ("COMPETING INTERESTS (declarations tool: \"I have nothing to declare\")",
         paragraph(avail, "Competing interests")),
        ("RESEARCH DATA (option C: data deposited, link given)", data),
        ("DECLARATION OF GENERATIVE AI (also in the manuscript, Declarations)",
         paragraph(avail, "Declaration of generative AI and AI-assisted technologies "
                          "in the manuscript preparation process")),
    ]
    out = ["JMGM submission: text for the submission form",
           "Generated by paper/scripts/make_jmgm.py from the manuscript sources.",
           "Each paragraph is one line, so that it pastes into a form field as it is.", ""]
    for head, text in sections:
        out += [head, text, ""]
    (OUT / "submission-form.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"  submission-form.txt (abstract {words} words, {len(keywords)} keywords)")


# ----------------------------------------------------------------------------
# Flat LaTeX sources and figure files


def figure_names(main: str, sections: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Image name -> Figure_<n>[letter]; names of the TikZ figures in order."""
    aux = (PAPER / "main.aux").read_text(encoding="latin-1")
    numbers = dict(re.findall(r"\\newlabel\{([^}@]+)\}\{\{(\d+)\}", aux))
    images: dict[str, str] = {}
    tikz: list[str] = []
    n = 0
    for text in sections.values():
        for m in FIGURE.finditer(strip_comments(text)):
            n += 1
            body = m.group(1)
            label = re.findall(r"\\label\{([^}]+)\}", body)[-1]
            assert numbers[label] == str(n), (label, numbers[label], n)
            graphics = [g for _, g in GRAPHIC.findall(body)]
            for i, g in enumerate(graphics):
                images[g] = f"Figure_{n}" + ("abcdefgh"[i] if len(graphics) > 1 else "")
            k = body.count("\\begin{tikzpicture}")
            tikz += [f"Figure_{n}" + ("abcdefgh"[i] if k > 1 else "") for i in range(k)]
    every = sum(strip_comments(t).count("\\begin{tikzpicture}") for t in sections.values())
    assert every == len(tikz), "a tikzpicture outside a figure would be cut out too"
    assert "tikzpicture" not in strip_comments(main.split("\\begin{document}")[0])
    return images, tikz


def flatten(text: str, images: dict[str, str]) -> str:
    text = text.replace("\\input{sections/", "\\input{").replace("{generated/", "{")
    return GRAPHIC.sub(lambda m: m.group(1) + "{" + images.get(m.group(2), m.group(2)) + "}",
                       text)


def put(name: str, data: bytes | str) -> None:
    dest = LATEX / name
    assert not dest.exists(), f"two source files named {name}"
    if isinstance(data, str):
        dest.write_text(data, encoding="utf-8")
    else:
        dest.write_bytes(data)


def latex_sources() -> tuple[str, list[str]]:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    order = re.findall(r"\\input\{sections/([\w-]+)\}", main)
    sections = {s: (PAPER / "sections" / f"{s}.tex").read_text(encoding="utf-8") for s in order}
    images, tikz = figure_names(main, sections)

    manuscript = flatten(main, images)
    for old, new in [("\\graphicspath{{figures/}}\n", ""),
                     ("\\date{Draft of \\today{}", "\\date{\\today{}")]:
        assert manuscript.count(old) == 1, old
        manuscript = manuscript.replace(old, new)
    LATEX.mkdir(parents=True)
    put("manuscript.tex", manuscript)
    for s, text in sections.items():
        put(f"{s}.tex", flatten(text, images))
    for f in sorted((PAPER / "generated").iterdir()):
        if f.suffix in (".tex", ".dat"):
            put(f.name, f.read_bytes())
    put("references.bib", (PAPER / "references.bib").read_bytes())
    put("manuscript.bbl", (PAPER / "main.bbl").read_bytes())
    for image, name in images.items():
        src = PAPER / "figures" / f"{image}.jpg"
        put(f"{name}.jpg", src.read_bytes())
    return main, tikz


def check_build(tikz: list[str]) -> None:
    """Build latex/ in a scratch copy: the manuscript, then the TikZ figures."""
    main_pages = pages((PAPER / "main.log").read_text(encoding="latin-1"))
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "latex"
        shutil.copytree(LATEX, build)
        log = ""
        for _ in range(4):
            log = pdflatex("manuscript.tex", build)
            if not re.search(r"Rerun to get|Label\(s\) may have changed", log):
                break
        bad = re.findall(r"^(?:LaTeX|Package \w+) Warning: .*(?:undefined|Undefined).*$", log, re.M)
        assert not bad, bad
        assert pages(log) == main_pages, (pages(log), main_pages)
        shutil.copy(build / "manuscript.pdf", OUT / "manuscript-preview.pdf")
        print(f"  manuscript-preview.pdf ({main_pages} pages, as main.pdf)")

        # Each tikzpicture on its own tightly cropped page (preview package);
        # \orcidlink draws its icon with TikZ too, so it is switched off here.
        src = (build / "manuscript.tex").read_text(encoding="utf-8")
        preview = ("\\usepackage[active,tightpage]{preview}\n"
                   "\\PreviewEnvironment{tikzpicture}\n"
                   "\\setlength\\PreviewBorder{2pt}\n"
                   "\\renewcommand{\\orcidlink}[1]{}\n\\begin{document}")
        (build / "figures.tex").write_text(src.replace("\\begin{document}", preview, 1),
                                           encoding="utf-8")
        pdflatex("figures.tex", build)
        assert pages(pdflatex("figures.tex", build)) == len(tikz)
        # One file per page; pdfTeX copies the page with its embedded fonts.
        (build / "split.tex").write_text(
            "\\documentclass[border=0pt]{standalone}\n\\usepackage{graphicx}\n"
            "\\begin{document}\\includegraphics[page=\\pg]{figures.pdf}\\end{document}\n",
            encoding="utf-8")
        for i, name in enumerate(tikz, start=1):
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={name}",
                 f"\\def\\pg{{{i}}}\\input{{split}}"], build)
            shutil.copy(build / f"{name}.pdf", LATEX / f"{name}.pdf")
    print(f"  latex/ ({len(list(LATEX.iterdir()))} files; TikZ figures {', '.join(tikz)})")


def zip_latex() -> None:
    with zipfile.ZipFile(OUT / "latex.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(LATEX.iterdir()):
            info = zipfile.ZipInfo(f.name, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, f.read_bytes())


# ----------------------------------------------------------------------------
# Highlights, graphical abstract, cover letter


def docx(path: Path, heading: str, paragraphs: list[str]) -> None:
    """A minimal Word document: a bold heading, then one paragraph per item."""
    def para(text: str, bold: bool = False) -> str:
        rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
        return f'<w:p><w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'

    w = "http://schemas.openxmlformats.org"
    parts = {
        "[Content_Types].xml":
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Types xmlns="{w}/package/2006/content-types">'
            f'<Default Extension="rels" ContentType="application/'
            f'vnd.openxmlformats-package.relationships+xml"/>'
            f'<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/word/document.xml" ContentType="application/'
            f'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        "_rels/.rels":
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Relationships xmlns="{w}/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{w}/officeDocument/2006/relationships/'
            f'officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml":
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{w}/wordprocessingml/2006/main"><w:body>'
            + para(heading, bold=True) + "".join(para(p) for p in paragraphs)
            + "<w:sectPr/></w:body></w:document>",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in parts.items():
            z.writestr(zipfile.ZipInfo(name, ZIP_DATE), text.encode("utf-8"),
                       compress_type=zipfile.ZIP_DEFLATED)


def highlights() -> None:
    items = (PAPER / "generated" / "highlights.txt").read_text(encoding="utf-8").splitlines()
    assert 3 <= len(items) <= 5 and all(i.startswith("\u2022 ") for i in items), items
    assert all(len(i[2:]) <= 85 for i in items)
    docx(OUT / "highlights.docx", "Highlights", items)
    print(f"  highlights.docx ({len(items)} bullets)")


def tiff_size(path: Path) -> tuple[int, int]:
    """Width and height from the first image directory of a TIFF file."""
    b = path.read_bytes()
    e = "<" if b[:2] == b"II" else ">"
    ifd = struct.unpack(e + "I", b[4:8])[0]
    tags = {}
    for i in range(struct.unpack(e + "H", b[ifd:ifd + 2])[0]):
        entry = b[ifd + 2 + 12 * i:ifd + 14 + 12 * i]
        tag, kind = struct.unpack(e + "HH", entry[:4])
        tags[tag] = struct.unpack(e + "H", entry[8:10])[0] if kind == 3 else \
            struct.unpack(e + "I", entry[8:12])[0]
    return tags[256], tags[257]


def graphical_abstract_and_letter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp)
        pdflatex("graphical-abstract.tex", JMGM, build)
        pdf = build / "graphical-abstract.pdf"
        shutil.copy(pdf, OUT / "graphical-abstract.pdf")
        run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=tiff24nc",
             "-sCompression=lzw", f"-r{GA_DPI}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
             f"-sOutputFile={OUT / 'graphical-abstract.tiff'}", str(pdf)], build)
        width, height = tiff_size(OUT / "graphical-abstract.tiff")
        assert width >= 1328 and width > height, (width, height)
        print(f"  graphical-abstract.pdf, .tiff ({width} x {height} px)")

        log = pdflatex("cover-letter.tex", JMGM, build)
        assert "Undefined value" not in log
        shutil.copy(build / "cover-letter.pdf", OUT / "cover-letter.pdf")
        print(f"  cover-letter.pdf ({pages(log)} page{'s' if pages(log) > 1 else ''})")


def main() -> None:
    for f in ("main.bbl", "main.aux", "main.log"):
        assert (PAPER / f).exists(), f"{f} missing: run `make pdf` first"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    print(f"writing {OUT.relative_to(PAPER)}/")
    main_tex, tikz = latex_sources()
    check_build(tikz)
    zip_latex()
    highlights()
    graphical_abstract_and_letter()
    submission_form(main_tex)


if __name__ == "__main__":
    main()
