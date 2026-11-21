#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Build the arXiv source bundle and the text for the arXiv submission form.

Run by `make arxiv` after `make pdf` (it needs main.bbl, main.aux and
main.log). Needs pdfLaTeX; Python standard library only. Outputs (paper/):

  arxiv/              what arXiv compiles: main.tex (with \\pdfoutput=1 and a
                      fixed date instead of \\today), main.bbl, the sections,
                      the generated tables and plot data the paper reads, and
                      only the figures it includes
  arxiv.tar.gz        the same files, to upload
  arxiv-metadata.txt  title, authors, ASCII abstract, comments and categories

arXiv's metadata fields accept ASCII only; the abstract may use TeX math,
which arXiv renders with MathJax, and at most 1920 characters.
"""

from __future__ import annotations

import datetime
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_jmgm as common  # noqa: E402  (shared helpers: values, figures, checks)

PAPER = common.PAPER
OUT = PAPER / "arxiv"
TARBALL = PAPER / "arxiv.tar.gz"
METADATA = PAPER / "arxiv-metadata.txt"
PRIMARY, CROSS_LISTS = "q-bio.BM", ("cs.CG", "physics.comp-ph")
ABSTRACT_MAX = 1920
ANGSTROM = r"\,\mathring{\mathrm{A}}"
JOURNAL = "Journal of Molecular Graphics and Modelling"


def bundle() -> str:
    """Write arxiv/; return the bundled main.tex."""
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    order = re.findall(r"\\input\{sections/([\w-]+)\}", main)
    sections = {s: (PAPER / "sections" / f"{s}.tex").read_text(encoding="utf-8")
                for s in order}
    images, _ = common.figure_names(main, sections)

    # \pdfoutput=1 within the first five lines (after the licence header);
    # a fixed date, because arXiv rebuilds the PDF later.
    today = datetime.date.today()
    header = "% SPDX-License-Identifier: MIT OR Apache-2.0\n"
    old_date = "\\date{Draft of \\today{}"
    assert main.count(header) == 1 and main.count(old_date) == 1
    main = main.replace(header, header + "\\pdfoutput=1\n")
    main = main.replace(old_date, f"\\date{{{today:%B} {today.day}, {today.year}")

    text = main + "".join(sections.values())
    generated = set()
    for name in re.findall(r"\{generated/([\w.-]+)\}", common.strip_comments(text)):
        generated.add(name if Path(name).suffix else f"{name}.tex")

    for d in ("sections", "generated", "figures"):
        (OUT / d).mkdir(parents=True)
    (OUT / "main.tex").write_text(main, encoding="utf-8")
    shutil.copy(PAPER / "main.bbl", OUT / "main.bbl")
    for s in order:
        shutil.copy(PAPER / "sections" / f"{s}.tex", OUT / "sections")
    for name in sorted(generated):
        shutil.copy(PAPER / "generated" / name, OUT / "generated")
    for image in images:
        shutil.copy(PAPER / "figures" / f"{image}.jpg", OUT / "figures")
    return main


def check() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "arxiv"
        shutil.copytree(OUT, build)
        return common.compile_check(build, "main.tex")


def tarball() -> None:
    def clean(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(TARBALL, "w:gz") as tar:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=str(f.relative_to(OUT)), filter=clean)


def outside_math(s: str, f) -> str:
    parts = s.split("$")
    return "$".join(f(p) if i % 2 == 0 else p for i, p in enumerate(parts))


def ascii_value(value: str) -> str:
    return value.replace("$-$", "-").replace("$+$", "+").replace("\\,", ",")


def ascii_abstract() -> str:
    """sections/abstract.tex as arXiv metadata: ASCII, TeX math only in $...$."""
    tex = (PAPER / "sections" / "abstract.tex").read_text(encoding="utf-8")
    s = common.strip_comments(tex.split("\\textbf{Keywords:}")[0])
    s = re.sub(r"\\(?:begin|end)\{abstract\}|\\noindent\b", "", s)
    s = re.sub(r"\\val\{([^}]+)\}", lambda m: ascii_value(common.VALUES[m.group(1)]), s)
    s = re.sub(r"\$([^$]*)\$\\Ang\b(?:\{\})?", lambda m: f"${m.group(1)}{ANGSTROM}$", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\\Ang\b(?:\{\})?", lambda m: f"${m.group(1)}{ANGSTROM}$", s)
    s = re.sub(r"\\sesmesh\b(?:\{\})?", "sesmesh", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,\\%", "%").replace("\\%", "%")
    s = outside_math(s, lambda t: t.replace("\\,", " ").replace("~", " "))
    s = re.sub(r"\s+", " ", s).strip()
    assert s.isascii(), s
    assert s.count("$") % 2 == 0
    outside = "".join(p for i, p in enumerate(s.split("$")) if i % 2 == 0)
    assert "\\" not in outside and "{" not in outside, s
    assert len(s) <= ABSTRACT_MAX, len(s)
    return s


def metadata(main: str, n_pages: int) -> None:
    title = common.plain(common.search(r"\\title\{(.*?)\}\s*\\author", main, re.S).group(1))
    assert title.isascii(), title
    abstract = ascii_abstract()
    aux = (PAPER / "main.aux").read_text(encoding="latin-1")
    figures = aux.count("\\contentsline {figure}")
    tables = aux.count("\\contentsline {table}")
    avail = (PAPER / "sections" / "availability.tex").read_text(encoding="utf-8")
    doi = common.search(r"doi:(10\.5281/zenodo\.\d+)", avail).group(1)
    repo = common.search(r"Source code & \\url\{([^}]+)\}", avail).group(1)
    comments = (f"{n_pages} pages, {figures} figures, {tables} tables. "
                f"Code: {repo} (version 1.1.2 archived at doi:{doi})")
    fields = [
        ("UPLOAD", f"{TARBALL.name} (arXiv compiles main.tex with pdfLaTeX)"),
        ("TITLE", title),
        ("AUTHORS", "Ugo Varetto"),
        (f"ABSTRACT ({len(abstract)} of at most {ABSTRACT_MAX} characters; "
         "TeX math is rendered with MathJax)", abstract),
        ("COMMENTS", comments),
        ("COMMENTS, once the paper is submitted to the journal",
         f"{comments}. Submitted to the {JOURNAL}"),
        ("PRIMARY CATEGORY", PRIMARY),
        ("CROSS-LISTS", ", ".join(CROSS_LISTS)),
        ("LICENCE", "CC BY 4.0 (Creative Commons Attribution license); "
                    "it cannot be changed later (see paper/README.md, arXiv)"),
    ]
    out = ["arXiv submission: text for the submission form",
           "Generated by paper/scripts/make_arxiv.py from the paper sources.",
           "Each field is one line, so that it pastes as it is.", ""]
    for head, text in fields:
        out += [head, text, ""]
    METADATA.write_text("\n".join(out), encoding="ascii")
    print(f"  {METADATA.name} (abstract {len(abstract)} characters, "
          f"{figures} figures, {tables} tables)")


def main() -> None:
    for f in ("main.bbl", "main.aux", "main.log"):
        assert (PAPER / f).exists(), f"{f} missing: run `make pdf` first"
    for path in (OUT, TARBALL, METADATA):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    main_tex = bundle()
    n_pages = check()
    tarball()
    files = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"  {OUT.name}/ and {TARBALL.name} ({files} files, builds to {n_pages} pages)")
    metadata(main_tex, n_pages)


if __name__ == "__main__":
    main()
