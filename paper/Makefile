# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# make          tables/plots from the CSV data, then the PDF (latexmk)
# make data     re-run the analytic reference program (builds the library)
# make arxiv    arXiv bundle (arxiv/, arxiv.tar.gz) and form text (arxiv-metadata.txt)
# make jmgm     JMGM submission files in jmgm/submission/ (see jmgm/README.md)
# make clean    remove LaTeX build files

PYTHON  ?= python3
LATEXMK ?= latexmk
TARGET_DIR ?= $(CURDIR)/scripts/analytic/target

.PHONY: all pdf tables data arxiv jmgm clean

all: pdf

tables:
	$(PYTHON) -I scripts/make_data.py

pdf: tables
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

data:
	cd scripts/analytic && CARGO_TARGET_DIR=$(TARGET_DIR) cargo run --release --quiet > ../../data/analytic.csv

arxiv: pdf
	$(PYTHON) -I scripts/make_arxiv.py

jmgm: pdf
	$(PYTHON) -I scripts/make_jmgm.py

clean:
	$(LATEXMK) -C main.tex
	rm -f main.bbl main.tdo
	rm -rf arxiv arxiv.tar.gz arxiv-metadata.txt
	rm -rf jmgm/submission
