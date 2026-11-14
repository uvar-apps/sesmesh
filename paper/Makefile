# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# make          tables/plots from the CSV data, then the PDF (latexmk)
# make data     re-run the analytic reference program (builds the library)
# make arxiv    source tarball for arXiv (main.bbl included)
# make clean    remove LaTeX build files

PYTHON  ?= python3
LATEXMK ?= latexmk
TARGET_DIR ?= $(CURDIR)/scripts/analytic/target

.PHONY: all pdf tables data arxiv clean

all: pdf

tables:
	$(PYTHON) -I scripts/make_data.py

pdf: tables
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

data:
	cd scripts/analytic && CARGO_TARGET_DIR=$(TARGET_DIR) cargo run --release --quiet > ../../data/analytic.csv

arxiv: pdf
	rm -rf arxiv arxiv.tar.gz
	mkdir -p arxiv/sections arxiv/generated arxiv/figures
	cp main.tex main.bbl references.bib arxiv/
	cp sections/*.tex arxiv/sections/
	cp generated/* arxiv/generated/
	cp figures/*.jpg arxiv/figures/
	tar -czf arxiv.tar.gz -C arxiv .
	@echo "wrote arxiv.tar.gz"

clean:
	$(LATEXMK) -C main.tex
	rm -f main.bbl main.tdo
	rm -rf arxiv arxiv.tar.gz
