#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Composes site/img/og.png (1200 × 630 social preview) from a transparent
render (site/render.html …&w=1200&h=630&format=png) and the site's style.

    python3 tools/site/og-card.py site/img/gallery/og-source.png site/img/og.png
"""
import sys

from PIL import Image, ImageDraw, ImageFont

PAPER, INK = (236, 235, 228), (13, 13, 12)
CPK = [((83, 91, 100), 0.618), ((223, 81, 83), 0.196), ((64, 123, 210), 0.168), ((220, 195, 61), 0.018)]
DISPLAY = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"

src, dst = sys.argv[1], sys.argv[2]
W, H = 1200, 630
card = Image.new("RGB", (W, H), PAPER)
d = ImageDraw.Draw(card)
for y in range(11, H, 22):
    for x in range(11, W, 22):
        d.point((x, y), fill=(196, 195, 189))
render = Image.open(src).convert("RGBA")
card.paste(render, (250, 0), render)
d.rectangle([0, 0, W - 1, H - 1], outline=INK, width=6)
x = 0
for color, frac in CPK:
    w = round(W * frac)
    d.rectangle([x, H - 26, x + w, H - 6], fill=color)
    x += w
d.line([0, H - 28, W, H - 28], fill=INK, width=4)
big = ImageFont.truetype(DISPLAY, 150)
mono = ImageFont.truetype(MONO, 24)
label = "RUST LIBRARY + CLI"  # no version: the card outlives releases
d.rectangle([40, 40, 68 + round(d.textlength(label, font=mono)), 82], fill=INK)
d.text((54, 47), label, font=mono, fill=PAPER)
d.text((40, 100), "SESMESH", font=big, fill=INK)
for i, line in enumerate(["Solvent Excluded", "Surface meshes.", "Exact. Watertight.", "Deterministic."]):
    d.text((44, 270 + 40 * i), line, font=mono, fill=INK)
d.text((44, 530), "uvar-apps.github.io/sesmesh", font=mono, fill=(85, 85, 79))
card.save(dst, optimize=True)
print(dst)
