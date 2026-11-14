# Cross-tool comparison: summary

## Status and topology per campaign

| campaign | tool | h (Å) | runs | ok | closed+oriented as written | after welding | with non-manifold edges | with vertices > 0.1 Å off | median p99 distance (Å) | worst distance (Å) |
|---|---|---|---|---|---|---|---|---|---|---|
| acc | sesmesh | 0.25 | 53 | 53 | 53/53 | 53/53 | 0 | 0 | 9.08e-06 | 0.0829 |
| acc | sesmesh | 0.5 | 53 | 53 | 53/53 | 53/53 | 0 | 1 | 9.26e-06 | 0.172 |
| acc | nanoshaper | 0.25 | 53 | 52 | 52/52 | 44/52 | 0 | 2 | 7.52e-04 | 0.256 |
| acc | nanoshaper | 0.5 | 53 | 53 | 53/53 | 50/53 | 0 | 2 | 7.53e-04 | 0.679 |
| acc | eses | 0.25 | 53 | 52 | 52/52 | 51/52 | 0 | 0 | 4.42e-05 | 0.0005 |
| acc | eses | 0.5 | 53 | 53 | 53/53 | 53/53 | 0 | 0 | 4.42e-05 | 0.0005 |
| acc | msms | 0.25 | 53 | 52 | 27/52 | 26/52 | 24 | 42 | 4.01e-02 | 3.4 |
| acc | msms | 0.5 | 53 | 52 | 45/52 | 35/52 | 7 | 42 | 7.66e-04 | 3.39 |
| acc | chimerax | 0.25 | 53 | 53 | 0/53 | 50/53 | 0 | 53 | 3.40e-01 | 1.4 |
| acc | chimerax | 0.5 | 53 | 53 | 0/53 | 52/53 | 0 | 53 | 6.37e-01 | 1.4 |
| acc | edtsurf | 0.25 | 53 | 53 | 53/53 | 53/53 | 0 | 53 | 7.44e-01 | 1.72 |
| acc | edtsurf | 0.5 | 53 | 53 | 53/53 | 53/53 | 0 | 53 | 9.53e-01 | 1.72 |
| acc | sesa | 0.5 | 53 | 53 | 0/0 | 0/0 | 0 | 0 | – | – |
| det | sesmesh | 0.5 | 10 | 10 | 10/10 | 10/10 | 0 | 0 | 9.38e-06 | 0.0938 |
| det | nanoshaper | 0.5 | 12 | 12 | 12/12 | 12/12 | 0 | 5 | 7.60e-04 | 0.205 |
| det | eses | 0.5 | 10 | 10 | 10/10 | 10/10 | 0 | 0 | 2.08e-04 | 0.0005 |
| det | msms | 0.5 | 6 | 6 | 3/6 | 0/6 | 3 | 6 | 3.53e-01 | 1.85 |
| det | chimerax | 0.5 | 10 | 10 | 0/10 | 10/10 | 0 | 10 | 7.74e-01 | 1.4 |
| det | edtsurf | 0.5 | 6 | 6 | 6/6 | 6/6 | 0 | 6 | 1.17e+00 | 1.4 |
| perf | sesmesh | 0.25 | 48 | 48 | 48/48 | 48/48 | 0 | 0 | 9.06e-06 | 0.00365 |
| perf | sesmesh | 0.5 | 45 | 45 | 45/45 | 45/45 | 0 | 0 | 9.46e-06 | 0.0938 |
| perf | sesmesh | 1.0 | 6 | 6 | 6/6 | 6/6 | 0 | 3 | 1.94e-05 | 0.106 |
| perf | nanoshaper | 0.25 | 15 | 15 | 15/15 | 12/15 | 0 | 3 | 7.58e-04 | 0.198 |
| perf | nanoshaper | 0.5 | 12 | 12 | 12/12 | 6/12 | 0 | 6 | 7.58e-04 | 0.679 |
| perf | nanoshaper | 1.0 | 2 | 1 | 1/1 | 1/1 | 0 | 0 | 4.61e-04 | 0.00242 |
| perf | eses | 0.25 | 7 | 7 | 7/7 | 7/7 | 0 | 0 | 4.42e-05 | 0.0005 |
| perf | eses | 0.5 | 4 | 4 | 4/4 | 4/4 | 0 | 0 | 3.50e-04 | 0.000499 |
| perf | eses | 1.0 | 2 | 2 | 2/2 | 2/2 | 0 | 0 | 1.74e-03 | 0.00496 |
| perf | msms | 0.25 | 7 | 7 | 3/7 | 3/7 | 4 | 4 | 2.97e-01 | 1.93 |
| perf | msms | 0.5 | 4 | 4 | 1/4 | 0/4 | 3 | 4 | 2.65e-01 | 2.08 |
| perf | msms | 1.0 | 2 | 1 | 0/1 | 0/1 | 1 | 1 | 1.15e-03 | 1.83 |
| perf | chimerax | 0.25 | 7 | 7 | 0/7 | 7/7 | 0 | 7 | 3.87e-01 | 1.4 |
| perf | chimerax | 0.5 | 4 | 4 | 0/4 | 3/4 | 0 | 4 | 7.48e-01 | 1.4 |
| perf | chimerax | 1.0 | 2 | 2 | 0/2 | 1/2 | 0 | 2 | 1.29e+00 | 1.4 |
| perf | edtsurf | 0.25 | 9 | 9 | 9/9 | 9/9 | 0 | 9 | 8.75e-01 | 1.4 |
| perf | edtsurf | 0.5 | 12 | 12 | 12/12 | 12/12 | 0 | 12 | 1.28e+00 | 1.72 |
| perf | edtsurf | 1.0 | 2 | 0 | 0/0 | 0/0 | 0 | 0 | – | – |
| perf | sesa | 0.5 | 7 | 6 | 0/0 | 0/0 | 0 | 0 | – | – |
| sw | sesmesh | 0.5 | 200 | 200 | 200/200 | 200/200 | 0 | 3 | 9.25e-06 | 0.218 |
| sw | nanoshaper | 0.5 | 200 | 200 | 200/200 | 182/200 | 0 | 8 | 7.53e-04 | 1.43 |
| sw | eses | 0.5 | 200 | 200 | 200/200 | 200/200 | 0 | 0 | 4.43e-05 | 0.0005 |
| sw | msms | 0.5 | 200 | 195 | 176/195 | 131/195 | 18 | 190 | 6.00e-02 | 3.39 |
| sw | chimerax | 0.5 | 200 | 200 | 0/200 | 198/200 | 0 | 200 | 6.82e-01 | 1.4 |
| sw | edtsurf | 0.5 | 200 | 200 | 200/200 | 200/200 | 0 | 200 | 1.02e+00 | 1.82 |
| sw | sesa | 0.5 | 200 | 200 | 0/0 | 0/0 | 0 | 0 | – | – |
| synth | sesmesh | 0.25 | 9 | 9 | 9/9 | 9/9 | 0 | 0 | 9.33e-06 | 0.00714 |
| synth | nanoshaper | 0.25 | 9 | 7 | 7/7 | 7/7 | 0 | 0 | 4.66e-04 | 0.000689 |
| synth | eses | 0.25 | 9 | 8 | 8/8 | 8/8 | 0 | 2 | 4.63e-06 | 1.28 |
| synth | msms | 0.25 | 9 | 9 | 9/9 | 8/9 | 0 | 1 | 6.12e-04 | 0.492 |
| synth | chimerax | 0.25 | 9 | 8 | 2/8 | 8/8 | 0 | 6 | 1.24e-01 | 0.361 |
| synth | edtsurf | 0.25 | 9 | 9 | 9/9 | 9/9 | 0 | 9 | 7.22e-01 | 1.4 |
| synth | sesa | 0.25 | 9 | 9 | 0/0 | 0/0 | 0 | 0 | – | – |

## Area and volume against the sesmesh h → 0 extrapolation (same spheres)

| tool | h (Å) | entries | area: median | area: worst | volume: median | volume: worst |
|---|---|---|---|---|---|---|
| sesmesh | 0.25 | 53 | -0.20 | -0.23 | -0.01 | -0.08 |
| sesmesh | 0.5 | 53 | -0.83 | -0.93 | -0.04 | -0.34 |
| nanoshaper | 0.25 | 52 | -0.19 | -0.21 | -0.01 | -0.08 |
| nanoshaper | 0.5 | 53 | -0.78 | -0.86 | -0.03 | -0.33 |
| eses | 0.25 | 52 | -0.20 | -0.22 | -0.01 | -0.08 |
| eses | 0.5 | 53 | -0.80 | -0.91 | -0.03 | -0.35 |
| msms | 0.25 | 52 | +0.71 | +2.67 | -0.05 | -0.40 |
| msms | 0.5 | 52 | -0.03 | +1.82 | -0.04 | -0.22 |
| chimerax | 0.25 | 53 | -4.55 | -6.56 | +1.92 | +2.86 |
| chimerax | 0.5 | 53 | -8.97 | -12.75 | +3.90 | +5.92 |

MSMS analytic SES area vs sesmesh A0 (52 entries): median +0.911 %, range -0.334 … +2.793 %.

## Cavities on the accuracy corpus (matched to sesmesh by centroid within 1 Å)

| tool | h (Å) | entries | cavities | sesmesh cavities | not matched | median max dist of unmatched (Å) | sesmesh cavities not found |
|---|---|---|---|---|---|---|---|
| nanoshaper | 0.5 | 53 | 1592 | 1590 | 55 | 0.001 | 53 |
| eses | 0.5 | 53 | 1814 | 1590 | 263 | 0.000 | 39 |
| msms | 0.5 | 52 | 3098 | 1563 | 1671 | 0.516 | 141 |
| chimerax | 0.5 | 53 | 1446 | 1590 | 805 | 1.079 | 949 |
| edtsurf | 0.5 | 53 | 3 | 1590 | 3 | 0.830 | 1590 |
| nanoshaper | 0.25 | 52 | 1338 | 1344 | 10 | 0.001 | 16 |
| eses | 0.25 | 52 | 1398 | 1344 | 65 | 0.000 | 11 |
| msms | 0.25 | 52 | 3095 | 1769 | 1456 | 0.587 | 135 |
| chimerax | 0.25 | 53 | 1720 | 1802 | 570 | 0.604 | 652 |
| edtsurf | 0.25 | 53 | 2 | 1802 | 2 | 0.681 | 1802 |

2PTN at 0.5 Å: sesmesh 14 cavities, 695.0 Å³; nanoshaper 14 cavities, 694.9 Å³; eses 14 cavities, 694.9 Å³; msms 16 cavities, 699.5 Å³; chimerax 6 cavities, 385.9 Å³; edtsurf 0 cavities, 0.0 Å³

2PTN at 0.25 Å: sesmesh 14 cavities, 707.8 Å³; nanoshaper 14 cavities, 707.8 Å³; eses 14 cavities, 707.8 Å³; msms 16 cavities, 712.7 Å³; chimerax 11 cavities, 534.7 Å³; edtsurf 0 cavities, 0.0 Å³

## Determinism (h = 0.5 Å; distinct mesh contents over thread counts and repeats)

| tool | entry | runs (threads) | distinct outputs |
|---|---|---|---|
| sesmesh | 1AON | 5 (1, 2, 4, 8, 16) | 1 |
| sesmesh | 4HHB | 5 (1, 2, 4, 8, 16) | 1 |
| nanoshaper | 1AON | 5 (1, 2, 4, 8, 16) | 1 |
| nanoshaper | 4HHB | 7 (1, 2, 4, 8, 16) | 1 |
| eses | 1AON | 5 (1, 2, 4, 8, 16) | 5 |
| eses | 4HHB | 5 (1, 2, 4, 8, 16) | 1 |
| msms | 1AON | 3 (1) | 1 |
| msms | 4HHB | 3 (1) | 1 |
| chimerax | 1AON | 5 (1, 2, 4, 8, 16) | 1 |
| chimerax | 4HHB | 5 (1, 2, 4, 8, 16) | 1 |
| edtsurf | 1AON | 3 (1) | 1 |
| edtsurf | 4HHB | 3 (1) | 1 |

## Performance (one idle 16-core Zen 5 host; median of repeats)

| tool | entry | h (Å) | threads | opts | runs | status | wall (s) | peak RSS (MB) | triangles |
|---|---|---|---|---|---|---|---|---|---|
| sesmesh | 1AON | 0.25 | 16 |  | 3 | ok | 3.16 | 1323 | 17470776 |
| sesmesh | 1AON | 0.25 | 16 | --simd scalar | 3 | ok | 3.43 | 1306 | 17470776 |
| sesmesh | 1AON | 0.25 | 16 | --simd sse4.2 | 3 | ok | 3.39 | 1319 | 17470776 |
| sesmesh | 1AON | 0.25 | 16 | --simd avx2 | 3 | ok | 3.26 | 1421 | 17470776 |
| sesmesh | 1AON | 0.25 | 16 | --simd avx512 | 3 | ok | 3.2 | 1385 | 17470776 |
| nanoshaper | 1AON | 0.25 | 16 |  | 3 | ok | 41.8 | 7603 | 17470796 |
| eses | 1AON | 0.25 | 16 |  | 1 | ok | 356 | 21291 | 17473784 |
| msms | 1AON | 0.25 | 1 |  | 1 | ok | 65.8 | 1467 | 17098490 |
| chimerax | 1AON | 0.25 | 16 |  | 1 | ok | 91.2 | 3755 | 22670454 |
| edtsurf | 1AON | 0.25 | 1 |  | 3 | ok | 2.79 | 337 | 893788 |
| sesmesh | 1AON | 0.5 | 16 |  | 3 | ok | 1.32 | 482 | 4337900 |
| sesmesh | 1AON | 0.5 | 12 |  | 3 | ok | 1.65 | 478 | 4337900 |
| sesmesh | 1AON | 0.5 | 8 |  | 3 | ok | 2.37 | 478 | 4337900 |
| sesmesh | 1AON | 0.5 | 4 |  | 3 | ok | 4.51 | 422 | 4337900 |
| sesmesh | 1AON | 0.5 | 2 |  | 3 | ok | 8.08 | 386 | 4337900 |
| sesmesh | 1AON | 0.5 | 1 |  | 3 | ok | 14.9 | 353 | 4337900 |
| nanoshaper | 1AON | 0.5 | 16 |  | 3 | ok | 8.24 | 1512 | 4338396 |
| eses | 1AON | 0.5 | 16 |  | 1 | ok | 58.4 | 13265 | 4338336 |
| msms | 1AON | 0.5 | 1 |  | 1 | ok | 22.9 | 1076 | 4106081 |
| chimerax | 1AON | 0.5 | 16 |  | 1 | ok | 22.7 | 994 | 6965364 |
| edtsurf | 1AON | 0.5 | 1 |  | 3 | ok | 2.75 | 340 | 898144 |
| sesa | 1AON | 0.5 | 1 |  | 1 | ok | 15.2 | 732 | None |
| sesmesh | 1CRN | 0.25 | 16 |  | 3 | ok | 0.02 | 20 | 111076 |
| sesmesh | 1CRN | 0.25 | 1 |  | 3 | ok | 0.19 | 18 | 111076 |
| nanoshaper | 1CRN | 0.25 | 16 |  | 3 | ok | 0.18 | 100 | 110816 |
| nanoshaper | 1CRN | 0.25 | 1 |  | 3 | ok | 0.18 | 97 | 110816 |
| eses | 1CRN | 0.25 | 16 |  | 3 | ok | 6.32 | 747 | 110768 |
| msms | 1CRN | 0.25 | 1 |  | 3 | ok | 0.09 | 7 | 105912 |
| chimerax | 1CRN | 0.25 | 16 |  | 3 | ok | 1.47 | 226 | 143128 |
| edtsurf | 1CRN | 0.25 | 1 |  | 3 | ok | 0.2 | 48 | 56350 |
| sesa | 1CRN | 0.5 | 1 |  | 1 | ok | 0.06 | 55 | None |
| sesmesh | 1JJ2 | 0.5 | 16 |  | 6 | ok | 1.9 | 657 | 5980688 |
| sesmesh | 1JJ2 | 0.5 | 12 |  | 3 | ok | 2.43 | 653 | 5980688 |
| sesmesh | 1JJ2 | 0.5 | 8 |  | 3 | ok | 3.46 | 653 | 5980688 |
| sesmesh | 1JJ2 | 0.5 | 4 |  | 3 | ok | 6.34 | 550 | 5980688 |
| sesmesh | 1JJ2 | 0.5 | 2 |  | 3 | ok | 11.6 | 603 | 5980688 |
| sesmesh | 1JJ2 | 0.5 | 1 |  | 3 | ok | 22 | 481 | 5980688 |
| nanoshaper | 1JJ2 | 0.5 | 16 |  | 3 | ok | 11.3 | 1997 | 5979792 |
| eses | 1JJ2 | 0.5 | 16 |  | 1 | ok | 100 | 16229 | 5979304 |
| msms | 1JJ2 | 0.5 | 1 |  | 1 | ok | 34 | 1479 | 5637337 |
| chimerax | 1JJ2 | 0.5 | 16 |  | 1 | ok | 32 | 1391 | 9773146 |
| edtsurf | 1JJ2 | 0.5 | 1 |  | 3 | ok | 3.52 | 457 | 1058042 |
| sesa | 1JJ2 | 0.5 | 1 |  | 1 | ok | 7.48 | 377 | None |
| sesa | 1K4R:asm1 | 0.5 | 1 |  | 1 | ok | 135 | 5941 | None |
| sesmesh | 1K4R:asm1 | 1.0 | 16 |  | 3 | ok | 5.58 | 1217 | 9883344 |
| nanoshaper | 1K4R:asm1 | 1.0 | 16 |  | 1 | ok | 29.1 | 4879 | 9881792 |
| eses | 1K4R:asm1 | 1.0 | 16 |  | 1 | ok | 153 | 18245 | 9879564 |
| msms | 1K4R:asm1 | 1.0 | 1 |  | 1 | ok | 1.29e+03 | 47820 | 10504334 |
| chimerax | 1K4R:asm1 | 1.0 | 16 |  | 1 | ok | 76.2 | 3498 | 21903292 |
| edtsurf | 1K4R:asm1 | 1.0 | 1 |  | 1 | 134 | – | – | – |
| sesa | 3J3Q | 0.5 | 1 |  | 1 | 139 | – | – | – |
| sesmesh | 3J3Q | 1.0 | 16 |  | 3 | ok | 27.5 | 4581 | 48655876 |
| nanoshaper | 3J3Q | 1.0 | 16 |  | 1 | 255 | – | – | – |
| eses | 3J3Q | 1.0 | 16 |  | 1 | ok | 770 | 42710 | 48633948 |
| msms | 3J3Q | 1.0 | 1 |  | 1 | 1 | – | – | – |
| chimerax | 3J3Q | 1.0 | 16 |  | 1 | ok | 369 | 15753 | 108752274 |
| edtsurf | 3J3Q | 1.0 | 1 |  | 1 | 134 | – | – | – |
| sesmesh | 4HHB | 0.25 | 16 |  | 3 | ok | 0.25 | 139 | 1282944 |
| sesmesh | 4HHB | 0.25 | 16 | --simd scalar | 3 | ok | 0.25 | 139 | 1282944 |
| sesmesh | 4HHB | 0.25 | 16 | --simd sse4.2 | 3 | ok | 0.24 | 136 | 1282944 |
| sesmesh | 4HHB | 0.25 | 16 | --simd avx2 | 3 | ok | 0.25 | 142 | 1282944 |
| sesmesh | 4HHB | 0.25 | 16 | --simd avx512 | 3 | ok | 0.23 | 137 | 1282944 |
| sesmesh | 4HHB | 0.25 | 1 |  | 3 | ok | 2.51 | 124 | 1282944 |
| sesmesh | 4HHB | 0.25 | 1 | --simd scalar | 3 | ok | 2.87 | 116 | 1282944 |
| sesmesh | 4HHB | 0.25 | 1 | --simd avx2 | 3 | ok | 2.54 | 116 | 1282944 |
| sesmesh | 4HHB | 0.25 | 1 | --simd avx512 | 3 | ok | 2.57 | 116 | 1282944 |
| nanoshaper | 4HHB | 0.25 | 16 |  | 3 | ok | 2.31 | 432 | 1282716 |
| nanoshaper | 4HHB | 0.25 | 1 |  | 3 | ok | 2.29 | 433 | 1282716 |
| eses | 4HHB | 0.25 | 16 |  | 3 | ok | 36 | 7444 | 1281920 |
| msms | 4HHB | 0.25 | 1 |  | 3 | ok | 2.18 | 63 | 1262841 |
| chimerax | 4HHB | 0.25 | 16 |  | 3 | ok | 7.47 | 474 | 1636200 |
| edtsurf | 4HHB | 0.25 | 1 |  | 3 | ok | 2.64 | 368 | 557350 |
| sesmesh | 4HHB | 0.5 | 16 |  | 3 | ok | 0.11 | 45 | 318812 |
| nanoshaper | 4HHB | 0.5 | 16 |  | 3 | ok | 0.48 | 169 | 318408 |
| eses | 4HHB | 0.5 | 16 |  | 1 | ok | 17 | 1264 | 318252 |
| msms | 4HHB | 0.5 | 1 |  | 1 | ok | 0.37 | 35 | 306108 |
| chimerax | 4HHB | 0.5 | 16 |  | 1 | ok | 2.36 | 259 | 497766 |
| edtsurf | 4HHB | 0.5 | 1 |  | 3 | ok | 0.3 | 58 | 129588 |
| sesa | 4HHB | 0.5 | 1 |  | 1 | ok | 1.13 | 97 | None |
| sesmesh | 4V6X | 0.5 | 16 |  | 3 | ok | 5.54 | 1805 | 18009428 |
| nanoshaper | 4V6X | 0.5 | 16 |  | 3 | ok | 40.6 | 7094 | 18008244 |
| eses | 4V6X | 0.5 | 16 |  | 1 | ok | 280 | 26734 | 18005424 |
| msms | 4V6X | 0.5 | 1 |  | 1 | ok | 507 | 9018 | 17105388 |
| chimerax | 4V6X | 0.5 | 16 |  | 1 | ok | 94.9 | 3830 | 29496852 |
| edtsurf | 4V6X | 0.5 | 1 |  | 3 | ok | 5.95 | 560 | 1397592 |
| sesa | 4V6X | 0.5 | 1 |  | 1 | ok | 37.6 | 1319 | None |
