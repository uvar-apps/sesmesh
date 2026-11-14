# Website (`site/`)

The project website, published to <https://uvar-apps.github.io/sesmesh/> by
`.github/workflows/pages.yml` on every push to `main` that touches `site/`.
Plain static HTML/CSS/JS: no build step, no generator. three.js (viewer only)
and the two fonts load from jsDelivr and Google Fonts.

| Path | Content |
|---|---|
| `site/index.html` | Home: hero with the live 3D viewer, features, CLI/library teaser, how it works, documentation links |
| `site/cli.html`, `site/library.html` | Guides (every Rust snippet on the library page compiles against 1.1.0) |
| `site/tutorials.html` | Four walkthroughs with real inputs and outputs |
| `site/gallery.html` | Live viewer + figures with the command behind each |
| `site/render.html` | Still renderer for the display meshes (not linked from the menu) |
| `site/404.html` | Not-found page (absolute `/sesmesh/` links, `noindex`) |
| `site/sitemap.xml` | Public pages for search engines (add new pages here) |
| `site/assets/` | `site.css` (all styles), `site.js` (theme, copy buttons, missing-image slots), `viewer.js` (WebGL viewer), favicon |
| `site/models/*.sesm.gz` | Compact display meshes for the viewer |
| `site/img/gallery/*.webp` | Figures; `site/img/og.png` social preview card |

## One-time setup (GitHub)

Repository **Settings → Pages → Build and deployment → Source: GitHub
Actions**. The workflow then deploys on the next push to `main` (or run it by
hand: Actions → pages → Run workflow).

## Search engines

Every public page has a canonical link, a description and Open Graph tags
(title, description, `img/og.png`); the home page also has a JSON-LD
`SoftwareApplication` block (keep `softwareVersion` current, see
`docs/release.md`). `render.html` and `404.html` are `noindex`. A
`robots.txt` here would be ignored: crawlers only read it at the domain root
(`uvar-apps.github.io/robots.txt`), so the sitemap is submitted in Google
Search Console instead (URL-prefix property
`https://uvar-apps.github.io/sesmesh/`; its HTML verification file goes in
`site/`, then submit `sitemap.xml`). Bing Webmaster Tools can import the
Search Console property.

Repository metadata that helps people find the project: the description and
topics (`gh repo edit --add-topic …`), the social preview image (Settings →
General → Social preview: upload `site/img/og.png`; no API), `CITATION.cff`
(cite button, Zenodo) and the crates' `homepage`, keywords and categories
(changed only by publishing a new version).

## Preview locally

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open <http://127.0.0.1:8765/>. Browsers cache `site.css` aggressively; reload
with the cache disabled after editing it.

## Pictures

All figures are rendered from sesmesh output by `site/render.html` (below).
They live in `site/img/gallery/` under the names the gallery expects; a
missing file shows a hatched box with the expected path.

| File | Figure | Size (px) |
|---|---|---|
| `1crn-element.webp` | Crambin, element colours (wide) | 2000 × 1125 (16:9) |
| `1crn-patch.webp` | Patch types | 1600 × 1200 (4:3) |
| `1crn-vdw.webp`, `1crn-sas.webp` | vdW and SAS surfaces | 1600 × 1200 |
| `1crn-coarse.webp`, `1crn-decimated.webp` | Same triangle budget, wireframe close-ups | 1600 × 1200 |
| `4hhb-chain.webp`, `1aon-chain.webp`, `1aon-cavities.webp` | Haemoglobin, GroEL–GroES, its cavities | 1600 × 1200 |
| `1k4r-capsid.webp` | Dengue virus capsid (full width) | 2400 × 1350 (16:9) |

Frames crop to 4:3 (16:9 for the wide figures) with `object-fit: cover`, so
other sizes work too. Keep files under about 600 KB; WebP or JPEG for
photographic renders, PNG only when needed. A transparent background
picks up the page colour in both themes; otherwise use the paper colour
`#ecebe4`. To use another format, change the `src` in `site/gallery.html`.
To add a figure, copy one `<figure class="fig …">` block there (`w4`, `w6`,
`w8`, `w12` = 4, 6, 8, 12 of 12 columns; keep each row summing to 12).

### Rendering with another viewer

The command under each figure produces its mesh; `tools/site/gallery-meshes.sh`
runs them all into `target/site-work/` (PLY with vertex colours). To replace
a figure with a render from another tool, for example:

* **Blender**: File ▸ Import ▸ Stanford PLY; in the material connect a
  *Color Attribute* node to *Base Color*; render with Cycles or EEVEE and a
  transparent film (Render Properties ▸ Film ▸ Transparent).
* **MeshLab / ParaView**: open the PLY, export a screenshot at the size above.
* **ChimeraX**: write OBJ instead (`-o name.obj`) and `open name.obj`.

### Regenerating with the site's renderer

`site/render.html` draws a display mesh with the site's lighting at an
exact size and offers a download; with the capture server it saves straight
into `site/img/gallery/`:

```bash
sh tools/site/gallery-meshes.sh              # meshes → target/site-work/*.sesm.gz
python3 tools/site/capture.py                # http://127.0.0.1:8777
```

Then open, for example,
`http://127.0.0.1:8777/site/render.html?src=/target/site-work/1crn-element.sesm.gz&w=2000&h=1125&az=30&el=15&zoom=1.28&name=1crn-element.webp&save=1`.
Parameters: `src`, `w`, `h`, `az`/`el` (degrees), `zoom`, `wire=1` (edges
over the surface) or `wire=2` (edges only), `flat=1`, `format=webp|png`,
`name`, `save=1`. The values used for the current figures:

| Figure | Parameters |
|---|---|
| `1crn-element` | `w=2000&h=1125&az=30&el=15&zoom=1.28` |
| `1crn-patch`, `1crn-vdw` | `w=1600&h=1200&az=30&el=15&zoom=1.2` |
| `1crn-sas` | `w=1600&h=1200&az=30&el=15&zoom=1.15` |
| `1crn-coarse`, `1crn-decimated` | `w=1600&h=1200&az=30&el=15&zoom=2.4&wire=1&flat=1` |
| `4hhb-chain` | `w=1600&h=1200&az=20&el=20&zoom=1.12` |
| `1aon-chain`, `1aon-cavities` | `w=1600&h=1200&az=20&el=22&zoom=1.08` |
| `1k4r-capsid` | `w=2400&h=1350&az=0&el=10&zoom=1.35` |

The social card: render `1aon-chain` with `w=1200&h=630&zoom=1.0&format=png&name=og-source.png&save=1`,
then `python3 tools/site/og-card.py site/img/gallery/og-source.png site/img/og.png`
(uses macOS system fonts; any 1200 × 630 PNG works) and delete the source.

## Viewer meshes

The home and gallery viewers load `site/models/{1crn,4hhb,1aon}.sesm.gz`
(names, captions and statistics in `MODELS` at the top of
`site/assets/viewer.js`). They are display copies: 16-bit quantized
positions, 8-bit normals, RGB colours, gzip-compressed (format in
`tools/site/ply2web.py`). Regenerate:

```bash
target/release/sesmesh data/1crn.cif -q high --decimate 0.3 --vertex-colors element -o 1crn.ply
target/release/sesmesh tests/data/cache/4HHB.cif.gz -s 0.4 --max-triangles 90000 --vertex-colors chain -o 4hhb.ply
target/release/sesmesh tests/data/cache/1AON.cif.gz -s 0.7 --max-triangles 220000 --vertex-colors chain -o 1aon.ply
for m in 1crn 4hhb 1aon; do python3 tools/site/ply2web.py $m.ply site/models/$m.sesm.gz; done
```

Keep each under about 2 MB; it is downloaded on page load (1CRN) or on click.
