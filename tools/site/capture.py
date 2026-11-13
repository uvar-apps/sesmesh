#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Local server for site/render.html: serves the repository root and saves
rendered images posted to /__save?name=<file> into site/img/gallery/.

    python3 tools/site/capture.py            # http://127.0.0.1:8777/site/render.html?...&save=1

Listens on 127.0.0.1 only. Accepts .webp / .png names without paths.
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "site", "img", "gallery")
NAME = re.compile(r"^[A-Za-z0-9._-]+\.(webp|png)$")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self):
        url = urlparse(self.path)
        name = parse_qs(url.query).get("name", [""])[0]
        if url.path != "/__save" or not NAME.match(name):
            self.send_error(400, "POST /__save?name=<file.webp|file.png>")
            return
        size = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(size)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(data)
        self.send_response(204)
        self.end_headers()
        sys.stderr.write(f"saved {name} ({size} bytes)\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    print(f"serving {ROOT} on http://127.0.0.1:{port}/site/render.html")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
