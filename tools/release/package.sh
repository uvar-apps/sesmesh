#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Builds and packages a release binary locally (docs/release.md; nothing runs
# on GitHub):
#
#   tools/release/package.sh TARGET [OUT_DIR]
#
# Builds `sesmesh` with the `dist` profile for TARGET (a Rust target triple
# this machine can build and run, e.g. aarch64-apple-darwin on an Apple
# Silicon Mac, x86_64-unknown-linux-gnu on Linux), smoke-tests it, and writes
# OUT_DIR/sesmesh-VERSION-TARGET.tar.gz (.zip for Windows) with the binary,
# README, CHANGELOG, licences, the CLI manual and output formats, the man
# page and shell completions, plus a `.sha256` file. OUT_DIR defaults to
# `dist`. Extra cargo arguments can be passed in CARGO_ARGS.
set -euo pipefail

if [ $# -lt 1 ]; then
  awk 'NR >= 6 && /^#/ { sub(/^# ?/, ""); print; next } NR >= 6 { exit }' "$0"
  exit 2
fi
target=$1
out=${2:-dist}
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

# shellcheck disable=SC2086
cargo build --profile dist --locked -p sesmesh-cli --target "$target" ${CARGO_ARGS:-}
exe=sesmesh
[[ $target == *windows* ]] && exe=sesmesh.exe
bin="target/$target/dist/$exe"

# Smoke test.
"$bin" --version
"$bin" tests/data/fixtures/1CRN.cif.gz -q draft -o - --quiet > /dev/null

version=$(cargo metadata --no-deps --format-version 1 |
  python3 -c "import json,sys; print(next(p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='sesmesh-cli'))")
name="sesmesh-$version-$target"
mkdir -p "$out"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
dir="$stage/$name"
mkdir -p "$dir/docs" "$dir/completions"
cp "$bin" "$dir/"
cp README.md CHANGELOG.md LICENSE-MIT LICENSE-APACHE "$dir/"
cp docs/cli.md docs/04-output-formats.md "$dir/docs/"
"$bin" man > "$dir/sesmesh.1"
"$bin" completions bash > "$dir/completions/sesmesh.bash"
"$bin" completions zsh > "$dir/completions/_sesmesh"
"$bin" completions fish > "$dir/completions/sesmesh.fish"
"$bin" completions powershell > "$dir/completions/_sesmesh.ps1"

out=$(cd "$out" && pwd)
if [[ $target == *windows* ]]; then
  file="$name.zip"
  (cd "$stage" && zip -qr "$out/$file" "$name")
else
  file="$name.tar.gz"
  if tar --version | grep -q bsdtar; then
    # macOS: no extended attributes or AppleDouble files in the archive.
    COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -C "$stage" -czf "$out/$file" "$name"
  else
    tar -C "$stage" -czf "$out/$file" "$name"
  fi
fi
cd "$out"
if command -v sha256sum > /dev/null; then
  sha256sum "$file" > "$file.sha256"
else
  shasum -a 256 "$file" > "$file.sha256"
fi
cat "$file.sha256"
