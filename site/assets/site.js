// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// sesmesh site: theme toggle, copy buttons, gallery slots. No dependencies.
(() => {
  const root = document.documentElement;
  const KEY = "sesmesh-theme";

  // Theme: follows the system unless the visitor chose one.
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch (_) {
    /* storage unavailable */
  }

  const isDark = () =>
    root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.querySelector(".theme-toggle");
    const label = () => {
      if (toggle) toggle.textContent = isDark() ? "Light" : "Dark";
    };
    label();
    if (toggle) {
      toggle.addEventListener("click", () => {
        root.dataset.theme = isDark() ? "light" : "dark";
        try {
          localStorage.setItem(KEY, root.dataset.theme);
        } catch (_) {
          /* storage unavailable */
        }
        label();
        document.dispatchEvent(new CustomEvent("themechange"));
      });
    }

    // Copy buttons on code blocks (prompt markers are not copied).
    for (const block of document.querySelectorAll(".code")) {
      const button = block.querySelector(".code-head button");
      const code = block.querySelector("pre code");
      if (!button || !code) continue;
      button.addEventListener("click", async () => {
        const clone = code.cloneNode(true);
        for (const p of clone.querySelectorAll(".p")) p.remove();
        const text = clone.textContent.replace(/\n$/, "");
        try {
          await navigator.clipboard.writeText(text);
          button.textContent = "Copied";
        } catch (_) {
          button.textContent = "Select + copy";
        }
        setTimeout(() => (button.textContent = "Copy"), 1400);
      });
    }

    // Gallery: show the expected file name where a picture is missing.
    for (const frame of document.querySelectorAll(".fig .frame")) {
      const img = frame.querySelector("img");
      if (!img) continue;
      const mark = () => frame.classList.add("missing");
      if (img.complete && img.naturalWidth === 0) mark();
      img.addEventListener("error", mark);
    }
  });
})();
