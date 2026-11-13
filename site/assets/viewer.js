// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

// sesmesh site: WebGL viewer for the compact "SESM" display meshes written by
// tools/site/ply2web.py (see the format there). three.js comes from the
// import map of the page.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export const MODELS = {
  "1crn": {
    url: "models/1crn.sesm.gz",
    title: "1CRN crambin",
    atoms: "327 atoms",
    detail: "SES · 0.25 Å · element colours · 111 076 → 33 322 triangles (decimated)",
    area: 2310.1,
    volume: 4658.6,
  },
  "4hhb": {
    url: "models/4hhb.sesm.gz",
    title: "4HHB haemoglobin",
    atoms: "4 558 atoms",
    detail: "SES · 0.4 Å · chain colours · 498 732 → 90 000 triangles (decimated)",
    area: 26251.5,
    volume: 70359.0,
  },
  "1aon": {
    url: "models/1aon.sesm.gz",
    title: "1AON GroEL–GroES",
    atoms: "58 870 atoms",
    detail: "SES · 0.7 Å · chain colours · 2 197 024 → 220 000 triangles (decimated)",
    area: 341532.6,
    volume: 881182.6,
  },
};

const fmt = (x) => Math.round(x).toLocaleString("en-US").replace(/,/g, " ");

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  let buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    if (!("DecompressionStream" in window)) {
      throw new Error("this browser cannot decompress the mesh (DecompressionStream)");
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    buf = await new Response(stream).arrayBuffer();
  }
  return buf;
}

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

/** Parses a SESM buffer into a centred BufferGeometry. */
export function parseSesm(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "SESM" || dv.getUint32(4, true) !== 1) throw new Error("not a SESM v1 mesh");
  const nv = dv.getUint32(8, true);
  const nt = dv.getUint32(12, true);
  const lo = [0, 1, 2].map((i) => dv.getFloat32(16 + 4 * i, true));
  const hi = [0, 1, 2].map((i) => dv.getFloat32(28 + 4 * i, true));
  let off = 40;
  const index = new Uint32Array(buf, off, 3 * nt);
  off += 12 * nt;
  const q = new Uint16Array(buf, off, 3 * nv);
  off += 6 * nv;
  const nrm = new Int8Array(buf, off, 3 * nv);
  off += 3 * nv;
  const rgb = new Uint8Array(buf, off, 3 * nv);

  const centre = lo.map((l, i) => 0.5 * (l + hi[i]));
  const scale = lo.map((l, i) => (hi[i] - l) / 65535);
  const pos = new Float32Array(3 * nv);
  const col = new Float32Array(3 * nv);
  for (let i = 0; i < 3 * nv; i++) {
    const a = i % 3;
    pos[i] = lo[a] + q[i] * scale[a] - centre[a];
    col[i] = toLinear(rgb[i] / 255);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(nrm), 3, true));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
  g.computeBoundingSphere();
  return { geometry: g, vertices: nv, triangles: nt };
}

/**
 * Mounts a viewer. `el` holds `.viewer-stage`, optional `.viewer-msg`,
 * buttons `[data-model]`, `[data-toggle="spin|wire"]`, `[data-reset]` and a
 * `.readout`. Options: `preserveDrawingBuffer` (snapshots), `spin`.
 */
export function mountViewer(el, options = {}) {
  const stage = el.querySelector(".viewer-stage");
  const msg = el.querySelector(".viewer-msg");
  const readout = el.querySelector(".readout");
  const caption = el.querySelector("[data-caption]");
  const say = (text) => {
    if (msg) {
      msg.textContent = text;
      msg.hidden = !text;
    }
  };

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: !!options.preserveDrawingBuffer,
    });
  } catch (e) {
    say("WebGL is not available in this browser — see the gallery for pictures.");
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 5000);
  scene.add(camera);
  // Headlight rig: lights move with the camera, so every view is lit.
  camera.add(new THREE.HemisphereLight(0xffffff, 0x2a2a26, 0.95));
  const key = new THREE.DirectionalLight(0xffffff, 2.6);
  key.position.set(-0.6, 0.9, 1.0);
  camera.add(key);
  const rim = new THREE.DirectionalLight(0xffffff, 0.9);
  rim.position.set(0.8, -0.4, -1.0);
  camera.add(rim);

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.6,
    metalness: 0.0,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const inkColor = () =>
    new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#0d0d0c");
  const wireMaterial = new THREE.MeshBasicMaterial({
    color: inkColor(),
    wireframe: true,
    transparent: true,
    opacity: 0.28,
  });

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.rotateSpeed = 0.9;
  controls.zoomSpeed = 0.9;
  controls.autoRotateSpeed = 1.1;
  // The wheel scrolls the page until the visitor presses on the viewer.
  controls.enableZoom = false;
  renderer.domElement.addEventListener("pointerdown", () => (controls.enableZoom = true));
  renderer.domElement.addEventListener("pointerleave", () => (controls.enableZoom = false));

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let spin = options.spin ?? !reduced;
  let wire = !!options.wire;
  let mesh = null;
  let wireMesh = null;
  let radius = 1;
  let visible = true;
  let current = null;

  const render = () => renderer.render(scene, camera);

  const resize = () => {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  };

  const loop = () => {
    controls.update();
    render();
  };

  const setLoop = () => {
    controls.autoRotate = spin;
    renderer.setAnimationLoop(spin && visible ? loop : null);
    if (!spin) render();
  };

  /** Places the camera; azimuth/elevation in degrees. */
  const frame = (azimuth = 35, elevation = 18, zoom = 1) => {
    const fov = (camera.fov * Math.PI) / 180;
    const fit = Math.min(fov, 2 * Math.atan(Math.tan(fov / 2) * camera.aspect));
    const dist = (radius / Math.sin(fit / 2)) * 1.04 / zoom;
    const az = (azimuth * Math.PI) / 180;
    const el2 = (elevation * Math.PI) / 180;
    camera.position.set(
      dist * Math.cos(el2) * Math.sin(az),
      dist * Math.sin(el2),
      dist * Math.cos(el2) * Math.cos(az)
    );
    camera.near = Math.max(dist / 100, 0.05);
    camera.far = dist * 10;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.update();
    render();
  };

  const setWire = (on) => {
    wire = on;
    if (wireMesh) wireMesh.visible = wire;
    for (const b of el.querySelectorAll('[data-toggle="wire"]')) b.setAttribute("aria-pressed", String(wire));
    render();
  };

  const setSpin = (on) => {
    spin = on;
    for (const b of el.querySelectorAll('[data-toggle="spin"]')) b.setAttribute("aria-pressed", String(spin));
    setLoop();
  };

  async function load(name, view = {}) {
    const info = MODELS[name] || { url: name, title: name, atoms: "", detail: "" };
    current = name;
    for (const b of el.querySelectorAll("[data-model]")) {
      b.setAttribute("aria-pressed", String(b.dataset.model === name));
    }
    say(`Loading ${info.title} …`);
    try {
      const { geometry, triangles } = parseSesm(await fetchBytes(info.url));
      if (current !== name) return;
      if (mesh) {
        scene.remove(mesh, wireMesh);
        mesh.geometry.dispose();
      }
      mesh = new THREE.Mesh(geometry, material);
      wireMesh = new THREE.Mesh(geometry, wireMaterial);
      wireMesh.visible = wire;
      scene.add(mesh, wireMesh);
      radius = geometry.boundingSphere.radius;
      controls.minDistance = radius * 0.6;
      controls.maxDistance = radius * 12;
      resize();
      frame(view.azimuth, view.elevation, view.zoom);
      say("");
      if (readout) {
        const area = info.area ? ` · A ${fmt(info.area)} Å²` : "";
        const vol = info.volume ? ` · V ${fmt(info.volume)} Å³` : "";
        readout.textContent = `${fmt(triangles)} triangles${area}${vol}`;
      }
      if (caption) caption.textContent = `${info.title}${info.atoms ? " · " + info.atoms : ""}`;
      el.title = info.detail || "";
      setLoop();
    } catch (e) {
      say(`Could not load ${info.title}: ${e.message}`);
    }
  }

  // Wiring.
  for (const b of el.querySelectorAll("[data-model]")) {
    b.addEventListener("click", () => load(b.dataset.model));
  }
  for (const b of el.querySelectorAll('[data-toggle="spin"]')) {
    b.addEventListener("click", () => setSpin(!spin));
  }
  for (const b of el.querySelectorAll('[data-toggle="wire"]')) {
    b.addEventListener("click", () => setWire(!wire));
  }
  for (const b of el.querySelectorAll("[data-reset]")) {
    b.addEventListener("click", () => frame());
  }
  renderer.domElement.addEventListener("dblclick", () => frame());
  controls.addEventListener("start", () => {
    if (spin) setSpin(false);
  });
  controls.addEventListener("change", () => {
    if (!spin) render();
  });
  new ResizeObserver(resize).observe(stage);
  new IntersectionObserver((entries) => {
    visible = entries.some((e) => e.isIntersecting);
    setLoop();
  }).observe(stage);
  document.addEventListener("visibilitychange", () => {
    visible = !document.hidden;
    setLoop();
  });
  document.addEventListener("themechange", () => {
    wireMaterial.color = inkColor();
    render();
  });

  setSpin(spin);
  setWire(wire);
  return { load, frame, render, renderer, setWire, setSpin };
}

// Auto-mount every [data-viewer] element.
for (const el of document.querySelectorAll("[data-viewer]")) {
  const v = mountViewer(el);
  if (v) v.load(el.dataset.viewer || "1crn");
}
