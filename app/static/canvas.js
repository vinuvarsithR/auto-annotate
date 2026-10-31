// canvas.js - draws one image plus its boxes and handles mouse editing.
// All box coordinates stay in IMAGE pixels ([x, y, w, h], COCO style);
// only drawing and mouse events convert to screen pixels.
"use strict";

const CLASS_COLORS = { person: "#7FC8D6", rb: "#E3C9A0", sb: "#E6D27A", biax: "#C9A2D9" };
// Other class names get a stable pick from this muted palette (same name -> same colour).
const FALLBACK_COLORS = ["#A3C99A", "#D9A38F", "#9FB4E0", "#D9A5C0", "#B9C97F", "#8FD1C0", "#E0B98A", "#B3A8E0"];
const HANDLE = 6;      // handle half-size in screen px
const MIN_SIZE = 4;    // smallest box in image px
const CANVAS_COLORS = { base: "#1A2026", panel: "#222A31", line: "#2E3840", text: "#DCE3E8", muted: "#8996A0",
            accept: "143,184,168" };
const LABEL_FONT = '500 12px "IBM Plex Sans", system-ui, sans-serif';

function classColor(name) {
  if (CLASS_COLORS[name]) return CLASS_COLORS[name];
  let h = 0;
  for (const ch of String(name)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return FALLBACK_COLORS[h % FALLBACK_COLORS.length];
}

// ---- motion helpers: every effect is started by a user action, lasts < 250ms, and is
// skipped entirely when the OS asks for reduced motion.
const reducedMotion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const lerp = (a, b, t) => a + (b - a) * t;
const fxNow = () => performance.now();

class BoxCanvas {
  constructor(canvas, callbacks) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.cb = callbacks; // onSelect(ann|null), onBoxChanged(ann, bbox), onBoxDrawn(bbox)
    this.img = null;          // HTMLImageElement or null (placeholder)
    this.imgW = 1280;
    this.imgH = 720;
    this.anns = [];
    this.selectedId = null;
    this.drawMode = false;
    this.drawColor = "#fff";
    this.showRejected = true;
    this.drag = null;
    this.scale = 1; this.ox = 0; this.oy = 0;
    this.fx = [];             // running effects: {kind, start, dur, delay?, ...}
    this.prevImg = null;      // what is faded out during a crossfade: {img} or {w, h} (placeholder)
    this.shown = false;       // has anything been drawn yet (no crossfade from an empty canvas)
    this.raf = 0;

    new ResizeObserver(() => this.render()).observe(canvas.parentElement);
    if (document.fonts) document.fonts.ready.then(() => this.render());
    canvas.addEventListener("mousedown", (e) => this.onDown(e));
    window.addEventListener("mousemove", (e) => this.onMove(e));
    window.addEventListener("mouseup", (e) => this.onUp(e));
  }

  // Load an image; on any error draw a grey placeholder of the given size.
  // The old image stays on screen until the new one has loaded (or failed), then the two
  // crossfade over 200ms, so there is no grey flash between frames.
  setImage(url, width, height) {
    this.drag = null;
    const show = (im) => {
      const old = this.shown ? (this.img ? { img: this.img } : { w: this.imgW, h: this.imgH }) : null;
      this.fx = this.fx.filter((f) => f.kind !== "fade-img");
      this.img = im;
      this.imgW = im ? im.naturalWidth : (width || 1280);
      this.imgH = im ? im.naturalHeight : (height || 720);
      this.prevImg = old;
      if (old) this.startFx({ kind: "fade-img", dur: 200 }); else this.render();
    };
    this._pending = null;
    if (!url) { show(null); return Promise.resolve(false); }
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => { if (this._pending !== im) return resolve(false); show(im); resolve(true); };
      im.onerror = () => { if (this._pending !== im) return resolve(false); show(null); resolve(false); };
      this._pending = im;
      im.src = url;
    });
  }

  setAnnotations(anns) { this.anns = anns; this.render(); }
  // glide: the selection frame slides from the old box to the new one.
  setSelected(id, glide = false) {
    const from = this.selected();
    this.selectedId = id;
    const to = this.selected();
    this.fx = this.fx.filter((f) => f.kind !== "glide");
    if (glide && from && to && from.id !== to.id) {
      this.startFx({ kind: "glide", id: to.id, from: from.bbox.slice(), dur: 180, delay: 50 });
    } else this.render();
  }

  // Accept: the box fills sage, then settles back to its outline.
  flashAccept(id) { this.startFx({ kind: "accept", id, dur: 240 }); }

  // Reject: the box (as it looked before) fades to the rejected look, or out if rejected are hidden.
  fadeReject(ann) { this.startFx({ kind: "reject", id: ann.id, ann, dur: 200 }); }

  startFx(f) {
    if (reducedMotion.matches) { this.prevImg = null; this.render(); return; }
    f.start = fxNow();
    this.fx.push(f);
    if (!this.raf) this.raf = requestAnimationFrame(() => this.tick());
  }

  tick() {
    this.raf = 0;
    this.render();
    if (this.fx.length) this.raf = requestAnimationFrame(() => this.tick());
  }

  // Eased 0..1 progress of an effect, or null while its delay is still running.
  fxT(f, t) {
    const p = (t - f.start - (f.delay || 0)) / f.dur;
    return p < 0 ? null : easeOut(Math.min(1, p));
  }
  setDrawMode(on, color) {
    this.drawMode = on;
    if (color) this.drawColor = color;
    this.cv.style.cursor = on ? "crosshair" : "default";
    this.render();
  }

  // ------------------------------------------------------------ geometry
  layout() {
    const box = this.cv.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(box.width)), h = Math.max(1, Math.floor(box.height));
    if (this.cv.width !== w * dpr || this.cv.height !== h * dpr) {
      this.cv.width = w * dpr; this.cv.height = h * dpr;
      this.cv.style.width = w + "px"; this.cv.style.height = h + "px";
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.scale = Math.min(w / this.imgW, h / this.imgH);
    this.ox = (w - this.imgW * this.scale) / 2;
    this.oy = (h - this.imgH * this.scale) / 2;
    return { w, h };
  }
  toScreen(x, y) { return [this.ox + x * this.scale, this.oy + y * this.scale]; }
  toImage(e) {
    const r = this.cv.getBoundingClientRect();
    return [(e.clientX - r.left - this.ox) / this.scale, (e.clientY - r.top - this.oy) / this.scale];
  }
  clampBox([x, y, w, h]) {
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    x = Math.max(0, Math.min(x, this.imgW)); y = Math.max(0, Math.min(y, this.imgH));
    w = Math.min(w, this.imgW - x); h = Math.min(h, this.imgH - y);
    return [x, y, w, h].map((v) => Math.round(v * 10) / 10);
  }
  visible() { return this.anns.filter((a) => this.showRejected || a.status !== "rejected"); }
  selected() { return this.anns.find((a) => a.id === this.selectedId) || null; }

  // Which handle of box b (image coords) is under point p? Returns e.g. "nw", "e", or null.
  handleAt(b, px, py) {
    const [x, y, w, h] = b;
    const t = HANDLE / this.scale + 2 / this.scale;
    const xs = { w: x, c: x + w / 2, e: x + w }, ys = { n: y, c: y + h / 2, s: y + h };
    const small = this.isSmall(w, h);
    for (const [hy, vy] of Object.entries(ys)) {
      for (const [hx, vx] of Object.entries(xs)) {
        if (hx === "c" && hy === "c") continue;
        if (small && (hx === "c" || hy === "c")) continue;   // corners only on small boxes
        if (Math.abs(px - vx) <= t && Math.abs(py - vy) <= t)
          return (hy === "c" ? "" : hy) + (hx === "c" ? "" : hx);
      }
    }
    return null;
  }
  // A box is "small" when its shorter side is under 4 handles wide on screen.
  isSmall(w, h) { return Math.min(w, h) * this.scale < 4 * HANDLE; }
  inside(b, px, py) { return px >= b[0] && px <= b[0] + b[2] && py >= b[1] && py <= b[1] + b[3]; }

  // ------------------------------------------------------------ mouse
  onDown(e) {
    if (e.button !== 0) return;
    const [px, py] = this.toImage(e);
    if (this.drawMode) {
      this.drag = { kind: "draw", sx: px, sy: py, box: [px, py, 0, 0] };
      return;
    }
    const sel = this.selected();
    // Rejected boxes can be selected (to accept them again) but not moved or resized.
    if (sel && sel.status !== "rejected") {
      const h = this.handleAt(sel.bbox, px, py);
      if (h) { this.startEdit(sel, "resize", h, px, py); return; }
      if (this.inside(sel.bbox, px, py)) { this.startEdit(sel, "move", null, px, py); return; }
    }
    // Smallest box under the cursor wins, so small boxes inside big ones stay clickable.
    const hits = this.visible().filter((a) => this.inside(a.bbox, px, py))
                               .sort((a, b) => a.bbox[2] * a.bbox[3] - b.bbox[2] * b.bbox[3]);
    const hit = hits[0] || null;
    this.selectedId = hit ? hit.id : null;
    this.cb.onSelect(hit);
    if (hit && hit.status !== "rejected") this.startEdit(hit, "move", null, px, py);
    this.render();
  }

  startEdit(ann, kind, handle, px, py) {
    this.drag = { kind, handle, ann, sx: px, sy: py, orig: ann.bbox.slice(), box: ann.bbox.slice(), moved: false };
  }

  onMove(e) {
    if (!this.drag) { this.hoverCursor(e); return; }
    const [px, py] = this.toImage(e);
    const d = this.drag;
    const dx = px - d.sx, dy = py - d.sy;
    if (d.kind === "draw") {
      d.box = this.clampBox([d.sx, d.sy, dx, dy]);
    } else if (d.kind === "move") {
      let [x, y, w, h] = d.orig;
      x = Math.max(0, Math.min(x + dx, this.imgW - w));
      y = Math.max(0, Math.min(y + dy, this.imgH - h));
      d.box = this.clampBox([x, y, w, h]);
      if (Math.abs(dx) * this.scale > 2 || Math.abs(dy) * this.scale > 2) d.moved = true;
    } else {
      let [x0, y0, w, h] = d.orig;
      let x1 = x0 + w, y1 = y0 + h;
      if (d.handle.includes("w")) x0 += dx;
      if (d.handle.includes("e")) x1 += dx;
      if (d.handle.includes("n")) y0 += dy;
      if (d.handle.includes("s")) y1 += dy;
      d.box = this.clampBox([x0, y0, x1 - x0, y1 - y0]);
      d.moved = true;
    }
    this.render();
  }

  onUp() {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    if (d.kind === "draw") {
      if (d.box[2] >= MIN_SIZE && d.box[3] >= MIN_SIZE) this.cb.onBoxDrawn(d.box);
      this.render();
      return;
    }
    const changed = d.moved && d.box.some((v, i) => Math.abs(v - d.orig[i]) > 0.05);
    if (changed && d.box[2] >= MIN_SIZE && d.box[3] >= MIN_SIZE) this.cb.onBoxChanged(d.ann, d.box);
    this.render();
  }

  hoverCursor(e) {
    if (this.drawMode) return;
    const sel = this.selected();
    let cur = "default";
    if (sel) {
      const [px, py] = this.toImage(e);
      const h = this.handleAt(sel.bbox, px, py);
      if (h) cur = { n: "ns", s: "ns", e: "ew", w: "ew", nw: "nwse", se: "nwse", ne: "nesw", sw: "nesw" }[h] + "-resize";
      else if (this.inside(sel.bbox, px, py)) cur = "move";
    }
    this.cv.style.cursor = cur;
  }

  // ------------------------------------------------------------ drawing
  render() {
    const { w, h } = this.layout();
    const ctx = this.ctx;
    const t = fxNow();
    // Effects still running this frame; finished ones are dropped after this frame.
    const fx = this.fx;
    this.fx = fx.filter((f) => t - f.start - (f.delay || 0) < f.dur);
    const fxOf = (kind, id) => fx.find((f) => f.kind === kind && (id === undefined || f.id === id));

    ctx.clearRect(0, 0, w, h);
    const fade = fxOf("fade-img");
    const ft = fade ? (this.fxT(fade, t) || 0) : 1;
    if (fade && this.prevImg) {
      const p = this.prevImg;
      ctx.globalAlpha = 1 - ft;
      this.drawFrame(p.img || null, p.img ? p.img.naturalWidth : p.w, p.img ? p.img.naturalHeight : p.h, w, h);
    }
    if (!fade) this.prevImg = null;
    ctx.globalAlpha = ft;
    this.drawFrame(this.img, this.imgW, this.imgH, w, h);
    ctx.globalAlpha = 1;
    this.shown = true;

    const anns = this.visible();
    const selId = this.selectedId;
    const glide = fxOf("glide");
    const gt = glide ? this.fxT(glide, t) : null;
    // Selected box is drawn last so it sits on top.
    const ordered = anns.filter((a) => a.id !== selId).concat(anns.filter((a) => a.id === selId));
    for (const a of ordered) {
      if (fxOf("reject", a.id)) continue; // drawn below from its snapshot
      const bbox = this.drag && this.drag.ann && this.drag.ann.id === a.id ? this.drag.box : a.bbox;
      const gliding = glide && glide.id === a.id && (gt === null || gt < 1);
      this.drawBox(a, bbox, a.id === selId && !gliding, ft, fxOf("accept", a.id), t);
    }
    for (const f of fx) {
      if (f.kind !== "reject") continue;
      const p = this.fxT(f, t) || 0;
      const end = this.showRejected ? 0.35 : 0;
      this.drawBox(f.ann, f.ann.bbox, false, ft * lerp(1, end, p), null, t);
    }
    if (glide) {
      const to = anns.find((a) => a.id === glide.id);
      if (to && (gt === null || gt < 1)) {
        const b = gt === null ? glide.from : glide.from.map((v, i) => lerp(v, to.bbox[i], gt));
        this.drawSelectionFrame(b, classColor(to.class_name));
      }
    }
    if (this.drag && this.drag.kind === "draw") {
      const [x, y] = this.toScreen(this.drag.box[0], this.drag.box[1]);
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = this.drawColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, this.drag.box[2] * this.scale, this.drag.box[3] * this.scale);
      ctx.setLineDash([]);
    }
  }

  // Image (or placeholder) of size iw x ih, fitted and centred in the canvas.
  drawFrame(img, iw, ih, cw, ch) {
    const ctx = this.ctx;
    const s = Math.min(cw / iw, ch / ih);
    const x = (cw - iw * s) / 2, y = (ch - ih * s) / 2;
    if (img) { ctx.drawImage(img, x, y, iw * s, ih * s); return; }
    ctx.fillStyle = CANVAS_COLORS.panel;
    ctx.fillRect(x, y, iw * s, ih * s);
    ctx.fillStyle = CANVAS_COLORS.muted;
    ctx.font = '400 14px "IBM Plex Sans", system-ui, sans-serif';
    ctx.textAlign = "center";
    ctx.fillText(`Image not available (placeholder ${iw}\u00d7${ih})`, x + iw * s / 2, y + ih * s / 2);
    ctx.textAlign = "left";
  }

  // The moving frame shown while the selection glides between boxes.
  drawSelectionFrame(b, color) {
    const ctx = this.ctx;
    const [x, y] = this.toScreen(b[0], b[1]);
    const w = b[2] * this.scale, h = b[3] * this.scale;
    ctx.save();
    ctx.strokeStyle = "rgba(14,18,22,0.75)";
    ctx.lineWidth = 5;
    ctx.strokeRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  drawBox(a, bbox, isSel, alpha = 1, accept = null, t = 0) {
    const ctx = this.ctx;
    const color = classColor(a.class_name);
    const [x, y] = this.toScreen(bbox[0], bbox[1]);
    const w = bbox[2] * this.scale, h = bbox[3] * this.scale;
    ctx.save();
    ctx.globalAlpha = alpha * (a.status === "rejected" ? 0.35 : 1);
    if (accept) { // sage fill: quick rise, then settles back to the plain outline
      const p = Math.min(1, (t - accept.start) / accept.dur);
      const k = p < 0.25 ? p / 0.25 : 1 - easeOut((p - 0.25) / 0.75);
      ctx.fillStyle = `rgba(${CANVAS_COLORS.accept},${0.42 * Math.max(0, Math.min(1, k))})`;
      ctx.fillRect(x, y, w, h);
    }
    ctx.setLineDash(a.status === "needs_review" ? [6, 4] : []);
    ctx.lineWidth = isSel ? 3 : 2;
    if (isSel) { // dark outline so the selected box stands out on any background
      ctx.strokeStyle = "rgba(14,18,22,0.75)";
      ctx.lineWidth = 5;
      ctx.strokeRect(x, y, w, h);
      ctx.lineWidth = 3;
    }
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    const label = a.class_name + (a.score != null ? " " + a.score.toFixed(2) : "") +
                  (a.status === "rejected" ? " \u2715" : "");
    ctx.font = LABEL_FONT;
    const tw = ctx.measureText(label).width + 10;
    const ly = y - 18 < 0 ? y : y - 18;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, 18);
    ctx.fillStyle = CANVAS_COLORS.base;
    ctx.fillText(label, x + 5, ly + 13);

    if (isSel && a.status !== "rejected") {
      ctx.fillStyle = CANVAS_COLORS.text;
      ctx.strokeStyle = CANVAS_COLORS.base;
      ctx.lineWidth = 1;
      const corners = [[0, 0], [1, 0], [0, 1], [1, 1]];
      const mids = [[0.5, 0], [0, 0.5], [1, 0.5], [0.5, 1]];
      // w, h are already screen px here; same rule as isSmall(): corners only on small boxes.
      const spots = Math.min(w, h) < 4 * HANDLE ? corners : corners.concat(mids);
      for (const [hx, hy] of spots) {
        const cx = x + w * hx, cy = y + h * hy;
        ctx.fillRect(cx - HANDLE / 2 - 1, cy - HANDLE / 2 - 1, HANDLE + 2, HANDLE + 2);
        ctx.strokeRect(cx - HANDLE / 2 - 1, cy - HANDLE / 2 - 1, HANDLE + 2, HANDLE + 2);
      }
    }
    ctx.restore();
  }
}

window.BoxCanvas = BoxCanvas;
window.classColor = classColor;
