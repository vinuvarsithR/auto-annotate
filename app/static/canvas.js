// canvas.js - draws one image plus its boxes and handles mouse editing, zoom and pan.
// All box coordinates stay in IMAGE pixels ([x, y, w, h], COCO style); polygons are normalized
// 0..1 (contract); only drawing and mouse events convert to screen pixels.
"use strict";

const CLASS_COLORS = { person: "#7FC8D6", rb: "#E3C9A0", sb: "#E6D27A", biax: "#C9A2D9" };
// Other class names get a stable pick from this muted palette (same name -> same colour).
const FALLBACK_COLORS = ["#A3C99A", "#D9A38F", "#9FB4E0", "#D9A5C0", "#B9C97F", "#8FD1C0", "#E0B98A", "#B3A8E0"];
const HANDLE = 6;      // handle half-size in screen px
const MIN_SIZE = 4;    // smallest box in image px
const VERTEX_R = 4;    // half the side of a polygon point's square, in screen px
const EDGE_HIT = 5;    // how close to a polygon edge a click counts, in screen px
const CLICK_SLOP = 4;  // less mouse travel than this (screen px) is a click, not a drag
const ZOOM_MAX = 8;    // 800% of "fit"
const ZOOM_GLIDE_MS = 60; // time constant of the wheel zoom glide
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
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

class BoxCanvas {
  constructor(canvas, callbacks) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    // onSelect(ann|null), onBoxChanged(ann, bbox), onBoxDrawn(bbox), onView(zoom, scale), onRender(),
    // onPolygonStart() -> may start?, onPolygonDrawn(polygon) -> accepted?, onPolygonChanged(ann, polygon, label),
    // onVertexClicked(ann, index)
    this.cb = callbacks;
    this.img = null;          // HTMLImageElement or null (placeholder)
    this.imgW = 1280;
    this.imgH = 720;
    this.anns = [];
    this.selectedId = null;
    this.drawMode = false;    // Box tool (B): every drag draws a box until the tool changes
    this.tempDraw = false;    // draw while Ctrl is held
    this.polyMode = false;    // Polygon tool (G): every click adds a point
    this.polyDraft = null;    // polygon being drawn: {pts: [[x, y], ...] in image px, cursor: [x, y]}
    this.drawColor = "#fff";
    this.showRejected = true;
    this.hideAll = false;     // H held: no boxes at all
    this.hiddenClasses = new Set(); // class ids switched off in the legend
    this.imgFilter = "none";  // display-only brightness / contrast
    this.drag = null;
    // View: zoom 1 = whole image fitted; pan = screen px offset from the centred position.
    this.zoom = 1; this.panX = 0; this.panY = 0;
    this.fit = 1; this.vw = 1; this.vh = 1;
    this.scale = 1; this.ox = 0; this.oy = 0;
    this.zoomGoal = null;     // wheel zoom glide: where it is heading (null = not gliding)
    this.zoomAnchor = null; this.zoomRaf = 0; this.zoomTime = 0;
    this.fx = [];             // running effects: {kind, start, dur, delay?, ...}
    this.prevImg = null;      // what is faded out during a crossfade: {img | w,h, view}
    this.shown = false;       // has anything been drawn yet (no crossfade from an empty canvas)
    this.raf = 0;

    new ResizeObserver(() => { this.render(); this.notifyView(); }).observe(canvas.parentElement);
    if (document.fonts) document.fonts.ready.then(() => this.render());
    canvas.addEventListener("mousedown", (e) => this.onDown(e));
    canvas.addEventListener("dblclick", (e) => { if (!this.drawing() && !e.ctrlKey) this.resetView(); });
    window.addEventListener("mousemove", (e) => this.onMove(e));
    window.addEventListener("mouseup", (e) => this.onUp(e));
  }

  // Load an image; on any error draw a grey placeholder of the given size.
  // The old image stays on screen until the new one has loaded (or failed), then the two
  // crossfade over 200ms, so there is no grey flash between frames. Every new image starts at "fit".
  setImage(url, width, height) {
    this.drag = null;
    this.polyDraft = null;
    this.stopZoomAnim();
    const show = (im) => {
      let old = null;
      if (this.shown) {
        this.layout();
        old = this.img ? { img: this.img } : { w: this.imgW, h: this.imgH };
        old.view = { s: this.scale, x: this.ox, y: this.oy };
      }
      this.fx = this.fx.filter((f) => f.kind !== "fade-img");
      this.img = im;
      this.imgW = im ? im.naturalWidth : (width || 1280);
      this.imgH = im ? im.naturalHeight : (height || 720);
      this.prevImg = old;
      this.zoom = 1; this.panX = 0; this.panY = 0;
      this.notifyView();
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
    if (to) this.ensureVisible(to.bbox);
    if (glide && from && to && from.id !== to.id) {
      this.startFx({ kind: "glide", id: to.id, from: from.bbox.slice(), dur: 180, delay: 50 });
    } else this.render();
  }

  // Accept: the box fills sage, then settles back to its outline.
  flashAccept(id) { this.startFx({ kind: "accept", id, dur: 240 }); }

  // Reject: the box (as it looked before) fades to the rejected look, or out if rejected are hidden.
  // gone = the box was deleted: always fade out completely.
  fadeReject(ann, gone = false) { this.startFx({ kind: "reject", id: ann.id, ann, gone, dur: 200 }); }

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

  // ------------------------------------------------------------ modes
  drawing() { return this.drawMode || this.tempDraw || this.polyMode; }
  setDrawMode(on, color) {
    this.drawMode = on;
    if (color) this.drawColor = color;
    if (!this.drag) this.cv.style.cursor = this.drawing() ? "crosshair" : "default";
    this.render();
  }
  setPolyMode(on) {
    this.polyMode = on;
    if (!on) this.polyDraft = null;
    if (!this.drag) this.cv.style.cursor = this.drawing() ? "crosshair" : "default";
    this.render();
  }
  setDrawColor(color) { this.drawColor = color; }
  // Ctrl held: the cursor becomes a crosshair and a drag draws a box (no button press needed).
  setTempDraw(on) {
    if (this.tempDraw === on) return;
    this.tempDraw = on;
    if (!this.drag) this.cv.style.cursor = this.drawing() ? "crosshair" : "default";
  }
  // Esc while dragging out a new box drops it. Returns true if there was one.
  cancelDraw() {
    if (!this.drag || this.drag.kind !== "draw") return false;
    this.drag = null;
    this.render();
    return true;
  }
  // Esc: drop the polygon being drawn. Returns true if there was one.
  cancelPolygon() {
    if (!this.polyDraft) return false;
    this.polyDraft = null;
    this.render();
    return true;
  }
  removeDraftPoint() {
    this.polyDraft.pts.pop();
    if (!this.polyDraft.pts.length) this.polyDraft = null;
    this.render();
  }
  // Enter or a click on the first point. The drawing stays open if the page refuses the shape.
  closeDraft() {
    const pts = this.polyDraft.pts.map(([x, y]) => [roundNorm(x / this.imgW), roundNorm(y / this.imgH)]);
    if (this.cb.onPolygonDrawn(pts)) this.polyDraft = null;
    this.render();
  }
  setHideAll(on) { if (this.hideAll !== on) { this.hideAll = on; this.render(); } }
  setClassHidden(classId, hidden) {
    if (hidden) this.hiddenClasses.add(classId); else this.hiddenClasses.delete(classId);
    this.render();
  }
  // brightness / contrast as factors (1 = unchanged). Display only; the stored image is not touched.
  setImageFilter(brightness, contrast) {
    this.imgFilter = brightness === 1 && contrast === 1 ? "none" : `brightness(${brightness}) contrast(${contrast})`;
    this.render();
  }

  // ------------------------------------------------------------ geometry + view
  layout() {
    const box = this.cv.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(box.width)), h = Math.max(1, Math.floor(box.height));
    // Rounded: at Chrome zoom 125% / 150% w * dpr can be fractional, and a canvas size never is.
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (this.cv.width !== bw || this.cv.height !== bh) {
      this.cv.width = bw; this.cv.height = bh;
      this.cv.style.width = w + "px"; this.cv.style.height = h + "px";
      // Resizing clears the canvas; outside render() (e.g. notifyView, zoomAt) draw it again.
      if (!this.inRender && !this.raf) this.raf = requestAnimationFrame(() => this.tick());
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.vw = w; this.vh = h;
    this.fit = Math.min(w / this.imgW, h / this.imgH);
    this.scale = this.fit * this.zoom;
    // At "fit": centred. Zoomed: it may shift so the point under the cursor stays put, but an edge
    // comes no further into the view than the empty margin that side has at "fit".
    const iw = this.imgW * this.scale, ih = this.imgH * this.scale;
    const cx = (w - iw) / 2, cy = (h - ih) / 2;
    const place = (c, pan, view, size, fitSize) => {
      if (this.zoom <= 1) return c;
      const margin = Math.max(0, (view - fitSize) / 2);
      return clamp(c + pan, view - size - margin, margin);
    };
    this.ox = place(cx, this.panX, w, iw, this.imgW * this.fit);
    this.oy = place(cy, this.panY, h, ih, this.imgH * this.fit);
    this.panX = this.ox - cx; this.panY = this.oy - cy;
    return { w, h };
  }
  toScreen(x, y) { return [this.ox + x * this.scale, this.oy + y * this.scale]; }
  toImage(e) {
    const r = this.cv.getBoundingClientRect();
    return [(e.clientX - r.left - this.ox) / this.scale, (e.clientY - r.top - this.oy) / this.scale];
  }

  // Zoom by `factor` keeping the image point under (clientX, clientY) still.
  // Without a point, zoom around the centre of the view.
  // Smallest zoom: "fit", or 100% (1 image px = 1 screen px) when the image is smaller than the view.
  minZoom() { return Math.min(1, 1 / this.fit); }
  zoomAt(factor, clientX = null, clientY = null) {
    this.stopZoomAnim();
    this.zoomBy(factor, clientX, clientY);
  }
  // Mouse wheel: glide to the new zoom over a few frames, so a notch is not one jump. More wheel
  // while it glides moves the goal; the anchor follows the cursor.
  wheelZoom(factor, clientX, clientY) {
    if (reducedMotion.matches) { this.zoomAt(factor, clientX, clientY); return; }
    this.layout();
    const hi = Math.max(ZOOM_MAX, 1 / this.fit);
    this.zoomGoal = clamp((this.zoomGoal == null ? this.zoom : this.zoomGoal) * factor, this.minZoom(), hi);
    this.zoomAnchor = [clientX, clientY];
    if (!this.zoomRaf) {
      this.zoomTime = fxNow();
      this.zoomRaf = requestAnimationFrame(() => this.zoomStep());
    }
  }
  zoomStep() {
    this.zoomRaf = 0;
    if (this.zoomGoal == null) return;
    const t = fxNow();
    const k = 1 - Math.exp(-(t - this.zoomTime) / ZOOM_GLIDE_MS);
    this.zoomTime = t;
    let z = Math.exp(lerp(Math.log(this.zoom), Math.log(this.zoomGoal), k));
    const arrived = Math.abs(this.zoomGoal / z - 1) < 0.002;
    if (arrived) z = this.zoomGoal;
    this.zoomBy(z / this.zoom, this.zoomAnchor[0], this.zoomAnchor[1]);
    if (arrived) this.zoomGoal = null;
    else this.zoomRaf = requestAnimationFrame(() => this.zoomStep());
  }
  stopZoomAnim() {
    this.zoomGoal = null;
    if (this.zoomRaf) cancelAnimationFrame(this.zoomRaf);
    this.zoomRaf = 0;
  }
  zoomBy(factor, clientX, clientY) {
    this.layout();
    const z = clamp(this.zoom * factor, this.minZoom(), Math.max(ZOOM_MAX, 1 / this.fit));
    if (Math.abs(z - this.zoom) < 1e-6) return;
    const r = this.cv.getBoundingClientRect();
    const sx = clientX == null ? this.vw / 2 : clientX - r.left;
    const sy = clientY == null ? this.vh / 2 : clientY - r.top;
    const ix = (sx - this.ox) / this.scale, iy = (sy - this.oy) / this.scale;
    this.zoom = z;
    const s2 = this.fit * z;
    this.panX = sx - ix * s2 - (this.vw - this.imgW * s2) / 2;
    this.panY = sy - iy * s2 - (this.vh - this.imgH * s2) / 2;
    this.render();
    this.notifyView();
  }
  // Zoom so that 1 image pixel = 1 screen pixel ("100%"), around the centre of the view.
  zoomToActual() { this.layout(); this.zoomAt(1 / this.fit / this.zoom); }
  panBy(dx, dy) { if (this.zoom > 1) { this.panX += dx; this.panY += dy; this.render(); } }
  resetView() {
    this.stopZoomAnim();
    if (this.zoom === 1 && !this.panX && !this.panY) return;
    this.zoom = 1; this.panX = 0; this.panY = 0;
    this.render();
    this.notifyView();
  }
  notifyView() { if (this.cb.onView) { this.layout(); this.cb.onView(this.zoom, this.scale); } }
  // When zoomed in, bring a box that is off screen into the middle of the view.
  ensureVisible(b) {
    if (this.zoom === 1) return;
    this.layout();
    const [x, y] = this.toScreen(b[0], b[1]);
    const w = b[2] * this.scale, h = b[3] * this.scale;
    if (x >= 0 && y >= 0 && x + w <= this.vw && y + h <= this.vh) return;
    this.panX += this.vw / 2 - (x + w / 2);
    this.panY += this.vh / 2 - (y + h / 2);
  }

  clampBox([x, y, w, h]) {
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    x = Math.max(0, Math.min(x, this.imgW)); y = Math.max(0, Math.min(y, this.imgH));
    w = Math.min(w, this.imgW - x); h = Math.min(h, this.imgH - y);
    return [x, y, w, h].map((v) => Math.round(v * 10) / 10);
  }
  visible() {
    if (this.hideAll) return [];
    return this.anns.filter((a) => (this.showRejected || a.status !== "rejected") && !this.hiddenClasses.has(a.class_id));
  }
  selected() { return this.anns.find((a) => a.id === this.selectedId) || null; }
  // Screen rectangle (canvas px) of the selected box as drawn now (follows a drag), or null when hidden.
  selectedRect() {
    const a = this.selected();
    if (!a || !this.visible().includes(a)) return null;
    const b = this.drag && this.drag.ann && this.drag.ann.id === a.id ? this.drag.box : a.bbox;
    const [x, y] = this.toScreen(b[0], b[1]);
    return { x, y, w: b[2] * this.scale, h: b[3] * this.scale };
  }

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

  polygonPx(poly) { return poly.map(([x, y]) => [x * this.imgW, y * this.imgH]); }
  // Index of the point of a's polygon under (px, py), or null.
  vertexAt(a, px, py) {
    let best = null, bestD = (VERTEX_R + 3) / this.scale;
    this.polygonPx(a.polygon).forEach(([x, y], i) => {
      const d = Math.hypot(px - x, py - y);
      if (d <= bestD) { best = i; bestD = d; }
    });
    return best;
  }
  // The edge of a's polygon under (px, py): {index where a new point goes, point (normalized)}, or null.
  edgeAt(a, px, py) {
    const pts = this.polygonPx(a.polygon);
    let best = null, bestD = EDGE_HIT / this.scale;
    pts.forEach(([x0, y0], i) => {
      const [x1, y1] = pts[(i + 1) % pts.length];
      const dx = x1 - x0, dy = y1 - y0, len2 = dx * dx + dy * dy;
      const t = len2 ? clamp(((px - x0) * dx + (py - y0) * dy) / len2, 0, 1) : 0;
      const qx = x0 + t * dx, qy = y0 + t * dy, d = Math.hypot(px - qx, py - qy);
      if (d <= bestD) {
        bestD = d;
        best = { index: i + 1, point: [roundNorm(qx / this.imgW), roundNorm(qy / this.imgH)] };
      }
    });
    return best;
  }
  insidePolygon(a, px, py) {
    const pts = this.polygonPx(a.polygon);
    let hit = false;
    pts.forEach(([x0, y0], i) => {
      const [x1, y1] = pts[(i + 1) % pts.length];
      if ((y0 > py) !== (y1 > py) && px < x0 + ((py - y0) / (y1 - y0)) * (x1 - x0)) hit = !hit;
    });
    return hit;
  }
  // Clicks select a polygon on its shape, not on the empty corners of its bbox.
  hits(a, px, py) {
    return a.polygon ? this.insidePolygon(a, px, py) || !!this.edgeAt(a, px, py) : this.inside(a.bbox, px, py);
  }

  // ------------------------------------------------------------ mouse
  onDown(e) {
    if (e.button === 1) { e.preventDefault(); this.startPan(e); return; } // middle button pans
    if (e.button !== 0) return;
    const [px, py] = this.toImage(e);
    if (this.polyMode) { // no default: a close opens the popup, and the click must not take its focus
      e.preventDefault();
      this.draftClick(px, py);
      return;
    }
    if (this.drawing() || e.ctrlKey) {
      this.drag = { kind: "draw", sx: px, sy: py, box: [px, py, 0, 0] };
      return;
    }
    const sel = this.selected();
    const visibleSel = sel && this.visible().includes(sel);
    // Rejected boxes can be selected (to accept them again) but not moved or resized.
    if (visibleSel && sel.status !== "rejected") {
      const v = sel.polygon ? this.vertexAt(sel, px, py) : null;
      if (v != null) { this.startVertex(sel, sel.polygon.map((p) => p.slice()), v, false, px, py); return; }
      const h = this.handleAt(sel.bbox, px, py);
      if (h) { this.startEdit(sel, "resize", h, px, py); return; }
      const edge = sel.polygon ? this.edgeAt(sel, px, py) : null;
      if (edge) {
        const poly = sel.polygon.map((p) => p.slice());
        poly.splice(edge.index, 0, edge.point);
        this.startVertex(sel, poly, edge.index, true, px, py);
        return;
      }
      if (this.inside(sel.bbox, px, py)) { this.startEdit(sel, "move", null, px, py); return; }
    }
    // Smallest box under the cursor wins, so small boxes inside big ones stay clickable.
    const hits = this.visible().filter((a) => this.hits(a, px, py))
                               .sort((a, b) => a.bbox[2] * a.bbox[3] - b.bbox[2] * b.bbox[3]);
    const hit = hits[0] || null;
    this.selectedId = hit ? hit.id : null;
    this.cb.onSelect(hit);
    if (hit && hit.status !== "rejected") this.startEdit(hit, "move", null, px, py);
    else if (!hit && this.zoom > 1) this.startPan(e); // drag on empty image pans when zoomed
    this.render();
  }

  startEdit(ann, kind, handle, px, py) {
    this.drag = { kind, handle, ann, sx: px, sy: py, orig: ann.bbox.slice(), box: ann.bbox.slice(), moved: false };
  }
  // Mouse down on a point: poly is the working copy; added = this click made the point.
  // On release a drag saves it; a click deletes an old point and keeps a new one.
  startVertex(ann, poly, index, added, px, py) {
    this.drag = { kind: "vertex", ann, poly, index, added, moved: false, sx: px, sy: py,
                  box: polygonBbox(poly, this.imgW, this.imgH) };
    this.render();
  }
  // Polygon tool: first click starts, a click on the first point closes, others add a point.
  draftClick(px, py) {
    const x = clamp(px, 0, this.imgW), y = clamp(py, 0, this.imgH);
    const d = this.polyDraft;
    if (!d) {
      if (this.cb.onPolygonStart()) { this.polyDraft = { pts: [[x, y]], cursor: [x, y] }; this.render(); }
      return;
    }
    if (this.nearDraftStart(x, y)) { if (d.pts.length > 1) this.closeDraft(); return; }
    const [lx, ly] = d.pts[d.pts.length - 1];
    if (Math.hypot(x - lx, y - ly) * this.scale >= 2) d.pts.push([x, y]); // not a double click
    this.render();
  }
  nearDraftStart(x, y) {
    const [fx, fy] = this.polyDraft.pts[0];
    return Math.hypot(x - fx, y - fy) * this.scale <= VERTEX_R + 3;
  }
  startPan(e) {
    this.drag = { kind: "pan", sx: e.clientX, sy: e.clientY, px: this.panX, py: this.panY };
    this.cv.style.cursor = "grabbing";
  }

  onMove(e) {
    if (this.polyDraft && !this.drag) {
      const [px, py] = this.toImage(e);
      this.polyDraft.cursor = [clamp(px, 0, this.imgW), clamp(py, 0, this.imgH)];
      this.render();
    }
    if (!this.drag) { this.hoverCursor(e); return; }
    const d = this.drag;
    if (d.kind === "pan") {
      this.panX = d.px + (e.clientX - d.sx);
      this.panY = d.py + (e.clientY - d.sy);
      this.render();
      return;
    }
    const [px, py] = this.toImage(e);
    const dx = px - d.sx, dy = py - d.sy;
    // A shaky click on a box, handle or point must not count as an edit (it would mark the box edited).
    if (d.kind !== "draw" && !d.moved) {
      if (Math.hypot(dx, dy) * this.scale < CLICK_SLOP) return;
      d.moved = true;
    }
    if (d.kind === "draw") {
      d.box = this.clampBox([d.sx, d.sy, dx, dy]);
    } else if (d.kind === "vertex") {
      d.poly[d.index] = [roundNorm(px / this.imgW), roundNorm(py / this.imgH)];
      d.box = polygonBbox(d.poly, this.imgW, this.imgH);
    } else if (d.kind === "move") {
      let [x, y, w, h] = d.orig;
      x = Math.max(0, Math.min(x + dx, this.imgW - w));
      y = Math.max(0, Math.min(y + dy, this.imgH - h));
      d.box = this.clampBox([x, y, w, h]);
    } else {
      let [x0, y0, w, h] = d.orig;
      let x1 = x0 + w, y1 = y0 + h;
      if (d.handle.includes("w")) x0 += dx;
      if (d.handle.includes("e")) x1 += dx;
      if (d.handle.includes("n")) y0 += dy;
      if (d.handle.includes("s")) y1 += dy;
      d.box = this.clampBox([x0, y0, x1 - x0, y1 - y0]);
    }
    this.render();
  }

  onUp(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    if (d.kind === "pan") { this.hoverCursor(e); return; }
    if (d.kind === "draw") {
      if (d.box[2] >= MIN_SIZE && d.box[3] >= MIN_SIZE) this.cb.onBoxDrawn(d.box);
      this.render();
      return;
    }
    if (d.kind === "vertex") {
      if (d.moved || d.added) this.cb.onPolygonChanged(d.ann, d.poly, d.added ? "add point to" : "move point of");
      else this.cb.onVertexClicked(d.ann, d.index);
      this.render();
      return;
    }
    const changed = d.moved && d.box.some((v, i) => Math.abs(v - d.orig[i]) > 0.05);
    if (changed && d.box[2] >= MIN_SIZE && d.box[3] >= MIN_SIZE) this.cb.onBoxChanged(d.ann, d.box);
    this.render();
  }

  hoverCursor(e) {
    const [px, py] = this.toImage(e);
    if (this.polyDraft && this.polyDraft.pts.length > 2 && this.nearDraftStart(px, py)) {
      this.cv.style.cursor = "pointer"; // a click closes the shape
      return;
    }
    if (this.drawing()) { this.cv.style.cursor = "crosshair"; return; }
    const sel = this.selected();
    let cur = "default";
    if (sel && sel.status !== "rejected" && this.visible().includes(sel)) {
      const h = this.handleAt(sel.bbox, px, py);
      if (sel.polygon && this.vertexAt(sel, px, py) != null) cur = "grab";
      else if (h) cur = { n: "ns", s: "ns", e: "ew", w: "ew", nw: "nwse", se: "nwse", ne: "nesw", sw: "nesw" }[h] + "-resize";
      else if (sel.polygon && this.edgeAt(sel, px, py)) cur = "copy";
      else if (this.inside(sel.bbox, px, py)) cur = "move";
    }
    if (cur === "default" && this.zoom > 1 && !this.visible().some((a) => this.hits(a, px, py))) cur = "grab";
    this.cv.style.cursor = cur;
  }

  // ------------------------------------------------------------ drawing
  render() {
    this.inRender = true;
    try { this.draw(); } finally { this.inRender = false; }
    if (this.cb.onRender) this.cb.onRender();
  }

  draw() {
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
      const p = this.prevImg, v = p.view;
      ctx.globalAlpha = 1 - ft;
      this.drawFrame(p.img || null, p.img ? p.img.naturalWidth : p.w, p.img ? p.img.naturalHeight : p.h, v.s, v.x, v.y);
    }
    if (!fade) this.prevImg = null;
    ctx.globalAlpha = ft;
    this.drawFrame(this.img, this.imgW, this.imgH, this.scale, this.ox, this.oy);
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
      if (f.kind !== "reject" || this.hideAll) continue;
      const p = this.fxT(f, t) || 0;
      const end = f.gone || !this.showRejected ? 0 : 0.35;
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
    if (this.polyDraft) this.drawDraft();
  }

  // The polygon being drawn: its points, plus a dashed line to the cursor and back to the start.
  drawDraft() {
    const ctx = this.ctx, d = this.polyDraft;
    const pts = d.pts.map(([x, y]) => this.toScreen(x, y));
    const cur = this.toScreen(d.cursor[0], d.cursor[1]);
    ctx.save();
    ctx.strokeStyle = this.drawColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pts[pts.length - 1][0], pts[pts.length - 1][1]);
    ctx.lineTo(cur[0], cur[1]);
    if (pts.length > 1) ctx.lineTo(pts[0][0], pts[0][1]);
    ctx.stroke();
    ctx.setLineDash([]);
    pts.forEach(([x, y], i) => { // the first point, where a click closes, is a little larger
      this.drawVertex(x, y, i ? VERTEX_R : VERTEX_R + 1, this.drawColor);
    });
    ctx.restore();
  }

  // A polygon point: a square in the class colour with a dark edge, so it shows on any image.
  drawVertex(x, y, r, color) {
    const ctx = this.ctx;
    ctx.fillStyle = color;
    ctx.strokeStyle = CANVAS_COLORS.base;
    ctx.lineWidth = 1;
    ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    ctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
  }

  // The polygon of box a as drawn now (follows a drag), or null for a plain box.
  shownPolygon(a) {
    if (!a.polygon) return null;
    const d = this.drag;
    if (!d || !d.ann || d.ann.id !== a.id) return a.polygon;
    return d.kind === "vertex" ? d.poly : scalePolygon(a.polygon, a.bbox, d.box, this.imgW, this.imgH);
  }

  // Image (or placeholder) of size iw x ih, drawn at scale s with its top-left at (x, y).
  drawFrame(img, iw, ih, s, x, y) {
    const ctx = this.ctx;
    if (img) {
      ctx.filter = this.imgFilter;
      ctx.drawImage(img, x, y, iw * s, ih * s);
      ctx.filter = "none";
      return;
    }
    ctx.fillStyle = CANVAS_COLORS.panel;
    ctx.fillRect(x, y, iw * s, ih * s);
    ctx.fillStyle = CANVAS_COLORS.muted;
    ctx.font = '400 14px "IBM Plex Sans", system-ui, sans-serif';
    ctx.textAlign = "center";
    ctx.fillText(`Image not available (placeholder ${iw}×${ih})`, x + iw * s / 2, y + ih * s / 2);
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
    const poly = this.shownPolygon(a);
    const screenPts = poly ? poly.map(([px, py]) => this.toScreen(px * this.imgW, py * this.imgH)) : null;
    // The shape: the polygon's outline when it has one, else the box.
    const outline = () => {
      ctx.beginPath();
      if (screenPts) {
        screenPts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
        ctx.closePath();
      } else ctx.rect(x, y, w, h);
    };
    ctx.save();
    ctx.globalAlpha = alpha * (a.status === "rejected" ? 0.35 : 1);
    if (accept) { // sage fill: quick rise, then settles back to the plain outline
      const p = Math.min(1, (t - accept.start) / accept.dur);
      const k = p < 0.25 ? p / 0.25 : 1 - easeOut((p - 0.25) / 0.75);
      ctx.fillStyle = `rgba(${CANVAS_COLORS.accept},${0.42 * Math.max(0, Math.min(1, k))})`;
      outline();
      ctx.fill();
    }
    ctx.setLineDash(a.status === "needs_review" ? [6, 4] : []);
    ctx.lineWidth = isSel ? 3 : 2;
    if (isSel) { // dark outline so the selected box stands out on any background
      ctx.strokeStyle = "rgba(14,18,22,0.75)";
      ctx.lineWidth = 5;
      outline();
      ctx.stroke();
      ctx.lineWidth = 3;
    }
    ctx.strokeStyle = color;
    outline();
    ctx.stroke();
    ctx.setLineDash([]);

    const label = a.class_name + (a.score != null ? " " + a.score.toFixed(2) : "") +
                  (a.status === "rejected" ? " ✕" : "");
    ctx.font = LABEL_FONT;
    const tw = ctx.measureText(label).width + 10;
    const ly = y - 18 < 0 ? y : y - 18;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, 18);
    ctx.fillStyle = CANVAS_COLORS.base;
    ctx.fillText(label, x + 5, ly + 13);

    if (isSel && a.status !== "rejected") {
      if (screenPts) { // a thin frame, so the resize handles below have something to belong to
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
      }
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
      if (screenPts) {
        const held = this.drag && this.drag.kind === "vertex" ? this.drag.index : null;
        screenPts.forEach(([px, py], i) => this.drawVertex(px, py, i === held ? VERTEX_R + 2 : VERTEX_R, color));
      }
    }
    ctx.restore();
  }
}

window.BoxCanvas = BoxCanvas;
window.classColor = classColor;
