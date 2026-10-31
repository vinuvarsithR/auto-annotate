// canvas.js - draws one image plus its boxes and handles mouse editing.
// All box coordinates stay in IMAGE pixels ([x, y, w, h], COCO style);
// only drawing and mouse events convert to screen pixels.
"use strict";

const CLASS_COLORS = { person: "#00e5ff", rb: "#ff8c1a", sb: "#ffe600", biax: "#ff3df2" };
const HANDLE = 6;      // handle half-size in screen px
const MIN_SIZE = 4;    // smallest box in image px

function classColor(name) {
  if (CLASS_COLORS[name]) return CLASS_COLORS[name];
  let h = 0;
  for (const ch of String(name)) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${h}, 85%, 60%)`;
}

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

    new ResizeObserver(() => this.render()).observe(canvas.parentElement);
    canvas.addEventListener("mousedown", (e) => this.onDown(e));
    window.addEventListener("mousemove", (e) => this.onMove(e));
    window.addEventListener("mouseup", (e) => this.onUp(e));
  }

  // Load an image; on any error draw a grey placeholder of the given size.
  setImage(url, width, height) {
    this.imgW = width || 1280;
    this.imgH = height || 720;
    this.img = null;
    this.drag = null;
    this.render();
    if (!url) return Promise.resolve(false);
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => {
        if (this._pending !== im) return resolve(false);
        this.img = im;
        this.imgW = im.naturalWidth;
        this.imgH = im.naturalHeight;
        this.render();
        resolve(true);
      };
      im.onerror = () => resolve(false);
      this._pending = im;
      im.src = url;
    });
  }

  setAnnotations(anns) { this.anns = anns; this.render(); }
  setSelected(id) { this.selectedId = id; this.render(); }
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
    ctx.clearRect(0, 0, w, h);
    const [ix, iy] = this.toScreen(0, 0);
    const iw = this.imgW * this.scale, ih = this.imgH * this.scale;
    if (this.img) {
      ctx.drawImage(this.img, ix, iy, iw, ih);
    } else {
      ctx.fillStyle = "#6b6b6b";
      ctx.fillRect(ix, iy, iw, ih);
      ctx.fillStyle = "#9a9a9a";
      ctx.font = "16px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`image not available (placeholder ${this.imgW}x${this.imgH})`, ix + iw / 2, iy + ih / 2);
      ctx.textAlign = "left";
    }

    const anns = this.visible();
    const selId = this.selectedId;
    // Selected box is drawn last so it sits on top.
    const ordered = anns.filter((a) => a.id !== selId).concat(anns.filter((a) => a.id === selId));
    for (const a of ordered) {
      const bbox = this.drag && this.drag.ann && this.drag.ann.id === a.id ? this.drag.box : a.bbox;
      this.drawBox(a, bbox, a.id === selId);
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

  drawBox(a, bbox, isSel) {
    const ctx = this.ctx;
    const color = classColor(a.class_name);
    const [x, y] = this.toScreen(bbox[0], bbox[1]);
    const w = bbox[2] * this.scale, h = bbox[3] * this.scale;
    ctx.save();
    if (a.status === "rejected") ctx.globalAlpha = 0.35;
    ctx.setLineDash(a.status === "needs_review" ? [6, 4] : []);
    ctx.lineWidth = isSel ? 3 : 2;
    if (isSel) { // dark outline so the selected box stands out on any background
      ctx.strokeStyle = "rgba(0,0,0,0.8)";
      ctx.lineWidth = 5;
      ctx.strokeRect(x, y, w, h);
      ctx.lineWidth = 3;
    }
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    const label = a.class_name + (a.score != null ? " " + a.score.toFixed(2) : "") +
                  (a.status === "rejected" ? " ✕" : "");
    ctx.font = "12px system-ui, sans-serif";
    const tw = ctx.measureText(label).width + 6;
    const ly = y - 16 < 0 ? y : y - 16;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, 16);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x + 3, ly + 12);

    if (isSel && a.status !== "rejected") {
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "#000";
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
