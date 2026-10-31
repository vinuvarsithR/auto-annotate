// app.js - screens + the review page (state, image list, box panel, undo/redo, keyboard, zoom).
"use strict";

const PAGE = 50;
const UNDO_MAX = 50;          // undo steps kept per image
const COPY_IOU = 0.5;         // "repeat previous" skips a box this much like an existing one
// Boxes "repeat previous" copies: the ones a reviewer kept (not rejected, not still unreviewed).
const COPY_STATUSES = new Set(["auto_accepted", "accepted", "edited"]);
const $ = (id) => document.getElementById(id);

const state = {
  projectId: null,
  project: null,
  classes: [],
  images: [], total: 0, offset: 0,
  imageIndex: -1,     // index into state.images
  image: null,
  anns: [],
  selectedId: null,
  drawMode: false,
  loadToken: 0,       // ignore late responses from an older image
  prevImageId: null,  // image open before this one (source for "repeat previous")
  busy: false,        // undo / redo / repeat request in flight
  screen: "projects",
};

// Undo / redo of box changes on the image that is open. Cleared when another image opens.
// Ops: {kind: "patch", aid, before, after, label} | {kind: "create" | "delete", anns: [...], label}
const undoLog = { imageId: null, undo: [], redo: [] };

const canvas = new BoxCanvas($("canvas"), {
  onSelect: (ann) => select(ann ? ann.id : null, false),
  onBoxChanged: (ann, bbox) => editBox(ann, { bbox }, "move box"),
  onBoxDrawn: (bbox) => createBox(bbox),
  onView: (zoom) => renderZoom(zoom),
});

// ------------------------------------------------------------ helpers
function showBanner(msg, kind = "error") {
  const b = $("banner");
  b.textContent = msg;
  b.className = "banner" + (kind === "info" ? " info" : "");
  b.hidden = false;
  clearTimeout(showBanner.t);
  showBanner.t = setTimeout(() => (b.hidden = true), kind === "info" ? 2500 : 8000);
}

async function guard(fn) {
  try { return await fn(); }
  catch (e) {
    console.error(e);
    showBanner(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    return undefined;
  }
}

function reviewer() {
  const v = $("reviewerInput").value.trim();
  return v || null;
}

function classById(id) { return state.classes.find((c) => c.id === id); }

function fillClassSelect(sel) {
  sel.innerHTML = "";
  state.classes.forEach((c, i) => {
    const o = document.createElement("option");
    o.value = c.id;
    o.textContent = `${i + 1}  ${c.name}`;
    sel.appendChild(o);
  });
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

// ------------------------------------------------------------ screens
const SCREENS = { projects: "projects", setup: "setup", review: "layout" };

function showScreen(name) {
  if (!SCREENS[name]) name = "projects";
  state.screen = name;
  for (const [n, id] of Object.entries(SCREENS)) $(id).hidden = n !== name;
  $("tabProjects").classList.toggle("active", name === "projects");
  $("tabSetup").classList.toggle("active", name === "setup");
  $("tabReview").classList.toggle("active", name === "review");
  // The project picker and review progress belong to Setup / Review; Projects lists them all.
  $("projectField").hidden = name === "projects";
  $("progress").style.visibility = name === "review" ? "visible" : "hidden";
  if (name !== "review") { canvas.setTempDraw(false); canvas.setHideAll(false); closeAdjust(); }
  document.dispatchEvent(new CustomEvent("screen-shown", { detail: name }));
}

// ------------------------------------------------------------ projects
function fillProjectPicker(projects) {
  const sel = $("projectSelect");
  sel.innerHTML = "";
  for (const p of projects) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.name;
    sel.appendChild(o);
  }
  if (state.projectId != null) sel.value = state.projectId;
}

// Refill the picker (e.g. after a rename) without reopening the project.
async function refreshProjectPicker() {
  const data = await guard(() => api.listProjects());
  if (data) fillProjectPicker(data.projects);
  return data;
}

// preferId: open this project (used after "Create project").
async function loadProjects(preferId = null) {
  const data = await guard(() => api.listProjects());
  if (!data) return;
  fillProjectPicker(data.projects);
  if (!data.projects.length) {
    clearReview();
    document.dispatchEvent(new CustomEvent("project-opened", { detail: null }));
    return;
  }
  let saved = preferId != null ? String(preferId) : null;
  if (saved == null) {
    try { saved = localStorage.getItem("review.project"); } catch (e) { /* ignore */ }
  }
  const pick = data.projects.find((p) => String(p.id) === saved) || data.projects[0];
  $("projectSelect").value = pick.id;
  await openProject(pick.id);
}

async function openProject(pid) {
  const changed = state.projectId !== Number(pid);
  state.projectId = Number(pid);
  $("projectSelect").value = pid;
  try { localStorage.setItem("review.project", pid); } catch (e) { /* ignore */ }
  const cls = await guard(() => api.listClasses(pid));
  state.classes = cls ? cls.classes : [];
  if (changed) { canvas.hiddenClasses.clear(); state.prevImageId = null; }
  fillClassSelect($("bClass"));
  fillClassSelect($("newClass"));
  syncDrawColor();
  state.offset = 0;
  await refreshProgress();
  await loadImages(0);
  // Lets setup.js follow the project picker and refresh its own panels.
  document.dispatchEvent(new CustomEvent("project-opened", { detail: state.project }));
}

// No project at all (e.g. the last one was deleted): empty review screen.
function clearReview() {
  Object.assign(state, { projectId: null, project: null, classes: [], images: [], total: 0, offset: 0,
                         imageIndex: -1, image: null, anns: [], prevImageId: null });
  fillClassSelect($("bClass"));
  fillClassSelect($("newClass"));
  buildImageList();
  renderLegend();
  $("imageTitle").textContent = "No project.";
  canvas.setAnnotations([]);
  canvas.setImage(null);
  resetHistory(null);
  select(null);
  $("progressFill").style.width = "0%";
}

async function refreshProgress() {
  const pid = state.projectId;
  if (pid == null) return;
  const [p, nr] = await Promise.all([
    guard(() => api.getProject(pid)),
    guard(() => api.listImages(pid, { status: "needs_review", offset: 0, limit: 1 })),
  ]);
  if (!p || pid !== state.projectId) return;
  state.project = p;
  const n = p.counts.images;
  const left = nr ? nr.total : 0;
  const c = p.counts;
  const pct = n ? Math.round(((n - left) / n) * 1000) / 10 : 0;
  const text = `${n - left} / ${n} images reviewed\n` +
    `Boxes: ${c.needs_review} needs review, ${c.auto_accepted} auto accepted, ` +
    `${c.accepted} accepted, ${c.edited} edited, ${c.rejected} rejected`;
  const bar = $("progress");
  $("progressFill").style.width = pct + "%";
  bar.title = text;
  bar.setAttribute("aria-valuenow", String(pct));
  bar.setAttribute("aria-valuetext", text.replace("\n", ". "));
}

const STATUS_LABEL = {
  needs_review: "Needs review", auto_accepted: "Auto accepted", accepted: "Accepted",
  edited: "Edited", rejected: "Rejected",
};

// Legend = the project's classes with this image's box count (rejected not counted).
// Clicking one hides / shows that class on the canvas.
function renderLegend() {
  const el = $("legend");
  el.innerHTML = "";
  for (const c of state.classes) {
    const n = state.anns.filter((a) => a.class_id === c.id && a.status !== "rejected").length;
    const hidden = canvas.hiddenClasses.has(c.id);
    const b = document.createElement("button");
    b.type = "button";
    b.className = "lg" + (hidden ? " off" : "");
    b.setAttribute("aria-pressed", String(!hidden));
    b.innerHTML = `<i></i><span class="nm"></span><span class="ct num-tab"></span>`;
    b.querySelector("i").style.background = classColor(c.name);
    b.querySelector(".nm").textContent = c.name;
    b.querySelector(".ct").textContent = String(n);
    b.title = `${c.description}\n${n} box${n === 1 ? "" : "es"} on this image. Click to ${hidden ? "show" : "hide"}.`;
    b.addEventListener("click", () => toggleClass(c.id));
    el.appendChild(b);
  }
}

function toggleClass(classId) {
  const hide = !canvas.hiddenClasses.has(classId);
  canvas.setClassHidden(classId, hide);
  const a = selectedAnn();
  if (hide && a && a.class_id === classId) select(null);
  renderLegend();
}

// ------------------------------------------------------------ image list
async function loadImages(offset, selectIndex = 0) {
  const status = $("statusFilter").value || null;
  const data = await guard(() => api.listImages(state.projectId, { status, offset, limit: PAGE }));
  if (!data) return;
  state.images = data.images;
  state.total = data.total;
  state.offset = data.offset;
  state.imageIndex = -1;
  buildImageList();
  if (state.images.length) {
    const i = selectIndex < 0 ? state.images.length - 1 : Math.min(selectIndex, state.images.length - 1);
    await openImage(i);
  } else {
    state.imageIndex = -1;
    if (state.image) state.prevImageId = state.image.id;
    state.image = null;
    state.anns = [];
    $("imageTitle").textContent = status ? `No images with ${STATUS_LABEL[status].toLowerCase()} boxes.` : "No images.";
    canvas.setAnnotations([]);
    canvas.setImage(null);
    resetHistory(null);
    renderLegend();
    renderRepeat();
    select(null);
  }
}

// The filmstrip is built once per page (so thumbnails are not re-fetched), then updated in place.
function buildImageList() {
  const ul = $("imageList");
  ul.innerHTML = "";
  state.images.forEach((img, i) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<button type="button" class="thumb"><span class="frame"><img alt="" loading="lazy" decoding="async">` +
      `<span class="ph"></span><span class="dot"></span></span><span class="name"></span></button>`;
    const frame = li.querySelector(".frame");
    const im = li.querySelector("img");
    im.addEventListener("error", () => frame.classList.add("missing"));
    li.querySelector(".ph").textContent = `${img.width}×${img.height}`;
    im.src = api.imageUrl(img);
    const name = li.querySelector(".name");
    name.textContent = img.file_name;
    li.querySelector(".thumb").title = img.file_name;
    li.querySelector(".thumb").addEventListener("click", () => openImage(i));
    ul.appendChild(li);
  });
  renderImageList();
}

function renderImageList() {
  const ul = $("imageList");
  state.images.forEach((img, i) => {
    const li = ul.children[i];
    if (!li) return;
    li.classList.toggle("current", i === state.imageIndex);
    const nr = img.counts ? img.counts.needs_review : 0;
    const dot = li.querySelector(".dot");
    dot.classList.toggle("todo", nr > 0);
    dot.textContent = nr > 0 ? String(nr) : "";
    dot.title = nr > 0 ? `${nr} box(es) need review` : "Nothing left to review";
  });
  const end = state.offset + state.images.length;
  $("pageInfo").textContent = state.total ? `${state.offset + 1}–${end} of ${state.total}` : "0";
  $("pagePrev").disabled = state.offset === 0;
  $("pageNext").disabled = end >= state.total;
}

async function openImage(i) {
  if (i < 0 || i >= state.images.length) return;
  const img = state.images[i];
  if (state.image && state.image.id !== img.id) state.prevImageId = state.image.id;
  state.imageIndex = i;
  const token = ++state.loadToken;
  renderImageList();
  const cur = $("imageList").children[i];
  if (cur) cur.scrollIntoView({ block: "nearest" });
  $("imageTitle").textContent = `${img.file_name} · ${img.width}×${img.height} · id ${img.id}`;
  setDrawMode(false);
  resetHistory(img.id);
  state.anns = [];
  canvas.setAnnotations([]);
  // Keep the box panel where it is while the next image loads, so it does not close and reopen.
  state.loadingImage = true;
  select(null);
  canvas.setImage(api.imageUrl(img), img.width, img.height); // also resets zoom to fit
  const data = await guard(() => api.getAnnotations(img.id));
  if (token !== state.loadToken) return;
  state.loadingImage = false;
  if (!data) { renderPanel(); return; }
  state.image = data.image;
  state.anns = data.annotations;
  canvas.setAnnotations(state.anns);
  renderLegend();
  renderRepeat();
  // Start on the first box that still needs review.
  const first = state.anns.find((a) => a.status === "needs_review" && !canvas.hiddenClasses.has(a.class_id));
  select(first ? first.id : null);
}

async function stepImage(delta) {
  const i = state.imageIndex + delta;
  if (i >= 0 && i < state.images.length) return openImage(i);
  if (delta > 0 && state.offset + state.images.length < state.total) return loadImages(state.offset + PAGE, 0);
  if (delta < 0 && state.offset > 0) return loadImages(Math.max(0, state.offset - PAGE), -1);
  showBanner(delta > 0 ? "Last image." : "First image.", "info");
}

// After any box change: update this image's counts in the list, then the top-bar progress.
function afterChange() {
  const img = state.images[state.imageIndex];
  if (img) img.counts = countStatuses(state.anns);
  renderImageList();
  canvas.setAnnotations(state.anns);
  renderLegend();
  renderPanel();
  refreshProgress();
}

// ------------------------------------------------------------ boxes
// glide: animate the selection from the old box to the new one (used after accept / reject and Tab).
function select(id, redraw = true, glide = false) {
  state.selectedId = id;
  if (redraw) canvas.setSelected(id, glide); else canvas.selectedId = id;
  renderPanel();
}

function selectedAnn() { return state.anns.find((a) => a.id === state.selectedId) || null; }

function renderPanel() {
  const a = selectedAnn();
  const right = $("right");
  $("boxEmpty").hidden = !!a;
  if (!a && state.loadingImage && right.classList.contains("open")) {
    right.classList.add("stale");   // next image is loading; leave the panel open, dimmed
    return;
  }
  right.classList.remove("stale");
  right.classList.toggle("open", !!a);
  if (!a) return;   // the old content stays inside while the panel slides shut
  $("boxPanel").hidden = false;
  $("bId").textContent = `#${a.id}`;
  $("bStatus").className = `status-line st-${a.status}`;
  $("bStatus").textContent = STATUS_LABEL[a.status] || a.status;
  $("bScore").textContent = a.score == null ? "-" : a.score.toFixed(3);
  $("bSource").textContent = a.source;
  $("bBbox").textContent = a.bbox.map((v) => Math.round(v)).join(", ");
  $("bReviewer").textContent = a.reviewer || "-";
  $("bClass").value = a.class_id;
  $("deleteNote").textContent = a.source === "human"
    ? "Delete removes this drawn box."
    : "Delete marks this model box rejected (kept for precision stats).";
}

function cycleBox(delta) {
  const list = canvas.visible();
  if (!list.length) return;
  const i = list.findIndex((a) => a.id === state.selectedId);
  const next = i < 0 ? (delta > 0 ? 0 : list.length - 1) : (i + delta + list.length) % list.length;
  select(list[next].id, true, true);
}

function replaceAnn(updated) {
  const k = state.anns.findIndex((a) => a.id === updated.id);
  if (k >= 0) state.anns[k] = updated;
}

async function patch(ann, body) {
  const updated = await guard(() => api.patchAnnotation(ann.id, { ...body, reviewer: reviewer() }));
  if (!updated) { canvas.render(); return null; }
  replaceAnn(updated);
  afterChange();
  return updated;
}

const snapOf = (a) => ({ status: a.status, bbox: a.bbox.slice(), class_id: a.class_id });
const sameBbox = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 0.05);
const sameSnap = (a, b) => a.status === b.status && a.class_id === b.class_id && sameBbox(a.bbox, b.bbox);

// A PATCH that can be undone.
async function editBox(ann, body, label) {
  const before = snapOf(ann);
  const updated = await patch(ann, body);
  if (!updated) return null;
  const after = snapOf(updated);
  if (!sameSnap(before, after)) record({ kind: "patch", aid: updated.id, before, after, label: `${label} #${updated.id}` });
  return updated;
}

function selectNextToReview() {
  const next = state.anns.find((x) => x.status === "needs_review" && canvas.visible().includes(x));
  if (next) select(next.id, true, true);
  else showBanner("No more boxes to review on this image. Press → for the next one.", "info");
}

async function setStatus(status, label = null) {
  const a = selectedAnn();
  if (!a) { showBanner("Select a box first (click it or press Tab).", "info"); return; }
  const before = { ...a };
  const updated = await editBox(a, { status }, label || (status === "accepted" ? "accept" : "reject"));
  if (!updated) return;
  if (status === "accepted") canvas.flashAccept(updated.id);
  else if (status === "rejected") canvas.fadeReject(before);
  selectNextToReview();
}

async function changeClass(classId) {
  const a = selectedAnn();
  if (!a || a.class_id === classId) return;
  await editBox(a, { class_id: classId }, "change class of");
}

// Delete: a drawn (human) box is really removed; a model box is rejected, so it still counts
// toward precision. A server without DELETE /annotations (contract #19) answers 405: then the
// drawn box is rejected instead.
async function deleteBox() {
  const a = selectedAnn();
  if (!a) { showBanner("Select a box first (click it or press Tab).", "info"); return; }
  if (a.source !== "human") { if (a.status !== "rejected") await setStatus("rejected", "delete"); return; }
  if (state.busy) return;
  const iid = state.image && state.image.id;
  let gone = false;
  try {
    await api.deleteAnnotation(a.id);
    gone = true;
  } catch (e) {
    console.error(e);
    if (e instanceof ApiError && e.httpStatus === 405) {
      showBanner("This server can't delete boxes yet (contract #19), so the box was rejected instead.");
      if (a.status !== "rejected") await setStatus("rejected", "delete");
    } else showBanner(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
  }
  if (!gone || !state.image || state.image.id !== iid) return;
  state.anns = state.anns.filter((x) => x.id !== a.id);
  canvas.fadeReject(a, true);
  record({ kind: "delete", anns: [a], label: `delete box #${a.id}` });
  state.selectedId = null;
  afterChange();
  selectNextToReview();
}

async function createBox(bbox) {
  if (!state.image) return;
  const classId = Number($("newClass").value);
  if (!classId) { showBanner("No classes in this project.", "error"); return; }
  const iid = state.image.id;
  const created = await guard(() => api.createAnnotation(iid,
    { class_id: classId, bbox, mask: null, reviewer: reviewer() }));
  setDrawMode(false);
  if (!created || !state.image || state.image.id !== iid) return;
  state.anns.push(created);
  record({ kind: "create", anns: [created], label: `draw box #${created.id}` });
  select(created.id);
  afterChange();
}

function syncDrawColor() {
  const cls = classById(Number($("newClass").value));
  canvas.setDrawColor(cls ? classColor(cls.name) : "#fff");
}

function setDrawMode(on) {
  state.drawMode = on;
  const cls = classById(Number($("newClass").value));
  canvas.setDrawMode(on, cls ? classColor(cls.name) : "#fff");
  $("drawBtn").classList.toggle("active", on);
  $("drawBtn").innerHTML = on ? "Drawing… <kbd>Esc</kbd>" : "Draw <kbd>N</kbd>";
}

// ------------------------------------------------------------ repeat previous
function repeatSourceId() {
  if (state.prevImageId != null && (!state.image || state.prevImageId !== state.image.id)) return state.prevImageId;
  const before = state.images[state.imageIndex - 1];
  return before ? before.id : null;
}

function renderRepeat() {
  const id = repeatSourceId();
  const src = id == null ? null : state.images.find((i) => i.id === id);
  $("repeatBtn").disabled = id == null || !state.image;
  $("repeatBtn").title = id == null ? "No previous image yet"
    : `Copy the kept boxes of ${src ? src.file_name : "image " + id} onto this image (P)`;
}

// Copy the previous image's kept boxes (auto accepted / accepted / edited) onto this one as
// human boxes. A box that overlaps an existing box of the same class (IoU >= 0.5) is skipped,
// so model boxes are not doubled. One undo removes the whole copy.
async function repeatPrevious() {
  if (!state.image || state.busy) return;
  const srcId = repeatSourceId();
  if (srcId == null) { showBanner("No previous image to copy from yet.", "info"); return; }
  const img = state.image;
  state.busy = true;
  try {
    const data = await guard(() => api.getAnnotations(srcId));
    if (!data || state.image !== img) return;
    const src = data.annotations.filter((a) => COPY_STATUSES.has(a.status) && classById(a.class_id));
    if (!src.length) { showBanner(`${data.image.file_name} has no kept boxes to copy.`, "info"); return; }
    // Same camera, so normally the same size; if not, scale the boxes to this image.
    const kx = img.width / data.image.width, ky = img.height / data.image.height;
    const existing = state.anns.filter((a) => a.status !== "rejected");
    const todo = [];
    let skipped = 0;
    for (const a of src) {
      const b = canvas.clampBox([a.bbox[0] * kx, a.bbox[1] * ky, a.bbox[2] * kx, a.bbox[3] * ky]);
      if (b[2] < MIN_SIZE || b[3] < MIN_SIZE) { skipped++; continue; }
      const dup = existing.concat(todo).some((e) => e.class_id === a.class_id && iou(e.bbox, b) >= COPY_IOU);
      if (dup) skipped++; else todo.push({ class_id: a.class_id, bbox: b });
    }
    const created = [];
    for (const t of todo) {
      const c = await guard(() => api.createAnnotation(img.id, { ...t, mask: null, reviewer: reviewer() }));
      if (!c) break;
      created.push(c);
      if (state.image === img) state.anns.push(c);
    }
    if (state.image !== img) return; // the user moved on; those boxes are on the other image
    if (created.length) {
      record({ kind: "create", anns: created, label: `repeat (${created.length} box${created.length === 1 ? "" : "es"})` });
      afterChange();
      select(created[created.length - 1].id);
    }
    showBanner(`Copied ${created.length} box(es) from ${data.image.file_name}` +
               (skipped ? `, skipped ${skipped} already here.` : "."), "info");
  } finally {
    state.busy = false;
  }
}

// ------------------------------------------------------------ undo / redo
function resetHistory(imageId) {
  undoLog.imageId = imageId;
  undoLog.undo = [];
  undoLog.redo = [];
  renderHistory();
}

function record(op) {
  undoLog.undo.push(op);
  if (undoLog.undo.length > UNDO_MAX) undoLog.undo.shift();
  undoLog.redo = [];
  renderHistory();
}

function renderHistory() {
  const u = undoLog.undo[undoLog.undo.length - 1], r = undoLog.redo[undoLog.redo.length - 1];
  $("undoBtn").disabled = !u;
  $("redoBtn").disabled = !r;
  $("undoBtn").title = u ? `Undo ${u.label} (Ctrl+Z)` : "Nothing to undo (Ctrl+Z)";
  $("redoBtn").title = r ? `Redo ${r.label} (Ctrl+Shift+Z)` : "Nothing to redo (Ctrl+Shift+Z)";
}

// A re-created box gets a new id; point every op on this image at the new one.
function remapId(oldId, newAnn) {
  for (const op of undoLog.undo.concat(undoLog.redo)) {
    if (op.aid === oldId) op.aid = newAnn.id;
    if (op.anns) op.anns = op.anns.map((a) => (a.id === oldId ? newAnn : a));
  }
}

// Put a deleted box back: POST it again, then restore its status if it was not "accepted".
async function recreate(snap) {
  const who = snap.reviewer || reviewer();
  const c = await guard(() => api.createAnnotation(snap.image_id,
    { class_id: snap.class_id, bbox: snap.bbox, mask: snap.mask || null, reviewer: who }));
  if (!c) return null;
  if (c.status === snap.status) return c;
  const u = await guard(() => api.patchAnnotation(c.id, { status: snap.status, reviewer: who }));
  return u || c;
}

// Returns whether the whole op went through, and for box groups which boxes did and did not,
// so a half-done group can be split.
async function applyOp(op, dir) {
  if (op.kind === "patch") {
    const target = dir === "undo" ? op.before : op.after;
    const other = dir === "undo" ? op.after : op.before;
    const body = { status: target.status, reviewer: reviewer() };
    if (!sameBbox(target.bbox, other.bbox)) body.bbox = target.bbox;
    if (target.class_id !== other.class_id) body.class_id = target.class_id;
    let u;
    try { u = await api.patchAnnotation(op.aid, body); }
    catch (e) {
      console.error(e);
      // Older server: #13 only takes accepted / rejected (requested change in docs/agents/ui.md).
      const oldServer = e instanceof ApiError && e.httpStatus === 400 && !["accepted", "rejected"].includes(body.status);
      showBanner(oldServer ? `Can't undo: this server can't set a box back to "${STATUS_LABEL[body.status]}" yet (contract #13 change).`
                           : (e instanceof ApiError ? `${e.code}: ${e.message}` : String(e)));
      return { done: false };
    }
    replaceAnn(u);
    state.selectedId = u.id;
    return { done: true };
  }
  const removing = (op.kind === "create") === (dir === "undo");
  const done = [], left = [];
  for (const a of op.anns) {
    if (left.length) { left.push(a); continue; }
    if (removing) {
      const ok = await guard(async () => { await api.deleteAnnotation(a.id); return true; });
      if (!ok) { left.push(a); continue; }
      state.anns = state.anns.filter((x) => x.id !== a.id);
      if (state.selectedId === a.id) state.selectedId = null;
      done.push(a);
    } else {
      const c = await recreate(a);
      if (!c) { left.push(a); continue; }
      remapId(a.id, c);
      state.anns.push(c);
      state.selectedId = c.id;
      done.push(c);
    }
  }
  return { done: !left.length, doneAnns: done, leftAnns: left };
}

async function runHistory(dir) {
  if (state.busy || !state.image) return;
  const from = dir === "undo" ? undoLog.undo : undoLog.redo;
  const to = dir === "undo" ? undoLog.redo : undoLog.undo;
  const op = from.pop();
  if (!op) { showBanner(dir === "undo" ? "Nothing to undo on this image." : "Nothing to redo.", "info"); return; }
  const imageId = undoLog.imageId;
  state.busy = true;
  let res;
  try { res = await applyOp(op, dir); } finally { state.busy = false; }
  if (undoLog.imageId !== imageId) return; // another image opened meanwhile; its history is fresh
  if (res.done) to.push(op.kind === "patch" ? op : { ...op, anns: res.doneAnns });
  else if (res.doneAnns && res.doneAnns.length) { // half of a group went through
    to.push({ ...op, anns: res.doneAnns });
    from.push({ ...op, anns: res.leftAnns });
  } else from.push(op);
  renderHistory();
  afterChange();
  canvas.setSelected(state.selectedId);
  if (res.done) showBanner(`${dir === "undo" ? "Undid" : "Redid"} ${op.label}.`, "info");
}

// ------------------------------------------------------------ zoom + adjust
function renderZoom(zoom) {
  const b = $("zoomBtn");
  b.textContent = zoom <= 1.001 ? "Fit" : `${Math.round(zoom * 100)}%`;
  b.classList.toggle("active", zoom > 1.001);
}

function inCanvas(e) {
  const r = $("canvasWrap").getBoundingClientRect();
  return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
}

// Ctrl+wheel (mouse) and pinch (touchpad; Chrome sends it as a wheel event with ctrlKey) zoom
// the image anywhere on the review screen and never the page. Plain wheel pans when zoomed.
window.addEventListener("wheel", (e) => {
  if (state.screen !== "review" || $("helpDialog").open) return;
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
  if (e.ctrlKey) {
    e.preventDefault();
    const dy = Math.max(-120, Math.min(120, e.deltaY * unit));
    const over = inCanvas(e);
    canvas.zoomAt(Math.exp(-dy * 0.0025), over ? e.clientX : null, over ? e.clientY : null);
  } else if (inCanvas(e) && canvas.zoom > 1) {
    e.preventDefault();
    let dx = e.deltaX * unit, dy = e.deltaY * unit;
    if (e.shiftKey && !dx) { dx = dy; dy = 0; }
    canvas.panBy(-dx, -dy);
  }
}, { passive: false });

const adjust = { brightness: 100, contrast: 100 };
function applyAdjust() {
  $("brightness").value = adjust.brightness;
  $("contrast").value = adjust.contrast;
  $("brightnessOut").textContent = adjust.brightness + "%";
  $("contrastOut").textContent = adjust.contrast + "%";
  canvas.setImageFilter(adjust.brightness / 100, adjust.contrast / 100);
  $("adjustBtn").classList.toggle("active", adjust.brightness !== 100 || adjust.contrast !== 100);
}
function openAdjust(on) {
  $("adjustPop").hidden = !on;
  $("adjustBtn").setAttribute("aria-expanded", String(on));
}
function closeAdjust() { openAdjust(false); }

// ------------------------------------------------------------ events
$("tabProjects").addEventListener("click", () => showScreen("projects"));
$("tabSetup").addEventListener("click", () => showScreen("setup"));
$("tabReview").addEventListener("click", () => showScreen("review"));
$("projectSelect").addEventListener("change", (e) => openProject(e.target.value));
$("statusFilter").addEventListener("change", () => loadImages(0));
$("pagePrev").addEventListener("click", () => loadImages(Math.max(0, state.offset - PAGE)));
$("pageNext").addEventListener("click", () => loadImages(state.offset + PAGE));
$("acceptBtn").addEventListener("click", () => setStatus("accepted"));
$("rejectBtn").addEventListener("click", () => setStatus("rejected"));
$("deleteBtn").addEventListener("click", deleteBox);
$("undoBtn").addEventListener("click", () => runHistory("undo"));
$("redoBtn").addEventListener("click", () => runHistory("redo"));
$("repeatBtn").addEventListener("click", repeatPrevious);
$("zoomBtn").addEventListener("click", () => canvas.resetView());
$("bClass").addEventListener("change", (e) => changeClass(Number(e.target.value)));
$("newClass").addEventListener("change", () => { syncDrawColor(); if (state.drawMode) setDrawMode(true); });
$("drawBtn").addEventListener("click", () => setDrawMode(!state.drawMode));
$("showRejected").addEventListener("change", (e) => {
  canvas.showRejected = e.target.checked;
  const a = selectedAnn();
  if (a && a.status === "rejected" && !e.target.checked) select(null); else canvas.render();
});
$("adjustBtn").addEventListener("click", () => openAdjust($("adjustPop").hidden));
$("brightness").addEventListener("input", (e) => { adjust.brightness = Number(e.target.value); applyAdjust(); });
$("contrast").addEventListener("input", (e) => { adjust.contrast = Number(e.target.value); applyAdjust(); });
$("adjustReset").addEventListener("click", () => { adjust.brightness = 100; adjust.contrast = 100; applyAdjust(); });
document.addEventListener("mousedown", (e) => {
  if (!$("adjustPop").hidden && !e.target.closest(".pop-wrap")) closeAdjust();
});
$("boxClose").addEventListener("click", () => select(null));
$("helpBtn").addEventListener("click", () => $("helpDialog").showModal());
$("banner").addEventListener("click", () => ($("banner").hidden = true));
$("reviewerInput").addEventListener("change", (e) => {
  try { localStorage.setItem("review.reviewer", e.target.value.trim()); } catch (err) { /* ignore */ }
});

document.addEventListener("keydown", (e) => {
  if (state.screen !== "review") return; // review shortcuts only on the review screen
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || $("helpDialog").open) return;
  const k = e.key;
  // Hold Ctrl = draw; hold H = hide boxes. Work even while a dropdown has focus.
  if (k === "Control") { canvas.setTempDraw(true); return; }
  if ((k === "h" || k === "H") && !e.ctrlKey && !e.metaKey && !e.altKey) {
    canvas.setHideAll(true);
    e.preventDefault();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && !e.altKey) {
    const key = k.toLowerCase();
    if (key === "z" && !e.shiftKey) runHistory("undo");
    else if ((key === "z" && e.shiftKey) || key === "y") runHistory("redo");
    else return;
    e.preventDefault();
    return;
  }
  if (tag === "select" && !["Tab", "Escape"].includes(k)) return;
  if (e.altKey || e.metaKey) return;
  let handled = true;
  if (k === "a" || k === "A") setStatus("accepted");
  else if (k === "r" || k === "R") setStatus("rejected");
  else if (k === "Delete" || k === "Backspace") deleteBox();
  else if (k === "Tab") cycleBox(e.shiftKey ? -1 : 1);
  else if (k === "ArrowRight" || k === "d" || k === "D") stepImage(1);
  else if (k === "ArrowLeft" || k === "s" || k === "S") stepImage(-1);
  else if (k === "n" || k === "N") setDrawMode(!state.drawMode);
  else if (k === "p" || k === "P") repeatPrevious();
  else if (k === "0") canvas.resetView();
  else if (k === "Escape") {
    if (!$("adjustPop").hidden) closeAdjust();
    else if (canvas.cancelDraw()) { /* dropped the box being drawn */ }
    else if (state.drawMode) setDrawMode(false);
    else select(null);
  }
  else if (k === "?") $("helpDialog").showModal();
  else if (/^[1-9]$/.test(k)) {
    const cls = state.classes[Number(k) - 1];
    if (!cls) handled = false;
    else if (selectedAnn()) changeClass(cls.id);
    else { $("newClass").value = cls.id; syncDrawColor(); if (state.drawMode) setDrawMode(true); }
  } else handled = false;
  if (handled) { e.preventDefault(); if (tag === "select") e.target.blur(); }
});

document.addEventListener("keyup", (e) => {
  if (e.key === "Control") canvas.setTempDraw(false);
  if (e.key === "h" || e.key === "H") canvas.setHideAll(false);
});
// A held key is never released if the window loses focus (Alt+Tab, click into devtools).
window.addEventListener("blur", () => { canvas.setTempDraw(false); canvas.setHideAll(false); });

// ------------------------------------------------------------ start
(function init() {
  $("mockBadge").hidden = !api.mock;
  if (api.mock) document.title += " (mock)";
  try { $("reviewerInput").value = localStorage.getItem("review.reviewer") || ""; } catch (e) { /* ignore */ }
  renderHistory();
  renderZoom(1);
  showScreen("projects"); // the start page
  loadProjects();
})();
