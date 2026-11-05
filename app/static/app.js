// app.js - screens + the review page (state, labels panel, image strip, tools, box popup,
// undo/redo, copy/paste, keyboard, zoom).
"use strict";

const PAGE = 50;
const UNDO_MAX = 50;          // undo steps kept per image
const COPY_IOU = 0.5;         // "repeat previous" skips a box this much like an existing one
const PENDING = -1;           // id of a box that was drawn but not saved yet (only in the browser)
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
  tool: "drag",       // "drag" (select / move) or "box" (every drag draws a box)
  pending: null,      // new box being drawn, not saved yet: an Annotation-like object with id PENDING
  lastClassId: null,  // class a new box starts with (the last one saved)
  clipboard: null,    // Ctrl+C: {width, height, boxes: [{class_id, bbox}]}
  loadToken: 0,       // ignore late responses from an older image
  listToken: 0,       // ignore late image lists from an older filter or page
  prevImageId: null,  // image open before this one (source for "repeat previous")
  busy: false,        // undo / redo / repeat / paste request in flight
  screen: "projects",
};

// Undo / redo of box changes on the image that is open. Cleared when another image opens.
// Ops: {kind: "patch", aid, before, after, label} | {kind: "create" | "delete", anns: [...], label}
const undoLog = { imageId: null, undo: [], redo: [] };

// Box popup: which box it was opened for, the filter text and the highlighted class (index in the filtered list).
// pos: where the user dragged it, in canvas px (null = beside the box); kept until the page reloads.
const pop = { forId: null, filter: "", hl: 0, pos: null };

const canvas = new BoxCanvas($("canvas"), {
  onSelect: (ann) => {
    if (state.pending && (!ann || ann.id !== PENDING)) dropPending();
    select(ann ? ann.id : null, false);
  },
  onBoxChanged: (ann, bbox) => {
    if (ann.id === PENDING) {
      const p = state.pending;
      if (p.polygon) setPendingPolygon(scalePolygon(p.polygon, p.bbox, bbox, canvas.imgW, canvas.imgH));
      else { p.bbox = bbox; canvas.render(); }
      return;
    }
    editBox(ann, { bbox }, "move box");
  },
  onBoxDrawn: (bbox) => startPending(bbox),
  onPolygonStart: () => startPolygon(),
  onPolygonDrawn: (polygon) => finishPolygon(polygon),
  onPolygonChanged: (ann, polygon, label) => changePolygon(ann, polygon, label),
  onVertexClicked: (ann, index) => deleteVertex(ann, index),
  onView: (zoom, scale) => renderZoom(zoom, scale),
  onRender: () => positionPopup(),
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

function showReviewEmpty(message, actionLabel, action) {
  $("imageTitle").hidden = true;
  $("reviewEmptyMessage").textContent = message;
  const button = $("reviewEmptyAction");
  button.textContent = actionLabel;
  button.onclick = action;
  $("reviewEmpty").hidden = false;
}

function hideReviewEmpty() {
  $("reviewEmpty").hidden = true;
  $("imageTitle").hidden = false;
}

// The class a new box starts with: the last one saved, else the project's first class.
function newBoxClass() {
  return classById(state.lastClassId) || state.classes[0] || null;
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

function lsGet(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
function lsSet(key, v) { try { localStorage.setItem(key, v); } catch (e) { /* ignore */ } }

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
  if (name !== "review") { canvas.setTempDraw(false); canvas.setHideAll(false); canvas.cancelPolygon(); closeAdjust(); }
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
  const saved = preferId != null ? String(preferId) : lsGet("review.project");
  const pick = data.projects.find((p) => String(p.id) === saved) || data.projects[0];
  $("projectSelect").value = pick.id;
  await openProject(pick.id);
}

async function openProject(pid) {
  const changed = state.projectId !== Number(pid);
  state.projectId = Number(pid);
  $("projectSelect").value = pid;
  lsSet("review.project", pid);
  const cls = await guard(() => api.listClasses(pid));
  state.classes = cls ? cls.classes : [];
  if (changed) { canvas.hiddenClasses.clear(); state.prevImageId = null; state.clipboard = null; }
  if (!classById(state.lastClassId)) state.lastClassId = state.classes.length ? state.classes[0].id : null;
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
                         imageIndex: -1, image: null, anns: [], prevImageId: null, pending: null,
                         lastClassId: null, clipboard: null });
  buildImageList();
  renderLabels();
  hideReviewEmpty();
  $("imageTitle").textContent = "No project.";
  syncCanvas();
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

// ------------------------------------------------------------ labels panel
const EYE = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" fill="none" stroke="currentColor" stroke-width="1.3"/><circle cx="8" cy="8" r="2" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M2.5 13.5l11-11" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>';

// Classes with boxes on this image (rejected not counted) get a count and an eye toggle;
// the project's other classes are listed greyed under "Unused classes".
function renderLabels() {
  const used = $("labelList"), unused = $("unusedList");
  used.innerHTML = "";
  unused.innerHTML = "";
  let total = 0;
  for (const c of state.classes) {
    const n = state.anns.filter((a) => a.class_id === c.id && a.status !== "rejected").length;
    total += n;
    const li = document.createElement("li");
    li.innerHTML = `<i></i><span class="nm"></span>`;
    li.querySelector("i").style.background = classColor(c.name);
    li.querySelector(".nm").textContent = c.name;
    li.title = c.description;
    if (n) {
      const hidden = canvas.hiddenClasses.has(c.id);
      li.classList.toggle("off", hidden);
      li.insertAdjacentHTML("beforeend", `<span class="ct"></span><button type="button" class="icon eye"></button>`);
      li.querySelector(".ct").textContent = String(n);
      const eye = li.querySelector(".eye");
      eye.innerHTML = hidden ? EYE_OFF : EYE;
      eye.setAttribute("aria-pressed", String(!hidden));
      eye.setAttribute("aria-label", `${hidden ? "Show" : "Hide"} ${c.name} boxes`);
      eye.title = `${hidden ? "Show" : "Hide"} ${c.name}`;
      eye.addEventListener("click", () => toggleClass(c.id));
      used.appendChild(li);
    } else unused.appendChild(li);
  }
  $("unusedHead").hidden = !unused.children.length;
  $("labelsTotal").textContent = state.image ? `${total} box${total === 1 ? "" : "es"}` : "";
}

function toggleClass(classId) {
  const hide = !canvas.hiddenClasses.has(classId);
  canvas.setClassHidden(classId, hide);
  const a = selectedAnn();
  if (hide && a && a.class_id === classId && a.id !== PENDING) select(null);
  renderLabels();
}

function setLabelsOpen(on) {
  $("labelsPanel").hidden = !on;
  $("labelsBtn").setAttribute("aria-expanded", String(on));
  lsSet("review.labels", on ? "1" : "0");
}

// ------------------------------------------------------------ image strip
async function loadImages(offset, selectIndex = 0) {
  const status = $("statusFilter").value || null;
  const token = ++state.listToken;
  const data = await guard(() => api.listImages(state.projectId, { status, offset, limit: PAGE }));
  if (!data || token !== state.listToken) return;
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
    state.pending = null;
    const projectHasImages = !!(state.project && state.project.counts.images > 0);
    if (projectHasImages) {
      const filterName = $("statusFilter").selectedOptions[0].textContent;
      showReviewEmpty(`No images match '${filterName}'.`, "Show all images", () => {
        $("statusFilter").value = "";
        loadImages(0);
      });
    } else {
      showReviewEmpty("No images yet", "Upload images", () => {
        showScreen("setup");
        showStep(2);
      });
    }
    syncCanvas();
    canvas.setImage(null);
    resetHistory(null);
    renderLabels();
    renderRepeat();
    renderImageList();
    select(null);
  }
}

// The strip is built once per page (so thumbnails are not re-fetched), then updated in place.
function buildImageList() {
  const ul = $("imageList");
  ul.innerHTML = "";
  state.images.forEach((img, i) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<button type="button" class="thumb"><span class="frame"><img alt="" loading="lazy" decoding="async">` +
      `<span class="ph"></span><span class="marks"><span class="nr"></span><span class="dot"></span></span>` +
      `<span class="idx"></span></span><span class="name"></span></button>`;
    const frame = li.querySelector(".frame");
    const im = li.querySelector("img");
    im.addEventListener("error", () => frame.classList.add("missing"));
    li.querySelector(".ph").textContent = `${img.width}×${img.height}`;
    li.querySelector(".idx").textContent = String(state.offset + i + 1);
    im.src = api.imageUrl(img);
    li.querySelector(".name").textContent = img.file_name;
    li.querySelector(".thumb").title = img.file_name;
    li.querySelector(".thumb").addEventListener("click", () => openImage(i));
    ul.appendChild(li);
  });
  renderImageList();
}

// Strip marks come from the stored review_status (#5 / #20): check = done, empty-set = empty, none = todo.
const REVIEW_MARK = { done: "Reviewed (done)", empty: "Marked empty: nothing to label" };
function renderImageList() {
  const ul = $("imageList");
  state.images.forEach((img, i) => {
    const li = ul.children[i];
    if (!li) return;
    li.classList.toggle("current", i === state.imageIndex);
    const rs = img.review_status || "todo";
    const dot = li.querySelector(".dot");
    dot.classList.toggle("done", rs === "done");
    dot.classList.toggle("empty", rs === "empty");
    dot.hidden = !REVIEW_MARK[rs];
    dot.title = REVIEW_MARK[rs] || "";
    const n = img.counts ? img.counts.needs_review : 0;
    const nr = li.querySelector(".nr");
    nr.hidden = !n;
    nr.textContent = n ? String(n) : "";
    nr.title = n ? `${n} box(es) need review` : "";
  });
  const pos = state.imageIndex >= 0 ? state.offset + state.imageIndex + 1 : 0;
  $("imgCounter").textContent = `${pos} / ${state.total}`;
  $("imgPrev").disabled = pos <= 1;
  $("imgNext").disabled = pos >= state.total;
  renderReviewStatusBtns();
}

function setStripOpen(on) {
  $("strip").classList.toggle("collapsed", !on);
  const t = $("stripToggle");
  t.setAttribute("aria-expanded", String(on));
  t.setAttribute("aria-label", on ? "Hide image strip" : "Show image strip");
  t.dataset.tip = on ? "Hide images" : "Show images";
  lsSet("review.strip", on ? "1" : "0");
}

async function openImage(i) {
  if (i < 0 || i >= state.images.length) return;
  const img = state.images[i];
  if (state.image && state.image.id !== img.id) state.prevImageId = state.image.id;
  state.imageIndex = i;
  hideReviewEmpty();
  const token = ++state.loadToken;
  renderImageList();
  const cur = $("imageList").children[i];
  if (cur) cur.scrollIntoView({ block: "nearest" });
  $("imageTitle").textContent = `${img.file_name} · ${img.width}×${img.height} · id ${img.id}`;
  resetHistory(img.id);
  state.anns = [];
  state.pending = null;
  syncCanvas();
  state.loadingImage = true;
  select(null);
  canvas.setImage(api.imageUrl(img), img.width, img.height); // also resets zoom to fit
  const data = await guard(() => api.getAnnotations(img.id));
  if (token !== state.loadToken) return;
  state.loadingImage = false;
  if (!data) { renderPopup(); return; }
  state.image = data.image;
  state.anns = data.annotations;
  syncCanvas();
  renderReviewStatusBtns();
  renderLabels();
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

// After any box change: update this image's counts in the strip, then the top-bar progress.
function afterChange() {
  const img = state.images[state.imageIndex];
  if (img) img.counts = countStatuses(state.anns);
  renderImageList();
  syncCanvas();
  renderLabels();
  renderPopup();
  refreshProgress();
}

// ------------------------------------------------------------ boxes
// The canvas shows the saved boxes plus the unsaved new one, if any.
function syncCanvas() {
  canvas.setAnnotations(state.pending ? state.anns.concat([state.pending]) : state.anns);
}

// glide: animate the selection from the old box to the new one (used after accept / reject and Tab).
function select(id, redraw = true, glide = false) {
  state.selectedId = id;
  if (redraw) canvas.setSelected(id, glide); else canvas.selectedId = id;
  renderPopup();
}

function selectedAnn() {
  if (state.pending && state.selectedId === PENDING) return state.pending;
  return state.anns.find((a) => a.id === state.selectedId) || null;
}

function cycleBox(delta) {
  const list = canvas.visible();
  if (!list.length) return;
  const i = list.findIndex((a) => a.id === state.selectedId);
  const next = i < 0 ? (delta > 0 ? 0 : list.length - 1) : (i + delta + list.length) % list.length;
  if (state.pending && list[next].id !== PENDING) dropPending();
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

const clonePolygon = (poly) => (poly ? poly.map((p) => p.slice()) : null);
const snapOf = (a) => ({ status: a.status, bbox: a.bbox.slice(), class_id: a.class_id, polygon: clonePolygon(a.polygon) });
const sameBbox = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 0.05);
const samePolygon = (a, b) => (!a || !b ? !a && !b : a.length === b.length &&
  a.every((p, i) => Math.abs(p[0] - b[i][0]) < 1e-7 && Math.abs(p[1] - b[i][1]) < 1e-7));
const sameSnap = (a, b) => a.status === b.status && a.class_id === b.class_id && sameBbox(a.bbox, b.bbox) &&
  samePolygon(a.polygon, b.polygon);

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
  else { select(null); showBanner("No more boxes to review on this image. Press → for the next one.", "info"); }
}

// A saved box is needed for review actions; an unsaved new one is saved with Enter first.
function savedSelection() {
  const a = selectedAnn();
  if (!a) { showBanner("Select a box first (click it or press Tab).", "info"); return null; }
  if (a.id === PENDING) { showBanner("Save the new box first (Enter), or Esc to drop it.", "info"); return null; }
  return a;
}

async function setStatus(status, label = null) {
  const a = savedSelection();
  if (!a) return;
  const before = { ...a };
  const updated = await editBox(a, { status }, label || (status === "accepted" ? "approve" : "reject"));
  if (!updated) return;
  if (status === "accepted") canvas.flashAccept(updated.id);
  else if (status === "rejected") canvas.fadeReject(before);
  selectNextToReview();
}

// U: put an approved / rejected box back to the status it had before. Uses this image's undo
// history when it knows; otherwise a model box goes back to "needs review" and a drawn box to
// "accepted" (the status a new human box gets). Sent as an explicit status (contract #13).
async function unreview() {
  const a = savedSelection();
  if (!a) return;
  if (a.status !== "accepted" && a.status !== "rejected") {
    showBanner("This box is not approved or rejected.", "info");
    return;
  }
  const op = undoLog.undo.slice().reverse().find((o) => o.kind === "patch" && o.aid === a.id &&
                                                        o.after.status === a.status && o.before.status !== a.status);
  const target = op ? op.before.status : (a.source === "human" ? "accepted" : "needs_review");
  if (target === a.status) { showBanner("Nothing to undo: this drawn box was never approved or rejected.", "info"); return; }
  const updated = await editBox(a, { status: target }, `undo ${a.status === "accepted" ? "approve" : "reject"} of`);
  if (updated) showBanner(`Box #${updated.id} is back to ${STATUS_LABEL[updated.status].toLowerCase()}.`, "info");
}

// Delete: an unsaved new box is dropped; a drawn (human) box is really removed; a model box is
// rejected, so it still counts toward precision. A server without DELETE /annotations
// (contract #19) answers 405: then the drawn box is rejected instead.
async function deleteBox() {
  const a = selectedAnn();
  if (!a) { showBanner("Select a box first (click it or press Tab).", "info"); return; }
  if (a.id === PENDING) { dropPending(); return; }
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

// ---- new boxes: drawing makes an unsaved box and opens the popup on it; Save (Enter) POSTs it,
// Esc drops it.
function startPending(bbox, polygon = null) {
  if (!state.image) return;
  const cls = newBoxClass();
  if (!cls) { showBanner("This project has no classes yet. Add them in Setup.", "error"); return; }
  state.pending = {
    id: PENDING, image_id: state.image.id, class_id: cls.id, class_name: cls.name, bbox,
    mask: null, polygon, score: null, source: "human", status: "accepted", reviewer: null,
  };
  syncCanvas();
  pop.forId = null; // a fresh popup, even if another unsaved box was open
  select(PENDING);
  const input = $("bpClass");
  input.focus();
  input.select();
}

function dropPending() {
  if (!state.pending) return;
  state.pending = null;
  if (state.selectedId === PENDING) state.selectedId = null;
  syncCanvas();
  canvas.setSelected(state.selectedId);
  renderPopup();
}

// ---- polygons
// What stops this polygon being kept (too few points, no area, smaller than a box may be), or null.
function polygonProblem(polygon) {
  const err = polygonError(polygon);
  if (err) return err[0].toUpperCase() + err.slice(1) + ".";
  const b = polygonBbox(polygon, canvas.imgW, canvas.imgH);
  return b[2] < MIN_SIZE || b[3] < MIN_SIZE ? "This polygon is too small to keep." : null;
}

// The first click of a new polygon. Whatever was open closes, so the keys act on the drawing.
function startPolygon() {
  if (!state.image) return false;
  if (!newBoxClass()) { showBanner("This project has no classes yet. Add them in Setup.", "error"); return false; }
  dropPending();
  select(null);
  return true;
}

// The shape was closed: same class choice as a new box. False keeps the drawing open.
function finishPolygon(polygon) {
  const problem = polygonProblem(polygon);
  if (problem) { showBanner(problem, "info"); return false; }
  startPending(polygonBbox(polygon, canvas.imgW, canvas.imgH), polygon);
  return true;
}

function setPendingPolygon(polygon) {
  state.pending.polygon = polygon;
  state.pending.bbox = polygonBbox(polygon, canvas.imgW, canvas.imgH);
  canvas.render();
}

// A point was moved, added or deleted: #13 with the whole polygon (one undo step).
async function changePolygon(ann, polygon, label) {
  const problem = polygonProblem(polygon);
  if (problem) { showBanner(problem, "info"); return; }
  if (ann.id === PENDING) { setPendingPolygon(polygon); return; }
  await editBox(ann, { polygon }, label);
}

// A click on a point of the selected polygon.
async function deleteVertex(ann, index) {
  if (ann.polygon.length <= 3) { showBanner("A polygon needs at least 3 points, so this one stays.", "info"); return; }
  await changePolygon(ann, ann.polygon.filter((_, k) => k !== index), "delete point of");
}

// Body of a #12 POST: a polygon goes without a bbox (the server computes it).
function boxBody(classId, bbox, polygon) {
  return polygon ? { class_id: classId, polygon, mask: null, reviewer: reviewer() }
                 : { class_id: classId, bbox, mask: null, reviewer: reviewer() };
}

const EMPTY_MSG = "Image is marked empty - press N to unmark it first.";

// POST #12. On an image marked empty the server answers 409: say how to fix it.
async function postBox(iid, body) {
  try { return await api.createAnnotation(iid, body); }
  catch (e) {
    if (e instanceof ApiError && e.httpStatus === 409) { showBanner(EMPTY_MSG); return null; } // expected, handled
    console.error(e);
    showBanner(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
    return null;
  }
}

async function savePending(cls) {
  const p = state.pending;
  const iid = p.image_id;
  const created = await postBox(iid, boxBody(cls.id, p.bbox, p.polygon));
  if (!created && state.pending === p) { dropPending(); return null; } // nothing saved: remove the unsaved box
  if (!created || !state.image || state.image.id !== iid || state.pending !== p) return null;
  state.pending = null;
  state.anns.push(created);
  record({ kind: "create", anns: [created], label: `draw box #${created.id}` });
  state.selectedId = null;
  afterChange();
  canvas.setSelected(null);
  return created;
}

function syncDrawColor() {
  const cls = newBoxClass();
  canvas.setDrawColor(cls ? classColor(cls.name) : "#fff");
}

// ------------------------------------------------------------ box popup
function popClasses() {
  const f = pop.filter.trim().toLowerCase();
  return f ? state.classes.filter((c) => c.name.toLowerCase().includes(f)) : state.classes.slice();
}

// Approve / Reject buttons above the image strip: only for a saved, selected box.
function renderReviewButtons() {
  const a = selectedAnn();
  const off = !a || a.id === PENDING;
  $("approveBtn").disabled = off;
  $("rejectBtn").disabled = off;
}

function renderPopup() {
  renderReviewButtons();
  const a = selectedAnn();
  const el = $("boxPop");
  if (!a || state.loadingImage) {
    if (el.contains(document.activeElement)) document.activeElement.blur();
    el.hidden = true;
    pop.forId = null;
    return;
  }
  if (pop.forId !== a.id) { // opened for another box: start from its own class
    pop.forId = a.id;
    pop.filter = "";
    $("bpClass").value = "";
    pop.hl = Math.max(0, state.classes.findIndex((c) => c.id === a.class_id));
  }
  if (a.id === PENDING) {
    $("bpStatus").className = "status-line st-needs_review";
    $("bpStatus").textContent = a.polygon ? "New polygon, not saved" : "New box, not saved";
    $("bpMeta").textContent = "";
  } else {
    $("bpStatus").className = `status-line st-${a.status}`;
    $("bpStatus").textContent = STATUS_LABEL[a.status] || a.status;
    const source = a.source === "human" ? "drawn" : a.source;
    $("bpMeta").textContent = `#${a.id} · ${source}` + (a.score != null ? ` · ${a.score.toFixed(2)}` : "");
    $("bpMeta").title = `source: ${a.source}${a.reviewer ? ", reviewer: " + a.reviewer : ""}`;
  }
  $("bpDelete").title = a.id === PENDING ? "Drop this new box"
    : a.source === "human" ? "Delete this drawn box (Del)" : "Reject this model box; it is kept for precision stats (Del)";
  renderPopList();
  el.hidden = false;
  positionPopup();
}

function renderPopList() {
  const a = selectedAnn();
  const list = popClasses();
  pop.hl = list.length ? Math.max(0, Math.min(pop.hl, list.length - 1)) : 0;
  const ul = $("bpList");
  ul.innerHTML = "";
  if (!list.length) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = state.classes.length ? "No class matches" : "No classes in this project";
    ul.appendChild(li);
  }
  list.forEach((c, i) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.id = `bpOpt${c.id}`;
    li.innerHTML = `<i></i><span></span>`;
    li.querySelector("i").style.background = classColor(c.name);
    li.querySelector("span").textContent = c.name;
    li.classList.toggle("hl", i === pop.hl);
    li.classList.toggle("cur", !!a && a.id !== PENDING && c.id === a.class_id);
    li.setAttribute("aria-selected", String(i === pop.hl));
    li.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus in the field
    li.addEventListener("click", () => { pop.hl = i; savePopup(); });
    ul.appendChild(li);
  });
  const hl = ul.children[pop.hl];
  $("bpClass").setAttribute("aria-activedescendant", list.length ? hl.id : "");
  if (hl) hl.scrollIntoView({ block: "nearest" });
  // An unsaved box shows the highlighted class right away (colour and tag).
  const p = state.pending;
  if (p && a === p && list[pop.hl] && list[pop.hl].id !== p.class_id) {
    p.class_id = list[pop.hl].id;
    p.class_name = list[pop.hl].name;
    canvas.render();
  }
}

function movePopHighlight(delta) {
  const n = popClasses().length;
  if (!n) return;
  pop.hl = (pop.hl + delta + n) % n;
  renderPopList();
}

// Place the popup beside the selected box (right, else left, else below / above), inside the view.
// Once the user has dragged it, it stays where they put it (still inside the view).
function positionPopup() {
  const el = $("boxPop");
  if (pop.forId == null) return;
  const r = canvas.selectedRect();
  if (!r || canvas.hideAll) { el.style.visibility = "hidden"; return; }
  el.style.visibility = "";
  const W = canvas.vw, H = canvas.vh, pw = el.offsetWidth, ph = el.offsetHeight, gap = 10, m = 8;
  const clampv = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
  if (pop.pos) {
    el.style.left = clampv(pop.pos.x, 0, Math.max(0, W - pw)) + "px";
    el.style.top = clampv(pop.pos.y, 0, Math.max(0, H - ph)) + "px";
    return;
  }
  let x, y;
  if (r.x + r.w + gap + pw <= W - m) { x = r.x + r.w + gap; y = r.y; }
  else if (r.x - gap - pw >= m) { x = r.x - gap - pw; y = r.y; }
  else {
    x = r.x;
    y = r.y + r.h + gap + ph <= H - m ? r.y + r.h + gap : r.y - gap - ph;
  }
  el.style.left = clampv(x, m, Math.max(m, W - pw - m)) + "px";
  el.style.top = clampv(y, m, Math.max(m, H - ph - m)) + "px";
}

// Save: a new box is created with the highlighted class; an existing box gets that class
// (if it changed). Either way the popup closes.
async function savePopup() {
  const a = selectedAnn();
  if (!a) return;
  const cls = popClasses()[pop.hl];
  if (!cls) { showBanner("No class matches. Change the text or press Esc.", "info"); return; }
  if (document.activeElement === $("bpClass")) $("bpClass").blur();
  state.lastClassId = cls.id;
  syncDrawColor();
  if (a.id === PENDING) { await savePending(cls); return; }
  if (cls.id !== a.class_id) await editBox(a, { class_id: cls.id }, "change class of");
  if (state.selectedId === a.id) select(null);
}

// Esc: a new box is dropped; for an existing box nothing is changed and the popup closes.
function cancelPopup() {
  if (document.activeElement === $("bpClass")) $("bpClass").blur();
  const a = selectedAnn();
  if (a && a.id === PENDING) dropPending(); else select(null);
}

function popupOpen() { return !$("boxPop").hidden && pop.forId != null; }

// Drag the popup by its header. No default on mouse down: the class field keeps its focus.
function startPopupDrag(e) {
  if (e.button !== 0) return;
  e.preventDefault();
  const el = $("boxPop");
  const dx = e.clientX - el.offsetLeft, dy = e.clientY - el.offsetTop;
  const move = (m) => { pop.pos = { x: m.clientX - dx, y: m.clientY - dy }; positionPopup(); };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    // Keep what the view allowed, so a drag past the edge does not leave a stored position outside it.
    pop.pos = { x: el.offsetLeft, y: el.offsetTop };
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
}

// ------------------------------------------------------------ copy / paste, repeat previous
// POST boxes ({class_id, bbox, polygon}) onto image img as human boxes, one by one; one undo step.
async function createBoxes(img, boxes, label) {
  const created = [];
  for (const t of boxes) {
    const c = await postBox(img.id, boxBody(t.class_id, t.bbox, t.polygon));
    if (!c) break;
    created.push(c);
    if (state.image === img) state.anns.push(c);
  }
  if (state.image !== img) return created; // the user moved on; those boxes are on the other image
  if (created.length) {
    record({ kind: "create", anns: created, label: `${label} (${created.length} box${created.length === 1 ? "" : "es"})` });
    afterChange();
    select(created[created.length - 1].id);
  }
  return created;
}

// Boxes from an image of another size are scaled to this one.
function scaleBox(b, fromW, fromH) {
  const img = state.image;
  const kx = img.width / fromW, ky = img.height / fromH;
  return canvas.clampBox([b[0] * kx, b[1] * ky, b[2] * kx, b[3] * ky]);
}

function copyBox() {
  const a = selectedAnn();
  if (!a || a.id === PENDING || !state.image) return false;
  state.clipboard = { width: state.image.width, height: state.image.height,
                      boxes: [{ class_id: a.class_id, bbox: a.bbox.slice(), polygon: clonePolygon(a.polygon) }] };
  showBanner(`Copied box #${a.id} (${a.class_name}). Ctrl+V pastes it on any image.`, "info");
  return true;
}

async function pasteBoxes() {
  const cb = state.clipboard;
  if (!cb) { showBanner("Nothing copied yet. Select a box and press Ctrl+C.", "info"); return; }
  if (!state.image || state.busy) return;
  const boxes = cb.boxes.filter((b) => classById(b.class_id))
    .map((b) => ({ ...b, bbox: scaleBox(b.bbox, cb.width, cb.height) })) // a polygon is normalized: no scaling
    .filter((b) => b.bbox[2] >= MIN_SIZE && b.bbox[3] >= MIN_SIZE);
  if (!boxes.length) { showBanner("The copied box does not fit this image.", "info"); return; }
  dropPending();
  state.busy = true;
  try { await createBoxes(state.image, boxes, "paste"); } finally { state.busy = false; }
}

function repeatSourceId() {
  if (state.prevImageId != null && (!state.image || state.prevImageId !== state.image.id)) return state.prevImageId;
  const before = state.images[state.imageIndex - 1];
  return before ? before.id : null;
}

function renderRepeat() {
  const id = repeatSourceId();
  const src = id == null ? null : state.images.find((i) => i.id === id);
  const b = $("repeatBtn");
  b.disabled = id == null || !state.image;
  b.dataset.tip = id == null ? "Repeat previous (P): no previous image yet"
    : `Repeat previous (P): copy the kept boxes of ${src ? src.file_name : "image " + id}`;
}

// Copy the previous image's kept boxes (auto accepted / accepted / edited) onto this one as
// human boxes. A box that overlaps an existing box of the same class (IoU >= 0.5) is skipped,
// so model boxes are not doubled. One undo removes the whole copy.
async function repeatPrevious() {
  if (!state.image || state.busy) return;
  const srcId = repeatSourceId();
  if (srcId == null) { showBanner("No previous image to copy from yet.", "info"); return; }
  const img = state.image;
  dropPending();
  state.busy = true;
  try {
    const data = await guard(() => api.getAnnotations(srcId));
    if (!data || state.image !== img) return;
    const src = data.annotations.filter((a) => COPY_STATUSES.has(a.status) && classById(a.class_id));
    if (!src.length) { showBanner(`${data.image.file_name} has no kept boxes to copy.`, "info"); return; }
    const existing = state.anns.filter((a) => a.status !== "rejected");
    const todo = [];
    let skipped = 0;
    for (const a of src) {
      const b = scaleBox(a.bbox, data.image.width, data.image.height);
      if (b[2] < MIN_SIZE || b[3] < MIN_SIZE) { skipped++; continue; }
      const dup = existing.concat(todo).some((e) => e.class_id === a.class_id && iou(e.bbox, b) >= COPY_IOU);
      if (dup) skipped++; else todo.push({ class_id: a.class_id, bbox: b, polygon: clonePolygon(a.polygon) });
    }
    const created = await createBoxes(img, todo, "repeat");
    if (state.image !== img) return;
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
  $("undoBtn").dataset.tip = u ? `Undo ${u.label} (Ctrl+Z)` : "Undo (Ctrl+Z): nothing to undo";
  $("redoBtn").dataset.tip = r ? `Redo ${r.label} (Ctrl+Shift+Z)` : "Redo (Ctrl+Shift+Z): nothing to redo";
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
    { ...boxBody(snap.class_id, snap.bbox, snap.polygon), mask: snap.mask || null, reviewer: who }));
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
    // The old polygon itself, not a bbox: scaling back could leave a rounding error.
    if (!samePolygon(target.polygon, other.polygon)) body.polygon = target.polygon;
    if (!target.polygon && !sameBbox(target.bbox, other.bbox)) body.bbox = target.bbox;
    if (target.class_id !== other.class_id) body.class_id = target.class_id;
    let u;
    try { u = await api.patchAnnotation(op.aid, body); }
    catch (e) {
      console.error(e);
      showBanner(e instanceof ApiError ? `${e.code}: ${e.message}` : String(e));
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
  dropPending();
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

// ------------------------------------------------------------ tools
// Drag: click selects, drag moves / resizes, Ctrl+drag draws. Box: every drag draws a box.
// Polygon: every click adds a point.
function setTool(tool) {
  state.tool = tool;
  syncDrawColor();
  canvas.setDrawMode(tool === "box");
  canvas.setPolyMode(tool === "polygon");
  $("toolDrag").setAttribute("aria-pressed", String(tool === "drag"));
  $("toolBox").setAttribute("aria-pressed", String(tool === "box"));
  $("toolPolygon").setAttribute("aria-pressed", String(tool === "polygon"));
}

// Image review status via #20. N toggles "empty", Ctrl+Enter toggles "done"; pressing again
// sets "todo". A 409 (boxes that are not rejected, for "empty") shows the server's message and
// changes nothing here. Marking done moves on to the next image.
async function toggleReview(kind) {
  const img = state.image;
  if (!img || state.busy || state.loadingImage) return;
  const target = img.review_status === kind ? "todo" : kind;
  state.busy = true;
  let updated = null;
  try { updated = await api.patchImage(img.id, { review_status: target, reviewer: reviewer() }); }
  catch (e) {
    if (!(e instanceof ApiError && e.httpStatus === 409)) console.error(e); // 409 is expected, shown as a toast
    showBanner(e instanceof ApiError ? e.message : String(e));
  } finally { state.busy = false; }
  if (!updated) return;
  img.review_status = updated.review_status;
  const row = state.images.find((i) => i.id === updated.id);
  if (row) row.review_status = updated.review_status;
  if (updated.review_status === "empty") dropPending();
  renderImageList();
  const msg = { empty: "Image marked empty.", done: "Image marked done.", todo: "Image set back to to-do." };
  showBanner(msg[updated.review_status], "info");
  if (updated.review_status === "done" && state.image === img) await stepImage(1);
}

function renderReviewStatusBtns() {
  const img = state.image;
  const rs = img ? img.review_status : null;
  const set = (id, on, tipOn, tipOff) => {
    const b = $(id);
    b.disabled = !img;
    b.setAttribute("aria-pressed", String(on));
    b.dataset.tip = on ? tipOn : tipOff;
    b.setAttribute("aria-label", b.dataset.tip);
  };
  set("emptyBtn", rs === "empty", "Unmark empty (N): set back to to-do", "Mark image empty (N)");
  set("doneBtn", rs === "done", "Unmark done (Ctrl+Enter): set back to to-do", "Mark image done (Ctrl+Enter), then next image");
}

// ------------------------------------------------------------ zoom + adjust
function renderZoom(zoom, scale) {
  $("zoomPct").textContent = `${Math.round(scale * 100)}%`;
  $("zoomReset").classList.toggle("active", Math.abs(zoom - 1) > 0.001);
}

function inCanvas(e) {
  const r = $("canvasWrap").getBoundingClientRect();
  return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
}

// Ctrl+wheel (mouse) and pinch (touchpad; Chrome sends it as a wheel event with ctrlKey) zoom
// the image anywhere on the review screen and never the page. Plain wheel pans when zoomed.
// The zoom follows the wheel delta. Pinch sends many small deltas, so it applies each one at once,
// with a stronger rate; a mouse notch is one big delta, so it glides there over a few frames.
const NOTCH_DELTA = 40;       // a smaller delta (px) is a touchpad, not a mouse notch
const WHEEL_RATE = 0.002;     // 100 px of wheel = zoom x 1.22
const PINCH_RATE = 0.01;
window.addEventListener("wheel", (e) => {
  if (state.screen !== "review" || $("helpDialog").open) return;
  if (e.target.closest && e.target.closest(".bp-list, #imageList, #labelsPanel")) {
    if (!e.ctrlKey) return; // let those lists scroll
  }
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
  if (e.ctrlKey) {
    e.preventDefault();
    const dy = e.deltaY * unit;
    const over = inCanvas(e);
    const x = over ? e.clientX : null, y = over ? e.clientY : null;
    if (Math.abs(dy) < NOTCH_DELTA) canvas.zoomAt(Math.exp(-dy * PINCH_RATE), x, y);
    else canvas.wheelZoom(Math.exp(-dy * WHEEL_RATE), x, y);
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

// ------------------------------------------------------------ shortcuts dialog
function filterKeys() {
  const q = $("keySearch").value.trim().toLowerCase();
  let shown = 0;
  for (const row of $("keyList").children) {
    const hit = !q || row.textContent.toLowerCase().includes(q);
    row.hidden = !hit;
    if (hit) shown++;
  }
  $("keyNone").hidden = shown > 0;
}
function openHelp() {
  $("keySearch").value = "";
  filterKeys();
  $("helpDialog").showModal();
  $("keySearch").focus();
}

// ------------------------------------------------------------ events
$("tabProjects").addEventListener("click", () => showScreen("projects"));
$("tabSetup").addEventListener("click", () => showScreen("setup"));
$("tabReview").addEventListener("click", () => showScreen("review"));
$("projectSelect").addEventListener("change", (e) => openProject(e.target.value));
$("statusFilter").addEventListener("change", () => loadImages(0));
$("imgPrev").addEventListener("click", () => stepImage(-1));
$("imgNext").addEventListener("click", () => stepImage(1));
$("stripToggle").addEventListener("click", () => setStripOpen($("strip").classList.contains("collapsed")));
$("labelsBtn").addEventListener("click", () => setLabelsOpen($("labelsPanel").hidden));
$("toolDrag").addEventListener("click", () => setTool("drag"));
$("toolBox").addEventListener("click", () => setTool("box"));
$("toolPolygon").addEventListener("click", () => setTool("polygon"));
$("undoBtn").addEventListener("click", () => runHistory("undo"));
$("redoBtn").addEventListener("click", () => runHistory("redo"));
$("repeatBtn").addEventListener("click", repeatPrevious);
$("emptyBtn").addEventListener("click", () => toggleReview("empty"));
$("doneBtn").addEventListener("click", () => toggleReview("done"));
$("zoomIn").addEventListener("click", () => canvas.zoomAt(1.25));
$("zoomOut").addEventListener("click", () => canvas.zoomAt(0.8));
$("zoomReset").addEventListener("click", () => canvas.resetView());
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
$("bpSave").addEventListener("click", savePopup);
$("bpHead").addEventListener("mousedown", startPopupDrag);
$("bpDelete").addEventListener("click", deleteBox);
// Same as the A / R keys (and undone with U).
$("approveBtn").addEventListener("click", () => setStatus("accepted"));
$("rejectBtn").addEventListener("click", () => setStatus("rejected"));
$("bpClass").addEventListener("input", (e) => { pop.filter = e.target.value; pop.hl = 0; renderPopList(); });
$("bpClass").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); savePopup(); }
  else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancelPopup(); }
  else if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); movePopHighlight(e.key === "ArrowDown" ? 1 : -1); }
});
$("helpBtn").addEventListener("click", openHelp);
$("keysBtn").addEventListener("click", openHelp);
$("keySearch").addEventListener("input", filterKeys);
$("banner").addEventListener("click", () => ($("banner").hidden = true));
$("reviewerInput").addEventListener("change", (e) => lsSet("review.reviewer", e.target.value.trim()));

document.addEventListener("keydown", (e) => {
  if (state.screen !== "review") return; // review shortcuts only on the review screen
  const tag = (e.target.tagName || "").toLowerCase();
  // Never while typing in a text field (incl. the popup's class field, which has its own keys).
  if (tag === "input" || tag === "textarea" || e.target.isContentEditable || $("helpDialog").open) return;
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
    else if (key === "c" && !e.shiftKey) { if (!copyBox()) return; }
    else if (key === "v" && !e.shiftKey) pasteBoxes();
    else if (key === "enter" && !e.shiftKey) toggleReview("done");
    else return;
    e.preventDefault();
    return;
  }
  if (canvas.polyDraft && !e.altKey && ["Enter", "Backspace", "Escape"].includes(k)) {
    e.preventDefault();
    if (k === "Enter") canvas.closeDraft();
    else if (k === "Backspace") canvas.removeDraftPoint();
    else canvas.cancelPolygon();
    return;
  }
  if (tag === "select" && !["Tab", "Escape"].includes(k)) return;
  if (e.altKey || e.metaKey) return;
  const onButton = tag === "button";
  let handled = true;
  if (k === "a" || k === "A") setStatus("accepted");
  else if (k === "r" || k === "R") setStatus("rejected");
  else if (k === "u" || k === "U") unreview();
  else if (k === "Delete" || k === "Backspace") deleteBox();
  else if (k === "Tab") cycleBox(e.shiftKey ? -1 : 1);
  else if (k === "ArrowRight") stepImage(1);
  else if (k === "ArrowLeft") stepImage(-1);
  else if ((k === "ArrowDown" || k === "ArrowUp") && popupOpen()) movePopHighlight(k === "ArrowDown" ? 1 : -1);
  else if (k === "Enter" && popupOpen() && !onButton) savePopup();
  else if (k === "d" || k === "D") setTool("drag");
  else if (k === "b" || k === "B") setTool("box");
  else if (k === "g" || k === "G") setTool("polygon");
  else if (k === "n" || k === "N") toggleReview("empty");
  else if (k === "p" || k === "P") repeatPrevious();
  else if (k === "+" || k === "=") canvas.zoomAt(1.25);
  else if (k === "-" || k === "_") canvas.zoomAt(0.8);
  else if (k === "0") canvas.resetView();
  else if (k === "1") canvas.zoomToActual();
  else if (k === "Escape") {
    if (!$("adjustPop").hidden) closeAdjust();
    else if (canvas.cancelDraw()) { /* dropped the box being dragged out */ }
    else if (popupOpen()) cancelPopup();
    else if (state.tool !== "drag") setTool("drag");
    else select(null);
  }
  else if (k === "?") openHelp();
  else handled = false;
  if (handled) { e.preventDefault(); if (tag === "select" || onButton) e.target.blur(); }
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
  $("reviewerInput").value = lsGet("review.reviewer") || "";
  setLabelsOpen(lsGet("review.labels") !== "0");
  setStripOpen(lsGet("review.strip") !== "0");
  setTool("drag");
  renderHistory();
  renderZoom(1, 1);
  showScreen("projects"); // the start page
  loadProjects();
})();
