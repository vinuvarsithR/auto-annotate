// app.js - wires the review page together (state, image list, side panel, keyboard).
"use strict";

const PAGE = 50;
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
};

const canvas = new BoxCanvas($("canvas"), {
  onSelect: (ann) => select(ann ? ann.id : null, false),
  onBoxChanged: (ann, bbox) => patch(ann, { bbox }),
  onBoxDrawn: (bbox) => createBox(bbox),
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

// ------------------------------------------------------------ projects
async function loadProjects() {
  const data = await guard(() => api.listProjects());
  if (!data) return;
  const sel = $("projectSelect");
  sel.innerHTML = "";
  for (const p of data.projects) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.name;
    sel.appendChild(o);
  }
  if (!data.projects.length) { showBanner("No projects yet.", "info"); return; }
  let saved = null;
  try { saved = localStorage.getItem("review.project"); } catch (e) { /* ignore */ }
  const pick = data.projects.find((p) => String(p.id) === saved) || data.projects[0];
  sel.value = pick.id;
  await openProject(pick.id);
}

async function openProject(pid) {
  state.projectId = Number(pid);
  try { localStorage.setItem("review.project", pid); } catch (e) { /* ignore */ }
  const cls = await guard(() => api.listClasses(pid));
  state.classes = cls ? cls.classes : [];
  fillClassSelect($("bClass"));
  fillClassSelect($("newClass"));
  renderLegend();
  state.offset = 0;
  await refreshProgress();
  await loadImages(0);
}

async function refreshProgress() {
  const pid = state.projectId;
  const [p, nr] = await Promise.all([
    guard(() => api.getProject(pid)),
    guard(() => api.listImages(pid, { status: "needs_review", offset: 0, limit: 1 })),
  ]);
  if (!p || pid !== state.projectId) return;
  state.project = p;
  const n = p.counts.images;
  const left = nr ? nr.total : 0;
  const c = p.counts;
  $("progress").innerHTML =
    `Reviewed <b>${n - left} / ${n}</b> images &nbsp;|&nbsp; boxes: ` +
    `<span class="st-needs_review">${c.needs_review} needs_review</span>, ` +
    `<span class="st-auto_accepted">${c.auto_accepted} auto</span>, ` +
    `<span class="st-accepted">${c.accepted} accepted</span>, ` +
    `<span class="st-edited">${c.edited} edited</span>, ` +
    `<span class="st-rejected">${c.rejected} rejected</span>`;
}

function renderLegend() {
  const el = $("legend");
  el.innerHTML = "";
  for (const c of state.classes) {
    const d = document.createElement("div");
    d.innerHTML = `<i></i><span></span>`;
    d.querySelector("i").style.background = classColor(c.name);
    d.querySelector("span").textContent = c.name;
    d.title = c.description;
    el.appendChild(d);
  }
}

// ------------------------------------------------------------ image list
async function loadImages(offset, selectIndex = 0) {
  const status = $("statusFilter").value || null;
  const data = await guard(() => api.listImages(state.projectId, { status, offset, limit: PAGE }));
  if (!data) return;
  state.images = data.images;
  state.total = data.total;
  state.offset = data.offset;
  renderImageList();
  if (state.images.length) {
    const i = selectIndex < 0 ? state.images.length - 1 : Math.min(selectIndex, state.images.length - 1);
    await openImage(i);
  } else {
    state.imageIndex = -1;
    state.image = null;
    state.anns = [];
    $("imageTitle").textContent = status ? `No images with ${status} boxes.` : "No images.";
    canvas.setAnnotations([]);
    canvas.setImage(null);
    select(null);
  }
}

function renderImageList() {
  const ul = $("imageList");
  ul.innerHTML = "";
  state.images.forEach((img, i) => {
    const li = document.createElement("li");
    if (i === state.imageIndex) li.className = "current";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = img.file_name;
    name.title = img.file_name;
    li.appendChild(name);
    const nr = img.counts ? img.counts.needs_review : 0;
    const badge = document.createElement("span");
    if (nr > 0) { badge.className = "count"; badge.textContent = nr; badge.title = "needs_review boxes"; }
    else { badge.className = "done"; badge.textContent = "✓"; badge.title = "nothing left to review"; }
    li.appendChild(badge);
    li.addEventListener("click", () => openImage(i));
    ul.appendChild(li);
  });
  const end = state.offset + state.images.length;
  $("pageInfo").textContent = state.total ? `${state.offset + 1}-${end} of ${state.total}` : "0";
  $("pagePrev").disabled = state.offset === 0;
  $("pageNext").disabled = end >= state.total;
}

async function openImage(i) {
  if (i < 0 || i >= state.images.length) return;
  state.imageIndex = i;
  const img = state.images[i];
  const token = ++state.loadToken;
  renderImageList();
  const cur = $("imageList").children[i];
  if (cur) cur.scrollIntoView({ block: "nearest" });
  $("imageTitle").textContent = `${img.file_name}  (${img.width}x${img.height}, id ${img.id})`;
  setDrawMode(false);
  canvas.setAnnotations([]);
  select(null);
  canvas.setImage(api.imageUrl(img), img.width, img.height);
  const data = await guard(() => api.getAnnotations(img.id));
  if (!data || token !== state.loadToken) return;
  state.image = data.image;
  state.anns = data.annotations;
  canvas.setAnnotations(state.anns);
  // Start on the first box that still needs review.
  const first = state.anns.find((a) => a.status === "needs_review");
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
  renderPanel();
  refreshProgress();
}

// ------------------------------------------------------------ boxes
function select(id, redraw = true) {
  state.selectedId = id;
  if (redraw) canvas.setSelected(id); else canvas.selectedId = id;
  renderPanel();
}

function selectedAnn() { return state.anns.find((a) => a.id === state.selectedId) || null; }

function renderPanel() {
  const a = selectedAnn();
  $("boxPanel").hidden = !a;
  $("boxEmpty").hidden = !!a;
  if (!a) return;
  $("bId").textContent = a.id;
  $("bStatus").innerHTML = `<span class="st-${a.status}">${a.status}</span>`;
  $("bScore").textContent = a.score == null ? "-" : a.score.toFixed(3);
  $("bSource").textContent = a.source;
  $("bBbox").textContent = a.bbox.map((v) => Math.round(v)).join(", ");
  $("bReviewer").textContent = a.reviewer || "-";
  $("bClass").value = a.class_id;
}

function cycleBox(delta) {
  const list = state.anns.filter((a) => canvas.showRejected || a.status !== "rejected");
  if (!list.length) return;
  const i = list.findIndex((a) => a.id === state.selectedId);
  const next = i < 0 ? (delta > 0 ? 0 : list.length - 1) : (i + delta + list.length) % list.length;
  select(list[next].id);
}

async function patch(ann, body) {
  const updated = await guard(() => api.patchAnnotation(ann.id, { ...body, reviewer: reviewer() }));
  if (!updated) { canvas.render(); return null; }
  const k = state.anns.findIndex((a) => a.id === updated.id);
  if (k >= 0) state.anns[k] = updated;
  afterChange();
  return updated;
}

async function setStatus(status) {
  const a = selectedAnn();
  if (!a) { showBanner("Select a box first (click it or press Tab).", "info"); return; }
  const updated = await patch(a, { status });
  if (!updated) return;
  // Jump to the next box that still needs review on this image, if any.
  const next = state.anns.find((x) => x.status === "needs_review");
  if (next) select(next.id);
  else showBanner("No more needs_review boxes on this image. → for next image.", "info");
}

async function changeClass(classId) {
  const a = selectedAnn();
  if (!a || a.class_id === classId) return;
  await patch(a, { class_id: classId });
}

async function createBox(bbox) {
  if (!state.image) return;
  const classId = Number($("newClass").value);
  if (!classId) { showBanner("No classes in this project.", "error"); return; }
  const created = await guard(() => api.createAnnotation(state.image.id,
    { class_id: classId, bbox, mask: null, reviewer: reviewer() }));
  setDrawMode(false);
  if (!created) return;
  state.anns.push(created);
  select(created.id);
  afterChange();
}

function setDrawMode(on) {
  state.drawMode = on;
  const cls = classById(Number($("newClass").value));
  canvas.setDrawMode(on, cls ? classColor(cls.name) : "#fff");
  $("drawBtn").classList.toggle("active", on);
  $("drawBtn").textContent = on ? "Drawing... (Esc to cancel)" : "Draw box (N)";
}

// ------------------------------------------------------------ events
$("projectSelect").addEventListener("change", (e) => openProject(e.target.value));
$("statusFilter").addEventListener("change", () => loadImages(0));
$("pagePrev").addEventListener("click", () => loadImages(Math.max(0, state.offset - PAGE)));
$("pageNext").addEventListener("click", () => loadImages(state.offset + PAGE));
$("acceptBtn").addEventListener("click", () => setStatus("accepted"));
$("rejectBtn").addEventListener("click", () => setStatus("rejected"));
$("bClass").addEventListener("change", (e) => changeClass(Number(e.target.value)));
$("newClass").addEventListener("change", () => { if (state.drawMode) setDrawMode(true); });
$("drawBtn").addEventListener("click", () => setDrawMode(!state.drawMode));
$("showRejected").addEventListener("change", (e) => {
  canvas.showRejected = e.target.checked;
  const a = selectedAnn();
  if (a && a.status === "rejected" && !e.target.checked) select(null); else canvas.render();
});
$("helpBtn").addEventListener("click", () => $("helpDialog").showModal());
$("banner").addEventListener("click", () => ($("banner").hidden = true));
$("reviewerInput").addEventListener("change", (e) => {
  try { localStorage.setItem("review.reviewer", e.target.value.trim()); } catch (err) { /* ignore */ }
});

document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || $("helpDialog").open) return;
  if (tag === "select" && !["Tab", "Escape"].includes(e.key)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key;
  let handled = true;
  if (k === "a" || k === "A") setStatus("accepted");
  else if (k === "r" || k === "R") setStatus("rejected");
  else if (k === "Tab") cycleBox(e.shiftKey ? -1 : 1);
  else if (k === "ArrowRight" || k === "d" || k === "D") stepImage(1);
  else if (k === "ArrowLeft" || k === "s" || k === "S") stepImage(-1);
  else if (k === "n" || k === "N") setDrawMode(!state.drawMode);
  else if (k === "Escape") { if (state.drawMode) setDrawMode(false); else select(null); }
  else if (k === "?") $("helpDialog").showModal();
  else if (/^[1-9]$/.test(k)) {
    const cls = state.classes[Number(k) - 1];
    if (!cls) handled = false;
    else if (selectedAnn()) changeClass(cls.id);
    else { $("newClass").value = cls.id; if (state.drawMode) setDrawMode(true); }
  } else handled = false;
  if (handled) { e.preventDefault(); if (tag === "select") e.target.blur(); }
});

// ------------------------------------------------------------ start
(function init() {
  $("mockBadge").hidden = !api.mock;
  if (api.mock) document.title += " (mock)";
  try { $("reviewerInput").value = localStorage.getItem("review.reviewer") || ""; } catch (e) { /* ignore */ }
  loadProjects();
})();
