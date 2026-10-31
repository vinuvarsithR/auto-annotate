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
// preferId: open this project (used by the setup screen after "Create project").
async function loadProjects(preferId = null) {
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
  if (!data.projects.length) {
    showBanner("No projects yet. Create one on the Setup screen.", "info");
    document.dispatchEvent(new CustomEvent("project-opened", { detail: null }));
    return;
  }
  let saved = preferId != null ? String(preferId) : null;
  if (saved == null) {
    try { saved = localStorage.getItem("review.project"); } catch (e) { /* ignore */ }
  }
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
  // Lets setup.js follow the project picker and refresh its own panels.
  document.dispatchEvent(new CustomEvent("project-opened", { detail: state.project }));
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
  state.imageIndex = -1;
  buildImageList();
  if (state.images.length) {
    const i = selectIndex < 0 ? state.images.length - 1 : Math.min(selectIndex, state.images.length - 1);
    await openImage(i);
  } else {
    state.imageIndex = -1;
    state.image = null;
    state.anns = [];
    $("imageTitle").textContent = status ? `No images with ${STATUS_LABEL[status].toLowerCase()} boxes.` : "No images.";
    canvas.setAnnotations([]);
    canvas.setImage(null);
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
  state.imageIndex = i;
  const img = state.images[i];
  const token = ++state.loadToken;
  renderImageList();
  const cur = $("imageList").children[i];
  if (cur) cur.scrollIntoView({ block: "nearest" });
  $("imageTitle").textContent = `${img.file_name} · ${img.width}×${img.height} · id ${img.id}`;
  setDrawMode(false);
  canvas.setAnnotations([]);
  // Keep the box panel where it is while the next image loads, so it does not close and reopen.
  state.loadingImage = true;
  select(null);
  canvas.setImage(api.imageUrl(img), img.width, img.height);
  const data = await guard(() => api.getAnnotations(img.id));
  if (token !== state.loadToken) return;
  state.loadingImage = false;
  if (!data) { renderPanel(); return; }
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
}

function cycleBox(delta) {
  const list = state.anns.filter((a) => canvas.showRejected || a.status !== "rejected");
  if (!list.length) return;
  const i = list.findIndex((a) => a.id === state.selectedId);
  const next = i < 0 ? (delta > 0 ? 0 : list.length - 1) : (i + delta + list.length) % list.length;
  select(list[next].id, true, true);
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
  const before = { ...a };
  const updated = await patch(a, { status });
  if (!updated) return;
  if (status === "accepted") canvas.flashAccept(updated.id);
  else if (status === "rejected") canvas.fadeReject(before);
  // Jump to the next box that still needs review on this image, if any.
  const next = state.anns.find((x) => x.status === "needs_review");
  if (next) select(next.id, true, true);
  else showBanner("No more boxes to review on this image. Press → for the next one.", "info");
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
  $("drawBtn").innerHTML = on ? "Drawing… <kbd>Esc</kbd>" : "Draw box <kbd>N</kbd>";
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
$("boxClose").addEventListener("click", () => select(null));
$("helpBtn").addEventListener("click", () => $("helpDialog").showModal());
$("banner").addEventListener("click", () => ($("banner").hidden = true));
$("reviewerInput").addEventListener("change", (e) => {
  try { localStorage.setItem("review.reviewer", e.target.value.trim()); } catch (err) { /* ignore */ }
});

document.addEventListener("keydown", (e) => {
  if ($("layout").hidden) return; // review shortcuts are off while the setup screen shows
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
