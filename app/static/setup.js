// setup.js - the Setup screen: create project (1), upload images (4), edit classes (7, 8),
// run detection (9) and poll the job (10); bulk import (22) in step 2; the model list comes from endpoint 16.
// Shown as one guided 4-step flow (create, upload, classes, run); uploads start as soon as
// files are dropped or chosen, and thumbnails appear as each batch lands.
// Uses globals from app.js: $, state, guard, showBanner, showScreen, loadProjects, openProject.
"use strict";

const UPLOAD_BATCH = 10;   // files per POST, so one bad file does not lose a big upload
const POLL_MS = 1500;      // job status poll interval
const MAX_TILES = 120;     // upload thumbnails shown per upload (the rest are counted)
const MODE_LABELS = { text: "Text prompts", examples: "Examples (your boxes)" };
const NOTHING_RAN = "no class has examples, so nothing was run. Draw or accept some boxes first.";

const setup = {
  project: null,       // Project from the last "project-opened" event
  dirty: false,        // class table edited but not saved
  uploading: false,
  jobId: null,         // job being polled
  lastJob: null,       // last Job seen for this project (step 4 summary)
  modes: {},           // detector name -> its modes (#16), default first
  resultJob: null,     // id of the job whose result is on screen
  savedClasses: 0,     // classes saved on the server for this project
  step: 1,             // step on screen (1-4)
  tileUrls: [],        // object URLs of upload thumbnails (revoked on the next upload)
};

// ------------------------------------------------------------ project
// Setup always works on the project selected in the top bar (state.project, set by app.js).
function onProjectOpened(p) {
  const changed = (setup.project && setup.project.id) !== (p && p.id);
  setup.project = p;
  if (changed) {
    setup.lastJob = null;
    clearJobResult();
    clearTiles();
    if (!imp.sending && !imp.jobId) { setImportParts([], ""); $("importBox").hidden = true; $("importResult").hidden = true; }
  }
  renderProjectBox();
  updateButtons();
  if (changed) loadModels();
  if (p) loadClasses(p.id);
  else renderClassRows([]);
  // A selected project already covers step 1: go straight to Upload.
  if (changed) showStep(p ? 2 : 1);
}

// Step 1: "Project: NAME" when one is selected (with a small "create a different project"
// link), else the create form.
function renderProjectBox(showCreate = false) {
  const p = setup.project;
  $("currentProject").hidden = !p;
  $("createBox").hidden = !!p && !showCreate;
  $("setupProjectName").textContent = p ? p.name : "-";
  $("setupImageCount").textContent = p ? p.counts.images : 0;
}

// Opening Setup re-reads the selected project, in case a "project-opened" event was missed.
function onSetupShown() {
  const p = state.project && state.project.id === state.projectId ? state.project : null;
  if ((setup.project && setup.project.id) !== (p && p.id)) onProjectOpened(p);
  else { renderProjectBox(); showStep(p ? 2 : 1); }
}

// ------------------------------------------------------------ steps
const STEP_NEXT = { 1: "Next: upload", 2: "Next: classes", 3: "Next: run" };

function stepDone(n) {
  const p = setup.project;
  if (!p) return false;
  if (n === 1) return true;
  if (n === 2) return p.counts.images > 0;
  if (n === 3) return setup.savedClasses > 0 && !setup.dirty;
  const boxes = ["auto_accepted", "needs_review", "accepted", "rejected", "edited"]
    .reduce((t, k) => t + (p.counts[k] || 0), 0);
  const j = setup.lastJob;
  return (!!j && j.status === "done" && !nothingRan(j)) || boxes > 0;
}

function showStep(n) {
  if (!setup.project) n = 1;
  setup.step = n;
  document.querySelectorAll("#setup .step").forEach((el) => { el.hidden = Number(el.dataset.step) !== n; });
  renderStepper();
}

function renderStepper() {
  const p = setup.project;
  document.querySelectorAll("#stepper button").forEach((b) => {
    const k = Number(b.dataset.step);
    b.classList.toggle("current", k === setup.step);
    b.classList.toggle("done", k !== setup.step && stepDone(k));
    b.disabled = k > 1 && !p;
    if (k === setup.step) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    b.querySelector(".num").textContent = k !== setup.step && stepDone(k) ? "\u2713" : String(k);
  });
  $("sum1").textContent = p ? `Project: ${p.name}` : "No project";
  $("sum2").textContent = p ? `${p.counts.images} image${p.counts.images === 1 ? "" : "s"}` : "-";
  $("sum3").textContent = !p ? "-" : setup.dirty ? "Unsaved changes" :
    `${setup.savedClasses} class${setup.savedClasses === 1 ? "" : "es"}`;
  $("sum3").classList.toggle("warn", !!p && setup.dirty);
  const j = setup.lastJob;
  $("sum4").textContent = !p ? "-" : !j ? (stepDone(4) ? "Has boxes" : "Not run yet") :
    nothingRan(j) ? "Nothing run" : j.status === "done" ? "Done" : j.status === "failed" ? "Failed" :
    preparingExamples(j) ? "Preparing" : `${j.done} / ${j.total}`;
  $("stepBack").hidden = setup.step === 1;
  $("stepNext").hidden = setup.step === 4;
  $("stepNext").textContent = STEP_NEXT[setup.step] || "Next";
  $("stepNext").disabled = !p;
  // After a finished run (or when the project already has boxes): straight to review.
  $("openReviewBtn").hidden = !p || !!setup.jobId || !stepDone(4);
}

async function createProject(ev) {
  ev.preventDefault();
  const name = $("newProjectName").value.trim();
  if (!name) { showBanner("Type a project name first."); return; }
  const p = await guard(() => api.createProject(name));
  if (!p) return;
  $("newProjectName").value = "";
  showBanner(`Project "${p.name}" created.`, "info");
  await loadProjects(p.id); // fills the picker and opens the new project
}

// ------------------------------------------------------------ upload
function setBar(barId, textId, fraction, text) {
  $(barId).style.width = Math.round(Math.max(0, Math.min(1, fraction)) * 100) + "%";
  $(textId).textContent = text;
}

function clearTiles() {
  for (const u of setup.tileUrls) URL.revokeObjectURL(u);
  setup.tileUrls = [];
  $("uploadThumbs").innerHTML = "";
}

// One tile per file (up to MAX_TILES): dashed while sending, then the picture or the skip reason.
function addTiles(files) {
  const ul = $("uploadThumbs");
  const tiles = [];
  for (const f of files) {
    if (ul.querySelectorAll("li[data-name]").length >= MAX_TILES) {
      let more = ul.querySelector("li.more");
      if (!more) { more = document.createElement("li"); more.className = "more skipped"; more.dataset.n = "0"; ul.appendChild(more); }
      more.dataset.n = String(Number(more.dataset.n) + 1);
      more.textContent = `+${more.dataset.n} more`;
      tiles.push(null);
      continue;
    }
    const li = document.createElement("li");
    li.className = "sending";
    li.dataset.name = f.name;
    li.title = f.name;
    li.innerHTML = `<span class="tag"></span>`;
    li.querySelector(".tag").textContent = f.name;
    ul.appendChild(li);
    tiles.push(li);
  }
  return tiles;
}

// Batches of up to UPLOAD_BATCH files, never two files with the same name in one batch, so each
// file's result in a response is found by its name without doubt (the response has no per-file order).
function uploadBatches(files) {
  const batches = [];
  let cur = [], names = new Set();
  for (const f of files) {
    if (cur.length >= UPLOAD_BATCH || names.has(f.name)) { batches.push(cur); cur = []; names = new Set(); }
    cur.push(f);
    names.add(f.name);
  }
  if (cur.length) batches.push(cur);
  return batches;
}

// What happened to one file of a batch: { ok: true } or { ok: false, reason }.
// Renamed and skipped are checked first: a saved image's new name can equal another file's name.
function fileOutcome(name, res) {
  if (res.renamed.some((r) => r.from === name)) return { ok: true };
  const s = res.skipped.find((x) => x.file_name === name);
  if (s) return { ok: false, reason: s.reason };
  if (res.images.some((i) => i.file_name === name)) return { ok: true };
  return { ok: false, reason: "not uploaded" };
}

function landTiles(files, tiles, res, error) {
  if (res) res = { images: res.images, skipped: res.skipped, renamed: res.renamed || [] };
  files.forEach((f, i) => {
    const li = tiles[i];
    if (!li) return;
    li.className = "";
    const out = res ? fileOutcome(f.name, res) : { ok: false, reason: "failed" };
    if (out.ok) {
      const url = URL.createObjectURL(f);
      setup.tileUrls.push(url);
      li.innerHTML = `<img alt="">`;
      li.querySelector("img").src = url;
    } else {
      const reason = out.reason;
      li.className = error ? "failed" : "skipped";
      li.innerHTML = `Skipped<span class="tag"></span>`;
      li.querySelector(".tag").textContent = `${f.name}: ${reason}`;
      li.title = `${f.name}: ${reason}`;
    }
  });
}

// "Added 12 - 2 duplicates skipped - 1 renamed: a.jpg -> a_2.jpg" (parts left out when 0).
function uploadSummary(added, skipped, renamed) {
  const dups = skipped.filter((s) => s.reason === "duplicate").length;
  const other = skipped.length - dups;
  const parts = [`Added ${added}`];
  if (dups) parts.push(`${dups} duplicate${dups === 1 ? "" : "s"} skipped`);
  if (other) parts.push(`${other} other skipped`);
  if (renamed.length) parts.push(`${renamed.length} renamed: ` + renamed.map((r) => `${r.from} -> ${r.to}`).join(", "));
  return parts.join(" - ");
}

async function uploadFiles(list) {
  const pid = setup.project && setup.project.id;
  const files = Array.from(list || []);
  if (!pid || !files.length) return;
  if (setup.uploading) { showBanner("An upload is still running. Wait for it to finish.", "info"); return; }
  if (imp.sending || imp.jobId) { showBanner("An import is still running. Wait for it to finish.", "info"); return; }
  setup.uploading = true;
  updateButtons();
  clearTiles();
  $("uploadBox").hidden = false;
  $("skippedList").innerHTML = "";
  let uploaded = 0, done = 0;
  const skipped = [], renamed = [];
  let failed = null;
  for (const batch of uploadBatches(files)) {
    if (failed) break;
    const report = (frac) => setBar("uploadBar", "uploadText", (done + frac * batch.length) / files.length,
                                    `Uploading ${Math.min(files.length, done + Math.round(frac * batch.length))} / ${files.length} files`);
    report(0);
    const tiles = addTiles(batch);
    try {
      const res = await api.uploadImages(pid, batch, report);
      uploaded += res.images.length;
      skipped.push(...res.skipped);
      renamed.push(...(res.renamed || []));
      landTiles(batch, tiles, res, null);
    } catch (e) {
      console.error(e);
      failed = e instanceof ApiError ? `${e.code}: ${e.message}` : String(e);
      landTiles(batch, tiles, null, failed);
    }
    done += batch.length;
  }
  const msg = uploadSummary(uploaded, skipped, renamed) +
              (failed ? `. Stopped after ${done} / ${files.length} files: ${failed}` : ".");
  setBar("uploadBar", "uploadText", done / files.length, msg);
  for (const s of skipped) {
    const li = document.createElement("li");
    li.textContent = `${s.file_name}: ${s.reason}`;
    $("skippedList").appendChild(li);
  }
  if (failed) showBanner(failed);
  setup.uploading = false;
  updateButtons();
  if (pid === state.projectId) await openProject(pid); // refresh counts + review image list
  // Uploads add to the project (same content = skipped as duplicate); say the new total.
  if (setup.project && setup.project.id === pid) {
    const n = setup.project.counts.images;
    setBar("uploadBar", "uploadText", done / files.length, `${msg} Project now has ${n} image${n === 1 ? "" : "s"}.`);
  }
}

// ------------------------------------------------------------ bulk import (#22)
// The picked set is a list of [File, partName]. partName = relative path for a folder
// (webkitRelativePath, or the dropped entry path), the plain file name otherwise.
const imp = { parts: [], label: "", jobId: null, sending: false };

function fmtBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
  return Math.max(1, Math.round(n / 1e3)) + " kB";
}

function setImportParts(parts, label) {
  imp.parts = parts;
  imp.label = label;
  const bytes = parts.reduce((t, [f]) => t + f.size, 0);
  $("importPicked").textContent = parts.length ? `${label} (${fmtBytes(bytes)})` : "Nothing picked.";
  $("importClearBtn").hidden = !parts.length;
  updateButtons();
}

// What a set of plain files means: one .zip, or loose files (images; #22 skips anything else).
function pickFiles(files) {
  if (!files.length) return;
  if (files.length === 1 && /\.zip$/i.test(files[0].name)) {
    setImportParts([[files[0], files[0].name]], `Zip: ${files[0].name}`);
  } else {
    setImportParts(files.map((f) => [f, f.name]), `${files.length} file${files.length === 1 ? "" : "s"}`);
  }
}

function pickFolder(files) {
  if (!files.length) { showBanner("That folder is empty."); return; }
  const top = (files[0].webkitRelativePath || "").split("/")[0];
  setImportParts(files.map((f) => [f, f.webkitRelativePath || f.name]),
                 `Folder ${top || "?"}: ${files.length} file${files.length === 1 ? "" : "s"}`);
}

// Dropped folders: walk the entries (readEntries returns at most ~100 per call, so loop).
function readAllEntries(dir) {
  const reader = dir.createReader();
  const out = [];
  return new Promise((resolve, reject) => {
    const next = () => reader.readEntries((batch) => {
      if (!batch.length) return resolve(out);
      out.push(...batch);
      next();
    }, reject);
    next();
  });
}

async function walkEntry(entry, out) {
  if (entry.isFile) {
    const f = await new Promise((res, rej) => entry.file(res, rej));
    out.push([f, entry.fullPath.replace(/^\/+/, "")]);
  } else if (entry.isDirectory) {
    for (const e of await readAllEntries(entry)) await walkEntry(e, out);
  }
}

async function onImportDrop(dt) {
  // Entries must be taken synchronously, before the first await (the list is cleared after).
  const entries = Array.from(dt.items || []).map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  const files = Array.from(dt.files || []);
  if (!entries.some((e) => e.isDirectory)) { pickFiles(files); return; }
  const parts = [];
  try {
    for (const e of entries) await walkEntry(e, parts);
  } catch (err) {
    showBanner(`Could not read the dropped folder: ${err.message || err}`);
    return;
  }
  if (!parts.length) { showBanner("That folder is empty."); return; }
  const dirs = entries.filter((e) => e.isDirectory).map((e) => e.name);
  setImportParts(parts, `Folder ${dirs.join(", ")}: ${parts.length} file${parts.length === 1 ? "" : "s"}`);
}

async function startImport() {
  const pid = setup.project && setup.project.id;
  if (!pid || !imp.parts.length || imp.sending || imp.jobId) return;
  if (setup.uploading) { showBanner("An upload is still running. Wait for it to finish.", "info"); return; }
  imp.sending = true;
  updateButtons();
  $("importBox").hidden = false;
  $("importResult").hidden = true;
  $("importBar").classList.remove("failed");
  const n = imp.parts.length;
  setBar("importBar", "importText", 0, `Sending ${n} file${n === 1 ? "" : "s"}...`);
  let job;
  try {
    job = await api.startImport(pid, imp.parts, (frac) =>
      setBar("importBar", "importText", frac, `Sending ${imp.label}: ${Math.round(frac * 100)}%`));
  } catch (e) {
    console.error(e);
    // #22 400s (bad zip, unsafe path, too large) are shown as the server wrote them.
    const msg = e instanceof ApiError ? `${e.httpStatus ? e.httpStatus + " " : ""}${e.code}: ${e.message}` : String(e);
    $("importBar").classList.add("failed");
    setBar("importBar", "importText", 1, "Import not started.");
    showImportError(msg);
    imp.sending = false;
    updateButtons();
    return;
  }
  imp.sending = false;
  imp.jobId = job.id;
  setImportParts([], "");
  renderImportJob(job);
  pollImport(job.id, 0);
}

function showImportError(msg) {
  const box = $("importResult");
  box.hidden = false;
  box.innerHTML = `<p class="import-err"></p>`;
  box.querySelector("p").textContent = msg;
}

function renderImportJob(job) {
  const c = job.config || {};
  const what = c.upload === "zip" ? c.file_name : `${c.count} files`;
  const text = `Import job ${job.id} (${what}): ${job.status}, ${job.done} / ${job.total} images` +
               (job.error ? ` - ${job.error}` : "");
  setBar("importBar", "importText", job.status === "done" ? 1 : job.progress, text);
  $("importBar").classList.toggle("failed", job.status === "failed");
}

async function pollImport(jid, errors) {
  let job;
  try {
    job = await api.getJob(jid);
  } catch (e) {
    console.error(e);
    if (errors + 1 >= 3) { // give up after 3 failed polls in a row
      showBanner(`Lost track of import job ${jid}: ${e.message}`);
      imp.jobId = null;
      updateButtons();
      return;
    }
    setTimeout(() => pollImport(jid, errors + 1), POLL_MS);
    return;
  }
  renderImportJob(job);
  if (job.status === "queued" || job.status === "running") {
    setTimeout(() => pollImport(jid, 0), POLL_MS);
    return;
  }
  imp.jobId = null;
  updateButtons();
  $("importResult").innerHTML = "";
  if (job.status === "failed") showImportError(`Import job ${jid} failed: ${job.error || "no message"}`);
  if (job.result) renderImportResult(job.result);
  if (job.project_id === state.projectId) {
    await openProject(job.project_id); // counts, review list, class legend
    if (setup.project && setup.project.id === job.project_id) loadClasses(job.project_id);
  }
}

// A <details> list: open when short, collapsed when long.
function detailsList(title, items, openUpTo = 10) {
  const d = document.createElement("details");
  d.open = items.length <= openUpTo;
  const s = document.createElement("summary");
  s.textContent = `${title} (${items.length})`;
  const ul = document.createElement("ul");
  for (const t of items) { const li = document.createElement("li"); li.textContent = t; ul.appendChild(li); }
  d.append(s, ul);
  return d;
}

function numTable(head, rows) {
  const t = document.createElement("table");
  t.className = "grid";
  if (head) {
    const tr = t.createTHead().insertRow();
    for (const h of head) { const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); }
  }
  const body = t.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const v of row) {
      const td = tr.insertCell();
      td.textContent = String(v);
      if (typeof v === "number") td.className = "n";
    }
  }
  return t;
}

function renderImportResult(r) {
  const box = $("importResult");
  box.hidden = false;
  const dups = r.duplicates || [], ren = r.renamed || [], skip = r.skipped || [];
  const created = r.classes_created || [];
  box.appendChild(numTable(null, [
    ["Format", r.format],
    ["Images added", r.images_added],
    ["Duplicates (not added)", dups.length],
    ["Renamed", ren.length],
    ["Classes created", created.length ? created.join(", ") : "none"],
    ["Skipped files", skip.length],
  ]));
  const byClass = Object.entries(r.boxes_by_class || {});
  box.appendChild(numTable(["Class", "Boxes imported"],
                           byClass.length ? byClass : [["No boxes imported", ""]]));
  if (dups.length) box.appendChild(detailsList("Duplicates", dups));
  if (ren.length) box.appendChild(detailsList("Renamed", ren.map((x) => `${x.from} -> ${x.to}`)));
  if (skip.length) box.appendChild(detailsList("Skipped", skip.map((x) => `${x.file_name}: ${x.reason}`)));
}

// ------------------------------------------------------------ classes
function scoreText(v) { return v == null ? "" : String(v); }

function addClassRow(c = { name: "", description: "", min_score: null, accept_score: null }) {
  const tr = document.createElement("tr");
  tr.innerHTML =
    `<td><input class="c-name" type="text" size="10"></td>` +
    `<td><input class="c-desc" type="text"></td>` +
    `<td><input class="c-min" type="number" min="0" max="1" step="0.01"></td>` +
    `<td><input class="c-acc" type="number" min="0" max="1" step="0.01"></td>` +
    `<td><button type="button" class="bad c-del" title="Remove this class" aria-label="Remove this class">&times;</button></td>`;
  tr.querySelector(".c-name").value = c.name;
  tr.querySelector(".c-desc").value = c.description;
  tr.querySelector(".c-min").value = scoreText(c.min_score);
  tr.querySelector(".c-acc").value = scoreText(c.accept_score);
  tr.querySelector(".c-del").addEventListener("click", () => { tr.remove(); setDirty(true); });
  $("classTable").tBodies[0].appendChild(tr);
  return tr;
}

function renderClassRows(classes) {
  $("classTable").tBodies[0].innerHTML = "";
  classes.forEach((c) => addClassRow(c));
  setup.savedClasses = classes.length;
  setDirty(false);
}

async function loadClasses(pid) {
  const data = await guard(() => api.listClasses(pid));
  if (data && setup.project && setup.project.id === pid) renderClassRows(data.classes);
  renderStepper();
}

function setDirty(on) {
  setup.dirty = on;
  $("classDirty").hidden = !on;
  renderStepper();
}

// Reads the table. Returns the list for PUT, or throws a message for the first bad cell.
function readClassRows() {
  const out = [];
  const seen = new Set();
  const rows = Array.from($("classTable").tBodies[0].rows);
  rows.forEach((tr, i) => {
    const name = tr.querySelector(".c-name").value.trim();
    const description = tr.querySelector(".c-desc").value.trim();
    const where = `row ${i + 1}${name ? ` (${name})` : ""}`;
    if (!name) throw `${where}: name is empty`;
    if (seen.has(name)) throw `${where}: name "${name}" is used twice`;
    if (!description) throw `${where}: description is empty`;
    seen.add(name);
    const score = (cls, label) => {
      const raw = tr.querySelector(cls).value.trim();
      if (raw === "") return null;
      const v = Number(raw);
      if (!Number.isFinite(v) || v < 0 || v > 1) throw `${where}: ${label} must be between 0 and 1`;
      return v;
    };
    out.push({ name, description, min_score: score(".c-min", "min score"),
               accept_score: score(".c-acc", "accept score") });
  });
  return out;
}

async function saveClasses() {
  const pid = setup.project && setup.project.id;
  if (!pid) return;
  let list;
  try { list = readClassRows(); } catch (msg) { showBanner(String(msg)); return; }
  const data = await guard(() => api.putClasses(pid, list));
  if (!data) return;
  renderClassRows(data.classes);
  showBanner(`Saved ${data.classes.length} class(es).`, "info");
  if (pid === state.projectId) await openProject(pid); // review screen class lists + legend
}

// ------------------------------------------------------------ detection job
// With a project open, #16 ?project_id= adds its uploaded models (#24) after the built-in detectors.
async function loadModels() {
  const sel = $("modelSelect");
  const pid = setup.project ? setup.project.id : null;
  const keep = sel.value;
  const data = await guard(() => api.listModels(pid));
  if ((setup.project ? setup.project.id : null) !== pid) return; // project changed meanwhile
  sel.innerHTML = "";
  for (const m of (data ? data.models : [])) {
    const o = document.createElement("option");
    o.value = o.textContent = m;
    sel.appendChild(o);
  }
  if (data && data.models.includes(keep)) sel.value = keep;
  const note = $("modelNote");
  note.hidden = !(data && data.fallback);
  note.textContent = "Model list not available from the server (endpoint 16 missing); showing defaults.";
  setup.modes = (data && data.modes) || {};
  renderModes();
  updateButtons();
}

// The mode choice of the selected detector. With fewer than 2 modes there is nothing to choose,
// so it is hidden; a single mode stays in the select and is still sent.
function renderModes() {
  const sel = $("modeSelect");
  const keep = sel.value;
  const modes = setup.modes[$("modelSelect").value] || [];
  sel.innerHTML = "";
  for (const m of modes) {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = MODE_LABELS[m] || m;
    sel.appendChild(o);
  }
  if (modes.includes(keep)) sel.value = keep;
  $("modeField").hidden = modes.length < 2;
  renderMasks();
}

// #9 masks: only owlv2 in examples mode takes it (any other model / mode answers 400).
function masksAllowed() { return $("modelSelect").value === "owlv2" && $("modeSelect").value === "examples"; }
function renderMasks() { $("masksField").hidden = !masksAllowed(); }

async function runDetection() {
  const pid = setup.project && setup.project.id;
  const model = $("modelSelect").value;
  if (!pid || !model || setup.jobId) return;
  if (setup.dirty) { showBanner("Save the class list first (it has unsaved changes)."); return; }
  const mode = $("modeSelect").value || null;
  const body = { model, mode, image_ids: null, options: {} };
  if (masksAllowed()) body.masks = $("masksCheck").checked ? "sam2" : "none";
  await startDetect(pid, body);
}

// #9 rerun: the server copies detector, mode and options from the project's last examples job
// and runs it on the images still `todo`. There is no job list endpoint, so the UI cannot know
// beforehand whether such a job exists; the server's 409 message says so plainly.
async function rerunExamples() {
  const pid = setup.project && setup.project.id;
  if (!pid || setup.jobId) return;
  await startDetect(pid, { rerun: true });
}

async function startDetect(pid, body) {
  let job;
  try {
    job = await api.startJob(pid, body);
  } catch (e) {
    console.error(e);
    // 409: the message is meant for people ("no images with review_status todo"), show it as is.
    showBanner(e instanceof ApiError ? (e.httpStatus === 409 ? e.message : `${e.code}: ${e.message}`) : String(e));
    return;
  }
  clearJobResult();
  setup.jobId = job.id;
  $("jobBox").hidden = false;
  renderJob(job);
  updateButtons();
  pollJob(job.id, 0);
}

// Worker step 2 of #9: running at 0, and result is only written once the embedding is finished.
function preparingExamples(job) {
  return job.status === "running" && job.config.mode === "examples" && !job.progress && !job.result;
}

// An examples job where no class had examples ends "done" without running any image.
function nothingRan(job) {
  return job.status === "done" && !!job.result && job.result.mode === "examples" &&
    !Object.keys(job.result.examples_by_class || {}).length;
}

function renderJob(job) {
  const mine = !!setup.project && job.project_id === setup.project.id;
  if (mine) { setup.lastJob = job; renderStepper(); }
  const c = job.config;
  const head = `Job ${job.id} (${c.mode ? `${c.model}, ${c.mode}` : c.model}` +
    `${c.rerun_of != null ? `, re-run of job ${c.rerun_of}` : ""})`;
  const text = preparingExamples(job) ? `${head}: preparing examples...` :
    nothingRan(job) ? `${head}: done, ${NOTHING_RAN}` :
    `${head}: ${job.status}, ${job.done} / ${job.total} images` + (job.error ? ` - ${job.error}` : "");
  setBar("jobBar", "jobText", job.status === "done" ? 1 : job.progress, text);
  $("jobBar").classList.toggle("failed", job.status === "failed");
  if (mine && job.result && job.result.mode === "examples" && setup.resultJob !== job.id) {
    setup.resultJob = job.id;
    renderExamplesResult(job.result);
  }
}

// Example counts and skipped classes, once: result never changes after #9 step 3.
function renderExamplesResult(r) {
  const box = $("jobResult");
  box.innerHTML = "";
  box.hidden = false;
  const used = Object.entries(r.examples_by_class || {});
  const skipped = r.skipped_classes || [];
  if (used.length) box.appendChild(numTable(["Class", "Examples used"], used));
  if (skipped.length) {
    box.appendChild(numTable(["Skipped class", "Why"], skipped.map((s) => [s.class_name, s.reason])));
  }
  if (r.masks) box.appendChild(para(r.masks === "sam2" ? "Outlines: SAM 2 (polygons)." : "Outlines: off (boxes only)."));
  const h = r.hook;
  if (!h) return;
  const withPoint = Object.keys(h.points || {}).length;
  box.appendChild(para(h.load_classes.length
    ? `Hook filter: load classes ${h.load_classes.join(", ")}, radius ${h.radius} px, ` +
      `${withPoint} video(s) with a hook point.`
    : "Hook filter: off (no load classes)."));
  const warn = h.warnings || [];
  if (warn.length) {
    const p = para(`${warn.length} video(s) have no hook point, so their load boxes were not filtered. ` +
      "Set one on the Review page (Labels panel, Hook points), then re-run.");
    p.className = "warn";
    box.appendChild(p);
    box.appendChild(numTable(["Video without hook point", "Images"], warn.map((w) => [w.sequence, w.images])));
  }
}

function para(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function clearJobResult() {
  setup.resultJob = null;
  $("jobResult").innerHTML = "";
  $("jobResult").hidden = true;
}

async function pollJob(jid, errors) {
  let job;
  try {
    job = await api.getJob(jid);
  } catch (e) {
    console.error(e);
    if (errors + 1 >= 3) { // give up after 3 failed polls in a row
      showBanner(`Lost track of job ${jid}: ${e.message}`);
      setup.jobId = null;
      updateButtons();
      return;
    }
    setTimeout(() => pollJob(jid, errors + 1), POLL_MS);
    return;
  }
  renderJob(job);
  if (job.status === "queued" || job.status === "running") {
    setTimeout(() => pollJob(jid, 0), POLL_MS);
    return;
  }
  setup.jobId = null;
  updateButtons();
  if (job.status === "failed") showBanner(`Job ${jid} failed: ${job.error || "no message"}`);
  else if (nothingRan(job)) showBanner(`Job ${jid}: ${NOTHING_RAN}`, "info");
  else showBanner(`Job ${jid} done: ${job.total} image(s) processed.`, "info");
  if (job.project_id === state.projectId) await openProject(job.project_id); // refresh image list
}

// ------------------------------------------------------------ buttons
function updateButtons() {
  const hasProject = !!(setup.project && setup.project.id);
  const importBusy = imp.sending || !!imp.jobId;
  $("fileInput").disabled = !hasProject || setup.uploading || importBusy;
  $("importFileInput").disabled = $("importDirInput").disabled = !hasProject || importBusy;
  $("importZone").classList.toggle("disabled", !hasProject);
  $("importBtn").disabled = !hasProject || !imp.parts.length || importBusy || setup.uploading;
  $("importBtn").textContent = imp.sending ? "Sending..." : imp.jobId ? "Importing..." : "Import";
  $("dropZone").classList.toggle("disabled", !hasProject);
  $("addClassBtn").disabled = !hasProject;
  $("saveClassesBtn").disabled = !hasProject;
  $("runBtn").disabled = !hasProject || !!setup.jobId || !$("modelSelect").value;
  $("runBtn").textContent = setup.jobId ? "Running..." : "Run on all images";
  // Shown when some detector has an examples mode; the server decides if there is a job to re-run.
  $("rerunBtn").hidden = !Object.values(setup.modes).some((m) => (m || []).includes("examples"));
  $("rerunBtn").disabled = !hasProject || !!setup.jobId;
  renderStepper();
}

// ------------------------------------------------------------ wiring
$("createForm").addEventListener("submit", createProject);
$("otherProjectBtn").addEventListener("click", () => { renderProjectBox(true); $("newProjectName").focus(); });
document.addEventListener("screen-shown", (e) => { if (e.detail === "setup") onSetupShown(); });
$("fileInput").addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = "";
  uploadFiles(files);
});
const drop = $("dropZone");
drop.addEventListener("dragover", (e) => {
  if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = setup.project ? "copy" : "none";
  drop.classList.add("over");
});
drop.addEventListener("dragleave", (e) => { if (!drop.contains(e.relatedTarget)) drop.classList.remove("over"); });
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (!setup.project) { showBanner("Create or pick a project first."); return; }
  uploadFiles(e.dataTransfer.files);
});
$("importFileInput").addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = "";
  pickFiles(files);
});
$("importDirInput").addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = "";
  pickFolder(files);
});
const importZone = $("importZone");
importZone.addEventListener("dragover", (e) => {
  if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = setup.project ? "copy" : "none";
  importZone.classList.add("over");
});
importZone.addEventListener("dragleave", (e) => { if (!importZone.contains(e.relatedTarget)) importZone.classList.remove("over"); });
importZone.addEventListener("drop", (e) => {
  e.preventDefault();
  importZone.classList.remove("over");
  if (!setup.project) { showBanner("Create or pick a project first."); return; }
  if (imp.sending || imp.jobId) { showBanner("An import is still running.", "info"); return; }
  onImportDrop(e.dataTransfer);
});
$("importBtn").addEventListener("click", startImport);
$("importClearBtn").addEventListener("click", () => setImportParts([], ""));
// A file dropped anywhere else would make the browser open it and leave the page.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());
$("stepper").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-step]");
  if (b && !b.disabled) showStep(Number(b.dataset.step));
});
$("stepBack").addEventListener("click", () => showStep(Math.max(1, setup.step - 1)));
$("stepNext").addEventListener("click", () => showStep(Math.min(4, setup.step + 1)));
$("addClassBtn").addEventListener("click", () => { addClassRow().querySelector(".c-name").focus(); setDirty(true); });
$("saveClassesBtn").addEventListener("click", saveClasses);
$("classTable").addEventListener("input", () => setDirty(true));
$("modelSelect").addEventListener("change", renderModes);
$("modeSelect").addEventListener("change", renderMasks);
$("runBtn").addEventListener("click", runDetection);
$("rerunBtn").addEventListener("click", rerunExamples);
$("openReviewBtn").addEventListener("click", () => showScreen("review"));
document.addEventListener("project-opened", (e) => onProjectOpened(e.detail));

(function initSetup() {
  showStep(1);
  updateButtons();
  loadModels();
})();
