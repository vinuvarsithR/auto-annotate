// setup.js - the Setup screen: create project (1), upload images (4), edit classes (7, 8),
// run detection (9) and poll the job (10); the model list comes from endpoint 16.
// Shown as one guided 4-step flow (create, upload, classes, run); uploads start as soon as
// files are dropped or chosen, and thumbnails appear as each batch lands.
// Uses globals from app.js: $, state, guard, showBanner, showScreen, loadProjects, openProject.
"use strict";

const UPLOAD_BATCH = 10;   // files per POST, so one bad file does not lose a big upload
const POLL_MS = 1500;      // job status poll interval
const MAX_TILES = 120;     // upload thumbnails shown per upload (the rest are counted)

const setup = {
  project: null,       // Project from the last "project-opened" event
  dirty: false,        // class table edited but not saved
  uploading: false,
  jobId: null,         // job being polled
  lastJob: null,       // last Job seen for this project (step 4 summary)
  savedClasses: 0,     // classes saved on the server for this project
  step: 1,             // step on screen (1-4)
  autoStep: true,      // pick the first unfinished step once this project's classes are known
  tileUrls: [],        // object URLs of upload thumbnails (revoked on the next upload)
};

// ------------------------------------------------------------ project
function onProjectOpened(p) {
  const changed = (setup.project && setup.project.id) !== (p && p.id);
  setup.project = p;
  if (changed) { setup.autoStep = true; setup.lastJob = null; clearTiles(); }
  $("setupProjectName").textContent = p ? p.name : "(none - create one)";
  $("setupImageCount").textContent = p ? p.counts.images : 0;
  updateButtons();
  if (p) loadClasses(p.id);
  else { renderClassRows([]); showStep(1); }
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
  return (setup.lastJob && setup.lastJob.status === "done") || boxes > 0;
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
  $("sum1").textContent = p ? p.name : "No project";
  $("sum2").textContent = p ? `${p.counts.images} image${p.counts.images === 1 ? "" : "s"}` : "-";
  $("sum3").textContent = !p ? "-" : setup.dirty ? "Unsaved changes" :
    `${setup.savedClasses} class${setup.savedClasses === 1 ? "" : "es"}`;
  $("sum3").classList.toggle("warn", !!p && setup.dirty);
  const j = setup.lastJob;
  $("sum4").textContent = !p ? "-" : !j ? (stepDone(4) ? "Has boxes" : "Not run yet") :
    j.status === "done" ? "Done" : j.status === "failed" ? "Failed" : `${j.done} / ${j.total}`;
  $("stepBack").hidden = setup.step === 1;
  $("stepNext").hidden = setup.step === 4;
  $("stepNext").textContent = STEP_NEXT[setup.step] || "Next";
  $("stepNext").disabled = !p;
  // After a finished run (or when the project already has boxes): straight to review.
  $("openReviewBtn").hidden = !p || !!setup.jobId || !stepDone(4);
}

// First step that is not done yet (or Run when all are).
function firstOpenStep() {
  for (const n of [1, 2, 3]) if (!stepDone(n)) return n;
  return 4;
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

function landTiles(files, tiles, res, error) {
  const ok = new Set(res ? res.images.map((i) => i.file_name) : []);
  const why = new Map(res ? res.skipped.map((s) => [s.file_name, s.reason]) : []);
  files.forEach((f, i) => {
    const li = tiles[i];
    if (!li) return;
    li.className = "";
    if (ok.has(f.name)) {
      const url = URL.createObjectURL(f);
      setup.tileUrls.push(url);
      li.innerHTML = `<img alt="">`;
      li.querySelector("img").src = url;
    } else {
      const reason = error ? "failed" : (why.get(f.name) || "not uploaded");
      li.className = error ? "failed" : "skipped";
      li.innerHTML = `Skipped<span class="tag"></span>`;
      li.querySelector(".tag").textContent = `${f.name}: ${reason}`;
      li.title = `${f.name}: ${reason}`;
    }
  });
}

async function uploadFiles(list) {
  const pid = setup.project && setup.project.id;
  const files = Array.from(list || []);
  if (!pid || !files.length) return;
  if (setup.uploading) { showBanner("An upload is still running. Wait for it to finish.", "info"); return; }
  setup.uploading = true;
  updateButtons();
  clearTiles();
  $("uploadBox").hidden = false;
  $("skippedList").innerHTML = "";
  let uploaded = 0, done = 0;
  const skipped = [];
  let failed = null;
  for (let i = 0; i < files.length && !failed; i += UPLOAD_BATCH) {
    const batch = files.slice(i, i + UPLOAD_BATCH);
    const report = (frac) => setBar("uploadBar", "uploadText", (done + frac * batch.length) / files.length,
                                    `Uploading ${Math.min(files.length, done + Math.round(frac * batch.length))} / ${files.length} files`);
    report(0);
    const tiles = addTiles(batch);
    try {
      const res = await api.uploadImages(pid, batch, report);
      uploaded += res.images.length;
      skipped.push(...res.skipped);
      landTiles(batch, tiles, res, null);
    } catch (e) {
      console.error(e);
      failed = e instanceof ApiError ? `${e.code}: ${e.message}` : String(e);
      landTiles(batch, tiles, null, failed);
    }
    done += batch.length;
  }
  const msg = `Uploaded ${uploaded} image(s)` + (skipped.length ? `, skipped ${skipped.length}` : "") +
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
  if (setup.project && setup.project.id === pid && setup.autoStep) {
    setup.autoStep = false;
    showStep(firstOpenStep());
  } else renderStepper();
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
async function loadModels() {
  const sel = $("modelSelect");
  const data = await guard(() => api.listModels());
  sel.innerHTML = "";
  for (const m of (data ? data.models : [])) {
    const o = document.createElement("option");
    o.value = o.textContent = m;
    sel.appendChild(o);
  }
  const note = $("modelNote");
  note.hidden = !(data && data.fallback);
  note.textContent = "Model list not available from the server (endpoint 16 missing); showing defaults.";
  updateButtons();
}

async function runDetection() {
  const pid = setup.project && setup.project.id;
  const model = $("modelSelect").value;
  if (!pid || !model || setup.jobId) return;
  if (setup.dirty) { showBanner("Save the class list first (it has unsaved changes)."); return; }
  const job = await guard(() => api.startJob(pid, { model, image_ids: null, options: {} }));
  if (!job) return;
  setup.jobId = job.id;
  $("jobBox").hidden = false;
  renderJob(job);
  updateButtons();
  pollJob(job.id, 0);
}

function renderJob(job) {
  if (setup.project && job.project_id === setup.project.id) { setup.lastJob = job; renderStepper(); }
  const text = `Job ${job.id} (${job.config.model}): ${job.status}, ${job.done} / ${job.total} images` +
               (job.error ? ` - ${job.error}` : "");
  setBar("jobBar", "jobText", job.status === "done" ? 1 : job.progress, text);
  $("jobBar").classList.toggle("failed", job.status === "failed");
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
  else showBanner(`Job ${jid} done: ${job.total} image(s) processed.`, "info");
  if (job.project_id === state.projectId) await openProject(job.project_id); // refresh image list
}

// ------------------------------------------------------------ buttons
function updateButtons() {
  const hasProject = !!(setup.project && setup.project.id);
  $("fileInput").disabled = !hasProject || setup.uploading;
  $("dropZone").classList.toggle("disabled", !hasProject);
  $("addClassBtn").disabled = !hasProject;
  $("saveClassesBtn").disabled = !hasProject;
  $("runBtn").disabled = !hasProject || !!setup.jobId || !$("modelSelect").value;
  $("runBtn").textContent = setup.jobId ? "Running..." : "Run on all images";
  renderStepper();
}

// ------------------------------------------------------------ wiring
$("createForm").addEventListener("submit", createProject);
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
$("runBtn").addEventListener("click", runDetection);
$("openReviewBtn").addEventListener("click", () => showScreen("review"));
document.addEventListener("project-opened", (e) => onProjectOpened(e.detail));

(function initSetup() {
  showStep(1);
  updateButtons();
  loadModels();
})();
