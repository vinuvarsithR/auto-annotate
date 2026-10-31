// setup.js - the Setup screen: create project (1), upload images (4), edit classes (7, 8),
// run detection (9) and poll the job (10); the model list comes from endpoint 16.
// Uses globals from app.js: $, state, guard, showBanner, loadProjects, openProject.
"use strict";

const UPLOAD_BATCH = 10;   // files per POST, so one bad file does not lose a big upload
const POLL_MS = 1500;      // job status poll interval

const setup = {
  project: null,       // Project from the last "project-opened" event
  dirty: false,        // class table edited but not saved
  uploading: false,
  jobId: null,         // job being polled
};

// ------------------------------------------------------------ screen toggle
function showScreen(name) {
  const isSetup = name === "setup";
  $("setup").hidden = !isSetup;
  $("layout").hidden = isSetup;
  $("tabSetup").classList.toggle("active", isSetup);
  $("tabReview").classList.toggle("active", !isSetup);
  try { localStorage.setItem("ui.screen", name); } catch (e) { /* ignore */ }
}

// ------------------------------------------------------------ project
function onProjectOpened(p) {
  setup.project = p;
  $("setupProjectName").textContent = p ? p.name : "(none - create one)";
  $("setupImageCount").textContent = p ? p.counts.images : 0;
  updateButtons();
  if (p) loadClasses(p.id);
  else { renderClassRows([]); showScreen("setup"); }
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

async function uploadFiles() {
  const pid = setup.project && setup.project.id;
  const files = Array.from($("fileInput").files || []);
  if (!pid || !files.length) return;
  setup.uploading = true;
  updateButtons();
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
    try {
      const res = await api.uploadImages(pid, batch, report);
      uploaded += res.images.length;
      skipped.push(...res.skipped);
    } catch (e) {
      console.error(e);
      failed = e instanceof ApiError ? `${e.code}: ${e.message}` : String(e);
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
  $("fileInput").value = "";
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
    `<td><button type="button" class="bad c-del" title="Remove this class">x</button></td>`;
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
  setDirty(false);
}

async function loadClasses(pid) {
  const data = await guard(() => api.listClasses(pid));
  if (data && setup.project && setup.project.id === pid) renderClassRows(data.classes);
}

function setDirty(on) {
  setup.dirty = on;
  $("classDirty").hidden = !on;
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
    out.push({ name, description, min_score: score(".c-min", "min_score"),
               accept_score: score(".c-acc", "accept_score") });
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
  $("uploadBtn").disabled = !hasProject || setup.uploading || !($("fileInput").files || []).length;
  $("addClassBtn").disabled = !hasProject;
  $("saveClassesBtn").disabled = !hasProject;
  $("runBtn").disabled = !hasProject || !!setup.jobId || !$("modelSelect").value;
  $("runBtn").textContent = setup.jobId ? "Running..." : "Run on all images";
}

// ------------------------------------------------------------ wiring
$("tabSetup").addEventListener("click", () => showScreen("setup"));
$("tabReview").addEventListener("click", () => showScreen("review"));
$("createForm").addEventListener("submit", createProject);
$("fileInput").addEventListener("change", updateButtons);
$("uploadBtn").addEventListener("click", uploadFiles);
$("addClassBtn").addEventListener("click", () => { addClassRow().querySelector(".c-name").focus(); setDirty(true); });
$("saveClassesBtn").addEventListener("click", saveClasses);
$("classTable").addEventListener("input", () => setDirty(true));
$("runBtn").addEventListener("click", runDetection);
document.addEventListener("project-opened", (e) => onProjectOpened(e.detail));

(function initSetup() {
  let screen = "review";
  try { screen = localStorage.getItem("ui.screen") || "review"; } catch (e) { /* ignore */ }
  showScreen(screen);
  updateButtons();
  loadModels();
})();
