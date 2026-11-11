// projects.js - the Projects start page: list, create (#1), rename (#17), delete (#18), and
// open a project in Setup or Review; export a zip (#14), push to Roboflow (#21) and Train on Colab (#23 - #26). #17 and #18 are requested contract changes (see
// docs/agents/ui.md); a server without them answers 405 and the page says so.
// Uses globals from app.js / setup.js: $, state, guard, showBanner, showScreen, loadProjects,
// openProject, refreshProjectPicker, setup, renderStepper.
"use strict";

const projectsPage = {
  list: [],
  renaming: null,   // project id with the rename field open
  deleting: null,   // project id with the delete confirmation open
  panel: null,      // {pid, kind: "export" | "push" | "train"}: the open panel
  train: {},        // pid -> Train panel state, see trainState()
  exportOpts: { format: "yolo", include_images: false },
  ready: {},        // pid -> {done, empty} image counts (for the push confirmation)
  pushJobs: {},     // pid -> latest roboflow_push Job, polled with #10 until done / failed
};
const PUSH_POLL_MS = 1500;

const ICON_EDIT = `<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M10.5 2.5l3 3L6 13H3v-3z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>`;
const ICON_TRASH = `<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>`;

function plural(n, one, many = one + "s") { return `${n} ${n === 1 ? one : many}`; }

function fmtDate(iso) {
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

// A 405 means the server does not have the endpoint yet (requested contract change).
function notOnServer(e, what) {
  if (e instanceof ApiError && e.httpStatus === 405) {
    showBanner(`This server can't ${what} yet (needs the backend change in docs/agents/ui.md).`);
    return true;
  }
  return false;
}

async function loadProjectsPage() {
  const data = await guard(() => api.listProjects());
  if (!data) return;
  projectsPage.list = data.projects.slice().sort((a, b) => b.id - a.id); // newest first
  renderProjectsPage();
}

function renderProjectsPage() {
  const ul = $("projectList");
  ul.innerHTML = "";
  $("projectsEmpty").hidden = projectsPage.list.length > 0;
  for (const p of projectsPage.list) ul.appendChild(projectRow(p));
  const focus = ul.querySelector("[data-autofocus]");
  if (focus) { focus.focus(); if (focus.select) focus.select(); }
}

function projectRow(p) {
  const li = document.createElement("li");
  li.className = "proj" + (p.id === state.projectId ? " current" : "");
  const c = p.counts;
  const boxes = c.auto_accepted + c.needs_review + c.accepted + c.rejected + c.edited;
  const pct = boxes ? Math.round(((boxes - c.needs_review) / boxes) * 100) : 0;
  li.innerHTML = `
    <div class="p-main">
      <div class="p-title"></div>
      <div class="p-meta muted num-tab"></div>
    </div>
    <div class="p-progress">
      <div class="bar"><div></div></div>
      <span class="muted num-tab p-left"></span>
    </div>
    <div class="p-actions">
      <button type="button" class="p-export" aria-expanded="false">Export</button>
      <button type="button" class="p-push" aria-expanded="false">Push to Roboflow</button>
      <button type="button" class="p-train" aria-expanded="false">Train</button>
      <button type="button" class="p-setup">Setup</button>
      <button type="button" class="primary p-review">Review &rarr;</button>
    </div>`;
  li.querySelector(".p-meta").textContent =
    `${plural(c.images, "image")} · created ${fmtDate(p.created_at)}`;
  li.querySelector(".bar > div").style.width = pct + "%";
  li.querySelector(".p-left").textContent = !boxes ? "No boxes yet"
    : c.needs_review ? `${plural(c.needs_review, "box", "boxes")} to review` : "All boxes reviewed";
  li.querySelector(".p-progress").title = boxes
    ? `${boxes - c.needs_review} / ${boxes} boxes reviewed (${pct}%)` : "Run detection in Setup first";

  const title = li.querySelector(".p-title");
  if (projectsPage.renaming === p.id) title.appendChild(renameForm(p));
  else {
    title.innerHTML = `<button type="button" class="p-name"></button>` +
      `<button type="button" class="icon ghost p-ren" aria-label="Rename" title="Rename">${ICON_EDIT}</button>` +
      `<button type="button" class="icon ghost p-del" aria-label="Delete" title="Delete">${ICON_TRASH}</button>`;
    title.querySelector(".p-name").textContent = p.name;
    title.querySelector(".p-name").title = `Open ${p.name} in Review`;
    title.querySelector(".p-name").addEventListener("click", () => openFromList(p.id, "review"));
    title.querySelector(".p-ren").addEventListener("click", () => {
      projectsPage.renaming = p.id; projectsPage.deleting = null; projectsPage.panel = null; renderProjectsPage();
    });
    title.querySelector(".p-del").addEventListener("click", () => {
      projectsPage.deleting = projectsPage.deleting === p.id ? null : p.id;
      projectsPage.renaming = null;
      projectsPage.panel = null;
      renderProjectsPage();
    });
  }
  li.dataset.pid = p.id;
  const panel = projectsPage.panel && projectsPage.panel.pid === p.id ? projectsPage.panel.kind : null;
  li.querySelector(".p-export").setAttribute("aria-expanded", String(panel === "export"));
  li.querySelector(".p-push").setAttribute("aria-expanded", String(panel === "push"));
  li.querySelector(".p-export").addEventListener("click", () => togglePanel(p.id, "export"));
  li.querySelector(".p-push").addEventListener("click", () => togglePanel(p.id, "push"));
  li.querySelector(".p-train").setAttribute("aria-expanded", String(panel === "train"));
  li.querySelector(".p-train").addEventListener("click", () => togglePanel(p.id, "train"));
  li.querySelector(".p-setup").addEventListener("click", () => openFromList(p.id, "setup"));
  li.querySelector(".p-review").addEventListener("click", () => openFromList(p.id, "review"));
  if (projectsPage.deleting === p.id) li.appendChild(deleteConfirm(p, boxes));
  if (panel === "export") li.appendChild(exportPanel(p));
  if (panel === "push") li.appendChild(pushPanel(p));
  if (panel === "train") li.appendChild(trainPanel(p));
  return li;
}

// ---------------------------------------------------------------- export (#14) / push (#21)
function togglePanel(pid, kind) {
  const open = projectsPage.panel && projectsPage.panel.pid === pid && projectsPage.panel.kind === kind;
  projectsPage.panel = open ? null : { pid, kind };
  projectsPage.renaming = null;
  projectsPage.deleting = null;
  if (!open && kind === "push") { delete projectsPage.ready[pid]; loadReadyCount(pid); }
  if (!open && kind === "train") openTrain(pid);
  renderProjectsPage();
}

function apiMessage(e) { return e instanceof ApiError ? e.message : String(e); }

// Small menu: format + "include images", then one POST that downloads the zip.
function exportPanel(p) {
  const f = document.createElement("form");
  f.className = "p-panel";
  f.innerHTML = `
    <div class="row">
      <span class="seg" role="radiogroup" aria-label="Format">
        <label><input type="radio" name="fmt" value="yolo"> YOLO</label>
        <label><input type="radio" name="fmt" value="yolo-seg"> YOLO-seg</label>
        <label><input type="radio" name="fmt" value="coco"> COCO</label>
      </span>
      <label class="check"><input type="checkbox" class="p-incl"> Include images</label>
      <span class="spacer"></span>
      <button type="button" class="p-cancel">Close</button>
      <button type="submit" class="primary">Download zip</button>
    </div>
    <p class="muted p-note">Only images marked done or empty are exported, with their approved boxes.</p>
    <p class="p-msg" role="status" hidden></p>`;
  const o = projectsPage.exportOpts;
  f.querySelector(`input[value="${o.format}"]`).checked = true;
  f.querySelector(".p-incl").checked = o.include_images;
  f.addEventListener("change", () => {
    o.format = f.querySelector("input[name=fmt]:checked").value;
    o.include_images = f.querySelector(".p-incl").checked;
  });
  f.querySelector(".p-cancel").addEventListener("click", () => togglePanel(p.id, "export"));
  f.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); togglePanel(p.id, "export"); } });
  const msg = f.querySelector(".p-msg");
  const go = f.querySelector("button[type=submit]");
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    go.disabled = true;
    msg.hidden = false;
    msg.className = "p-msg muted";
    msg.textContent = o.include_images ? "Building the zip (with images, this can take a while)..." : "Building the zip...";
    try {
      const { blob, filename } = await api.exportZip(p.id, { format: o.format, include_images: o.include_images });
      downloadBlob(blob, filename);
      msg.textContent = `Downloaded ${filename}.`;
    } catch (err) {
      console.error(err);
      msg.className = "p-msg err";
      msg.textContent = apiMessage(err);   // e.g. the 409 "no done or empty images" message as-is
    }
    go.disabled = false;
  });
  return f;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Done + empty images = what #21 sends (minus images already pushed, which the server skips).
async function loadReadyCount(pid) {
  try {
    const [d, e] = await Promise.all([
      api.listImages(pid, { reviewStatus: "done", limit: 1 }),
      api.listImages(pid, { reviewStatus: "empty", limit: 1 })]);
    projectsPage.ready[pid] = { done: d.total, empty: e.total };
  } catch (err) {
    console.error(err);
    projectsPage.ready[pid] = { error: apiMessage(err) };
  }
  if (projectsPage.panel && projectsPage.panel.pid === pid && projectsPage.panel.kind === "push") renderProjectsPage();
}

function pushRunning(job) { return !!job && (job.status === "queued" || job.status === "running"); }

// Confirmation (how many images go) -> start #21 -> progress from #10 until done / failed.
// The API key stays on the server (.env); this page never shows or asks for it.
function pushPanel(p) {
  const f = document.createElement("form");
  f.className = "p-panel";
  f.innerHTML = `
    <p class="p-ask"></p>
    <div class="p-job" hidden>
      <div class="bar"><div></div></div>
      <div class="muted num-tab p-jobtext"></div>
    </div>
    <p class="p-msg err" role="status" hidden></p>
    <div class="row">
      <span class="spacer"></span>
      <button type="button" class="p-cancel">Close</button>
      <button type="submit" class="primary">Push</button>
    </div>`;
  const ask = f.querySelector(".p-ask");
  const go = f.querySelector("button[type=submit]");
  const r = projectsPage.ready[p.id];
  const job = projectsPage.pushJobs[p.id];
  if (!r) { ask.textContent = "Counting done and empty images..."; go.disabled = true; }
  else if (r.error) { ask.textContent = `Could not count images: ${r.error}`; }
  else {
    const n = r.done + r.empty;
    ask.textContent = `Push ${plural(n, "image")} (${r.done} done, ${r.empty} empty) with their approved boxes ` +
      `to the Roboflow project set in the server's .env? Images pushed before are skipped.`;
  }
  if (job) renderPushJob(f, job);
  if (pushRunning(job)) go.disabled = true;
  f.querySelector(".p-cancel").addEventListener("click", () => togglePanel(p.id, "push"));
  f.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); togglePanel(p.id, "push"); } });
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (pushRunning(projectsPage.pushJobs[p.id])) return;
    go.disabled = true;
    const msg = f.querySelector(".p-msg");
    msg.hidden = true;
    let started;
    try { started = await api.startPush(p.id); }
    catch (err) {
      console.error(err);
      msg.textContent = apiMessage(err);   // 409 as-is, e.g. "Roboflow settings missing from .env: ..."
      msg.hidden = false;
      go.disabled = false;
      return;
    }
    projectsPage.pushJobs[p.id] = started;
    renderPushJob(f, started);
    pollPush(p.id, started.id, 0);
  });
  return f;
}

function renderPushJob(root, job) {
  const box = root.querySelector(".p-job");
  box.hidden = false;
  const pct = job.total ? Math.round((job.done / job.total) * 100) : (job.status === "done" ? 100 : 0);
  box.querySelector(".bar > div").style.width = pct + "%";
  const t = box.querySelector(".p-jobtext");
  t.classList.toggle("err", job.status === "failed");
  t.textContent =
    job.status === "queued" ? "Push queued..." :
    job.status === "running" ? `Pushing: ${job.done} / ${job.total} images` :
    job.status === "done" ? (job.total ? `Push done: ${plural(job.total, "image")} sent to Roboflow.`
                                       : "Nothing new to push: every image was pushed before.") :
    `Push failed after ${job.done} / ${job.total} images: ${job.error || "unknown error"}. ` +
    `Press Push again to continue (pushed images are skipped).`;
}

async function pollPush(pid, jid, errors) {
  let job;
  try { job = await api.getJob(jid); }
  catch (err) {
    console.error(err);
    if (errors >= 4) { showBanner(`Lost track of push job ${jid}: ${apiMessage(err)}`); return; }
    setTimeout(() => pollPush(pid, jid, errors + 1), PUSH_POLL_MS);
    return;
  }
  projectsPage.pushJobs[pid] = job;
  const li = document.querySelector(`#projectList li[data-pid="${pid}"]`);
  const panel = li && li.querySelector(".p-panel .p-job") ? li.querySelector(".p-panel") : null;
  if (panel) {
    renderPushJob(panel, job);
    panel.querySelector("button[type=submit]").disabled = pushRunning(job);
  }
  if (pushRunning(job)) setTimeout(() => pollPush(pid, jid, 0), PUSH_POLL_MS);
  else if (job.status === "done") showBanner(`Roboflow push finished (${plural(job.total, "image")}).`, "info");
}

// ---------------------------------------------------------------- Train on Colab (#23 - #26)
// Download a training package (#23), train in Colab, upload best.pt (#24), list / delete (#25, #26).
// An uploaded model becomes a detector in Setup's model choice (#16 ?project_id=).
const TRAIN_MAX_BYTES = 200 * 1024 * 1024;   // #24 limit, checked here too to fail fast

function trainState(pid) {
  if (!projectsPage.train[pid]) {
    projectsPage.train[pid] = {
      task: "detect",
      weights: null,     // null = loading, [TrainedModel], or {error}
      file: null,        // the chosen .pt File
      upload: null,      // fraction 0..1 while uploading
      building: false,   // #23 request running
      pkgMsg: null,      // {text, err}
      upMsg: null,       // {text, err, mismatch: details}
      confirmDel: null,  // wid with the delete confirmation open
    };
  }
  return projectsPage.train[pid];
}

function trainOpen(pid) {
  return !!projectsPage.panel && projectsPage.panel.pid === pid && projectsPage.panel.kind === "train";
}

function rerenderTrain(pid) { if (trainOpen(pid)) renderProjectsPage(); }

function openTrain(pid) {
  const st = trainState(pid);
  st.pkgMsg = st.upMsg = null;
  st.confirmDel = null;
  loadWeightsList(pid);
}

async function loadWeightsList(pid) {
  const st = trainState(pid);
  try { st.weights = (await api.listWeights(pid)).weights; }
  catch (err) { console.error(err); st.weights = { error: apiMessage(err) }; }
  rerenderTrain(pid);
}

function fmtBytes(n) {
  return n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : n >= 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`;
}

function trainPanel(p) {
  const st = trainState(p.id);
  const f = document.createElement("form");
  f.className = "p-panel p-train-panel";
  f.innerHTML = `
    <div class="row">
      <span class="seg" role="radiogroup" aria-label="Task">
        <label><input type="radio" name="task" value="detect"> Detect (boxes)</label>
        <label><input type="radio" name="task" value="segment"> Segment (polygons)</label>
      </span>
      <span class="spacer"></span>
      <button type="button" class="p-cancel">Close</button>
      <button type="button" class="primary t-pkg">Download training package</button>
    </div>
    <p class="p-msg t-pkgmsg" role="status" hidden></p>
    <ol class="t-steps">
      <li>Download the training package (only images marked done or empty, with their approved boxes).</li>
      <li>Open <code>train.ipynb</code> from the zip in Colab; Runtime &gt; Change runtime type &gt; GPU.</li>
      <li>Run the 3 cells: upload (pick the zip), train, download (saves <code>best.pt</code>).</li>
      <li>Upload <code>best.pt</code> below with the same task. It then shows up as a model in Setup &gt; Run detection.</li>
    </ol>
    <div class="row t-up">
      <label class="field">Weights (.pt) <input type="file" accept=".pt" class="t-file"></label>
      <span class="muted t-fname"></span>
      <span class="spacer"></span>
      <button type="button" class="primary t-upload">Upload weights</button>
    </div>
    <p class="muted p-note">A .pt file can run code when the server loads it: upload only weights you trained yourself.</p>
    <div class="p-job t-prog" hidden><div class="bar"><div></div></div><div class="muted num-tab t-progtext"></div></div>
    <div class="p-msg t-upmsg" role="status" hidden></div>
    <h3 class="t-head">Uploaded models</h3>
    <div class="t-list"></div>`;

  // task choice
  // Both tasks are always offered; #23 answers 409 for segment without polygons (shown as is).
  f.querySelector(`input[value="${st.task}"]`).checked = true;
  f.querySelectorAll("input[name=task]").forEach((r) => r.addEventListener("change", () => {
    st.task = f.querySelector("input[name=task]:checked").value;
  }));

  const showMsg = (el, m) => {
    el.hidden = !m;
    if (!m) return;
    el.classList.toggle("err", !!m.err);
    el.classList.toggle("muted", !m.err);
    el.textContent = m.text;
    if (m.mismatch) el.appendChild(mismatchLists(m.mismatch));
  };
  showMsg(f.querySelector(".t-pkgmsg"), st.pkgMsg);
  showMsg(f.querySelector(".t-upmsg"), st.upMsg);

  // #23 download
  const pkg = f.querySelector(".t-pkg");
  pkg.disabled = st.building;
  pkg.textContent = st.building ? "Building..." : "Download training package";
  pkg.addEventListener("click", async () => {
    if (st.building) return;
    st.building = true;
    st.pkgMsg = { text: "Building the training package (images included, this can take a while)..." };
    rerenderTrain(p.id);
    try {
      const { blob, filename } = await api.trainPackage(p.id, st.task);
      downloadBlob(blob, filename);
      st.pkgMsg = { text: `Downloaded ${filename}. Next: open train.ipynb in Colab.` };
    } catch (err) {
      console.error(err);
      st.pkgMsg = { text: apiMessage(err), err: true };   // 409s as the server writes them
    }
    st.building = false;
    rerenderTrain(p.id);
  });

  // #24 upload
  const fileInput = f.querySelector(".t-file");
  f.querySelector(".t-fname").textContent = st.file ? `${st.file.name} (${fmtBytes(st.file.size)})` : "";
  fileInput.disabled = st.upload != null;
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    fileInput.value = "";
    st.file = file || null;
    st.upMsg = null;
    if (file && !/\.pt$/i.test(file.name)) { st.upMsg = { text: `${file.name} is not a .pt file.`, err: true }; st.file = null; }
    else if (file && !file.size) { st.upMsg = { text: `${file.name} is empty.`, err: true }; st.file = null; }
    else if (file && file.size > TRAIN_MAX_BYTES) {
      st.upMsg = { text: `${file.name} is ${fmtBytes(file.size)}; the limit is 200 MB.`, err: true };
      st.file = null;
    }
    rerenderTrain(p.id);
  });
  const up = f.querySelector(".t-upload");
  up.disabled = !st.file || st.upload != null;
  up.textContent = st.upload != null ? "Uploading..." : "Upload weights";
  if (st.upload != null) {
    const box = f.querySelector(".t-prog");
    box.hidden = false;
    box.querySelector(".bar > div").style.width = Math.round(st.upload * 100) + "%";
    box.querySelector(".t-progtext").textContent = st.upload < 1
      ? `Uploading: ${Math.round(st.upload * 100)}%` : "Checking the weights on the server...";
  }
  up.addEventListener("click", () => uploadTrainWeights(p.id));

  // #25 list, #26 delete
  f.querySelector(".t-list").appendChild(weightsList(p.id, st));

  f.querySelector(".p-cancel").addEventListener("click", () => togglePanel(p.id, "train"));
  f.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); togglePanel(p.id, "train"); } });
  f.addEventListener("submit", (e) => e.preventDefault());
  return f;
}

// #24 class_mismatch details as two lists.
function mismatchLists(d) {
  const div = document.createElement("div");
  div.className = "t-mismatch";
  const list = (title, names) => {
    const box = document.createElement("div");
    const h = document.createElement("div");
    h.className = "muted";
    h.textContent = title;
    const ul = document.createElement("ul");
    for (const n of names.length ? names : ["(none)"]) {
      const li = document.createElement("li");
      li.textContent = n;
      ul.appendChild(li);
    }
    box.append(h, ul);
    return box;
  };
  div.append(list("Project classes missing in the model", d.missing_in_model || []),
             list("Model classes not in the project", d.extra_in_model || []));
  return div;
}

async function uploadTrainWeights(pid) {
  const st = trainState(pid);
  if (!st.file || st.upload != null) return;
  st.upload = 0;
  st.upMsg = null;
  rerenderTrain(pid);
  let last = 0;
  try {
    const w = await api.uploadWeights(pid, st.file, st.task, (frac) => {
      st.upload = frac;
      if (frac === 1 || frac - last >= 0.05) { last = frac; rerenderTrain(pid); }
    });
    st.file = null;
    st.upMsg = { text: `Uploaded as model "${w.name}" (${w.task}, ${w.class_names.length} classes). ` +
                       `Pick it in Setup > Run detection.` };
    showBanner(`Model ${w.name} uploaded.`, "info");
    await modelsChanged(pid);
  } catch (err) {
    console.error(err);
    st.upMsg = { text: apiMessage(err), err: true,
                 mismatch: err instanceof ApiError && err.code === "class_mismatch" ? err.details : null };
  }
  st.upload = null;
  rerenderTrain(pid);
}

// The weights list changed: reload it, and Setup's model choice if this project is open there.
async function modelsChanged(pid) {
  await loadWeightsList(pid);
  if (setup.project && setup.project.id === pid) await loadModels();
}

function weightsList(pid, st) {
  const box = document.createElement("div");
  if (st.weights === null) { box.className = "muted"; box.textContent = "Loading..."; return box; }
  if (st.weights.error) { box.className = "err"; box.textContent = `Could not load the models: ${st.weights.error}`; return box; }
  if (!st.weights.length) { box.className = "muted"; box.textContent = "No uploaded models yet."; return box; }
  const t = document.createElement("table");
  t.className = "grid t-models";
  t.innerHTML = "<thead><tr><th>Model</th><th>Task</th><th>Classes</th><th>Image size</th><th>File</th>" +
                "<th>Uploaded</th><th></th></tr></thead><tbody></tbody>";
  for (const w of st.weights) {
    const tr = document.createElement("tr");
    const cells = [w.name, w.task, w.class_names.join(", "), w.imgsz != null ? String(w.imgsz) : "-",
                   `${w.file_name} (${fmtBytes(w.size_bytes)})`, fmtDate(w.uploaded_at)];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    const td = document.createElement("td");
    td.className = "t-act";
    if (st.confirmDel === w.id) {
      td.innerHTML = `<button type="button" class="danger t-yes"></button> <button type="button" class="t-no">Cancel</button>`;
      td.querySelector(".t-yes").textContent = `Delete ${w.name}`;
      td.querySelector(".t-yes").addEventListener("click", () => deleteTrainWeights(pid, w));
      td.querySelector(".t-no").addEventListener("click", () => { st.confirmDel = null; rerenderTrain(pid); });
    } else {
      td.innerHTML = `<button type="button" class="icon ghost t-del" aria-label="Delete" title="Delete">${ICON_TRASH}</button>`;
      td.querySelector(".t-del").setAttribute("aria-label", `Delete ${w.name}`);
      td.querySelector(".t-del").addEventListener("click", () => { st.confirmDel = w.id; rerenderTrain(pid); });
    }
    tr.appendChild(td);
    t.tBodies[0].appendChild(tr);
  }
  box.appendChild(t);
  return box;
}

async function deleteTrainWeights(pid, w) {
  const st = trainState(pid);
  try { await api.deleteWeights(w.id); }
  catch (err) {
    console.error(err);
    st.upMsg = { text: apiMessage(err), err: true };   // 409 while a job uses the model, as-is
    st.confirmDel = null;
    rerenderTrain(pid);
    return;
  }
  st.confirmDel = null;
  st.upMsg = { text: `Deleted model ${w.name}.` };
  await modelsChanged(pid);
  rerenderTrain(pid);
}

function renameForm(p) {
  const f = document.createElement("form");
  f.className = "p-rename";
  f.innerHTML = `<input type="text" size="24" aria-label="New name" data-autofocus>` +
    `<button type="submit" class="primary">Save</button><button type="button" class="p-cancel">Cancel</button>`;
  const input = f.querySelector("input");
  input.value = p.name;
  const cancel = () => { projectsPage.renaming = null; renderProjectsPage(); };
  input.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); cancel(); } });
  f.querySelector(".p-cancel").addEventListener("click", cancel);
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = input.value.trim();
    if (!name) { showBanner("The name can't be empty."); return; }
    if (name === p.name) { cancel(); return; }
    let updated;
    try { updated = await api.renameProject(p.id, name); }
    catch (err) {
      console.error(err);
      if (!notOnServer(err, "rename projects")) showBanner(err instanceof ApiError ? `${err.code}: ${err.message}` : String(err));
      return;
    }
    projectsPage.renaming = null;
    if (state.project && state.project.id === p.id) state.project.name = updated.name;
    if (setup.project && setup.project.id === p.id) {
      setup.project.name = updated.name;
      $("setupProjectName").textContent = updated.name;
      renderStepper();
    }
    await refreshProjectPicker();
    showBanner(`Renamed to "${updated.name}".`, "info");
    await loadProjectsPage();
  });
  return f;
}

// Deleting needs the project name typed in, since it removes every image and box for good.
function deleteConfirm(p, boxes) {
  const div = document.createElement("form");
  div.className = "p-confirm";
  div.innerHTML = `
    <p></p>
    <div class="row">
      <label class="field">Type the name to confirm <input type="text" size="20" data-autofocus></label>
      <span class="spacer"></span>
      <button type="button" class="p-cancel">Cancel</button>
      <button type="submit" class="danger" disabled>Delete project</button>
    </div>`;
  div.querySelector("p").textContent =
    `Delete "${p.name}"? This removes ${plural(p.counts.images, "image")} and ${plural(boxes, "box", "boxes")}, ` +
    `and can't be undone.`;
  const input = div.querySelector("input");
  input.setAttribute("aria-label", `Type ${p.name} to confirm`);
  const go = div.querySelector(".danger");
  input.addEventListener("input", () => { go.disabled = input.value.trim() !== p.name; });
  const cancel = () => { projectsPage.deleting = null; renderProjectsPage(); };
  input.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); cancel(); } });
  div.querySelector(".p-cancel").addEventListener("click", cancel);
  div.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (input.value.trim() !== p.name) return;
    go.disabled = true;
    try { await api.deleteProject(p.id); }
    catch (err) {
      console.error(err);
      if (!notOnServer(err, "delete projects")) showBanner(err instanceof ApiError ? `${err.code}: ${err.message}` : String(err));
      go.disabled = false;
      return;
    }
    projectsPage.deleting = null;
    showBanner(`Deleted project "${p.name}".`, "info");
    if (p.id === state.projectId) {
      try { localStorage.removeItem("review.project"); } catch (err) { /* ignore */ }
      state.projectId = null;
      await loadProjects();   // opens another project, or clears Setup / Review if none is left
    } else await refreshProjectPicker();
    await loadProjectsPage();
  });
  return div;
}

async function openFromList(pid, screen) {
  if (pid !== state.projectId) await openProject(pid);
  showScreen(screen);
}

$("projectCreateForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("projectCreateName").value.trim();
  if (!name) { showBanner("Type a project name first."); $("projectCreateName").focus(); return; }
  const p = await guard(() => api.createProject(name));
  if (!p) return;
  $("projectCreateName").value = "";
  showBanner(`Project "${p.name}" created. Next: upload images.`, "info");
  await loadProjects(p.id);   // selects + opens it in the top bar
  showScreen("setup");        // Setup uses the selected project and opens on step 2 Upload
});

document.addEventListener("screen-shown", (e) => { if (e.detail === "projects") loadProjectsPage(); });
document.addEventListener("project-opened", () => { if (state.screen === "projects") loadProjectsPage(); });

// app.js showed the start page before this file was loaded, so fill it now.
if (state.screen === "projects") loadProjectsPage();
