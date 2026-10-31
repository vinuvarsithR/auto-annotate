// projects.js - the Projects start page: list, create (#1), rename (#17), delete (#18), and
// open a project in Setup or Review. #17 and #18 are requested contract changes (see
// docs/agents/ui.md); a server without them answers 405 and the page says so.
// Uses globals from app.js / setup.js: $, state, guard, showBanner, showScreen, loadProjects,
// openProject, refreshProjectPicker, setup, renderStepper.
"use strict";

const projectsPage = {
  list: [],
  renaming: null,   // project id with the rename field open
  deleting: null,   // project id with the delete confirmation open
};

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
      projectsPage.renaming = p.id; projectsPage.deleting = null; renderProjectsPage();
    });
    title.querySelector(".p-del").addEventListener("click", () => {
      projectsPage.deleting = projectsPage.deleting === p.id ? null : p.id;
      projectsPage.renaming = null;
      renderProjectsPage();
    });
  }
  li.querySelector(".p-setup").addEventListener("click", () => openFromList(p.id, "setup"));
  li.querySelector(".p-review").addEventListener("click", () => openFromList(p.id, "review"));
  if (projectsPage.deleting === p.id) li.appendChild(deleteConfirm(p, boxes));
  return li;
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
  showScreen("setup");
  await loadProjects(p.id);   // opens it; Setup jumps to the first unfinished step (upload)
});

document.addEventListener("screen-shown", (e) => { if (e.detail === "projects") loadProjectsPage(); });
document.addEventListener("project-opened", () => { if (state.screen === "projects") loadProjectsPage(); });

// app.js showed the start page before this file was loaded, so fill it now.
if (state.screen === "projects") loadProjectsPage();
