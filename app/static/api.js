// api.js - one small data layer used by the UI.
// Real mode calls the FastAPI backend under /api (docs/CONTRACT.md section 2).
// Mock mode (?mock=1) reads docs/fixtures/api_samples.json once and answers
// from an in-memory copy, so edits show up until the page is reloaded.
"use strict";

const STATUSES = ["auto_accepted", "needs_review", "accepted", "rejected", "edited"];
const MOCK = new URLSearchParams(location.search).get("mock") === "1";
const FIXTURES_URL = "/docs/fixtures/api_samples.json";

class ApiError extends Error {
  constructor(code, message, httpStatus) {
    super(message);
    this.code = code;
    this.httpStatus = httpStatus;
  }
}

function countStatuses(anns) {
  const c = {};
  for (const s of STATUSES) c[s] = 0;
  for (const a of anns) if (a.status in c) c[a.status] += 1;
  return c;
}

// ---------------------------------------------------------------- real mode
async function http(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch("/api" + path, opts);
  } catch (e) {
    throw new ApiError("network", "cannot reach the server: " + e.message, 0);
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty or non-JSON body */ }
  if (!res.ok) {
    const err = (data && data.error) || {};
    throw new ApiError(err.code || "http_" + res.status,
                       err.message || res.status + " " + res.statusText, res.status);
  }
  return data;
}

const realApi = {
  mock: false,
  listProjects: () => http("GET", "/projects"),
  getProject: (pid) => http("GET", `/projects/${pid}`),
  listImages(pid, { status = null, offset = 0, limit = 50 } = {}) {
    const q = new URLSearchParams({ offset, limit });
    if (status) q.set("status", status);
    return http("GET", `/projects/${pid}/images?${q}`);
  },
  listClasses: (pid) => http("GET", `/projects/${pid}/classes`),
  getAnnotations: (iid) => http("GET", `/images/${iid}/annotations`),
  createAnnotation: (iid, body) => http("POST", `/images/${iid}/annotations`, body),
  patchAnnotation: (aid, body) => http("PATCH", `/annotations/${aid}`, body),
  imageUrl: (img) => img.url,

  // ---- requested contract changes (docs/agents/ui.md): #17 rename, #18 delete project,
  // #19 delete box. Until the backend has them the server answers 405 / 404.
  renameProject: (pid, name) => http("PATCH", `/projects/${pid}`, { name }),
  deleteProject: (pid) => http("DELETE", `/projects/${pid}`),
  deleteAnnotation: (aid) => http("DELETE", `/annotations/${aid}`),

  // ---- setup screen (endpoints 1, 4, 8, 9, 10, 16)
  createProject: (name) => http("POST", "/projects", { name }),
  // Upload one batch of File objects. onProgress(fraction 0..1) follows the bytes sent.
  uploadImages(pid, files, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      for (const f of files) form.append("files", f, f.name);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/projects/${pid}/images`);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
      xhr.onerror = () => reject(new ApiError("network", "cannot reach the server", 0));
      xhr.onload = () => {
        let data = null;
        try { data = JSON.parse(xhr.responseText); } catch (e) { /* non-JSON body */ }
        if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
        const err = (data && data.error) || {};
        reject(new ApiError(err.code || "http_" + xhr.status,
                            err.message || `${xhr.status} ${xhr.statusText}`, xhr.status));
      };
      xhr.send(form);
    });
  },
  putClasses: (pid, classes) => http("PUT", `/projects/${pid}/classes`, { classes }),
  startJob: (pid, body) => http("POST", `/projects/${pid}/jobs`, body),
  getJob: (jid) => http("GET", `/jobs/${jid}`),
  // Endpoint 16 (contract on main): GET /api/models -> {"models": [...]} = list_detectors().
  // If an older backend answers 404, fall back to the one detector that always exists.
  async listModels() {
    try {
      const d = await http("GET", "/models");
      return { models: d.models, fallback: false };
    } catch (e) {
      if (e.httpStatus === 404) return { models: FALLBACK_MODELS.slice(), fallback: true };
      throw e;
    }
  },
};

const FALLBACK_MODELS = ["fake"];
const MOCK_MODELS = ["fake"]; // same as the backend today (contract #16)

// ---------------------------------------------------------------- mock mode
const clone = (x) => JSON.parse(JSON.stringify(x));
const now = () => new Date().toISOString().replace(/\.\d+Z$/, "Z");
let mockDb = null;

async function loadMock() {
  if (mockDb) return mockDb;
  const res = await fetch(FIXTURES_URL);
  if (!res.ok) throw new ApiError("mock", `cannot load ${FIXTURES_URL} (${res.status})`, res.status);
  const fx = await res.json();
  mockDb = {
    projects: clone(fx["GET /api/projects"].projects),
    images: clone(fx["GET /api/projects/1/images"].images),
    // Fixture classes all belong to project 1; keep the project id on each row (stripped on output).
    classes: clone(fx["GET /api/projects/1/classes"].classes).map((c) => ({ ...c, project_id: 1 })),
    annotations: clone(fx["all_annotations"]),
    jobs: [],
  };
  const nextOf = (rows) => Math.max(0, ...rows.map((r) => r.id)) + 1;
  mockDb.nextId = nextOf(mockDb.annotations);
  mockDb.nextProjectId = nextOf(mockDb.projects);
  mockDb.nextImageId = nextOf(mockDb.images);
  mockDb.nextClassId = nextOf(mockDb.classes);
  mockDb.nextJobId = 100;
  return mockDb;
}

function mockClasses(db, pid) {
  return db.classes.filter((c) => c.project_id === Number(pid))
                   .map(({ project_id, ...c }) => clone(c));
}

function mockDelay(ms) { return new Promise((r) => setTimeout(r, ms)); }

// Pixel size of an uploaded file; falls back to the crane size if the browser cannot decode it.
async function imageSize(file) {
  try {
    const bmp = await createImageBitmap(file);
    const out = { width: bmp.width, height: bmp.height };
    bmp.close();
    return out;
  } catch (e) {
    return { width: 1280, height: 720 };
  }
}

// One step of a mock detect job. Acts like the contract's "fake" detector: 2 boxes per image
// (first two classes, scores 0.9 and 0.3). Earlier unreviewed model boxes are replaced.
function mockDetectOne(db, job, imageId) {
  const img = db.images.find((i) => i.id === imageId);
  if (!img) return;
  db.annotations = db.annotations.filter((a) => !(a.image_id === imageId && a.source !== "human" &&
    (a.status === "auto_accepted" || a.status === "needs_review")));
  const classes = db.classes.filter((c) => c.project_id === job.project_id).slice(0, 2);
  const scores = [0.9, 0.3];
  const t = now();
  classes.forEach((c, k) => {
    const score = scores[k];
    if (c.min_score != null && score < c.min_score) return;
    db.annotations.push({
      id: db.nextId++, image_id: imageId, class_id: c.id, class_name: c.name,
      bbox: [Math.round(img.width * (0.15 + 0.4 * k)), Math.round(img.height * 0.3),
             Math.round(img.width * 0.2), Math.round(img.height * 0.15)],
      mask: null, score, source: job.config.model,
      status: c.accept_score != null && score >= c.accept_score ? "auto_accepted" : "needs_review",
      job_id: job.id, reviewer: null, created_at: t, updated_at: t,
    });
  });
}

function stripJob(job) {
  const { _ids, ...out } = job;
  return clone(out);
}

function mockNotFound(what, id) {
  throw new ApiError("not_found", `${what} ${id} not found`, 404);
}

function mockImageWithCounts(db, img) {
  const out = clone(img);
  out.counts = countStatuses(db.annotations.filter((a) => a.image_id === img.id));
  return out;
}

function mockProject(db, pid) {
  const p = db.projects.find((x) => x.id === Number(pid));
  if (!p) mockNotFound("project", pid);
  const imgs = db.images.filter((i) => i.project_id === p.id);
  const ids = new Set(imgs.map((i) => i.id));
  const out = clone(p);
  out.counts = { images: imgs.length,
                 ...countStatuses(db.annotations.filter((a) => ids.has(a.image_id))) };
  return out;
}

const mockApi = {
  mock: true,
  async listProjects() {
    const db = await loadMock();
    return { projects: db.projects.map((p) => mockProject(db, p.id)) };
  },
  async getProject(pid) { return mockProject(await loadMock(), pid); },
  async listImages(pid, { status = null, offset = 0, limit = 50 } = {}) {
    const db = await loadMock();
    mockProject(db, pid);
    let imgs = db.images.filter((i) => i.project_id === Number(pid))
                        .map((i) => mockImageWithCounts(db, i));
    if (status) imgs = imgs.filter((i) => i.counts[status] > 0);
    return { images: imgs.slice(offset, offset + limit), total: imgs.length, offset, limit };
  },
  async listClasses(pid) {
    const db = await loadMock();
    mockProject(db, pid);
    return { classes: mockClasses(db, pid) };
  },
  async getAnnotations(iid) {
    const db = await loadMock();
    const img = db.images.find((i) => i.id === Number(iid));
    if (!img) mockNotFound("image", iid);
    return { image: mockImageWithCounts(db, img),
             annotations: clone(db.annotations.filter((a) => a.image_id === img.id)) };
  },
  async createAnnotation(iid, body) {
    const db = await loadMock();
    const img = db.images.find((i) => i.id === Number(iid));
    if (!img) mockNotFound("image", iid);
    const cls = db.classes.find((c) => c.id === body.class_id);
    if (!cls) throw new ApiError("bad_request", `class ${body.class_id} not found`, 400);
    const t = now();
    const a = { id: db.nextId++, image_id: img.id, class_id: cls.id, class_name: cls.name,
                bbox: body.bbox.map(Number), mask: body.mask || null, score: null,
                source: "human", status: "accepted", job_id: null,
                reviewer: body.reviewer || null, created_at: t, updated_at: t };
    db.annotations.push(a);
    return clone(a);
  },
  async patchAnnotation(aid, body) {
    const db = await loadMock();
    const a = db.annotations.find((x) => x.id === Number(aid));
    if (!a) mockNotFound("annotation", aid);
    let edited = false;
    if (body.bbox) { a.bbox = body.bbox.map(Number); edited = true; }
    if ("mask" in body) { a.mask = body.mask; edited = true; }
    if (body.class_id !== undefined && body.class_id !== a.class_id) {
      const cls = db.classes.find((c) => c.id === body.class_id);
      if (!cls) throw new ApiError("bad_request", `class ${body.class_id} not found`, 400);
      a.class_id = cls.id;
      a.class_name = cls.name;
      edited = true;
    }
    // Status rules (CONTRACT.md section 1 + requested #13 change): an explicit status (any of
    // the 5) wins, otherwise any edit -> edited.
    if (body.status !== undefined && !STATUSES.includes(body.status)) {
      throw new ApiError("bad_request", `unknown status "${body.status}"`, 400);
    }
    if (body.status) a.status = body.status;
    else if (edited) a.status = "edited";
    if ("reviewer" in body) a.reviewer = body.reviewer;
    a.updated_at = now();
    return clone(a);
  },
  imageUrl: (img) => img.url, // fixture urls are not served -> the canvas draws a placeholder

  async deleteAnnotation(aid) { // requested #19
    const db = await loadMock();
    const k = db.annotations.findIndex((x) => x.id === Number(aid));
    if (k < 0) mockNotFound("annotation", aid);
    db.annotations.splice(k, 1);
    return null;
  },
  async renameProject(pid, name) { // requested #17
    const db = await loadMock();
    mockProject(db, pid);
    name = String(name || "").trim();
    if (!name) throw new ApiError("bad_request", "name is required", 400);
    if (db.projects.some((p) => p.name === name && p.id !== Number(pid))) {
      throw new ApiError("name_taken", `project "${name}" already exists`, 409);
    }
    db.projects.find((p) => p.id === Number(pid)).name = name;
    return mockProject(db, pid);
  },
  async deleteProject(pid) { // requested #18
    const db = await loadMock();
    mockProject(db, pid);
    pid = Number(pid);
    if (db.jobs.some((j) => j.project_id === pid && (j.status === "queued" || j.status === "running"))) {
      throw new ApiError("conflict", "project has a running job", 409);
    }
    const ids = new Set(db.images.filter((i) => i.project_id === pid).map((i) => i.id));
    for (const i of db.images) if (ids.has(i.id) && String(i.url).startsWith("blob:")) URL.revokeObjectURL(i.url);
    db.annotations = db.annotations.filter((a) => !ids.has(a.image_id));
    db.images = db.images.filter((i) => i.project_id !== pid);
    db.classes = db.classes.filter((c) => c.project_id !== pid);
    db.jobs = db.jobs.filter((j) => j.project_id !== pid);
    db.projects = db.projects.filter((p) => p.id !== pid);
    return null;
  },

  // ---- setup screen
  async createProject(name) {
    const db = await loadMock();
    name = String(name || "").trim();
    if (!name) throw new ApiError("bad_request", "name is required", 400);
    if (db.projects.some((p) => p.name === name)) {
      throw new ApiError("name_taken", `project "${name}" already exists`, 409);
    }
    const p = { id: db.nextProjectId++, name, created_at: now() };
    db.projects.push(p);
    return mockProject(db, p.id);
  },
  async uploadImages(pid, files, onProgress) {
    const db = await loadMock();
    mockProject(db, pid);
    const images = [], skipped = [];
    let i = 0;
    for (const f of files) {
      await mockDelay(60); // so the progress bar is visible
      i += 1;
      if (onProgress) onProgress(i / files.length);
      if (!/\.(jpe?g|png)$/i.test(f.name)) { skipped.push({ file_name: f.name, reason: "not an image" }); continue; }
      if (db.images.some((x) => x.project_id === Number(pid) && x.file_name === f.name)) {
        skipped.push({ file_name: f.name, reason: "duplicate" });
        continue;
      }
      const { width, height } = await imageSize(f);
      // A blob: URL, so uploaded images really show on the review canvas in mock mode.
      const img = { id: db.nextImageId++, project_id: Number(pid), file_name: f.name, width, height,
                    url: URL.createObjectURL(f), uploaded_at: now() };
      db.images.push(img);
      images.push(mockImageWithCounts(db, img));
    }
    return { images, skipped };
  },
  async putClasses(pid, classes) {
    const db = await loadMock();
    mockProject(db, pid);
    pid = Number(pid);
    const names = new Set();
    for (const c of classes) {
      if (!c.name || !String(c.name).trim()) throw new ApiError("bad_request", "every class needs a name", 400);
      if (names.has(c.name)) throw new ApiError("bad_request", `class "${c.name}" is listed twice`, 400);
      names.add(c.name);
    }
    for (const c of db.classes.filter((x) => x.project_id === pid)) {
      if (!names.has(c.name) && db.annotations.some((a) => a.class_id === c.id)) {
        throw new ApiError("class_in_use", `class "${c.name}" still has annotations`, 409);
      }
    }
    db.classes = db.classes.filter((c) => c.project_id !== pid || names.has(c.name));
    for (const c of classes) {
      const row = { name: c.name, description: c.description || "",
                    min_score: c.min_score ?? null, accept_score: c.accept_score ?? null };
      const old = db.classes.find((x) => x.project_id === pid && x.name === c.name);
      if (old) Object.assign(old, row);
      else db.classes.push({ id: db.nextClassId++, project_id: pid, ...row });
    }
    return { classes: mockClasses(db, pid) };
  },
  async startJob(pid, body) {
    const db = await loadMock();
    mockProject(db, pid);
    pid = Number(pid);
    if (!MOCK_MODELS.includes(body.model)) {
      throw new ApiError("unknown_model", `unknown model "${body.model}"`, 400);
    }
    if (!db.classes.some((c) => c.project_id === pid)) {
      throw new ApiError("no_classes", "project has no classes", 400);
    }
    const ids = body.image_ids || db.images.filter((i) => i.project_id === pid).map((i) => i.id);
    const job = { id: db.nextJobId++, project_id: pid, kind: "detect", status: "queued", progress: 0,
                  done: 0, total: ids.length, error: null,
                  config: { model: body.model, image_ids: body.image_ids ?? null, options: body.options || {} },
                  created_at: now(), started_at: null, finished_at: null, _ids: ids };
    db.jobs.push(job);
    return stripJob(job);
  },
  // Each poll moves a mock job forward by one image, so the progress bar is visible.
  async getJob(jid) {
    const db = await loadMock();
    const job = db.jobs.find((j) => j.id === Number(jid));
    if (!job) mockNotFound("job", jid);
    if (job.status === "queued") {
      job.status = "running";
      job.started_at = now();
    } else if (job.status === "running" && job.done < job.total) {
      mockDetectOne(db, job, job._ids[job.done]);
      job.done += 1;
      job.progress = Math.round((job.done / job.total) * 10000) / 10000;
    }
    if (job.status === "running" && job.done >= job.total) {
      job.status = "done";
      job.progress = 1;
      job.finished_at = now();
    }
    return stripJob(job);
  },
  async listModels() { return { models: MOCK_MODELS.slice(), fallback: false }; },
};

window.api = MOCK ? mockApi : realApi;
window.ApiError = ApiError;
window.STATUSES = STATUSES;
window.countStatuses = countStatuses;
