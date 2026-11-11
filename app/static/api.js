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

// ---- polygons: [[x, y], ...] normalized 0..1 (contract conventions). Used by the canvas, the
// review page and the mock layer. Only the server's bbox is stored; these keep the browser in step.
const roundNorm = (v) => Math.round(Math.min(1, Math.max(0, v)) * 1e6) / 1e6;
const roundPx = (v) => Math.round(v * 100) / 100;

// Area in normalized units (shoelace formula).
function polygonArea(poly) {
  let sum = 0;
  poly.forEach(([x, y], i) => {
    const [nx, ny] = poly[(i + 1) % poly.length];
    sum += x * ny - nx * y;
  });
  return Math.abs(sum) / 2;
}

// The #12 / #13 rule. Returns what is wrong with a polygon, or null if it is fine.
function polygonError(poly) {
  if (!Array.isArray(poly) || poly.length < 3) return "a polygon needs at least 3 points";
  const ok = (p) => Array.isArray(p) && p.length === 2 &&
    p.every((v) => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1);
  if (!poly.every(ok)) return "every polygon point must be 2 numbers in 0..1";
  if (polygonArea(poly) < 1e-12) return "the polygon has zero area";
  return null;
}

// Bounding box in pixels of a polygon on an image of size w x h.
function polygonBbox(poly, w, h) {
  const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
  const x0 = Math.min(...xs) * w, y0 = Math.min(...ys) * h;
  return [roundPx(x0), roundPx(y0), roundPx(Math.max(...xs) * w - x0), roundPx(Math.max(...ys) * h - y0)];
}

// Scale and move a polygon so that its old bbox `from` fills `to` (both px); points stay in 0..1.
function scalePolygon(poly, from, to, w, h) {
  const kx = from[2] ? to[2] / from[2] : 1, ky = from[3] ? to[3] / from[3] : 1;
  return poly.map(([x, y]) => [roundNorm((to[0] + (x * w - from[0]) * kx) / w),
                               roundNorm((to[1] + (y * h - from[1]) * ky) / h)]);
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

// multipart POST of "files" parts with upload progress (fetch has none). onProgress(0..1).
function sendForm(url, parts, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const [f, name] of parts) form.append("files", f, name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
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
}

const realApi = {
  mock: false,
  listProjects: () => http("GET", "/projects"),
  getProject: (pid) => http("GET", `/projects/${pid}`),
  listImages(pid, { status = null, reviewStatus = null, offset = 0, limit = 50 } = {}) {
    const q = new URLSearchParams({ offset, limit });
    if (status) q.set("status", status);
    if (reviewStatus) q.set("review_status", reviewStatus);
    return http("GET", `/projects/${pid}/images?${q}`);
  },
  listClasses: (pid) => http("GET", `/projects/${pid}/classes`),
  getAnnotations: (iid) => http("GET", `/images/${iid}/annotations`),
  createAnnotation: (iid, body) => http("POST", `/images/${iid}/annotations`, body),
  patchAnnotation: (aid, body) => http("PATCH", `/annotations/${aid}`, body),
  // #20: set the image review status ("todo" | "done" | "empty"); 409 conflict when "empty"
  // and the image still has a box that is not rejected.
  patchImage: (iid, body) => http("PATCH", `/images/${iid}`, body),
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
    return sendForm(`/api/projects/${pid}/images`, files.map((f) => [f, f.name]), onProgress);
  },
  // #22 bulk import. parts = [[File, partName], ...]; partName is the relative path for a folder
  // upload (webkitRelativePath), the plain name for a zip or loose images. Response 202 is a Job.
  startImport: (pid, parts, onProgress) => sendForm(`/api/projects/${pid}/import`, parts, onProgress),
  putClasses: (pid, classes) => http("PUT", `/projects/${pid}/classes`, { classes }),
  startJob: (pid, body) => http("POST", `/projects/${pid}/jobs`, body),
  getJob: (jid) => http("GET", `/jobs/${jid}`),

  // ---- export (#14) and Roboflow push (#21)
  // #14 answers a zip, so this uses fetch directly. Returns {blob, filename}; errors -> ApiError.
  async exportZip(pid, body) {
    let res;
    try {
      res = await fetch(`/api/projects/${pid}/export`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (e) {
      throw new ApiError("network", "cannot reach the server: " + e.message, 0);
    }
    if (!res.ok) {
      let data = null;
      try { data = await res.json(); } catch (e) { /* non-JSON body */ }
      const err = (data && data.error) || {};
      throw new ApiError(err.code || "http_" + res.status, err.message || res.status + " " + res.statusText, res.status);
    }
    const m = /filename="?([^";]+)"?/.exec(res.headers.get("Content-Disposition") || "");
    return { blob: await res.blob(), filename: m ? m[1] : `project_${pid}_${body.format}.zip` };
  },
  // #21: no body = the default approved statuses. Response 202 is a Job, polled with getJob.
  startPush: (pid) => http("POST", `/projects/${pid}/roboflow/push`),
  // Endpoint 16: GET /api/models -> {"models": [...], "modes": {name: [...]}}, default mode first.
  // If an older backend answers 404, fall back to the one detector that always exists.
  async listModels() {
    try {
      const d = await http("GET", "/models");
      return { models: d.models, modes: d.modes || {}, fallback: false };
    } catch (e) {
      if (e.httpStatus === 404) return { models: FALLBACK_MODELS.slice(), modes: {}, fallback: true };
      throw e;
    }
  },
};

const FALLBACK_MODELS = ["fake"];
const MOCK_MODELS = ["fake"]; // same as the backend today (contract #16)
const MOCK_MODES = { fake: ["text", "examples"] };
const MOCK_PREP_POLLS = 2;    // polls an examples job stays running at 0 while it "embeds"
const EXAMPLE_CAP = 30;       // contract #9: at most 30 examples per class

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
    pushed: new Set(),   // image ids "sent to Roboflow" by a mock push job (no SDK call in mock mode)
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

// Contract #9: which boxes teach. Hand-drawn unless rejected, or any reviewed box.
function mockIsExample(a) {
  return (a.source === "human" && a.status !== "rejected") || a.status === "accepted" || a.status === "edited";
}

// The bbox clipped to the image has some area left. The image must be known (mock "file missing").
function mockUsableBox(a, img) {
  if (!img) return false;
  const [x, y, w, h] = a.bbox;
  return Math.min(x + w, img.width) > Math.max(x, 0) && Math.min(y + h, img.height) > Math.max(y, 0);
}

// Steps 1 and 3 of #9: count the examples of each class, write result, remember the classes that run.
// The fake detector rejects nothing, so the mock never gives "no usable examples".
function mockPrepareExamples(db, job) {
  const examples_by_class = {}, skipped_classes = [];
  job._ran = [];
  for (const c of db.classes.filter((k) => k.project_id === job.project_id)) {
    const n = db.annotations.filter((a) => a.class_id === c.id && mockIsExample(a) &&
      mockUsableBox(a, db.images.find((i) => i.id === a.image_id))).length;
    if (n) {
      examples_by_class[c.name] = Math.min(n, EXAMPLE_CAP);
      job._ran.push(c.id);
    } else {
      skipped_classes.push({ class_name: c.name, reason: "no examples" });
    }
  }
  job.result = { mode: "examples", examples_by_class, skipped_classes };
}

function mockFinishJob(job) {
  job.status = "done";
  job.progress = 1;
  job.finished_at = now();
}

// One step of a mock detect job. Acts like the contract's "fake" detector: 2 boxes per image
// (first two classes, scores 0.9 and 0.3). Earlier unreviewed model boxes are replaced.
// An examples job only touches the classes that ran.
function mockDetectOne(db, job, imageId) {
  const img = db.images.find((i) => i.id === imageId);
  if (!img) return;
  // A re-run skips an image reviewed while the job ran (it still counts in done).
  if (job.config.rerun_of != null && (img.review_status || "todo") !== "todo") return;
  const runs = (classId) => !job._ran || job._ran.includes(classId);
  db.annotations = db.annotations.filter((a) => !(a.image_id === imageId && a.source !== "human" &&
    a.source !== "import" && runs(a.class_id) &&
    (a.status === "auto_accepted" || a.status === "needs_review")));
  const classes = db.classes.filter((c) => c.project_id === job.project_id && runs(c.id)).slice(0, 2);
  const scores = [0.9, 0.3];
  const t = now();
  classes.forEach((c, k) => {
    const score = scores[k];
    if (c.min_score != null && score < c.min_score) return;
    db.annotations.push({
      id: db.nextId++, image_id: imageId, class_id: c.id, class_name: c.name,
      bbox: [Math.round(img.width * (0.15 + 0.4 * k)), Math.round(img.height * 0.3),
             Math.round(img.width * 0.2), Math.round(img.height * 0.15)],
      mask: null, polygon: null, score, source: job.config.model,
      status: c.accept_score != null && score >= c.accept_score ? "auto_accepted" : "needs_review",
      job_id: job.id, reviewer: null, created_at: t, updated_at: t,
    });
  });
}

function stripJob(job) {
  const { _ids, _ran, _prep, ...out } = job;
  return clone(out);
}

function mockNotFound(what, id) {
  throw new ApiError("not_found", `${what} ${id} not found`, 404);
}

function mockCheckPolygon(poly) {
  const err = polygonError(poly);
  if (err) throw new ApiError("bad_request", err, 400);
}

function mockImageWithCounts(db, img) {
  const out = clone(img);
  if (!out.review_status) out.review_status = "todo";
  out.counts = countStatuses(db.annotations.filter((a) => a.image_id === img.id));
  return out;
}

function mockReadyImages(db, pid) {
  return db.images.filter((i) => i.project_id === Number(pid) &&
                                 (i.review_status === "done" || i.review_status === "empty"));
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
  async listImages(pid, { status = null, reviewStatus = null, offset = 0, limit = 50 } = {}) {
    const db = await loadMock();
    mockProject(db, pid);
    let imgs = db.images.filter((i) => i.project_id === Number(pid))
                        .map((i) => mockImageWithCounts(db, i));
    if (status) imgs = imgs.filter((i) => i.counts[status] > 0);
    if (reviewStatus) imgs = imgs.filter((i) => i.review_status === reviewStatus);
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
    if (img.review_status === "empty") {
      throw new ApiError("conflict", `image ${img.id} is marked empty; set review_status to todo first`, 409);
    }
    const cls = db.classes.find((c) => c.id === body.class_id);
    if (!cls) throw new ApiError("bad_request", `class ${body.class_id} not found`, 400);
    // #12: with a polygon the server computes the bbox (a sent bbox is ignored).
    const polygon = body.polygon != null ? body.polygon : null;
    if (polygon) mockCheckPolygon(polygon);
    else if (!Array.isArray(body.bbox)) throw new ApiError("bad_request", "bbox is required", 400);
    const t = now();
    const a = { id: db.nextId++, image_id: img.id, class_id: cls.id, class_name: cls.name,
                bbox: polygon ? polygonBbox(polygon, img.width, img.height) : body.bbox.map(Number),
                mask: body.mask || null, polygon: polygon ? clone(polygon) : null, score: null,
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
    if (body.polygon != null) mockCheckPolygon(body.polygon); // before anything changes
    const img = db.images.find((i) => i.id === a.image_id);
    if ("polygon" in body) { // #13 table: a new polygon wins over a bbox; null keeps a sent bbox
      a.polygon = body.polygon != null ? clone(body.polygon) : null;
      if (a.polygon) a.bbox = polygonBbox(a.polygon, img.width, img.height);
      else if (body.bbox) a.bbox = body.bbox.map(Number);
      edited = true;
    } else if (body.bbox) {
      const to = body.bbox.map(Number);
      if (a.polygon) { // bbox only on a polygon row: the polygon fills the new bbox
        a.polygon = scalePolygon(a.polygon, a.bbox, to, img.width, img.height);
        a.bbox = polygonBbox(a.polygon, img.width, img.height);
      } else a.bbox = to;
      edited = true;
    }
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
  async patchImage(iid, body) { // #20
    const db = await loadMock();
    const img = db.images.find((i) => i.id === Number(iid));
    if (!img) mockNotFound("image", iid);
    if (!["todo", "done", "empty"].includes(body.review_status)) {
      throw new ApiError("bad_request", `unknown review_status "${body.review_status}"`, 400);
    }
    if (body.review_status === "empty") {
      const live = db.annotations.filter((a) => a.image_id === img.id && a.status !== "rejected").length;
      if (live) throw new ApiError("conflict", `image ${img.id} has ${live} box(es) that are not rejected`, 409);
    }
    img.review_status = body.review_status;
    return mockImageWithCounts(db, img);
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
    const images = [], skipped = [], renamed = [];
    let i = 0;
    for (const f of files) {
      await mockDelay(60); // so the progress bar is visible
      i += 1;
      if (onProgress) onProgress(i / files.length);
      if (!/\.(jpe?g|png)$/i.test(f.name)) { skipped.push({ file_name: f.name, reason: "not an image" }); continue; }
      // Mock stand-in for the server's sha256 check: same name + same size = duplicate,
      // same name + other size = renamed to stem_2.ext (stem_3, ...).
      const mine = db.images.filter((x) => x.project_id === Number(pid));
      const same = mine.find((x) => x.file_name === f.name);
      if (same && same.size === f.size) { skipped.push({ file_name: f.name, reason: "duplicate" }); continue; }
      let name = f.name;
      if (same) {
        const dot = f.name.lastIndexOf(".");
        const stem = f.name.slice(0, dot), ext = f.name.slice(dot);
        for (let n = 2; mine.some((x) => x.file_name === name); n++) name = `${stem}_${n}${ext}`;
        renamed.push({ from: f.name, to: name });
      }
      const { width, height } = await imageSize(f);
      // A blob: URL, so uploaded images really show on the review canvas in mock mode.
      const img = { id: db.nextImageId++, project_id: Number(pid), file_name: name, width, height, size: f.size,
                    url: URL.createObjectURL(f), uploaded_at: now(), review_status: "todo" };
      db.images.push(img);
      images.push(mockImageWithCounts(db, img));
    }
    return { images, skipped, renamed };
  },
  async startImport() {
    throw new ApiError("mock", "bulk import (#22) needs the real server; mock mode has no importer", 400);
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
    let rerunOf = null;
    if (body.rerun) { // #9 rerun: copy model, mode, options from the last examples job; todo images only
      const extra = ["model", "mode", "image_ids", "options"].filter((k) => k in body);
      if (extra.length) throw new ApiError("bad_request", `rerun takes no other fields; got: ${extra.join(", ")}`, 400);
      const src = db.jobs.filter((j) => j.project_id === pid && j.kind === "detect" && j.config.mode === "examples")
                         .sort((a, b) => b.id - a.id)[0];
      if (!src) throw new ApiError("conflict", `project ${pid} has no examples-mode job to re-run`, 409);
      rerunOf = src.id;
      body = { model: src.config.model, mode: src.config.mode, options: src.config.options };
    }
    if (!MOCK_MODELS.includes(body.model)) {
      throw new ApiError("unknown_model", `unknown model "${body.model}"`, 400);
    }
    const modes = MOCK_MODES[body.model] || [];
    if (body.mode != null && !modes.includes(body.mode)) {
      throw new ApiError("unknown_mode", `unknown mode "${body.mode}" for model "${body.model}"`, 400);
    }
    const mode = body.mode ?? modes[0] ?? null;
    if (!db.classes.some((c) => c.project_id === pid)) {
      throw new ApiError("no_classes", "project has no classes", 400);
    }
    let ids = body.image_ids || db.images.filter((i) => i.project_id === pid).map((i) => i.id);
    const config = { model: body.model, mode, image_ids: body.image_ids ?? null, options: body.options || {} };
    if (rerunOf != null) {
      ids = db.images.filter((i) => i.project_id === pid && (i.review_status || "todo") === "todo")
                     .map((i) => i.id).sort((a, b) => a - b);
      if (!ids.length) throw new ApiError("conflict", `project ${pid} has no images with review_status todo`, 409);
      Object.assign(config, { image_ids: ids.slice(), rerun_of: rerunOf });
    }
    const job = { id: db.nextJobId++, project_id: pid, kind: "detect", status: "queued", progress: 0,
                  done: 0, total: ids.length, error: null, config,
                  result: null, created_at: now(), started_at: null, finished_at: null, _ids: ids,
                  _prep: mode === "examples" ? MOCK_PREP_POLLS : 0 };
    db.jobs.push(job);
    return stripJob(job);
  },
  // #14 in mock mode: same 409 rule; the "zip" is a small text file, since mock mode has no exporter.
  async exportZip(pid, body) {
    const db = await loadMock();
    const p = mockProject(db, pid);
    const ready = mockReadyImages(db, pid);
    if (!ready.length) throw new ApiError("conflict", `project ${pid} has no done or empty images to export`, 409);
    const text = `Mock export of "${p.name}": ${ready.length} image(s), format ${body.format}, ` +
                 `include_images ${body.include_images}.
The real server sends a zip here.
`;
    return { blob: new Blob([text], { type: "text/plain" }), filename: `${p.name}_${body.format}_MOCK.txt` };
  },
  // #21 in mock mode: same 409 rule, a job that only marks images as pushed (no Roboflow call).
  async startPush(pid) {
    const db = await loadMock();
    mockProject(db, pid);
    pid = Number(pid);
    const ready = mockReadyImages(db, pid);
    if (!ready.length) throw new ApiError("conflict", `project ${pid} has no done or empty images to push`, 409);
    const ids = ready.filter((i) => !db.pushed.has(i.id)).map((i) => i.id);
    const job = { id: db.nextJobId++, project_id: pid, kind: "roboflow_push", status: "queued", progress: 0,
                  done: 0, total: ids.length, error: null,
                  config: { statuses: ["auto_accepted", "accepted", "edited"] },
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
    } else if (job.status === "running" && job._prep > 0) {
      job._prep -= 1;
      if (job._prep > 0) return stripJob(job);
      mockPrepareExamples(db, job);
      if (!job._ran.length) { // no class left: done at once, no image run, nothing deleted
        job.done = job.total;
        mockFinishJob(job);
      }
      return stripJob(job);
    } else if (job.status === "running" && job.done < job.total) {
      if (job.kind === "roboflow_push") db.pushed.add(job._ids[job.done]);
      else mockDetectOne(db, job, job._ids[job.done]);
      job.done += 1;
      job.progress = Math.round((job.done / job.total) * 10000) / 10000;
    }
    if (job.status === "running" && !job._prep && job.done >= job.total) mockFinishJob(job);
    return stripJob(job);
  },
  async listModels() { return { models: MOCK_MODELS.slice(), modes: clone(MOCK_MODES), fallback: false }; },
};

window.api = MOCK ? mockApi : realApi;
window.ApiError = ApiError;
window.STATUSES = STATUSES;
window.countStatuses = countStatuses;
window.roundNorm = roundNorm;
window.polygonError = polygonError;
window.polygonBbox = polygonBbox;
window.scalePolygon = scalePolygon;
