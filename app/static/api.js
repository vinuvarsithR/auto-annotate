// api.js - one small data layer used by the UI.
// Real mode calls the FastAPI backend under /api (docs/CONTRACT.md section 2).
// Mock mode (?mock=1) reads docs/fixtures/api_samples.json once and answers
// from an in-memory copy, so edits show up until the page is reloaded.
"use strict";

const STATUSES = ["auto_accepted", "needs_review", "accepted", "rejected", "edited"];
const MOCK = new URLSearchParams(location.search).get("mock") === "1";
const FIXTURES_URL = "/docs/fixtures/api_samples.json";

class ApiError extends Error {
  // details: only #24 class_mismatch sends it (model / project class lists).
  constructor(code, message, httpStatus, details = null) {
    super(message);
    this.code = code;
    this.httpStatus = httpStatus;
    this.details = details;
  }
}

function errorFrom(data, status, statusText) {
  const err = (data && data.error) || {};
  return new ApiError(err.code || "http_" + status, err.message || `${status} ${statusText}`,
                      status, err.details || null);
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

// ---- video of an image: a copy of autolabel/sequences.py sequence_of() (keep the two the same).
// The server keys hook points by it (#27 - #29); the review page uses it to find the open image's video.
// "rb_load__cam0_2026-09-29_18-24-11_00568_jpg.rf.ab12.jpg" -> "rb_load__cam0_2026-09-29_18-24-11".
function sequenceOf(fileName) {
  let stem = String(fileName).split("/").pop();
  if (stem.includes(".")) stem = stem.slice(0, stem.lastIndexOf("."));  // real extension
  stem = stem.replace(/\.rf\.[0-9a-fA-F]+$/, "").replace(/_(jpe?g|png|bmp|webp)$/i, "");
  const m = /^(.+)_t?(\d+)$/.exec(stem);
  return m ? m[1] : stem;
}

// OWLv2 + SAM 2 defaults (#27 / #30, used when the setting is NULL).
const HOOK_DEFAULT_LOAD = ["rb", "sb", "biax"];
const HOOK_DEFAULT_RADIUS = 130;

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
  if (!res.ok) throw errorFrom(data, res.status, res.statusText);
  return data;
}

// multipart POST with upload progress (fetch has none). onProgress(0..1).
function sendMultipart(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
    xhr.onerror = () => reject(new ApiError("network", "cannot reach the server", 0));
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) { /* non-JSON body */ }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
      reject(errorFrom(data, xhr.status, xhr.statusText));
    };
    xhr.send(form);
  });
}

// multipart POST of "files" parts (#4 upload, #22 import).
function sendForm(url, parts, onProgress) {
  const form = new FormData();
  for (const [f, name] of parts) form.append("files", f, name);
  return sendMultipart(url, form, onProgress);
}

// POST a JSON body to an endpoint that answers a zip (#14, #23). Returns {blob, filename}.
async function postForZip(path, body, fallbackName) {
  let res;
  try {
    res = await fetch("/api" + path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (e) {
    throw new ApiError("network", "cannot reach the server: " + e.message, 0);
  }
  if (!res.ok) {
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON body */ }
    throw errorFrom(data, res.status, res.statusText);
  }
  const m = /filename="?([^";]+)"?/.exec(res.headers.get("Content-Disposition") || "");
  return { blob: await res.blob(), filename: m ? m[1] : fallbackName };
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
  exportZip: (pid, body) => postForZip(`/projects/${pid}/export`, body, `project_${pid}_${body.format}.zip`),
  // #21: no body = the default approved statuses. Response 202 is a Job, polled with getJob.
  startPush: (pid) => http("POST", `/projects/${pid}/roboflow/push`),

  // ---- Train on Colab (#23 - #26)
  trainPackage: (pid, task) => postForZip(`/projects/${pid}/train-package`, { task }, `project_${pid}_train_${task}.zip`),
  // #24: one .pt part + the task form field (+ optional metrics .json part). Response 201 is a TrainedModel.
  uploadWeights(pid, file, task, onProgress, metricsFile = null) {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("task", task);
    if (metricsFile) form.append("metrics", metricsFile, metricsFile.name);
    return sendMultipart(`/api/projects/${pid}/weights`, form, onProgress);
  },
  listWeights: (pid) => http("GET", `/projects/${pid}/weights`),
  deleteWeights: (wid) => http("DELETE", `/weights/${wid}`),

  // ---- hook points (#27 - #30, OWLv2 + SAM 2)
  getHook: (pid) => http("GET", `/projects/${pid}/hook`),
  putHookPoint: (pid, body) => http("PUT", `/projects/${pid}/hook/points`, body),   // {image_id, x, y}
  deleteHookPoint: (pid, seq) => http("DELETE", `/projects/${pid}/hook/points/${encodeURIComponent(seq)}`),
  patchHook: (pid, body) => http("PATCH", `/projects/${pid}/hook`, body),           // {load_classes?, radius?}

  // ---- assistant sidebar (#31 - #34)
  assistantStatus: () => http("GET", "/assistant/status"),
  chat: (body) => http("POST", "/assistant/chat", body),       // {session_id, project_id, user, message}
  chatConfirm: (body) => http("POST", "/assistant/chat/confirm", body), // {session_id, confirm_id, decision, user}
  deleteChat: (sid) => http("DELETE", `/assistant/chat/${encodeURIComponent(sid)}`),

  // Endpoint 16: GET /api/models -> {"models": [...], "modes": {name: [...]}}, default mode first.
  // With pid, the project's uploaded models are listed too (modes []).
  // If an older backend answers 404, fall back to the one detector that always exists.
  async listModels(pid = null) {
    try {
      const d = await http("GET", pid ? `/models?project_id=${pid}` : "/models");
      return { models: d.models, modes: d.modes || {}, trained: d.trained || [], fallback: false };
    } catch (e) {
      if (e.httpStatus === 404) return { models: FALLBACK_MODELS.slice(), modes: {}, trained: [], fallback: true };
      throw e;
    }
  },
};

const FALLBACK_MODELS = ["fake"];
// fake as in contract #16; owlv2 so the "Outlines (SAM 2)" toggle and job result.hook can be tried.
// The mock owlv2 finds the same boxes as fake, then applies the hook rule and SAM-like outlines.
const MOCK_MODELS = ["fake", "owlv2"];
const MOCK_MODES = { fake: ["text", "examples"], owlv2: ["text", "examples"] };
const MASKS = ["sam2", "none"];   // #9 masks values
const MOCK_PREP_POLLS = 2;    // polls an examples job stays running at 0 while it "embeds"
const EXAMPLE_CAP = 30;       // contract #9: at most 30 examples per class
const TRAIN_TASKS = ["detect", "segment"];        // #23 / #24
const WEIGHTS_MAX_BYTES = 200 * 1024 * 1024;     // #24: at most 200 MB
const METRICS_MAX_BYTES = 1024 * 1024;           // #24 metrics part: at most 1 MB
const APPROVED = ["auto_accepted", "accepted", "edited"]; // statuses #14 / #23 export

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
    // TrainedModel rows (+ deleted_at); the mock keeps no .pt file.
    weights: clone((fx["GET /api/projects/1/weights"] || { weights: [] }).weights)
      .map((w) => ({ source: "colab", metrics: null, split: null, ...w, deleted_at: null })),
    pushed: new Set(),   // image ids "sent to Roboflow" by a mock push job (no SDK call in mock mode)
    // #27 - #30 per project id: {load_classes: null | [...], radius: null | number, points: [Point]}
    hooks: {},
    fx,                  // the raw fixtures (assistant status samples)
    chats: {},           // assistant sessions: id -> {messages, pending: [{confirm_id, tool, arguments}]}
  };
  const h = fx["GET /api/projects/1/hook"];
  if (h) {
    mockDb.hooks[1] = { load_classes: h.load_classes_default ? null : clone(h.load_classes),
                        radius: h.radius_default ? null : h.radius, points: clone(h.points) };
  }
  const nextOf = (rows) => Math.max(0, ...rows.map((r) => r.id)) + 1;
  mockDb.nextId = nextOf(mockDb.annotations);
  mockDb.nextProjectId = nextOf(mockDb.projects);
  mockDb.nextImageId = nextOf(mockDb.images);
  mockDb.nextClassId = nextOf(mockDb.classes);
  mockDb.nextJobId = 100;
  mockDb.nextWeightsId = nextOf(mockDb.weights);
  return mockDb;
}

// #24 name rule: "<safe project name>"; same safe-name idea as #14 (letters, digits, - and _).
function mockSafeName(name) { return String(name).replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "project"; }

function mockLiveWeights(db, pid) {
  return db.weights.filter((w) => w.project_id === Number(pid) && !w.deleted_at).sort((a, b) => a.version - b.version);
}

// The #24 metrics file rules (CONTRACT "Metrics file"), as read_metrics checks them; broken -> 400 bad_metrics.
function mockReadMetrics(text, task) {
  const bad = (msg) => new ApiError("bad_metrics", `metrics file: ${msg}`, 400);
  let d;
  try { d = JSON.parse(text); } catch (e) { throw bad("not valid JSON"); }
  if (!d || typeof d !== "object" || Array.isArray(d)) throw bad("must be a JSON object");
  if (d.format !== "autolabel-metrics-1") throw bad("format must be autolabel-metrics-1");
  if (d.task !== task) throw bad(`task is "${d.task}", not "${task}"`);
  const m = d.metrics;
  if (!m || typeof m !== "object") throw bad("metrics must be an object");
  const pair = (x, key) => {
    if (!x || typeof x !== "object") throw bad(`metrics.${key} must be {map50, map50_95}`);
    for (const k of ["map50", "map50_95"]) {
      if (typeof x[k] !== "number" || x[k] < 0 || x[k] > 1) throw bad(`metrics.${key}.${k} must be a number in [0, 1]`);
    }
  };
  pair(m.box, "box");
  if (task === "segment") pair(m.mask, "mask");
  else if (m.mask != null) throw bad("metrics.mask must be null for detect");
  for (const k of ["epochs", "imgsz"]) {
    if (!Number.isInteger(m[k]) || m[k] < 1) throw bad(`metrics.${k} must be a positive integer`);
  }
  if (typeof m.base_weights !== "string") throw bad("metrics.base_weights must be a string");
  if (typeof m.finished_at !== "string" || isNaN(new Date(m.finished_at))) throw bad("metrics.finished_at must be an ISO UTC time");
  return { metrics: clone(m), split: d.split === undefined ? null : clone(d.split) };
}

function stripWeights(w) {
  const { deleted_at, ...out } = w;
  return clone(out);
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
  if (job.config.model === "owlv2") mockHookResult(db, job);
}

// ---- hook points (#27 - #30) in mock mode
function mockHookRow(db, pid) {
  pid = Number(pid);
  if (!db.hooks[pid]) db.hooks[pid] = { load_classes: null, radius: null, points: [] };
  return db.hooks[pid];
}

function mockHookSettings(db, pid) {
  const h = mockHookRow(db, pid);
  const counts = {};
  for (const img of db.images.filter((i) => i.project_id === Number(pid))) {
    const s = sequenceOf(img.file_name);
    counts[s] = (counts[s] || 0) + 1;
  }
  const has = new Set(h.points.map((p) => p.sequence));
  return {
    load_classes: clone(h.load_classes ?? HOOK_DEFAULT_LOAD), load_classes_default: h.load_classes == null,
    radius: h.radius ?? HOOK_DEFAULT_RADIUS, radius_default: h.radius == null,
    points: clone(h.points).sort((a, b) => (a.sequence < b.sequence ? -1 : a.sequence > b.sequence ? 1 : 0)),
    sequences: Object.keys(counts).sort().map((s) => ({ sequence: s, images: counts[s], has_point: has.has(s) })),
  };
}

// #9 result.masks + result.hook of an owlv2 examples job. Settings and points are read once, here
// (the mock's "job start"), and kept on the job for mockDetectOne.
function mockHookResult(db, job) {
  const s = mockHookSettings(db, job.project_id);
  const names = new Set(db.classes.filter((c) => c.project_id === job.project_id).map((c) => c.name));
  const load = s.load_classes.filter((n) => names.has(n));
  const loadIds = db.classes.filter((c) => c.project_id === job.project_id && load.includes(c.name) &&
                                           job._ran.includes(c.id)).map((c) => c.id);
  const bySeq = {};
  for (const id of job._ids) {
    const img = db.images.find((i) => i.id === id);
    if (img) (bySeq[sequenceOf(img.file_name)] ||= []).push(id);
  }
  const points = {}, warnings = [];
  for (const seq of Object.keys(bySeq).sort()) {
    const p = s.points.find((q) => q.sequence === seq);
    if (p) points[seq] = [p.x, p.y];
    else if (loadIds.length) warnings.push({ sequence: seq, images: bySeq[seq].length, reason: "no hook point" });
  }
  job._hook = { loadIds, radius: s.radius, points };
  job.result.masks = job.config.masks || "none";
  job.result.hook = { load_classes: s.load_classes, radius: s.radius, points, warnings };
}

// Step 2 of the OWLv2 + SAM 2 section: load boxes farther than radius from the hook are dropped,
// then only the best-scoring load box is kept. No hook point for the video: nothing filtered.
function mockHookFilter(job, img, boxes) {
  const hk = job._hook;
  const pt = hk && hk.points[sequenceOf(img.file_name)];
  if (!pt || !hk.loadIds.length) return boxes;
  const isLoad = (b) => hk.loadIds.includes(b.class_id);
  const near = (b) => Math.hypot(b.bbox[0] + b.bbox[2] / 2 - pt[0], b.bbox[1] + b.bbox[3] / 2 - pt[1]) <= hk.radius;
  let best = null;
  for (const b of boxes) if (isLoad(b) && near(b) && (!best || b.score > best.score)) best = b;
  return boxes.filter((b) => !isLoad(b) || b === best);
}

// Steps 3-4 stand-in (like FakeSegmenter): the ellipse inscribed in the box, as a 16-point polygon.
function mockEllipse(bbox, w, h) {
  const [x, y, bw, bh] = bbox, pts = [];
  for (let k = 0; k < 16; k++) {
    const t = (k / 16) * 2 * Math.PI;
    pts.push([roundNorm((x + bw / 2 + (bw / 2) * Math.cos(t)) / w), roundNorm((y + bh / 2 + (bh / 2) * Math.sin(t)) / h)]);
  }
  return pts;
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
  const owlExamples = job.config.model === "owlv2" && job.config.mode === "examples";
  const source = owlExamples ? "owlv2_image" : MOCK_MODELS.includes(job.config.model) ? job.config.model : "yolo_trained";
  let boxes = [];
  classes.forEach((c, k) => {
    const score = scores[k];
    if (c.min_score != null && score < c.min_score) return;
    boxes.push({
      id: null, image_id: imageId, class_id: c.id, class_name: c.name,
      bbox: [Math.round(img.width * (0.15 + 0.4 * k)), Math.round(img.height * 0.3),
             Math.round(img.width * 0.2), Math.round(img.height * 0.15)],
      mask: null, polygon: null, score, source,
      status: c.accept_score != null && score >= c.accept_score ? "auto_accepted" : "needs_review",
      job_id: job.id, reviewer: null, created_at: t, updated_at: t,
    });
  });
  if (owlExamples) {
    boxes = mockHookFilter(job, img, boxes);
    if (job.config.masks === "sam2") {
      for (const b of boxes) {
        b.polygon = mockEllipse(b.bbox, img.width, img.height);
        b.bbox = polygonBbox(b.polygon, img.width, img.height); // section 1: bbox = polygon's bbox
      }
    }
  }
  for (const b of boxes) { b.id = db.nextId++; db.annotations.push(b); }
}

function stripJob(job) {
  const { _ids, _ran, _prep, _hook, ...out } = job;
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

// ---- assistant (#31 - #34) in mock mode. No Claude: a message equal to a chat_fake.json scenario's
// request.message gets that scenario's expect_response (or its error). "?mock=1&chat=off" = no key.
const CHAT_FIXTURES_URL = "/docs/fixtures/assistant/chat_fake.json";
const TOOLS_FIXTURES_URL = "/docs/fixtures/assistant/tools.json";
const CHAT_OFF = new URLSearchParams(location.search).get("chat") === "off";
const CHAT_MAX_MESSAGES = 40;
const USER_NAME_RE = /^[A-Za-z0-9 ._-]{1,40}$/;
// One extra mock-only prompt so a make_train_package download link can be tried (data from tools.json).
const MOCK_TRAIN_PROMPT = "Make a training package.";
let chatFx = null;

async function loadChatFixtures() {
  if (chatFx) return chatFx;
  const get = async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new ApiError("mock", `cannot load ${url} (${res.status})`, res.status);
    return res.json();
  };
  const [chat, tools] = await Promise.all([get(CHAT_FIXTURES_URL), get(TOOLS_FIXTURES_URL)]);
  chatFx = { scenarios: chat.scenarios, tools };
  return chatFx;
}

function mockChatOff() {
  return new ApiError("assistant_unavailable", "Claude chat is off: ANTHROPIC_API_KEY is not set in the server's .env", 503);
}

function mockCheckUser(user) {
  if (typeof user !== "string" || !USER_NAME_RE.test(user.trim())) {
    throw new ApiError("bad_request", "user: set your name first (1-40 letters, digits, space . _ -)", 400);
  }
}

function mockSession(db, sid) {
  const s = db.chats[sid];
  if (!s) {
    throw new ApiError("not_found", `conversation ${sid} not found (the server keeps chats in memory; a restart clears them)`, 404);
  }
  return s;
}

let mockChatSeq = 0;
const mockId = (p) => `${p}-${Date.now().toString(36)}-${++mockChatSeq}`;

// The scripted answer for one message: a fixture scenario, the train-package prompt, or a help reply.
function mockChatAnswer(fx, message) {
  const sc = fx.scenarios.find((s) => s.request.message === message &&
    !(s.env && s.env.ANTHROPIC_API_KEY === "")); // "no key" is ?chat=off, not a message
  if (sc) {
    if (sc.expect_status) {
      const e = sc.expect_response.error;
      throw new ApiError(e.code, e.message, sc.expect_status);
    }
    return clone(sc.expect_response);
  }
  if (message === MOCK_TRAIN_PROMPT) {
    const ex = (fx.tools.examples.make_train_package || [])[0];
    const url = ex && ex.output.ok ? ex.output.data.download_url : "/api/assistant/downloads/MOCK";
    return { reply: `The training package is ready: ${url} (kept for 24 h).`,
             tool_calls: [{ tool: "make_train_package", arguments: ex ? ex.input : { project_id: 1 }, outcome: "ok" }],
             pending: [], stopped: "end_turn", usage: { input_tokens: 1800, output_tokens: 40 } };
  }
  const prompts = fx.scenarios.filter((s) => !(s.env && s.env.ANTHROPIC_API_KEY === ""))
                              .map((s) => `"${s.request.message}"`).concat([`"${MOCK_TRAIN_PROMPT}"`]);
  return { reply: `Mock mode has no Claude. Try one of the scripted messages: ${prompts.join(", ")}. ` +
                  'Add &chat=off to the URL for "chat is off".',
           tool_calls: [], pending: [], stopped: "end_turn", usage: { input_tokens: 0, output_tokens: 0 } };
}

// A confirmed pending tool, run on the mock data. Returns the result envelope.
function mockRunConfirmed(db, p) {
  if (p.tool === "delete_annotation") {
    const id = p.arguments.annotation_id;
    if (!db.annotations.some((a) => a.id === id)) return { ok: false, error: { code: "not_found", message: `annotation ${id} not found` } };
    db.annotations = db.annotations.filter((a) => a.id !== id);
    return { ok: true, data: { deleted: id } };
  }
  return { ok: false, error: { code: "bad_request", message: `mock mode cannot run ${p.tool}` } };
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
    db.weights = db.weights.filter((w) => w.project_id !== pid);
    delete db.hooks[pid];
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
      const extra = ["model", "mode", "image_ids", "options", "masks"].filter((k) => k in body);
      if (extra.length) throw new ApiError("bad_request", `rerun takes no other fields; got: ${extra.join(", ")}`, 400);
      const src = db.jobs.filter((j) => j.project_id === pid && j.kind === "detect" && j.config.mode === "examples")
                         .sort((a, b) => b.id - a.id)[0];
      if (!src) throw new ApiError("conflict", `project ${pid} has no examples-mode job to re-run`, 409);
      rerunOf = src.id;
      body = { model: src.config.model, mode: src.config.mode, options: src.config.options };
      if ("masks" in src.config) body.masks = src.config.masks; // rerun copies config.masks too
    }
    // #9: a non-deleted uploaded model of THIS project is a detector name too (no modes).
    const trained = mockLiveWeights(db, pid).some((w) => w.name === body.model);
    if (!MOCK_MODELS.includes(body.model) && !trained) {
      throw new ApiError("unknown_model", `unknown model "${body.model}"`, 400);
    }
    const modes = trained ? [] : MOCK_MODES[body.model] || [];
    if (body.mode != null && !modes.includes(body.mode)) {
      throw new ApiError("unknown_mode", `unknown mode "${body.mode}" for model "${body.model}"`, 400);
    }
    const mode = body.mode ?? modes[0] ?? null;
    if ("masks" in body) { // #9 masks: after unknown_mode, before no_classes
      if (!MASKS.includes(body.masks)) throw new ApiError("bad_request", `masks must be "sam2" or "none", got ${JSON.stringify(body.masks)}`, 400);
      if (body.model !== "owlv2" || mode !== "examples") {
        throw new ApiError("bad_request", "masks is only for model owlv2 in mode examples", 400);
      }
    }
    if (!db.classes.some((c) => c.project_id === pid)) {
      throw new ApiError("no_classes", "project has no classes", 400);
    }
    let ids = body.image_ids || db.images.filter((i) => i.project_id === pid).map((i) => i.id);
    const config = { model: body.model, mode, image_ids: body.image_ids ?? null, options: body.options || {} };
    if ("masks" in body) config.masks = body.masks; // stored only when sent
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
  async listModels(pid = null) {
    const out = { models: MOCK_MODELS.slice(), modes: clone(MOCK_MODES), fallback: false };
    if (pid == null) return out;
    const db = await loadMock();
    mockProject(db, pid);
    out.trained = [];
    for (const w of mockLiveWeights(db, pid)) {
      out.models.push(w.name);
      out.modes[w.name] = [];
      out.trained.push(clone({ name: w.name, source: w.source, task: w.task, uploaded_at: w.uploaded_at, metrics: w.metrics }));
    }
    return out;
  },

  // ---- Train on Colab in mock mode. The "zip" is a small text file (no exporter in the browser).
  async trainPackage(pid, task) {
    const db = await loadMock();
    const p = mockProject(db, pid);
    if (!TRAIN_TASKS.includes(task)) throw new ApiError("bad_request", `task must be detect or segment, got "${task}"`, 400);
    const ready = mockReadyImages(db, pid);
    if (!ready.length) throw new ApiError("conflict", `project ${pid} has no done or empty images to export`, 409);
    const ids = new Set(ready.map((i) => i.id));
    if (task === "segment" && !db.annotations.some((a) => ids.has(a.image_id) && APPROVED.includes(a.status) && a.polygon)) {
      throw new ApiError("conflict", `project ${pid} has no exported box with a polygon`, 409);
    }
    const name = `${mockSafeName(p.name)}_train_${task}`;
    const text = `Mock training package of "${p.name}": ${ready.length} image(s), task ${task}.
The real server sends a zip with data.yaml, split.json, train.ipynb, images and labels.
`;
    return { blob: new Blob([text], { type: "text/plain" }), filename: `${name}_MOCK.txt` };
  },
  // #24 in mock mode. A browser cannot open a .pt, so the "checkpoint" is made up from the file name:
  // the task sent and the project's classes, unless the name contains "detect" / "segment" (that
  // task, for task_mismatch), "pose" (bad_weights) or "beam" (one class swapped, for class_mismatch).
  async uploadWeights(pid, file, task, onProgress, metricsFile = null) {
    const db = await loadMock();
    const p = mockProject(db, pid);
    pid = Number(pid);
    if (!file || !/\.pt$/i.test(file.name)) throw new ApiError("bad_request", "file must be one .pt file", 400);
    if (!file.size || file.size > WEIGHTS_MAX_BYTES) throw new ApiError("bad_request", "file must be 1 byte .. 200 MB", 400);
    if (!TRAIN_TASKS.includes(task)) throw new ApiError("bad_request", `task must be detect or segment, got "${task}"`, 400);
    if (metricsFile && (!/\.json$/i.test(metricsFile.name) || !metricsFile.size || metricsFile.size > METRICS_MAX_BYTES)) {
      throw new ApiError("bad_request", "metrics must be one .json file, 1 byte .. 1 MB", 400);
    }
    const metricsRead = metricsFile ? mockReadMetrics(await metricsFile.text(), task) : { metrics: null, split: null };
    const project_classes = mockClasses(db, pid).map((c) => c.name);
    if (!project_classes.length) throw new ApiError("no_classes", "project has no classes", 400);
    for (let k = 1; k <= 4; k++) { if (onProgress) onProgress(k / 4); await mockDelay(120); }
    const low = file.name.toLowerCase();
    if (low.includes("pose")) throw new ApiError("bad_weights", "the weights are a 'pose' model; only detect or segment can be used", 400);
    const modelTask = low.includes("segment") ? "segment" : low.includes("detect") ? "detect" : task;
    if (modelTask !== task) throw new ApiError("task_mismatch", `the weights are a '${modelTask}' model, not '${task}'`, 400);
    const model_classes = project_classes.slice();
    if (low.includes("beam")) model_classes[model_classes.length - 1] = "beam";
    const missing = project_classes.filter((n) => !model_classes.includes(n));
    const extra = model_classes.filter((n) => !project_classes.includes(n));
    if (missing.length || extra.length) {
      throw new ApiError("class_mismatch",
        `model classes do not match project classes: missing in model: ${missing.join(", ")}; extra in model: ${extra.join(", ")}`,
        400, { model_classes, project_classes, missing_in_model: missing, extra_in_model: extra });
    }
    const version = 1 + Math.max(0, ...db.weights.filter((w) => w.project_id === pid).map((w) => w.version));
    let name = `${mockSafeName(p.name)}_v${version}`;
    if (db.weights.some((w) => w.name === name)) name = `${mockSafeName(p.name)}_p${pid}_v${version}`;
    const w = { id: db.nextWeightsId++, project_id: pid, name, version, task, class_names: model_classes,
                imgsz: 960, file_name: file.name, size_bytes: file.size, uploaded_at: now(),
                source: "colab", metrics: metricsRead.metrics, split: metricsRead.split, deleted_at: null };
    db.weights.push(w);
    return stripWeights(w);
  },
  async listWeights(pid) {
    const db = await loadMock();
    mockProject(db, pid);
    return { weights: mockLiveWeights(db, pid).map(stripWeights) };
  },
  // ---- assistant (#31 - #34)
  async assistantStatus() {
    const db = await loadMock();
    return clone(db.fx[CHAT_OFF ? "GET /api/assistant/status (no key)" : "GET /api/assistant/status"]);
  },
  async chat(body) {
    const db = await loadMock();
    if (CHAT_OFF) throw mockChatOff();
    body = body || {};
    const extra = Object.keys(body).filter((k) => !["session_id", "project_id", "user", "message"].includes(k));
    const msg = typeof body.message === "string" ? body.message : "";
    if (extra.length || !msg.length || msg.length > 4000 || !Number.isInteger(body.project_id)) {
      throw new ApiError("bad_request", "body must be {session_id, project_id, user, message} with message 1-4000 characters", 400);
    }
    mockCheckUser(body.user);
    mockProject(db, body.project_id);
    let sid = body.session_id;
    if (sid != null) mockSession(db, sid);
    else { sid = mockId("s"); db.chats[sid] = { messages: 0, pending: [] }; }
    const s = db.chats[sid];
    if (s.messages >= CHAT_MAX_MESSAGES) throw new ApiError("conflict", "this chat has 40 messages; start a new chat", 409);
    s.pending = []; // a new message ends every earlier pending confirmation
    await mockDelay(350); // "thinking"
    const out = mockChatAnswer(await loadChatFixtures(), msg.trim());
    if (out.stopped !== "refusal") s.messages += 1; // a refused turn is removed from the history
    out.session_id = sid;
    out.pending = out.pending.map((p) => ({ ...p, confirm_id: mockId("c") }));
    s.pending = out.pending.map((p) => ({ confirm_id: p.confirm_id, tool: p.tool, arguments: clone(p.arguments) }));
    return out;
  },
  async chatConfirm(body) {
    const db = await loadMock();
    if (CHAT_OFF) throw mockChatOff();
    body = body || {};
    const keys = Object.keys(body).sort().join(",");
    if (keys !== "confirm_id,decision,session_id,user" || !["confirm", "cancel"].includes(body.decision)) {
      throw new ApiError("bad_request", "body must be {session_id, confirm_id, decision: confirm | cancel, user}", 400);
    }
    mockCheckUser(body.user);
    const s = mockSession(db, body.session_id);
    const p = s.pending.find((q) => q.confirm_id === body.confirm_id);
    if (!p) throw new ApiError("conflict", `confirmation ${body.confirm_id} is no longer valid`, 409);
    s.pending = s.pending.filter((q) => q !== p); // valid once
    const result = body.decision === "confirm" ? mockRunConfirmed(db, p) : null;
    return { confirm_id: p.confirm_id, tool: p.tool, decision: body.decision, result };
  },
  async deleteChat(sid) {
    const db = await loadMock();
    if (CHAT_OFF) throw mockChatOff();
    mockSession(db, sid);
    delete db.chats[sid];
    return null;
  },
  // Mock only: forget every chat, as a server restart does (to try the "conversation was lost" path).
  async _restartChats() { (await loadMock()).chats = {}; },

  // ---- hook points (#27 - #30); errors in the contract's order
  async getHook(pid) {
    const db = await loadMock();
    mockProject(db, pid);
    return mockHookSettings(db, pid);
  },
  async putHookPoint(pid, body) {
    const db = await loadMock();
    mockProject(db, pid);
    const keys = Object.keys(body || {}).sort().join(",");
    const num = (v) => typeof v === "number" && Number.isFinite(v);
    if (keys !== "image_id,x,y" || !Number.isInteger(body.image_id) || !num(body.x) || !num(body.y)) {
      throw new ApiError("bad_request", "body must be {image_id, x, y} with finite numbers", 400);
    }
    const img = db.images.find((i) => i.id === body.image_id && i.project_id === Number(pid));
    if (!img) throw new ApiError("bad_image_ids", `image ${body.image_id} is not in project ${pid}`, 400);
    if (body.x < 0 || body.x > img.width || body.y < 0 || body.y > img.height) {
      throw new ApiError("bad_request", `x must be in 0..${img.width} and y in 0..${img.height}`, 400);
    }
    const h = mockHookRow(db, pid);
    const point = { sequence: sequenceOf(img.file_name), x: body.x, y: body.y, image_id: img.id, updated_at: now() };
    h.points = h.points.filter((p) => p.sequence !== point.sequence).concat([point]);
    return clone(point);
  },
  async deleteHookPoint(pid, seq) {
    const db = await loadMock();
    mockProject(db, pid);
    const h = mockHookRow(db, pid);
    if (!h.points.some((p) => p.sequence === seq)) {
      throw new ApiError("not_found", `no hook point for sequence ${seq}`, 404);
    }
    h.points = h.points.filter((p) => p.sequence !== seq);
    return null;
  },
  async patchHook(pid, body) {
    const db = await loadMock();
    mockProject(db, pid);
    body = body || {};
    const extra = Object.keys(body).filter((k) => k !== "load_classes" && k !== "radius");
    if (extra.length) throw new ApiError("bad_request", `unknown field(s): ${extra.join(", ")}`, 400);
    const names = mockClasses(db, pid).map((c) => c.name);
    const lc = body.load_classes;
    if (lc != null && (!Array.isArray(lc) || lc.some((n) => !names.includes(n)) || new Set(lc).size !== lc.length)) {
      throw new ApiError("bad_request", "load_classes must be distinct class names of the project", 400);
    }
    const r = body.radius;
    if (r != null && !(typeof r === "number" && Number.isFinite(r) && r > 0)) {
      throw new ApiError("bad_request", "radius must be a finite number > 0", 400);
    }
    const h = mockHookRow(db, pid);
    if ("load_classes" in body) h.load_classes = lc == null ? null : lc.slice();
    if ("radius" in body) h.radius = r == null ? null : r;
    return mockHookSettings(db, pid);
  },
  async deleteWeights(wid) {
    const db = await loadMock();
    const w = db.weights.find((x) => x.id === Number(wid) && !x.deleted_at);
    if (!w) mockNotFound("weights", wid);
    if (db.jobs.some((j) => j.kind === "detect" && j.config.model === w.name && (j.status === "queued" || j.status === "running"))) {
      throw new ApiError("conflict", `model ${w.name} is used by a queued or running job`, 409);
    }
    w.deleted_at = now();
    return null;
  },
};

window.api = MOCK ? mockApi : realApi;
window.ApiError = ApiError;
window.sequenceOf = sequenceOf;
window.STATUSES = STATUSES;
window.countStatuses = countStatuses;
window.roundNorm = roundNorm;
window.polygonError = polygonError;
window.polygonBbox = polygonBbox;
window.scalePolygon = scalePolygon;
