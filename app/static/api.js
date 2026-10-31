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
};

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
    classes: clone(fx["GET /api/projects/1/classes"].classes),
    annotations: clone(fx["all_annotations"]),
  };
  mockDb.nextId = Math.max(...mockDb.annotations.map((a) => a.id)) + 1;
  return mockDb;
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
    return { classes: clone(db.classes) };
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
    // Status rules (CONTRACT.md section 1): an explicit status wins, otherwise any edit -> edited.
    if (body.status) a.status = body.status;
    else if (edited) a.status = "edited";
    if ("reviewer" in body) a.reviewer = body.reviewer;
    a.updated_at = now();
    return clone(a);
  },
  imageUrl: (img) => img.url, // not served by http.server -> the canvas draws a placeholder
};

window.api = MOCK ? mockApi : realApi;
window.ApiError = ApiError;
window.STATUSES = STATUSES;
window.countStatuses = countStatuses;
