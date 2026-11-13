-- Copied from docs/CONTRACT.md section 1. Do not change here without a contract change.

CREATE TABLE projects (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  created_at  TEXT NOT NULL
);

CREATE TABLE images (
  id           INTEGER PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  file_name    TEXT NOT NULL,             -- original upload name
  path         TEXT NOT NULL,             -- on disk: data/uploads/<project_id>/<id>_<file_name>
  width        INTEGER NOT NULL,
  height       INTEGER NOT NULL,
  uploaded_at  TEXT NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'todo'
                CHECK (review_status IN ('todo', 'done', 'empty')),  -- set only by #20
  roboflow_image_id TEXT,               -- NULL = not pushed yet; set only by the #21 push job
  sha256       TEXT,                    -- hex SHA-256 of the uploaded bytes (#4 duplicate check)
  UNIQUE (project_id, file_name)
);

CREATE TABLE classes (
  id            INTEGER PRIMARY KEY,
  project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,            -- short label, e.g. "biax"
  description   TEXT NOT NULL,            -- text prompt, e.g. "a short yellow crossbar ..."
  min_score     REAL,                     -- NULL = model default; boxes below are not stored
  accept_score  REAL,                     -- NULL = never auto-accept; boxes >= become auto_accepted
  UNIQUE (project_id, name)
);

CREATE TABLE jobs (
  id           INTEGER PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL CHECK (kind IN ('detect', 'roboflow_push', 'import')),
  status       TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
  progress     REAL NOT NULL DEFAULT 0,   -- 0..1 = done / total
  done         INTEGER NOT NULL DEFAULT 0,
  total        INTEGER NOT NULL DEFAULT 0,
  error        TEXT,                      -- one-line message when status = failed
  config       TEXT NOT NULL,             -- JSON: the request body of POST .../jobs
  result       TEXT,                      -- JSON summary; set by import jobs (#22) and examples-mode detect jobs, NULL otherwise
  created_at   TEXT NOT NULL,
  started_at   TEXT,
  finished_at  TEXT
);

CREATE TABLE annotations (
  id          INTEGER PRIMARY KEY,
  image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  class_id    INTEGER NOT NULL REFERENCES classes(id),
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  mask        TEXT,                       -- COCO RLE JSON or NULL
  polygon     TEXT,                       -- JSON [[x, y], ...] normalized 0..1, or NULL (plain box)
  score       REAL,                       -- NULL for human-drawn boxes
  source      TEXT NOT NULL,              -- 'owlv2' | 'owlv2_image' | 'gdino' | 'pipeline' | 'human' | 'import'
  status      TEXT NOT NULL CHECK (status IN
                ('auto_accepted', 'needs_review', 'accepted', 'rejected', 'edited')),
  job_id      INTEGER REFERENCES jobs(id),  -- NULL for human-drawn boxes
  reviewer    TEXT,                       -- free text name for now (no login until Phase 6)
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX idx_ann_image ON annotations(image_id);
CREATE INDEX idx_ann_status ON annotations(status);

CREATE TABLE trained_models (            -- #24 - #26; migrate() creates it if missing
  id           INTEGER PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name         TEXT NOT NULL UNIQUE,     -- "eot_crane_v1"; the #9 `model` value
  version      INTEGER NOT NULL,         -- 1, 2, ... per project, never reused
  task         TEXT NOT NULL CHECK (task IN ('detect', 'segment')),
  class_names  TEXT NOT NULL,            -- JSON list, model index order
  imgsz        INTEGER,                  -- from the checkpoint, or NULL
  file_name    TEXT NOT NULL,            -- uploaded name, e.g. "best.pt"
  path         TEXT NOT NULL,            -- data/weights/<project_id>/<name>.pt
  size_bytes   INTEGER NOT NULL,
  uploaded_at  TEXT NOT NULL,
  deleted_at   TEXT,                     -- set by #26; the row is kept
  UNIQUE (project_id, version)
);
CREATE TABLE hook_points (               -- #27 - #30 (OWLv2 + SAM 2); migrate() creates it if missing
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  sequence     TEXT NOT NULL,            -- autolabel.sequences.sequence_of(images.file_name)
  x            REAL NOT NULL,            -- hook point in image pixels
  y            REAL NOT NULL,
  image_id     INTEGER REFERENCES images(id) ON DELETE SET NULL,  -- the image it was clicked on
  updated_at   TEXT NOT NULL,
  PRIMARY KEY (project_id, sequence)
);
-- projects gets two columns (migrate() adds them if missing; not part of the Project JSON):
--   load_classes TEXT   -- JSON list of class names, NULL = default ["rb", "sb", "biax"]
--   hook_radius  REAL   -- px, NULL = default 130
