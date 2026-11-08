-- The database of the MLOps platform: what the pages are, which pages each dataset version holds, the jobs and
-- their logs, the models and the evaluations. The images and the PageXML files themselves are in the data volume
-- (pages/<id>.xml + image), named by an opaque id. Applied at every start: only "IF NOT EXISTS".

CREATE TABLE IF NOT EXISTS batches (
    id          serial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    status      text NOT NULL DEFAULT 'uploaded',          -- uploaded, ingested, failed
    report      jsonb                                      -- counts and reasons of rejection (no file name)
);

CREATE TABLE IF NOT EXISTS pages (
    id          text PRIMARY KEY,                          -- start of the SHA-256 of the image
    sha256      text NOT NULL UNIQUE,
    batch_id    integer NOT NULL REFERENCES batches(id),
    group_key   text NOT NULL,                             -- hash of the group (register, year): never the name
    split       text NOT NULL CHECK (split IN ('train', 'val', 'test')),
    image_name  text NOT NULL,
    lines       integer NOT NULL,
    characters  integer NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- page 1 (names) or page 2 (details) of an act, from the original file name (ingest.page_type_pattern)
ALTER TABLE pages ADD COLUMN IF NOT EXISTS page_type text;

-- PRIVATE: the original file names (they identify real records). Read by nothing but a person who needs to find
-- a page in eScriptorium again; never logged, never sent to W&B.
CREATE TABLE IF NOT EXISTS page_origins (
    page_id       text PRIMARY KEY REFERENCES pages(id),
    original_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id          serial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    digest      text NOT NULL UNIQUE,                      -- hash of (page, split) of all its pages
    counts      jsonb NOT NULL                             -- {"train": n, "val": n, "test": n}
);

CREATE TABLE IF NOT EXISTS dataset_pages (
    version_id  integer NOT NULL REFERENCES dataset_versions(id),
    page_id     text NOT NULL REFERENCES pages(id),
    split       text NOT NULL,
    PRIMARY KEY (version_id, page_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id               serial PRIMARY KEY,
    kind             text NOT NULL,                        -- ingest, train, evaluate, release
    task             text,                                 -- blla, htr (null for ingest)
    status           text NOT NULL DEFAULT 'queued',       -- queued, running, done, failed, cancelled
    args             jsonb NOT NULL DEFAULT '{}',
    result           jsonb,
    error            text,
    wandb_url        text,
    cancel_requested boolean NOT NULL DEFAULT false,
    created_at       timestamptz NOT NULL DEFAULT now(),
    started_at       timestamptz,
    finished_at      timestamptz
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, id);

CREATE TABLE IF NOT EXISTS job_logs (
    id      bigserial PRIMARY KEY,
    job_id  integer NOT NULL REFERENCES jobs(id),
    at      timestamptz NOT NULL DEFAULT now(),
    line    text NOT NULL
);
CREATE INDEX IF NOT EXISTS job_logs_job ON job_logs (job_id, id);

CREATE TABLE IF NOT EXISTS models (
    id                 serial PRIMARY KEY,
    task               text NOT NULL,
    status             text NOT NULL DEFAULT 'candidate',  -- candidate, proposed (pull request), rejected
    artifact           text,                               -- W&B: kraken-htr-bc:v4
    path               text NOT NULL,                      -- the exported .mlmodel in the runs volume
    dataset_version_id integer REFERENCES dataset_versions(id),
    job_id             integer REFERENCES jobs(id),
    val_metrics        jsonb,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluations (
    id              serial PRIMARY KEY,
    job_id          integer REFERENCES jobs(id),
    task            text NOT NULL,
    model_id        integer NOT NULL REFERENCES models(id),
    test_version_id integer REFERENCES dataset_versions(id),
    production_card jsonb,                                 -- MODEL_CARD.json entry of the production model
    candidate       jsonb NOT NULL,                        -- metrics of the candidate
    production      jsonb,                                 -- metrics of the production model
    comparison      jsonb NOT NULL,                        -- gain, confidence interval, old / new pages
    passed          boolean NOT NULL,
    message         text NOT NULL,
    pr_url          text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workers (
    name     text PRIMARY KEY,
    seen_at  timestamptz NOT NULL DEFAULT now(),
    job_id   integer
);

-- the settings changed in the app (training options per task, split), on top of params.yaml (mlops/settings.py)
CREATE TABLE IF NOT EXISTS settings (
    name        text PRIMARY KEY,
    value       jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- held-out batches: their pages are always in test (never trained on, never augmented, never moved by a split)
ALTER TABLE batches ADD COLUMN IF NOT EXISTS held_out boolean NOT NULL DEFAULT false;
-- a deleted batch: its pages leave the dataset (the files stay while a dataset version uses them)
ALTER TABLE batches ADD COLUMN IF NOT EXISTS deleted_at timestamptz;

-- the act of a page (keyed hash of "<register>_<year>_<act>"): both pages of an act always share their split
ALTER TABLE pages ADD COLUMN IF NOT EXISTS act_key text;
-- the writer of the page (keyed hash of the officer's canonical name, read on page 2 of its act); null: unknown
ALTER TABLE pages ADD COLUMN IF NOT EXISTS writer_key text;
-- a split no placement may change: "test" for a held-out page, the split given to "mlops.cli import --split"
ALTER TABLE pages ADD COLUMN IF NOT EXISTS pinned_split text;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS held_out boolean NOT NULL DEFAULT false;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
CREATE INDEX IF NOT EXISTS pages_live ON pages (split) WHERE deleted_at IS NULL;

-- PRIVATE: the name of the officer written in the page ("من طرفنا نحن ... موظف جماعي"), as transcribed. Shown in
-- the app (writers, aliases) to the person who runs it; never logged, never sent to W&B (only writer_key is).
CREATE TABLE IF NOT EXISTS page_writers (
    page_id  text PRIMARY KEY REFERENCES pages(id),
    raw_name text NOT NULL
);

-- PRIVATE: spellings of the same officer ("لحسن اشميطي" -> "لحسن شميطي"), keys and values normalised
CREATE TABLE IF NOT EXISTS writer_aliases (
    variant    text PRIMARY KEY,
    canonical  text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- the worker that runs a job: "kraken" (the main worker) or "baselines" (CRNN, TrOCR, Qwen: their own container)
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS queue text NOT NULL DEFAULT 'kraken';

-- who downloaded what from the app (the pseudonymised preview contains personal data)
CREATE TABLE IF NOT EXISTS downloads (
    id         serial PRIMARY KEY,
    at         timestamptz NOT NULL DEFAULT now(),
    username   text,
    what       text NOT NULL
);

-- the results of the experiments (cross-validation, other recognizers, full pipeline): numbers only, and the
-- folder of their figures
CREATE TABLE IF NOT EXISTS experiments (
    id         serial PRIMARY KEY,
    job_id     integer REFERENCES jobs(id),
    kind       text NOT NULL,                              -- crossval, pipeline
    name       text NOT NULL,
    summary    jsonb NOT NULL,
    figures    text,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- the folder of the figures of an evaluation (PNG / PDF, shown and downloaded in the app)
ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS figures text;
ALTER TABLE models ADD COLUMN IF NOT EXISTS figures text;