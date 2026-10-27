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
