"""The job queue, in PostgreSQL: the app adds jobs, the worker takes them one at a time.

A job is taken with SELECT ... FOR UPDATE SKIP LOCKED, so two workers could never take the same one. Its output
goes to job_logs line by line (the app shows it live), its result and error to the job's row.
"""

import time

from mlops import db

KINDS = ("ingest", "train", "evaluate", "release")


def enqueue(conn, kind, task=None, **args):
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind}")
    return db.one(conn, "INSERT INTO jobs (kind, task, args) VALUES (%s, %s, %s) RETURNING *",
                  kind, task, db.json(args))


def claim(conn):
    """The oldest queued job, now running -> its row, or None."""
    with conn.transaction():
        job = db.one(conn, "SELECT * FROM jobs WHERE status = 'queued' ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1")
        if job is None:
            return None
        return db.one(conn, "UPDATE jobs SET status = 'running', started_at = now() WHERE id = %s RETURNING *",
                      job["id"])


def finish(conn, job_id, status, result=None, error=None):
    conn.execute("UPDATE jobs SET status = %s, result = %s, error = %s, finished_at = now() WHERE id = %s",
                 (status, db.json(result) if result is not None else None, error, job_id))


def cancel(conn, job_id):
    """A queued job is cancelled at once; a running one is asked to stop (the worker stops its process)."""
    conn.execute("UPDATE jobs SET status = 'cancelled', finished_at = now() WHERE id = %s AND status = 'queued'",
                 (job_id,))
    conn.execute("UPDATE jobs SET cancel_requested = true WHERE id = %s AND status = 'running'", (job_id,))


def cancel_requested(conn, job_id):
    row = db.one(conn, "SELECT cancel_requested FROM jobs WHERE id = %s", job_id)
    return bool(row and row["cancel_requested"])


def recover(conn):
    """Jobs left running by a worker that stopped (restart, crash) are marked failed."""
    conn.execute("UPDATE jobs SET status = 'failed', error = 'the worker stopped during the job', "
                 "finished_at = now() WHERE status = 'running'")


class Log:
    """The output of a job, written to job_logs in small batches (the app reads it live)."""

    def __init__(self, conn, job_id, flush_every=2.0):
        self.conn, self.job_id, self.flush_every = conn, job_id, flush_every
        self.pending, self.last = [], time.monotonic()

    def __call__(self, line):
        line = line.rstrip("\n")
        if not line.strip():
            return
        print(f"[job {self.job_id}] {line}", flush=True)
        self.pending.append(line[:2000])
        if time.monotonic() - self.last >= self.flush_every:
            self.flush()

    def flush(self):
        if self.pending:
            with self.conn.cursor() as cur:
                cur.executemany("INSERT INTO job_logs (job_id, line) VALUES (%s, %s)",
                                [(self.job_id, line) for line in self.pending])
            self.pending = []
        self.last = time.monotonic()


def tail(conn, job_id, limit=300):
    rows = db.all_rows(conn, "SELECT line FROM (SELECT id, line FROM job_logs WHERE job_id = %s "
                             "ORDER BY id DESC LIMIT %s) t ORDER BY id", job_id, limit)
    return "\n".join(row["line"] for row in rows)


def heartbeat(conn, name, job_id=None):
    conn.execute("INSERT INTO workers (name, seen_at, job_id) VALUES (%s, now(), %s) "
                 "ON CONFLICT (name) DO UPDATE SET seen_at = now(), job_id = EXCLUDED.job_id", (name, job_id))
