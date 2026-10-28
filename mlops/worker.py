"""The worker: takes the jobs of the queue one at a time (one GPU) and runs them.

    python -m mlops.worker

Every 30 seconds it writes a heartbeat in the database (the app shows whether the worker is alive, the healthcheck
of its container fails when the heartbeat is old). A job cancelled in the app stops its process. A job left running
by a previous worker (restart, crash) is marked failed at start.
"""

import os
import shutil
import socket
import threading
import time
import traceback
from pathlib import Path

from mlops import db, jobs, release
from mlops.augment import run_augment
from mlops.config import path
from mlops.evaluate import evaluate
from mlops.ingest import ingest
from mlops.process import Cancelled
from mlops.train import train

NAME = os.environ.get("MLOPS_WORKER_NAME") or socket.gethostname()
HEARTBEAT_SECONDS = 30


def run_ingest(conn, job, log, should_stop, set_wandb_url):
    args = job["args"]
    upload = Path(args["upload"])
    try:
        report = ingest(conn, args["batch_id"], upload, args.get("force_split"), log)
        conn.execute("UPDATE batches SET status = 'ingested', report = %s WHERE id = %s",
                     (db.json(report), args["batch_id"]))
        return report
    except Exception:
        conn.execute("UPDATE batches SET status = 'failed' WHERE id = %s", (args["batch_id"],))
        raise
    finally:
        if upload.is_relative_to(path("data") / "uploads"):   # the upload is in the dataset now (or refused)
            upload.unlink(missing_ok=True) if upload.is_file() else shutil.rmtree(upload, ignore_errors=True)


def run_release(conn, job, log, should_stop, set_wandb_url):
    evaluation = db.one(conn, "SELECT e.*, v.digest AS test_digest FROM evaluations e "
                              "LEFT JOIN dataset_versions v ON v.id = e.test_version_id WHERE e.id = %s",
                        job["args"]["evaluation_id"])
    if evaluation is None:
        raise RuntimeError("no such evaluation")
    if not evaluation["passed"] and not job["args"].get("override"):
        raise RuntimeError("the candidate did not pass the gate (tick 'open anyway' in the app to override)")
    if evaluation["pr_url"]:
        return {"pr_url": evaluation["pr_url"], "note": "already opened"}
    model = db.one(conn, "SELECT * FROM models WHERE id = %s", evaluation["model_id"])
    version = db.one(conn, "SELECT * FROM dataset_versions WHERE id = %s", model["dataset_version_id"])
    url = release.open_pull_request(job["task"], model, evaluation, version, log)
    conn.execute("UPDATE evaluations SET pr_url = %s WHERE id = %s", (url, evaluation["id"]))
    conn.execute("UPDATE models SET status = 'proposed' WHERE id = %s", (model["id"],))
    log(f"pull request: {url}")
    return {"pr_url": url}


HANDLERS = {"ingest": run_ingest, "augment": run_augment, "train": train, "evaluate": evaluate, "release": run_release}


def heartbeat_loop(state, stop):
    with db.connection() as conn:
        while not stop.is_set():
            try:
                jobs.heartbeat(conn, NAME, state.get("job_id"))
            except Exception as e:
                print(f"heartbeat failed: {e}", flush=True)
            stop.wait(HEARTBEAT_SECONDS)


def run_job(conn, job):
    log = jobs.Log(conn, job["id"])
    log(f'{job["kind"]} {job["task"] or ""} started on {NAME}')

    def should_stop():
        return jobs.cancel_requested(conn, job["id"])

    def set_wandb_url(url):
        conn.execute("UPDATE jobs SET wandb_url = %s WHERE id = %s", (url, job["id"]))
        log(f"W&B: {url}")

    try:
        result = HANDLERS[job["kind"]](conn, job, log, should_stop, set_wandb_url)
        log("done")
        log.flush()
        jobs.finish(conn, job["id"], "done", result=result)
    except Cancelled:
        log("cancelled")
        log.flush()
        jobs.finish(conn, job["id"], "cancelled")
    except Exception as e:
        log(traceback.format_exc())
        log.flush()
        jobs.finish(conn, job["id"], "failed", error=f"{type(e).__name__}: {e}"[:2000])


def main(once=False):
    db.init_schema()
    state, stop = {}, threading.Event()
    threading.Thread(target=heartbeat_loop, args=(state, stop), daemon=True).start()
    with db.connection() as conn:
        jobs.recover(conn)
        try:
            from mlops.settings import publish_models
            publish_models(conn, release.sync())
            release.mark_production()
        except Exception as e:
            print(f"MaDoc clone not synced yet: {e}", flush=True)
        print(f"worker {NAME} ready", flush=True)
        from mlops.settings import paused
        while True:
            if paused(conn):   # a deployment is waiting for the running job: take no new one
                if once:
                    break
                time.sleep(3)
                continue
            job = jobs.claim(conn)
            if job is None:
                if once:
                    break
                time.sleep(3)
                continue
            state["job_id"] = job["id"]
            run_job(conn, job)
            state["job_id"] = None
    stop.set()


if __name__ == "__main__":
    main()