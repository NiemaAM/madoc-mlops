"""Commands of the platform, run inside a container:

    python -m mlops.cli health worker          # healthcheck of the worker container (its heartbeat is recent)
    python -m mlops.cli health ui              # healthcheck of the app container (the database answers)
    python -m mlops.cli import /import/train --split train     # import an existing split (copied, then queued)
    python -m mlops.cli jobs                   # the last jobs
    python -m mlops.cli normalize              # apply ingest.normalize to the pages already in the dataset
    python -m mlops.cli resplit                # split every group again, balanced on the characters (mlops/splitting.py)
    python -m mlops.cli missing-pages --list   # the acts with one page annotated: the missing page to annotate
    python -m mlops.cli pause                  # the worker takes no new job (the running one goes on)
    python -m mlops.cli wait-idle --timeout 36000    # wait until no job is running (0), or the timeout (1)
    python -m mlops.cli resume                 # the worker takes jobs again
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

from mlops import db, jobs
from mlops.config import SPLITS, path


def health(component):
    with db.connection() as conn:
        if component == "worker":
            row = db.one(conn, "SELECT max(seen_at) > now() - interval '5 minutes' AS alive FROM workers")
            return 0 if row and row["alive"] else 1
        db.one(conn, "SELECT 1 AS ok")
        return 0


def import_folder(source, split):
    """Copy a folder (or a zip) into the uploads of the data volume, then queue its ingestion: the worker, which
    does not see the folder of this command, reads it there (and deletes it once ingested)."""
    source = Path(source)
    if not source.exists():
        raise SystemExit(f"{source} does not exist (inside the container: mount it with -v)")
    with db.connection() as conn:
        batch = db.one(conn, "INSERT INTO batches (status) VALUES ('uploaded') RETURNING id")
        target = path("data") / "uploads" / (f'batch-{batch["id"]}' + (".zip" if source.is_file() else ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copyfile(source, target)
        else:
            shutil.copytree(source, target)
        job = jobs.enqueue(conn, "ingest", batch_id=batch["id"], upload=str(target), force_split=split)
        print(f"job {job['id']} queued: ingest of {source} ({split or 'split by group'}), batch {batch['id']}")


def list_jobs():
    with db.connection() as conn:
        for job in db.all_rows(conn, "SELECT id, kind, task, status, created_at, error FROM jobs ORDER BY id DESC "
                                     "LIMIT 20"):
            print(f'{job["id"]:>5} {job["kind"]:<9} {job["task"] or "":<5} {job["status"]:<10} '
                  f'{job["created_at"]:%Y-%m-%d %H:%M} {job["error"] or ""}')


def wait_idle(timeout, every=60):
    """Wait until no job is running -> 0, or 1 after `timeout` seconds."""
    end = time.time() + timeout
    while True:
        with db.connection() as conn:
            running = db.all_rows(conn, "SELECT id, kind, task, started_at FROM jobs WHERE status = 'running'")
        if not running:
            print("no job running", flush=True)
            return 0
        if time.time() >= end:
            print(f"still running after {timeout} s: " + ", ".join(f'job {j["id"]}' for j in running), flush=True)
            return 1
        print("waiting for " + ", ".join(f'job {j["id"]} ({j["kind"]} {j["task"] or ""}, started '
                                         f'{j["started_at"]:%H:%M})' for j in running), flush=True)
        time.sleep(every)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLOps platform commands")
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("health")
    h.add_argument("component", choices=("worker", "ui"))
    i = sub.add_parser("import")
    i.add_argument("source", help="a folder or a zip visible inside the container")
    i.add_argument("--split", choices=SPLITS, help="put every page in this split (keep an existing split)")
    sub.add_parser("jobs")
    sub.add_parser("normalize", help="apply ingest.normalize (params.yaml) to the pages already in the dataset")
    sub.add_parser("resplit", help="split every group again, balanced on the characters (before the first training)")
    m = sub.add_parser("missing-pages", help="the acts with only one of their two pages annotated")
    m.add_argument("--list", action="store_true", help="also print the file names of the missing pages")
    sub.add_parser("pause", help="the worker takes no new job (before a deployment)")
    sub.add_parser("resume", help="the worker takes jobs again")
    w = sub.add_parser("wait-idle", help="wait until no job is running")
    w.add_argument("--timeout", type=int, default=36000, help="seconds (default 10 h)")
    args = parser.parse_args(argv)
    if args.command == "health":
        return health(args.component)
    db.init_schema()
    if args.command == "import":
        import_folder(args.source, args.split)
    elif args.command == "normalize":
        from mlops.ingest import normalize_dataset
        with db.connection() as conn:
            normalize_dataset(conn)
    elif args.command == "resplit":
        from mlops.ingest import resplit
        with db.connection() as conn:
            resplit(conn)
    elif args.command in ("pause", "resume"):
        from mlops import settings
        with db.connection() as conn:
            (settings.pause if args.command == "pause" else settings.resume)(conn)
        print("worker paused: no new job" if args.command == "pause" else "worker resumed")
    elif args.command == "missing-pages":
        from mlops.missing import print_report
        with db.connection() as conn:
            print_report(conn, args.list)
    elif args.command == "wait-idle":
        return wait_idle(args.timeout)
    else:
        list_jobs()
    return 0


if __name__ == "__main__":
    sys.exit(main())