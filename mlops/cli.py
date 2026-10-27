"""Commands of the platform, run inside a container:

    python -m mlops.cli health worker          # healthcheck of the worker container (its heartbeat is recent)
    python -m mlops.cli health ui              # healthcheck of the app container (the database answers)
    python -m mlops.cli import /import/train --split train     # import an existing split (copied, then queued)
    python -m mlops.cli jobs                   # the last jobs
"""

import argparse
import shutil
import sys
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLOps platform commands")
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("health")
    h.add_argument("component", choices=("worker", "ui"))
    i = sub.add_parser("import")
    i.add_argument("source", help="a folder or a zip visible inside the container")
    i.add_argument("--split", choices=SPLITS, help="put every page in this split (keep an existing split)")
    sub.add_parser("jobs")
    args = parser.parse_args(argv)
    if args.command == "health":
        return health(args.component)
    db.init_schema()
    if args.command == "import":
        import_folder(args.source, args.split)
    else:
        list_jobs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
