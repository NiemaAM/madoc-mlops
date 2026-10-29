"""Dataset versions: an immutable snapshot of which pages are in which split.

A training records the version it used, an evaluation the version of its test pages: any result can be traced
back to its exact pages. A snapshot is made when a job needs one; the same pages give the same version.
"""

import hashlib

from mlops import db
from mlops.config import SPLITS, path


def snapshot(conn):
    """The version holding every page of the dataset now (made if it does not exist yet) -> its row."""
    rows = db.all_rows(conn, "SELECT id, split FROM pages ORDER BY id")
    digest = hashlib.sha256("\n".join(f'{r["id"]}:{r["split"]}' for r in rows).encode()).hexdigest()[:16]
    existing = db.one(conn, "SELECT * FROM dataset_versions WHERE digest = %s", digest)
    if existing:
        return existing
    counts = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    with conn.transaction():
        version = db.one(conn, "INSERT INTO dataset_versions (digest, counts) VALUES (%s, %s) "
                               "ON CONFLICT (digest) DO UPDATE SET digest = EXCLUDED.digest RETURNING *",
                         digest, db.json(counts))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO dataset_pages (version_id, page_id, split) VALUES (%s, %s, %s) "
                            "ON CONFLICT DO NOTHING", [(version["id"], r["id"], r["split"]) for r in rows])
    return version


def by_digest(conn, digest):
    return db.one(conn, "SELECT * FROM dataset_versions WHERE digest = %s", digest) if digest else None


def pages(conn, version_id, split):
    return db.all_rows(conn, "SELECT p.* FROM dataset_pages d JOIN pages p ON p.id = d.page_id "
                             "WHERE d.version_id = %s AND d.split = %s ORDER BY p.id", version_id, split)


def seen_page_ids(conn, version_id):
    """The pages a model trained on that version saw while training: its train and validation pages."""
    if not version_id:
        return set()
    return {row["page_id"] for row in db.all_rows(
        conn, "SELECT page_id FROM dataset_pages WHERE version_id = %s AND split IN ('train', 'val')", version_id)}


def leaking_groups(conn, version_id):
    """The groups of a version found in more than one split (a split by page, imported with --split and not
    split again): a page of such a group in test has near-copies in train."""
    return db.all_rows(conn, "SELECT p.group_key, count(DISTINCT d.split) AS splits, count(*) AS pages "
                             "FROM dataset_pages d JOIN pages p ON p.id = d.page_id WHERE d.version_id = %s "
                             "GROUP BY p.group_key HAVING count(DISTINCT d.split) > 1", version_id)


def xml_paths(rows):
    folder = path("data") / "pages"
    return [str(folder / f'{row["id"]}.xml') for row in rows]


def write_list(paths, target):
    """A ketos manifest: one file path per line."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return target