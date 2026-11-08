"""Deleting an uploaded batch.

Delete (soft): the pages of the batch leave the dataset at once: no split, no new dataset version, no training,
no evaluation sees them again. Their files and rows stay, so the dataset versions already made, and the models
trained on them, remain traceable; uploading the same pages again brings them back. "Restore" undoes it.

Purge: the files and the rows are removed for good. Only allowed when no model was trained, validated or
evaluated on a dataset version holding these pages (the dataset versions that only exist for nothing else are
removed with them). The augmented copies and the compiled lines of these pages are removed too (they are made
again when needed).
"""

import shutil

from mlops import db
from mlops.config import path


def delete(conn, batch_id):
    """Soft delete -> pages taken out of the dataset."""
    with conn.transaction():
        batch = db.one(conn, "UPDATE batches SET deleted_at = now() WHERE id = %s AND deleted_at IS NULL RETURNING id",
                       batch_id)
        if not batch:
            raise ValueError(f"batch {batch_id} does not exist or is already deleted")
        cursor = conn.execute("UPDATE pages SET deleted_at = now() WHERE batch_id = %s AND deleted_at IS NULL",
                              (batch_id,))
    return cursor.rowcount


def restore(conn, batch_id):
    with conn.transaction():
        if not db.one(conn, "UPDATE batches SET deleted_at = NULL WHERE id = %s AND deleted_at IS NOT NULL "
                            "AND status <> 'purged' RETURNING id", batch_id):
            raise ValueError(f"batch {batch_id} is not deleted (or was purged)")
        cursor = conn.execute("UPDATE pages SET deleted_at = NULL WHERE batch_id = %s", (batch_id,))
    return cursor.rowcount


def used_versions(conn, batch_id):
    """The dataset versions holding pages of the batch that a model or an evaluation used."""
    return db.all_rows(conn, """
        SELECT DISTINCT v.id, v.digest FROM dataset_versions v JOIN dataset_pages d ON d.version_id = v.id
        JOIN pages p ON p.id = d.page_id
        WHERE p.batch_id = %s AND (EXISTS (SELECT 1 FROM models m WHERE m.dataset_version_id = v.id)
                                   OR EXISTS (SELECT 1 FROM evaluations e WHERE e.test_version_id = v.id))
        ORDER BY v.id""", batch_id)


def purge(conn, batch_id, confirmation):
    """Remove the pages of a deleted batch for good. confirmation must be "PURGE <batch id>" -> pages removed."""
    if confirmation.strip() != f"PURGE {batch_id}":
        raise ValueError(f'type "PURGE {batch_id}" to confirm')
    batch = db.one(conn, "SELECT * FROM batches WHERE id = %s", batch_id)
    if not batch or batch["deleted_at"] is None:
        raise ValueError(f"batch {batch_id}: delete it first (a purge is only for deleted batches)")
    if db.one(conn, "SELECT 1 AS x FROM jobs WHERE status = 'running'"):
        raise ValueError("a job is running (it may be reading these files): purge once it is done")
    used = used_versions(conn, batch_id)
    if used:
        raise ValueError(f"models or evaluations used dataset versions with these pages "
                         f"({', '.join(v['digest'] for v in used)}): they stay, deleted but traceable")
    pages = db.all_rows(conn, "SELECT id, image_name FROM pages WHERE batch_id = %s", batch_id)
    ids = [p["id"] for p in pages]
    with conn.transaction():
        versions = [r["version_id"] for r in db.all_rows(
            conn, "SELECT DISTINCT version_id FROM dataset_pages WHERE page_id = ANY(%s)", ids)]
        conn.execute("DELETE FROM dataset_pages WHERE version_id = ANY(%s)", (versions,))
        conn.execute("DELETE FROM dataset_versions WHERE id = ANY(%s)", (versions,))
        for table in ("page_origins", "page_writers"):
            conn.execute(f"DELETE FROM {table} WHERE page_id = ANY(%s)", (ids,))
        conn.execute("DELETE FROM pages WHERE id = ANY(%s)", (ids,))
        conn.execute("UPDATE batches SET status = 'purged', report = coalesce(report, '{}'::jsonb) || %s "
                     "WHERE id = %s", (db.json({"purged_pages": len(ids)}), batch_id))
    folder = path("data") / "pages"
    for page in pages:
        (folder / f'{page["id"]}.xml').unlink(missing_ok=True)
        (folder / page["image_name"]).unlink(missing_ok=True)
    augmented = path("data") / "augmented"
    for page_id in ids:
        for file in augmented.glob(f"*/{page_id}_aug*"):
            file.unlink(missing_ok=True)
    shutil.rmtree(path("data") / "compiled", ignore_errors=True)
    return len(ids)