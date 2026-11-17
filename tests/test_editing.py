"""Editing the pages of the dataset (mlops/editing.py): lines, rotation, crop, revisions, restore, and what an
edit changes elsewhere (the dataset digest, the augmented copies, the purge)."""

import pytest
from conftest import make_page, make_zip
from PIL import Image


def ingested(env, pages=6):
    from mlops import db, ingest
    source = env / "export"
    for n in range(pages):
        make_page(source, f"03010108_1_1_I_20{10 + n % 3}_{n + 1}-2-1", (n * 20, 90, 120))
    with db.connection() as conn:
        batch = db.one(conn, "INSERT INTO batches (status) VALUES ('uploaded') RETURNING id")
        ingest.ingest(conn, batch["id"], make_zip(source, env / "export.zip"), log=lambda *_: None)
        return db.all_rows(conn, "SELECT * FROM pages ORDER BY id")


def lines_of(page_id):
    from mlops.editing import files_of
    from mlops.pagexml import read_page
    return read_page(files_of(page_id)[0])["lines"]


def as_payload(lines):
    return [{"id": line["id"], "polygon": line["polygon"], "baseline": line["baseline_points"], "text": line["text"]}
            for line in lines]


def test_lines_are_edited_in_place_and_every_change_is_a_revision(env):
    from mlops import datasets, db, editing
    page = ingested(env)[0]
    before = lines_of(page["id"])
    with db.connection() as conn:
        digest = datasets.current_digest(conn)
        version = datasets.snapshot(conn)
        payload = as_payload(before)
        payload[0]["text"] = "  رسم   ولادة رقم "          # spaces cleaned, the no-break space normalised
        payload[1]["polygon"] = [[1, 1], [150, 1], [150, 18], [1, 18]]
        del payload[2]
        payload.append({"polygon": [[0, 200], [180, 200], [180, 220], [0, 220]], "baseline": [[0, 218], [180, 218]],
                        "text": "سطر جديد", "id": "new-123"})
        result = editing.save_lines(conn, page["id"], payload, base_revision=0, user="ana")
        assert result["revision"] == 1 and result["lines"] == len(before)
        after = lines_of(page["id"])
        assert after[0]["text"] == "رسم ولادة رقم" and after[1]["polygon"][1] == (150.0, 1.0)
        assert [line["id"] for line in after[:2]] == [before[0]["id"], before[1]["id"]]
        assert before[2]["id"] not in [line["id"] for line in after] and after[-1]["text"] == "سطر جديد"
        assert after[-1]["id"] not in {line["id"] for line in before}
        row = db.one(conn, "SELECT * FROM pages WHERE id = %s", page["id"])
        assert row["revision"] == 1 and row["characters"] == sum(len(line["text"]) for line in after)
        # a new dataset version: the edited page counts with its revision; the old version keeps its digest
        assert datasets.current_digest(conn) != digest and datasets.snapshot(conn)["id"] != version["id"]
        assert db.one(conn, "SELECT revision FROM dataset_pages WHERE version_id = %s AND page_id = %s",
                      datasets.snapshot(conn)["id"], page["id"])["revision"] == 1
        history = editing.history(conn, page["id"])
        assert [(h["revision"], h["action"], h["username"]) for h in history] == [(1, "lines", "ana"),
                                                                              (0, "original", "ana")]
        # the same lines again: no new revision
        assert editing.save_lines(conn, page["id"], as_payload(after), base_revision=1)["unchanged"]
        # a stale editor is refused
        with pytest.raises(editing.Conflict):
            editing.save_lines(conn, page["id"], as_payload(after), base_revision=0)
        # restore the original: a new revision with the old files
        assert editing.restore(conn, page["id"], 0, base_revision=1, user="ana")["revision"] == 2
        assert [line["text"] for line in lines_of(page["id"])] == [line["text"] for line in before]
        with pytest.raises(editing.EditError):
            editing.restore(conn, page["id"], 7, base_revision=2)


def test_bad_lines_are_refused_and_the_file_is_left_as_it_was(env):
    from mlops import db, editing
    page = ingested(env)[0]
    xml, _ = editing.files_of(page["id"])
    original = xml.read_bytes()
    with db.connection() as conn:
        for bad in ([{"polygon": [[0, 0], [1, 1]], "baseline": [[0, 0], [1, 1]]}],
                    [{"polygon": [[0, 0], [9, 0], [9, 9]], "baseline": [[0, 0]]}],
                    [{"polygon": [["a", 0], [9, 0], [9, 9]], "baseline": [[0, 0], [9, 9]]}],
                    [{"polygon": [[0, 0], [9, 0], [9, 9]], "baseline": [[0, 0], [9, 9]], "text": 5}],
                    "not a list"):
            with pytest.raises(editing.EditError):
                editing.save_lines(conn, page["id"], bad, base_revision=0)
        assert xml.read_bytes() == original and editing.history(conn, page["id"]) == []
        # points outside the image are brought back inside it
        editing.save_lines(conn, page["id"], [{"polygon": [[-50, -5], [999, 0], [20, 999]],
                                               "baseline": [[-1, 5], [999, 5]], "text": "x"}], base_revision=0)
        line = lines_of(page["id"])[0]
        assert line["polygon"] == [(0, 0), (200, 0), (20, 300)] and line["baseline_points"] == [(0, 5), (200, 5)]


def test_rotation_and_crop_move_the_lines_with_the_image(env):
    from mlops import db, editing
    page = ingested(env)[0]
    xml, image = editing.files_of(page["id"])
    first = lines_of(page["id"])[1]           # polygon 0,10 100,10 100,18 0,18 on a 200 x 300 image (6 lines)
    with db.connection() as conn:
        editing.transform(conn, page["id"], {"op": "rotate", "degrees": 90}, base_revision=0)
        with Image.open(image) as picture:
            assert picture.size == (300, 200)
        moved = lines_of(page["id"])[1]
        assert moved["polygon"][0] == (300 - first["polygon"][0][1], first["polygon"][0][0])
        editing.transform(conn, page["id"], {"op": "rotate", "degrees": 270}, base_revision=1)
        assert lines_of(page["id"])[1]["polygon"] == first["polygon"]
        with Image.open(image) as picture:
            assert picture.size == (200, 300)
        result = editing.transform(conn, page["id"], {"op": "crop", "box": [0, 25, 200, 300]}, base_revision=2)
        with Image.open(image) as picture:
            assert picture.size == (200, 275)
        cropped = lines_of(page["id"])     # the two lines above y = 25 are gone, the third is cut
        assert result["lines"] == len(cropped) == 4 and cropped[0]["polygon"][0] == (0, 0)
        assert all(min(y for _, y in line["polygon"]) >= 0 for line in cropped)
        assert 'imageHeight="275"' in xml.read_text(encoding="utf-8")
        with pytest.raises(editing.EditError):
            editing.transform(conn, page["id"], {"op": "crop", "box": [0, 0, 10, 10]}, base_revision=3)
        with pytest.raises(editing.EditError):
            editing.transform(conn, page["id"], {"op": "rotate", "degrees": 45}, base_revision=3)
        # back to the page as ingested, image included
        editing.restore(conn, page["id"], 0, base_revision=3)
        with Image.open(image) as picture:
            assert picture.size == (200, 300)
        assert lines_of(page["id"])[1]["polygon"] == first["polygon"]


def test_an_edit_waits_for_a_job_reading_the_pages_and_removes_the_augmented_copies(env):
    from mlops import db, editing, jobs
    from mlops.config import path
    page = ingested(env)[0]
    augmented = path("data") / "augmented" / "abc-c1-s42"
    augmented.mkdir(parents=True)
    (augmented / f"{page['id']}_aug1.xml").write_text("x")
    (augmented / "otherpage_aug1.xml").write_text("x")
    with db.connection() as conn:
        job = jobs.enqueue(conn, "train", "htr")
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = %s", (job["id"],))
        with pytest.raises(editing.Conflict, match="reading the pages"):
            editing.save_lines(conn, page["id"], as_payload(lines_of(page["id"]))[:2], base_revision=0)
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job["id"],))
        editing.save_lines(conn, page["id"], as_payload(lines_of(page["id"]))[:2], base_revision=0)
    assert not (augmented / f"{page['id']}_aug1.xml").exists() and (augmented / "otherpage_aug1.xml").exists()


def test_a_purge_removes_the_revisions(env):
    from mlops import batches, db, editing
    from mlops.config import path
    page = ingested(env)[0]
    with db.connection() as conn:
        editing.save_lines(conn, page["id"], as_payload(lines_of(page["id"]))[:1], base_revision=0)
        assert (path("data") / "revisions" / page["id"]).is_dir()
        batches.delete(conn, page["batch_id"])
        batches.purge(conn, page["batch_id"], f"PURGE {page['batch_id']}")
        assert not (path("data") / "revisions" / page["id"]).exists()
        assert db.one(conn, "SELECT count(*) AS n FROM page_revisions")["n"] == 0
