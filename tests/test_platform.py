"""Tests of the platform. The database tests need MLOPS_TEST_DATABASE_URL (see README: Tests); the others run
anywhere. No GPU, no kraken, no W&B account: tests/fakes stand for ketos and kraken, W&B is offline."""

import json
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import make_page, make_zip
from mlops.kraken_output import blla_report, htr_report, train_metrics
from mlops.textmetrics import bootstrap, cer, word_f1

FIXTURES = Path(__file__).parent / "fixtures"


# --- without a database ------------------------------------------------------------------------------------
def test_text_metrics():
    assert cer(["رسم ولادة"], ["رسم ولادة"]) == 0
    assert cer(["abcd"], ["abed"]) == 25.0
    assert cer(["a  b"], ["a b"]) == 0                                    # spaces normalised
    assert word_f1(["a b", "c"], ["c", "a b"]) == 1.0                     # the order of the lines does not matter
    assert word_f1(["a b c d"], ["a b"]) == pytest.approx(2 / 3)
    mean, low, high = bootstrap([1.0] * 50, samples=200)
    assert mean == low == high == 1.0
    mean, low, high = bootstrap([1.0, -1.0] * 50, samples=500)
    assert low < 0 < high                                                  # noise: the interval crosses 0


def test_ketos_reports():
    result = htr_report((FIXTURES / "ketos_test_report.txt").read_text(encoding="utf-8"))
    assert result["metrics"]["character_accuracy"] == 97.27 and result["errors"][0] == (371, "SPACE", "")
    assert blla_report("│ Overall │ 0.95 │ 0.93 │ 0.94 │")["metrics"] == {"precision": 0.95, "recall": 0.93, "f1": 0.94}
    assert train_metrics("htr", "stage 2/∞ val_accuracy: 0.95") == {"val_accuracy": 0.95}


def test_group_and_split():
    from mlops.ingest import assign_groups, group_of
    pattern = r"^(.+?_\d{4})_"
    assert group_of("03010108_1_1_I_2016_162-2-1", pattern) == group_of("03010108_1_1_I_2016_7-2-1", pattern)
    assert group_of("03010108_1_1_I_2016_162-2-1", pattern) != group_of("03010108_1_1_I_2017_162-2-1", pattern)
    assert group_of("other", pattern) != group_of("other2", pattern)                  # no match: its own group
    assert "2016" not in group_of("03010108_1_1_I_2016_1", pattern)                   # a keyed hash, not the name
    # few large groups (7 registers, as in birth_certificates_v2): no split is left empty
    sizes = {"a": 120, "b": 100, "c": 90, "d": 80, "e": 70, "f": 60, "g": 50}
    splits = assign_groups({}, sizes, 0.15, 0.15)
    pages = {s: sum(n for g, n in sizes.items() if splits[g] == s) for s in ("train", "val", "test")}
    assert all(pages.values()) and pages["train"] > pages["val"] and pages["train"] > pages["test"]
    assert assign_groups({}, sizes, 0.15, 0.15) == splits                              # only depends on the data
    # many small groups: close to 70 / 15 / 15
    many = {f"g{i:03d}": 5 + i % 7 for i in range(200)}
    splits = assign_groups({}, many, 0.15, 0.15)
    total = sum(many.values())
    for split, share in (("train", 0.7), ("val", 0.15), ("test", 0.15)):
        assert abs(sum(n for g, n in many.items() if splits[g] == split) / total - share) < 0.02
    # later batches fill the split that is short
    assert set(assign_groups({"train": 70, "val": 15, "test": 0}, {"x": 10}, 0.15, 0.15).values()) == {"test"}


def test_gate():
    from mlops.evaluate import compare
    production = {f"p{i}": {"page_cer": 5.0} for i in range(40)}
    better = {f"p{i}": {"page_cer": 4.0} for i in range(40)}
    comparison, passed, message = compare("htr", better, production, old_ids={f"p{i}" for i in range(20)})
    assert passed and comparison["gain"] == 1.0 and comparison["old"]["pages"] == 20 and comparison["new"]["pages"] == 20
    mixed = {f"p{i}": {"page_cer": 4.0 if i % 2 else 6.0} for i in range(40)}
    assert not compare("htr", mixed, production, set())[1]                            # not better with confidence
    forgets = {f"p{i}": {"page_cer": 5.5 if i < 20 else 2.0} for i in range(40)}
    comparison, passed, message = compare("htr", forgets, production, {f"p{i}" for i in range(20)})
    assert not passed and "old test pages" in message                                 # worse on the old pages
    words = {f"p{i}": {"e2e_word_f1": 0.9} for i in range(30)}
    assert compare("blla", words, {k: {"e2e_word_f1": 0.8} for k in words}, set())[1]  # higher is better


# --- with the database -------------------------------------------------------------------------------------
def ingest_zip(conn, tmp, name, pages, force_split=None):
    from mlops import db
    from mlops.ingest import ingest
    folder = tmp / f"src-{name}"
    for stem, color in pages:
        make_page(folder, stem, color)
    batch = db.one(conn, "INSERT INTO batches DEFAULT VALUES RETURNING id")
    return ingest(conn, batch["id"], make_zip(folder, tmp / f"{name}.zip"), force_split, log=lambda *_: None)


def test_ingest_pseudonymises_checks_and_splits_by_group(env):
    from mlops import db
    with db.connection() as conn:
        folder = env / "src"
        for register in range(12):
            for act in range(4):
                make_page(folder, f"03010108_1_{register}_I_2016_{act}-2-1", (register * 20, act * 50, 10))
        make_page(folder, "few_lines", (1, 2, 3), lines=2)
        make_page(folder, "no_baseline", (4, 5, 6), baselines=False)
        make_page(folder, "untranscribed", (7, 8, 9), texts=["", "", "", "", "", "x"])
        (folder / "orphan.xml").write_text((folder / "few_lines.xml").read_text().replace("few_lines.jpg", "gone.jpg"))
        batch = db.one(conn, "INSERT INTO batches DEFAULT VALUES RETURNING id")
        from mlops.ingest import ingest
        report = ingest(conn, batch["id"], make_zip(folder, env / "a.zip"), log=lambda *_: None)
        assert sum(report["added"].values()) == 48 and report["rejected"] == 4
        assert report["reasons"] == {"too few lines": 1, "line without baseline or polygon": 1,
                                     "lines not transcribed": 1, "image not found": 1}
        # every page of a register lands in the same split
        groups = db.all_rows(conn, "SELECT group_key, count(DISTINCT split) AS n FROM pages GROUP BY group_key")
        assert len(groups) == 12 and all(g["n"] == 1 for g in groups)
        # opaque names in the dataset, the originals only in page_origins
        pages = env / "data" / "pages"
        assert all("03010108" not in p.name and "03010108" not in p.read_text(errors="ignore")
                   for p in pages.iterdir())
        assert db.one(conn, "SELECT count(*) AS n FROM page_origins WHERE original_name LIKE '%%03010108%%'")["n"] == 48
        page = db.one(conn, "SELECT * FROM pages LIMIT 1")
        assert f'imageFilename="{page["image_name"]}"' in (pages / f'{page["id"]}.xml').read_text()
        # the same export again: duplicates only
        again = ingest(conn, batch["id"], env / "a.zip", log=lambda *_: None)
        assert again["duplicates"] == 48 and sum(again["added"].values()) == 0


def test_a_group_keeps_its_split(env):
    from mlops import db
    with db.connection() as conn:
        ingest_zip(conn, env, "old", [(f"X_2010_{i}-1", (i * 40, 0, 0)) for i in range(3)], force_split="test")
        ingest_zip(conn, env, "new", [(f"X_2010_{i}-1", (i * 40, 200, 200)) for i in range(3, 6)])
        assert db.one(conn, "SELECT count(*) AS n FROM pages")["n"] == 6
        assert {r["split"] for r in db.all_rows(conn, "SELECT split FROM pages")} == {"test"}


def test_zip_slip_is_refused(env, tmp_path):
    import zipfile
    from mlops.ingest import extract
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as archive:
        archive.writestr("../../evil.txt", "x")
        archive.writestr("ok/page.txt", "y")
    target = tmp_path / "out"
    target.mkdir()
    extract(tmp_path / "evil.zip", target)
    assert (target / "ok" / "page.txt").is_file() and not (tmp_path / "evil.txt").exists()


def test_resplit_balances_the_groups(env):
    from mlops import db
    from mlops.ingest import resplit
    with db.connection() as conn:
        pages = [(f"03010108_1_{r}_I_2016_{a}-1", (r * 30, a * 8, 99)) for r in range(7) for a in range(10 + r)]
        ingest_zip(conn, env, "v2", pages, force_split="train")                        # everything in train
        counts = resplit(conn, log=lambda *_: None)
        assert all(counts.values()) and sum(counts.values()) == len(pages)
        groups = db.all_rows(conn, "SELECT group_key, count(DISTINCT split) AS n FROM pages GROUP BY group_key")
        assert all(g["n"] == 1 for g in groups)


def test_dataset_versions(env):
    from mlops import datasets, db
    with db.connection() as conn:
        ingest_zip(conn, env, "a", [(f"p{i}", (i * 20, 10, 10)) for i in range(10)])
        first = datasets.snapshot(conn)
        assert datasets.snapshot(conn)["id"] == first["id"]                    # same pages: same version
        ingest_zip(conn, env, "b", [(f"q{i}", (i * 20, 200, 200)) for i in range(5)])
        second = datasets.snapshot(conn)
        assert second["id"] != first["id"] and sum(second["counts"].values()) == 15
        assert sum(len(datasets.pages(conn, first["id"], s)) for s in ("train", "val", "test")) == 10


def test_job_queue(env):
    from mlops import db, jobs
    with db.connection() as a, db.connection() as b:
        first = jobs.enqueue(a, "train", "htr")
        second = jobs.enqueue(a, "train", "blla")
        with a.transaction():   # a holds the first job: b takes the next one (SKIP LOCKED)
            db.one(a, "SELECT * FROM jobs WHERE id = %s FOR UPDATE", first["id"])
            assert jobs.claim(b)["id"] == second["id"]
        assert jobs.claim(a)["id"] == first["id"] and jobs.claim(a) is None
        third = jobs.enqueue(a, "evaluate", "htr", model_id=1)
        jobs.cancel(a, third["id"])
        assert db.one(a, "SELECT status FROM jobs WHERE id = %s", third["id"])["status"] == "cancelled"
        jobs.cancel(a, first["id"])
        assert jobs.cancel_requested(a, first["id"])
        jobs.recover(a)
        assert db.one(a, "SELECT status FROM jobs WHERE id = %s", first["id"])["status"] == "failed"
        log = jobs.Log(a, second["id"], flush_every=0)
        log("line 1\n")
        log("line 2\n")
        log.flush()
        assert jobs.tail(a, second["id"]) == "line 1\nline 2"


def test_cancel_stops_the_process(env):
    from mlops.process import Cancelled, run
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    with pytest.raises(Cancelled):
        run(["sleep", "60"], log=lambda *_: None, should_stop=stop.is_set)


def test_whole_flow_with_the_worker(env, madoc_remote, monkeypatch):
    """upload -> ingest -> train htr and blla -> evaluate -> pull request, with fake ketos / kraken / GitHub."""
    from mlops import db, jobs, release, worker
    opened = []
    monkeypatch.setattr(release, "create_pull_request", lambda title, body, head: opened.append((title, body, head))
                        or "https://github.com/NiemaAM/MaDoc/pull/7")
    monkeypatch.setattr("mlops.resources.free_gpu", lambda log=print: None)
    monkeypatch.setattr("mlops.train.free_gpu", lambda log=print: None)
    monkeypatch.setattr("mlops.evaluate.free_gpu", lambda log=print: None)
    with db.connection() as conn:
        folder = env / "export"
        for register in range(20):
            for act in range(3):
                make_page(folder, f"03010108_1_{register}_I_2015_{act}-1", (register * 12, act * 60, 30))
        upload = env / "data" / "uploads" / "batch-1.zip"
        upload.parent.mkdir(parents=True)
        make_zip(folder, upload)
        batch = db.one(conn, "INSERT INTO batches DEFAULT VALUES RETURNING id")
        jobs.enqueue(conn, "ingest", batch_id=batch["id"], upload=str(upload))
        jobs.enqueue(conn, "train", "htr")
        jobs.enqueue(conn, "train", "blla")
        worker.main(once=True)
        done = db.all_rows(conn, "SELECT kind, task, status, error FROM jobs ORDER BY id")
        assert [j["status"] for j in done] == ["done"] * 3, done
        assert not upload.exists()                                                         # ingested, removed
        models = db.all_rows(conn, "SELECT * FROM models ORDER BY id")
        assert [m["task"] for m in models] == ["htr", "blla"] and all(Path(m["path"]).is_file() for m in models)
        for model in models:
            jobs.enqueue(conn, "evaluate", model["task"], model_id=model["id"])
        worker.main(once=True)
        evaluations = db.all_rows(conn, "SELECT * FROM evaluations ORDER BY id")
        assert [e["passed"] for e in evaluations] == [True, True], [e["message"] for e in evaluations]
        assert evaluations[0]["comparison"]["metric"] == "page_cer" and evaluations[0]["comparison"]["gain"] > 0
        assert evaluations[1]["comparison"]["metric"] == "e2e_word_f1"
        jobs.enqueue(conn, "release", "htr", evaluation_id=evaluations[0]["id"])
        worker.main(once=True)
        release_job = db.one(conn, "SELECT * FROM jobs WHERE kind = 'release'")
        assert release_job["status"] == "done", release_job["error"]
        assert db.one(conn, "SELECT pr_url FROM evaluations WHERE id = %s", evaluations[0]["id"])["pr_url"]
        title, body, head = opened[0]
        assert head.startswith("models/htr-") and "page_cer" in body and "| character_accuracy |" in body
        # the branch on "GitHub" has the new model and its card
        card = json.loads(subprocess.run(["git", "show", f"{head}:models/MODEL_CARD.json"], cwd=madoc_remote,
                                         check=True, capture_output=True, text=True).stdout)
        assert card["htr"]["artifact"] == models[0]["artifact"] and card["htr"]["test_version"]
        # no file name of a certificate in the logs
        assert not db.one(conn, "SELECT count(*) AS n FROM job_logs WHERE line LIKE '%%03010108%%'")["n"]


def test_the_app_builds(env):
    pytest.importorskip("gradio")
    from mlops import ui
    assert ui.build_app() is not None
    assert [row[0] for row in ui.dataset_table()] == ["train", "val", "test"]
    assert "CPU" in ui.server_html()