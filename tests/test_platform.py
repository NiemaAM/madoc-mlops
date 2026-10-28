"""Tests of the platform. The database tests need MLOPS_TEST_DATABASE_URL (see README: Tests); the others run
anywhere. No GPU, no kraken, no W&B account: tests/fakes stand for ketos and kraken, W&B is offline."""

import json
import sys
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import make_page, make_zip
from mlops.kraken_output import blla_report, htr_report, train_metrics
from mlops.textmetrics import bootstrap, cer, word_f1

FIXTURES = Path(__file__).parent / "fixtures"


# --- without a database ------------------------------------------------------------------------------------
def test_line_cer_ignores_the_order_of_the_lines():
    from mlops.textmetrics import cer, line_cer
    truth = ["رسم ولادة رقم", "في يوم عام", "جماعة خريبكة"]
    assert line_cer(truth, list(reversed(truth))) == 0.0 and cer(truth, list(reversed(truth))) > 30
    assert 0 < line_cer(truth, ["رسم ولادة رقم", "في يوم عام"]) < 100                 # a line missed
    assert line_cer(truth, truth + ["x"]) > 0                                         # a line invented
    assert line_cer([], []) == 0.0


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
    from collections import Counter
    from mlops.ingest import group_of
    from mlops.splitting import assign_groups, coverage
    pattern = r"^(.+?_\d{4})_"
    assert group_of("03010108_1_1_I_2016_162-2-1", pattern) == group_of("03010108_1_1_I_2016_7-2-1", pattern)
    assert group_of("03010108_1_1_I_2016_162-2-1", pattern) != group_of("03010108_1_1_I_2017_162-2-1", pattern)
    assert group_of("other", pattern) != group_of("other2", pattern)                  # no match: its own group
    assert "2016" not in group_of("03010108_1_1_I_2016_1", pattern)                   # a keyed hash, not the name

    def share(groups, splits):
        totals = {s: sum(sum(c.values()) for g, c in groups.items() if splits[g] == s) for s in ("train", "val", "test")}
        return {s: n / sum(totals.values()) for s, n in totals.items()}

    # few large groups (7 registers, as in birth_certificates_v2): no split is left empty
    sizes = {"a": 120, "b": 100, "c": 90, "d": 80, "e": 70, "f": 60, "g": 50}
    groups = {g: Counter({"ا": n * 30, "ل": n * 20}) for g, n in sizes.items()}
    splits = assign_groups({}, groups, 0.15, 0.15)
    shares = share(groups, splits)
    assert all(shares.values()) and shares["train"] > shares["val"] and shares["train"] > shares["test"]
    assert assign_groups({}, groups, 0.15, 0.15) == splits                            # only depends on the data
    # many small groups: close to 70 / 15 / 15 of the characters
    many = {f"g{i:03d}": Counter({"ر": 50 + 10 * (i % 7), "م": 30}) for i in range(200)}
    for split, target in share(many, assign_groups({}, many, 0.15, 0.15)).items():
        assert abs(target - {"train": 0.7, "val": 0.15, "test": 0.15}[split]) < 0.02
    # later batches fill the split that is short
    fixed = {"train": Counter({"a": 700}), "val": Counter({"a": 150}), "test": Counter()}
    assert assign_groups(fixed, {"x": Counter({"a": 150})}, 0.15, 0.15) == {"x": "test"}
    # coverage: a rare character in 3 groups ends up in every split, even if a size-only split would put the 3
    # groups in train; a character in a single group goes to train
    groups = {f"g{i}": Counter({"a": 100}) for i in range(14)}
    for i in (0, 1, 2):
        groups[f"g{i}"]["ڭ"] = 2
    groups["g3"]["ﻻ"] = 1
    splits = assign_groups({}, groups, 0.15, 0.15)
    by_split = {s: sum((c for g, c in groups.items() if splits[g] == s), Counter()) for s in ("train", "val", "test")}
    report = coverage(by_split)
    assert {splits[g] for g in ("g0", "g1", "g2")} == {"train", "val", "test"} and splits["g3"] == "train"
    assert report["missing"]["train"] == "" and report["in_all_splits"] == 2
    assert all(abs(report["share"][s] - t) <= 0.33 * t + 1e-9 for s, t in (("val", .15), ("test", .15)))
    # page types: val and test get their share of the page 2 (the pages MaDoc structures), within the tolerance
    pages = {"g0": (60, 10), "g1": (50, 15), "g2": (10, 60), "g3": (15, 55), "g4": (40, 40), "g5": (20, 30),
             "g6": (5, 25)}
    groups = {g: Counter({"a": 40 * (p1 + p2), "b": 20 * p2 + 5}) for g, (p1, p2) in pages.items()}
    types = {g: Counter({"1": p1, "2": p2}) for g, (p1, p2) in pages.items()}
    splits = assign_groups({}, groups, 0.15, 0.15, 0.33, None, types)
    page2 = sum(p2 for _, p2 in pages.values())
    for split in ("val", "test"):
        share = sum(p2 for g, (_, p2) in pages.items() if splits[g] == split) / page2
        assert 0.10 <= share <= 0.20, (split, share)


def test_evaluation_charts(tmp_path, monkeypatch):
    import wandb
    from mlops.eval_charts import log_model_run
    from mlops.kraken_output import blla_report, htr_report
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    # the tables of a real `ketos segtest` (kraken 7.0.2), as the worker logs them
    segtest = ("│ baselines │ default   │ 2         │ 2         │ 2424     │ 2424     │ ok     │\n"
               "│ aux      │ _start_separator │ 0.997          │ 0.340 │ N/A          │\n"
               "│ aux      │ _end_separator   │ 0.998          │ 0.344 │ N/A          │\n"
               "│ Overall │ 0.995     │ 0.954  │ 0.974 │\n"
               "│ default │ 0.995     │ 0.954  │ 0.974 │\n")
    report = blla_report(segtest + segtest)                     # ketos prints its tables twice in a pipe
    assert report["metrics"] == {"precision": 0.995, "recall": 0.954, "f1": 0.974}
    assert report["classes"] == [{"class": "default", "precision": 0.995, "recall": 0.954, "f1": 0.974}]
    assert [(r["class"], r["pixel_accuracy"], r["iou"]) for r in report["iou"]] == [
        ("_start_separator", 0.997, 0.34), ("_end_separator", 0.998, 0.344)]       # not the counts table
    pages = {"p1": {"e2e_word_f1": 0.9, "e2e_cer": 5.0}, "p2": {"e2e_word_f1": 0.8, "e2e_cer": 7.0}}
    result = {"metrics": {**report["metrics"], "mean_e2e_word_f1": 0.85, "mean_e2e_cer": 6.0}, "pages": pages,
              "report": report}
    run = wandb.init(project="t", dir=str(tmp_path))
    try:
        keys = log_model_run("blla", result)                                       # one model per run
        assert {"precision", "recall", "f1", "mean_e2e_word_f1", "precision_recall_f1", "pixel_accuracy_plot",
                "iou_plot", "precision_recall_f1_table", "iou_table", "pages"} <= set(keys)
        assert not any("candidate" in k or "production" in k for k in keys)
        text = (FIXTURES / "ketos_test_report.txt").read_text(encoding="utf-8")
        report = htr_report(text)
        htr = {"metrics": {**report["metrics"], "mean_page_cer": 3.0}, "pages": {"p1": {"page_cer": 3.0}},
               "report": report}
        keys = log_model_run("htr", htr)
        assert {"character_accuracy", "accuracy_comparison", "error_types", "CER_by_language", "language_accuracy",
                "character_error_table", "metrics", "pages"} <= set(keys)
    finally:
        run.finish()


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
        result = resplit(conn, log=lambda *_: None)
        counts = result["pages"]
        assert all(counts.values()) and sum(counts.values()) == len(pages)
        assert result["coverage"]["missing"]["train"] == "" and result["coverage"]["characters"] > 0
        groups = db.all_rows(conn, "SELECT group_key, count(DISTINCT split) AS n FROM pages GROUP BY group_key")
        assert all(g["n"] == 1 for g in groups)


def test_page_types_are_balanced_and_never_split_apart(env):
    from mlops import db
    from mlops.ingest import page_type_of, resplit
    from mlops.config import params
    from mlops.ingest import group_of
    pattern = params()["ingest"]["page_type_pattern"]
    assert page_type_of("03010108_1_1_I_2004_1-1-1", pattern) == "1"
    assert page_type_of("03010108_1_1_I_2004_1-2-1", pattern) == "2"
    assert page_type_of("scan", pattern) == "other"
    group = params()["ingest"]["group_pattern"]
    assert group_of("03010108_1_1_I_2004_1-1-1", group) == group_of("03010108_1_1_I_2004_1-2-1", group)
    with db.connection() as conn:
        # 9 register-years; in 3 of them only page 2 was transcribed (a split without page 1 would be easy to get)
        pages = []
        for r in range(9):
            for act in range(6 + r):
                for side in ((2,) if r < 3 else (1, 2)):
                    pages.append((f"03010108_1_{r}_I_2016_{act}-{side}-1", (r * 25, act * 9, side * 60)))
        report = ingest_zip(conn, env, "types", pages)
        assert sum(report["added"].values()) == len(pages)
        rows = db.all_rows(conn, "SELECT p.split, p.page_type, o.original_name FROM pages p "
                                 "JOIN page_origins o ON o.page_id = p.id")
        assert {r["page_type"] for r in rows} == {"1", "2"}
        # the two pages of an act are always in the same split (no leak of the names)
        acts = {}
        for r in rows:
            acts.setdefault(r["original_name"].split("-")[0], set()).add(r["split"])
        assert all(len(splits) == 1 for splits in acts.values())
        # every split has both types
        for result in (report["coverage"], resplit(conn, log=lambda *_: None)["coverage"]):
            assert all(set(result["page_types"][s]) == {"1", "2"} for s in ("train", "val", "test"))


def test_augment_job_and_browse(env):
    from mlops import augment, browse, db
    with db.connection() as conn:
        ingest_zip(conn, env, "b", [(f"03010108_1_{r}_I_2016_{a}-{s}-1", (r * 40, a * 30, s * 90))
                                    for r in range(7) for a in range(2) for s in (1, 2)])
        result = augment.run_augment(conn, {"args": {"copies": 1, "seed": 3}}, lambda *_: None, lambda: False, None)
        train = db.one(conn, "SELECT count(*) AS n FROM pages WHERE split = 'train'")["n"]
        assert result["pages"] == train and augment.augmentations()[0]["seed"] == "3"
        # browse: the pages of a split, of a type, the augmented ones; a page with its lines drawn
        options, note = browse.choices(conn, "train")
        assert len(options) == train and all("03010108" not in label for label, _ in options)
        assert all("page 1" in label for label, _ in browse.choices(conn, "test", "1")[0])
        augmented, note = browse.choices(conn, browse.AUGMENTED)
        assert len(augmented) == train and "_aug" in augmented[0][1]
        image, text, xml, info = browse.show(options[0][1])
        assert image.size == (200, 300) and "br-line" in text and "PcGts" in xml and "6 lines" in info
        assert browse.show(augmented[0][1])[0] is not None
        assert browse.step(options, options[-1][1], 1) == options[0][1]
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            browse.resolve("../../etc/passwd.xml")


def test_normalization():
    from mlops import normalize
    table = normalize.table()
    assert table["\u00a0"] == " " and table["\u200b"] == ""
    assert normalize.text("رسم\u00a0ولادة\u200b", table) == ("رسم ولادة", {"U+00A0": 1, "U+200B": 1})
    page = ('<PcGts><TextLine id="a"><TextEquiv><Unicode>A\u00a0&amp;\u00a0B</Unicode></TextEquiv></TextLine>'
            '<TextLine id="b"><TextEquiv><pc:Unicode>x&#160;y</pc:Unicode></TextEquiv></TextLine>'
            '<Metadata>\u00a0kept</Metadata></PcGts>')
    fixed, found = normalize.xml(page, table)
    assert "<Unicode>A &amp; B</Unicode>" in fixed and "<pc:Unicode>x y</pc:Unicode>" in fixed
    assert "\u00a0kept" in fixed and found == {"U+00A0": 3}                       # only the transcriptions
    assert normalize.xml(fixed, table) == (fixed, {})


def test_ingest_and_cli_normalize(env):
    from mlops import db
    from mlops.ingest import normalize_dataset
    from mlops.pagexml import read_page
    texts = ["رسم\u00a0ولادة", "في\u200bيوم", "عام", "سنة", "جماعة", "موافق"]
    with db.connection() as conn:
        folder = env / "src-n"
        make_page(folder, "03010108_1_1_I_2016_1-1-1", (10, 20, 30), texts=texts)
        batch = db.one(conn, "INSERT INTO batches DEFAULT VALUES RETURNING id")
        from mlops.ingest import ingest
        report = ingest(conn, batch["id"], make_zip(folder, env / "n.zip"), log=lambda *_: None)
        assert report["normalized"] == {"U+00A0": 1, "U+200B": 1}
        page = db.one(conn, "SELECT * FROM pages")
        stored = env / "data" / "pages" / f'{page["id"]}.xml'
        assert [line["text"] for line in read_page(stored)["lines"]][:2] == ["رسم ولادة", "فييوم"]
        # a page already in the dataset with a no-break space (ingested before the setting existed)
        stored.write_text(stored.read_text(encoding="utf-8").replace("عام", "عام\u00a0x"), encoding="utf-8")
        (env / "data" / "augmented" / "old").mkdir(parents=True)
        assert normalize_dataset(conn, log=lambda *_: None) == {"U+00A0": 1}
        assert "\u00a0" not in stored.read_text(encoding="utf-8") and not (env / "data" / "augmented").exists()
        assert normalize_dataset(conn, log=lambda *_: None) == {}


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
        # the metrics came live from the TensorBoard files of ketos (not only from its final progress bar)
        assert models[1]["val_metrics"]["val_bl_f1"] == pytest.approx(0.83) and "epoch" in models[1]["val_metrics"]
        for model in models:
            jobs.enqueue(conn, "evaluate", model["task"], model_id=model["id"])
        worker.main(once=True)
        evaluations = db.all_rows(conn, "SELECT * FROM evaluations ORDER BY id")
        assert [e["passed"] for e in evaluations] == [True, True], [e["message"] for e in evaluations]
        assert evaluations[0]["comparison"]["metric"] == "page_cer" and evaluations[0]["comparison"]["gain"] > 0
        assert evaluations[1]["comparison"]["metric"] == "e2e_word_f1"
        # one W&B run per model, in one group per evaluation
        assert evaluations[1]["comparison"]["wandb"]["group"].startswith("blla-eval-job")
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


def test_training_options_and_split_settings(env):
    from mlops import db, settings
    from mlops.ingest import resplit
    from mlops.train import ketos_command
    with db.connection() as conn:
        htr = settings.training(conn, "htr")                                   # params.yaml
        assert htr["batch_size"] == 30 and htr["quit"] == "early" and "batch_size" not in settings.training(conn, "blla")
        chosen = settings.clean("htr", {**htr, "epochs": "40", "lag": 7, "learning_rate": "1e-4", "schedule": "cosine",
                                        "cos_max": 40, "kraken_augment": True, "extra_args": "--warmup 200"})
        settings.save(conn, "train.htr", chosen)
        assert settings.training(conn, "htr")["learning_rate"] == 1e-4           # saved from the app
        cmd = ketos_command("htr", settings.training(conn, "htr"), "/m.mlmodel", "t.txt", "v.txt", "out")
        text = " ".join(cmd)
        for part in ("train --load /m.mlmodel", "--epochs 40", "--quit early", "--lag 7", "--lrate 0.0001",
                     "--schedule cosine", "--cos-max 40", "--batch-size 30", "--augment", "--warmup 200",
                     "--resize new"):
            assert part in text, part
        blla = ketos_command("blla", settings.training(conn, "blla"), "/b.mlmodel", "t", "v", "o")
        assert "segtrain" in blla and "--batch-size" not in blla and "--lrate" in blla
        settings.reset(conn, "train.htr")
        assert settings.training(conn, "htr") == htr
        with pytest.raises(ValueError):
            settings.clean("htr", {**htr, "resize": "both"})
        with pytest.raises(ValueError):
            settings.check_split({"val_fraction": 0.6, "test_fraction": 0.5, "split_tolerance": 0.3,
                                  "split_weights": {}})
        # the split: preview changes nothing, apply splits again with the chosen fractions
        pages = [(f"03010108_1_{r}_I_2016_{a}-{t}-1", (r * 25, a * 15, t * 70)) for r in range(10) for a in range(6)
                 for t in (1, 2)]
        ingest_zip(conn, env, "s", pages)
        before = {r["id"]: r["split"] for r in db.all_rows(conn, "SELECT id, split FROM pages")}
        split = {"val_fraction": 0.3, "test_fraction": 0.3, "split_tolerance": 0.5,
                 "split_weights": {"missing_character": 1000, "page_types": {"1": 1000, "2": 3000}}}
        preview = resplit(conn, log=lambda *_: None, split=split, apply=False)
        assert {r["id"]: r["split"] for r in db.all_rows(conn, "SELECT id, split FROM pages")} == before
        assert "preview" in preview["text"]
        settings.save(conn, "split", split)
        result = resplit(conn, log=lambda *_: None)                          # the saved settings
        assert result["groups"]["val"] >= 2 and result["groups"]["test"] >= 2


def test_a_paused_worker_takes_no_job(env):
    from mlops import cli, db, jobs, settings, worker
    with db.connection() as conn:
        job = jobs.enqueue(conn, "augment")
        settings.pause(conn)
        worker.main(once=True)                                   # paused: the queued job stays queued
        assert db.one(conn, "SELECT status FROM jobs WHERE id = %s", job["id"])["status"] == "queued"
        assert cli.wait_idle(timeout=0) == 0                     # nothing running
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = %s", (job["id"],))
        assert cli.wait_idle(timeout=0, every=0) == 1            # a job running: the deployment waits
        conn.execute("UPDATE jobs SET status = 'queued' WHERE id = %s", (job["id"],))
        settings.resume(conn)
        assert not settings.paused(conn)


def test_the_app_builds(env):
    pytest.importorskip("gradio")
    from mlops import ui
    assert ui.build_app() is not None
    assert [row[0] for row in ui.dataset_table()] == ["train", "val", "test"]
    assert "CPU" in ui.server_html()


def test_vocabulary(env):
    from collections import Counter
    from mlops import db, vocabulary
    with db.connection() as conn:
        ingest_zip(conn, env, "voc", [("03010108_1_1_I_2016_1", (10, 20, 30))])
        by_split = vocabulary.vocabulary(conn)
    total = sum(by_split.values(), Counter())
    assert total["ر"] > 0 and " " in total
    page = vocabulary.vocabulary_html(by_split)
    assert "voc-cell" in page and "␣" in page and f"{len(total)}</b> distinct" in page
    # red under the threshold, dashed when not in train, combining marks on a dotted circle
    page = vocabulary.vocabulary_html({"train": Counter({"a": 100, "\u064e": 3}), "val": Counter({"b": 2}),
                                       "test": Counter()})
    assert page.count("voc-rare'") + page.count("voc-rare voc") == 2 and "voc-no-train" in page
    assert "◌\u064e" in page and "LATIN SMALL LETTER B (U+0062) — train 0 · val 2 · test 0" in page
    # the split summary (as printed by ingest / resplit) and the text copied by the Copy button
    by_split = {"train": Counter({"a": 100, "b": 5}), "val": Counter({"b": 2}), "test": Counter({"a": 9})}
    types = {"train": Counter({"1": 3, "2": 5}), "val": Counter({"1": 1}), "test": Counter({"2": 1})}
    summary = vocabulary.coverage_html(by_split, types)
    assert "characters missing from this split" in summary and "missing: train 0, val 1 (&#x27;a&#x27;)" in summary
    page = vocabulary.vocabulary_html(by_split, "all")
    assert "id='voc-tsv'" in page and "characters missing" not in page
    tsv = vocabulary.vocabulary_tsv(by_split).splitlines()                          # what the copy button copies
    assert tsv[0].split("\t") == ["character", "code", "name", "total", "train", "val", "test"]
    assert tsv[1].split("\t") == ["a", "U+0061", "LATIN SMALL LETTER A", "109", "100", "0", "9"]


def test_ketos_metrics_reach_wandb_through_tensorboard(tmp_path, monkeypatch):
    """ketos run by the worker prints its progress bar only at the end: the metrics are read from its TensorBoard
    files while it trains."""
    pytest.importorskip("tensorboard")
    import os
    import subprocess
    import wandb
    from conftest import FAKES
    from mlops.tb_bridge import STEP, TensorBoardBridge
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    logged = []
    log_dir = tmp_path / "tb"
    log_dir.mkdir()
    run = wandb.init(project="test")
    monkeypatch.setattr(wandb, "log", lambda values: logged.append(values))
    bridge = TensorBoardBridge(log_dir, interval=0.2, log=lambda *_: None).start()
    subprocess.run([str(FAKES / "ketos"), "segtrain", "--output", str(tmp_path / "ck")],
                   check=True, capture_output=True, env={**os.environ, "MLOPS_TB_DIR": str(log_dir)})
    bridge.stop()
    run.finish()
    assert [row[STEP] for row in logged] == sorted(row[STEP] for row in logged)        # in the order of the steps
    assert {row[STEP] for row in logged} == {10, 20, 30}
    assert bridge.latest["val_bl_f1"] == pytest.approx(0.83) and bridge.latest["epoch"] == 2
    assert bridge.poll() == 0                                                          # nothing sent twice


def test_ketos_gets_a_tensorboard_logger(tmp_path, monkeypatch):
    """kraken 7.0.2 ignores "ketos --logger": mlops.ketos_logged gives the trainer its logger."""
    package = tmp_path / "kraken"
    (package / "train").mkdir(parents=True)
    (package / "ketos").mkdir()
    (package / "__init__.py").write_text("")
    (package / "train" / "__init__.py").write_text("from kraken.train.utils import KrakenTrainer\n")
    (package / "train" / "utils.py").write_text(
        "made = []\nclass KrakenTrainer:\n    def __init__(self, *args, **kwargs):\n        made.append(kwargs)\n")
    (package / "ketos" / "__init__.py").write_text(
        "def cli():\n    from kraken.train import KrakenTrainer\n    KrakenTrainer(max_epochs=-1)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == "kraken" or m.startswith("kraken.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("MLOPS_TB_DIR", str(tmp_path / "tb"))
    from mlops import ketos_logged
    monkeypatch.setattr(ketos_logged, "scalars_logger", lambda log_dir: ("scalars only", log_dir))
    ketos_logged.main()
    from kraken.train import utils
    assert utils.made == [{"max_epochs": -1, "pl_logger": ("scalars only", str(tmp_path / "tb")),
                           "log_dir": str(tmp_path / "tb"), "log_every_n_steps": ketos_logged.LOG_EVERY_N_STEPS}]


def test_the_logger_writes_numbers_but_no_images_or_texts(tmp_path):
    """kraken writes validation lines as images / texts when the logger can (and crashes on add_text in 7.0.2):
    the logger of mlops.ketos_logged cannot, but writes the metrics."""
    pytest.importorskip("lightning")
    from mlops.ketos_logged import scalars_logger
    logger = scalars_logger(str(tmp_path))
    assert getattr(logger.experiment, "add_image", None) is None
    assert getattr(logger.experiment, "add_text", None) is None
    logger.log_metrics({"val_accuracy": 0.9}, step=10)
    logger.finalize("success")
    events = list(tmp_path.rglob("events.out.tfevents.*"))
    assert events
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(events[0]))
    accumulator.Reload()
    assert accumulator.Scalars("val_accuracy")[0].value == pytest.approx(0.9)