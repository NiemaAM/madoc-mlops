"""Tests of the platform. The database tests need MLOPS_TEST_DATABASE_URL (see README: Tests); the others run
anywhere. No GPU, no kraken, no W&B account: tests/fakes stand for ketos and kraken, W&B is offline."""

import json
import sys
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import make_page, make_zip, page_xml
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
    # a loss close to 0 is printed with an exponent; val_bl_f1 has digits in its name
    assert train_metrics("blla", "train_loss_step: 4.5e-05 val_bl_f1: 0.973 val_mean_iu: .17") == {
        "train_loss_step": 4.5e-05, "val_bl_f1": 0.973, "val_mean_iu": 0.17}
    assert train_metrics("htr", "early_stopping: 3/20 train_loss_epoch: 52.1") == {
        "early_stopping": 3.0, "train_loss_epoch": 52.1}


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


def test_gate_by_group():
    """Pages of a register are not independent: with few groups the interval stays by page and says so; with
    enough groups it is computed by group, wider when the groups disagree."""
    from mlops.evaluate import compare
    from mlops.textmetrics import bootstrap, group_bootstrap
    production = {f"p{i}": {"page_cer": 5.0} for i in range(40)}
    better = {f"p{i}": {"page_cer": 4.0} for i in range(40)}
    two = {f"p{i}": f"g{i % 2}" for i in range(40)}
    comparison, passed, message = compare("htr", better, production, set(), two)
    assert passed and comparison["interval"] == "by page" and comparison["groups"] == 2
    assert "2 test groups only" in message and set(comparison["group_gains"]) == {"g0", "g1"}
    # 8 groups: 6 gain a lot, 2 lose: by page it looks sure, by group it is not
    eight = {f"p{i}": f"g{i % 8}" for i in range(40)}
    mixed = {p: {"page_cer": 2.0 if int(eight[p][1]) < 6 else 12.0} for p in production}
    comparison, passed, message = compare("htr", mixed, production, set(), eight)
    assert comparison["interval"] == "by group" and comparison["groups"] == 8 and "interval by group" in message
    gains = {p: production[p]["page_cer"] - mixed[p]["page_cer"] for p in production}
    _, page_low, _ = bootstrap(list(gains.values()))
    _, group_low, _ = group_bootstrap(gains, eight)
    assert group_low < page_low and comparison["low"] == round(group_low, 4)
    assert group_bootstrap({}, {}) == (0.0, 0.0, 0.0)


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


def test_augmentation_reuses_the_pages_already_augmented(env):
    """A new dataset version (new pages ingested): the train pages augmented for the previous version are linked,
    only the new ones go through the augmentation script, and the files are the same."""
    from mlops import augment, db
    with db.connection() as conn:
        ingest_zip(conn, env, "a", [(f"R{r}_2016_{a}-2-1", (r * 40, a * 30, 90)) for r in range(7) for a in range(2)])
        logs = []
        first = augment.run_augment(conn, {"args": {"copies": 1, "seed": 3}}, logs.append, lambda: False, None)
        old_files = {p.name: p.read_bytes() for p in augment.augmentations()[0]["folder"].glob("*_aug*")}
        assert len(old_files) == 2 * first["pages"]                            # image + PageXML of every copy
        assert any(f"0 train pages already augmented (reused), {first['pages']} to augment" in line for line in logs)
        conn.execute("UPDATE pages SET split = 'train'")                       # a new version, more train pages
        logs.clear()
        second = augment.run_augment(conn, {"args": {"copies": 1, "seed": 3}}, logs.append, lambda: False, None)
        new = second["pages"] - first["pages"]
        assert new > 0 and any(f"{first['pages']} train pages already augmented (reused), {new} to augment" in line
                               for line in logs)
        folder = augment.augmentations()[0]["folder"]
        assert len(augment.augmentations()) == 1                                # the older version's folder is gone
        from mlops import ui
        assert "the current dataset version" in ui.augmentations_md()
        conn.execute("UPDATE pages SET split = 'test' WHERE id = (SELECT min(id) FROM pages)")   # a new split
        assert "OLDER dataset version" in ui.augmentations_md() and "no augmentation yet" in ui.augmentations_md()
        assert all((folder / name).read_bytes() == data for name, data in old_files.items())


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


def test_read_page_odd_values(tmp_path):
    from mlops.pagexml import points, read_page
    assert points("1,2 nan,3 inf,4 a,b 5,6 7") == [(1.0, 2.0), (5.0, 6.0)]
    page = tmp_path / "p.xml"
    page.write_text(page_xml("p.jpg").replace('imageWidth="200"', 'imageWidth="inf"')
                    .replace('imageHeight="300"', 'imageHeight="-3"'), encoding="utf-8")
    result = read_page(page)
    assert result["width"] == 0 and result["height"] == 0 and len(result["lines"]) == 6


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
        assert normalize_dataset(conn, log=lambda *_: None, apply=False)["replaced"] == {"U+00A0": 1}
        assert "\u00a0" in stored.read_text(encoding="utf-8")                        # a preview changes nothing
        result = normalize_dataset(conn, log=lambda *_: None)
        assert result["replaced"] == {"U+00A0": 1} and result["pages"] == 1
        assert "\u00a0" not in stored.read_text(encoding="utf-8") and not (env / "data" / "augmented").exists()
        assert "\u00a0" in (Path(result["backup"]) / stored.name).read_text(encoding="utf-8")   # the old file kept
        assert normalize_dataset(conn, log=lambda *_: None)["replaced"] == {}


def test_normalize_rules_from_the_app(env):
    import pytest
    from mlops import db, normalize
    from mlops.ingest import ingest, normalize_dataset
    from mlops.pagexml import read_page
    assert normalize.character("U+0649") == "ى" and normalize.character("ى") == "ى"
    assert normalize.replacement("U+0020") == " " and normalize.replacement("") == ""
    assert normalize.replacement("ي") == "ي" and normalize.replacement("U+064A U+0627") == "يا"
    with pytest.raises(ValueError):
        normalize.character("ab")
    with db.connection() as conn:
        assert normalize.table(conn) == normalize.table()                      # nothing written in the app yet
        table = normalize.add_rule(conn, "ى", "ي")                             # a rule of the app
        assert table["ى"] == "ي"
        table = normalize.add_rule(conn, "U+00A0", "")                         # replaces the rule of params.yaml
        assert table[" "] == "" and normalize.table()[" "] == " "
        with pytest.raises(ValueError):
            normalize.add_rule(conn, "ى", "ى")
        rows = {r[0]: r for r in normalize.rules_rows(conn)}
        assert rows["U+0649"][3] == "ي" and rows["U+0649"][4] == "app" and rows["U+00A0"][3] == "(deleted)"
        normalize.remove_rule(conn, "U+00A0")      # the app's rule goes, then the rule of params.yaml is off
        assert " " not in normalize.table(conn)
        assert normalize.remove_rule(conn, "U+200B") and "​" not in normalize.table(conn)
        assert {r[0]: r[4] for r in normalize.rules_rows(conn)}["U+200B"] == "params.yaml (switched off)"
        normalize.add_rule(conn, "U+200B", "")                                 # on again
        assert normalize.table(conn)["​"] == ""
        with pytest.raises(ValueError):
            normalize.remove_rule(conn, "x")                                   # no rule for x
        # ingest uses the app's rules; Normalize applies them to the pages already in
        folder = env / "src-r"
        make_page(folder, "03010108_1_1_I_2016_1-1-1", (10, 20, 30),
                  texts=["مصطفى", "في يوم", "عام", "سنة", "جماعة", "موافق"])
        batch = db.one(conn, "INSERT INTO batches DEFAULT VALUES RETURNING id")
        report = ingest(conn, batch["id"], make_zip(folder, env / "r.zip"), log=lambda *_: None)
        assert report["normalized"] == {"U+0649": 1}                           # U+00A0 switched off: kept
        page = db.one(conn, "SELECT * FROM pages")
        stored = env / "data" / "pages" / f'{page["id"]}.xml'
        assert [line["text"] for line in read_page(stored)["lines"]][:2] == ["مصطفي", "في يوم"]
        normalize.add_rule(conn, "U+00A0", "U+0020")
        assert normalize_dataset(conn, log=lambda *_: None)["replaced"] == {"U+00A0": 1}
        assert read_page(stored)["lines"][1]["text"] == "في يوم"


def test_normalize_words_and_the_officer_name(env):
    from mlops import db, normalize, writers
    from mlops.ingest import normalize_dataset, resplit
    from mlops.pagexml import read_page
    rules = {"المصصفي": "المصطفى", "المصصفي عبادي": "المصطفى عبادي", "\u00a0": " "}
    fixed, found = normalize.text("والمصصفي عبادي\u00a0و المصصفي", rules)
    assert fixed == "والمصطفى عبادي و المصطفى"                       # inside a word, the longest first, one pass
    label = normalize.label("المصصفي عبادي")
    assert found == {label: 1, normalize.label("المصصفي"): 1, "U+00A0": 1} and label.startswith("R-")
    assert normalize.text("المصطفى", {"ى": "ي", "المصطفى": "X"})[0] == "X"  # a replacement is not replaced again
    names = ["المصطفى عبادي", "عبد الرحيم كلفاع", "محمد العلوي", "سعيد بنعلي", "رشيد الإدريسي"]
    with db.connection() as conn:
        pages = []
        for w, writer in enumerate(names):
            for act in range(5):
                pages += act_pages(w, act, writer, (w * 40, act * 12, 50))
        pages += act_pages(7, 1, "المصصفي عبادي", (250, 250, 1)) + act_pages(7, 2, "المصصفي عبادي", (250, 1, 250))
        ingest_texts(conn, env, "w", pages)
        assert len(writers.table(conn)) == 6                                 # the misspelling: one more writer
        conn.execute("UPDATE pages SET split = 'test' WHERE writer_key = (SELECT writer_key FROM pages WHERE id IN "
                     "(SELECT page_id FROM page_writers WHERE raw_name = 'المصصفي عبادي') LIMIT 1)")
        normalize.add_rule(conn, "المصصفي عبادي", "المصطفى عبادي")
        rows = {r[1]: r for r in normalize.rules_rows(conn)}
        assert rows["المصصفي عبادي"][0] == label and rows["المصصفي عبادي"][3] == "المصطفى عبادي"
        logs = []
        result = normalize_dataset(conn, log=logs.append)
        assert result["replaced"] == {label: 2} and result["pages"] == 2      # the officer line of the 2 page 2s
        assert "المصصفي" not in " ".join(logs) and label in " ".join(logs)   # the text never in the log
        assert len(writers.table(conn)) == 5                                 # the officer read again: one writer
        splits = db.all_rows(conn, "SELECT writer_key, split FROM pages WHERE writer_key IS NOT NULL")
        assert all(len({r["split"] for r in splits if r["writer_key"] == w}) == 1 for w in {r["writer_key"] for r in splits})
        page = db.one(conn, "SELECT page_id FROM page_writers WHERE raw_name = 'المصطفى عبادي' LIMIT 1")["page_id"]
        assert "المصطفى عبادي" in " ".join(line["text"] for line in read_page(env / "data" / "pages" / f"{page}.xml")["lines"])
        normalize.remove_rule(conn, "المصصفي عبادي")
        assert "المصصفي عبادي" not in normalize.table(conn)
        assert resplit(conn, log=lambda *_: None)["pages"]


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


def test_evaluation_leaves_out_the_test_pages_a_model_trained_on(env):
    """The split changed since the candidate was trained: its old train pages now in test are not used to judge
    it, nor the pages the production model trained on (its card names its version)."""
    import pytest
    from mlops import datasets, db
    from mlops.evaluate import unseen_test_pages
    with db.connection() as conn:
        ingest_zip(conn, env, "a", [(f"G{g}_2010_{i}-1", (g * 40, i * 20, 10)) for g in range(6) for i in range(3)])
        before = datasets.snapshot(conn)
        trained = datasets.seen_page_ids(conn, before["id"])
        conn.execute("UPDATE pages SET split = 'train'")
        all_train = datasets.snapshot(conn)
        conn.execute("UPDATE pages SET split = 'test'")                          # everything in test now
        rows = datasets.pages(conn, datasets.snapshot(conn)["id"], "test")
        notes = []
        kept, excluded = unseen_test_pages(conn, rows, {"dataset_version_id": None}, {}, notes.append)
        assert len(kept) == 18 and excluded == {"candidate": 0, "production": 0} and not notes
        kept, excluded = unseen_test_pages(conn, rows, {"dataset_version_id": before["id"]}, {}, notes.append)
        assert {r["id"] for r in kept} == {r["id"] for r in rows} - trained
        assert excluded["candidate"] == len(trained) > 0 and "candidate" in notes[0]
        with pytest.raises(RuntimeError, match="every test page"):
            unseen_test_pages(conn, rows, {"dataset_version_id": None}, {"dataset_version": all_train["digest"]},
                              notes.append)
        # a page-level split: the groups found in several splits are reported
        assert datasets.leaking_groups(conn, before["id"]) == []
        conn.execute("UPDATE pages SET split = 'train' WHERE id IN (SELECT id FROM pages ORDER BY id LIMIT 5)")
        assert datasets.leaking_groups(conn, datasets.snapshot(conn)["id"])


def test_job_queue(env):
    from mlops import db, jobs
    with db.connection() as a, db.connection() as b:
        first = jobs.enqueue(a, "train", "htr")
        second = jobs.enqueue(a, "train", "blla")
        other = jobs.enqueue(a, "crossval", None, recognizer="crnn")       # the baselines container's job
        assert other["queue"] == "baselines" and first["queue"] == "kraken"
        with a.transaction():   # a holds the first job: b takes the next one (SKIP LOCKED)
            db.one(a, "SELECT * FROM jobs WHERE id = %s FOR UPDATE", first["id"])
            assert jobs.claim(b)["id"] == second["id"]
        # one job on the GPU at a time: nothing is taken while one runs, by either worker
        assert jobs.claim(a) is None and jobs.claim(a, ("baselines",)) is None
        jobs.finish(a, second["id"], "done")
        assert jobs.claim(a)["id"] == first["id"] and jobs.claim(a) is None
        jobs.finish(a, first["id"], "done")
        assert jobs.claim(a) is None                                            # only a baselines job is left
        assert jobs.claim(b, ("baselines",))["id"] == other["id"]
        jobs.recover(a, ("kraken",))                                            # the other queue's job goes on
        assert db.one(a, "SELECT status FROM jobs WHERE id = %s", other["id"])["status"] == "running"
        jobs.finish(a, other["id"], "done")
        a.execute("UPDATE jobs SET status = 'running', finished_at = NULL WHERE id = %s", (first["id"],))
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


def test_a_command_printing_bytes_that_are_not_utf8(tmp_path):
    import sys
    from mlops.process import run
    out = run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'ok \\xff\\xfe end\\n')"], lambda _: None)
    assert out.startswith("ok ") and out.rstrip().endswith("end")


def test_old_job_logs_are_pruned(env):
    from mlops import db, jobs
    with db.connection() as conn:
        old, recent, running = (jobs.enqueue(conn, "augment") for _ in range(3))
        for job in (old, recent, running):
            jobs.Log(conn, job["id"], flush_every=0)("a line")
        conn.execute("UPDATE jobs SET status = 'done', finished_at = now() - interval '200 days' WHERE id = %s",
                     (old["id"],))
        conn.execute("UPDATE jobs SET status = 'done', finished_at = now() WHERE id = %s", (recent["id"],))
        assert jobs.prune_logs(conn, 0) == 0
        assert jobs.prune_logs(conn, 180) == 1
        left = {r["job_id"] for r in db.all_rows(conn, "SELECT job_id FROM job_logs")}
        assert left == {recent["id"], running["id"]}


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
        # HTR trains on compiled lines (ketos compile, then --format-type binary); the segmentation on the pages
        logs = [r["line"] for r in db.all_rows(conn, "SELECT line FROM job_logs ORDER BY id")]
        assert sum("compile -f page" in line for line in logs) == 2                      # train and val
        assert any(" train " in line and "--format-type binary" in line for line in logs)
        assert any(" segtrain " in line and "--format-type xml" in line and "--workers 4" in line for line in logs)
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
        assert evaluations[0]["comparison"]["groups"] >= 1 and "group_gains" in evaluations[0]["comparison"]
        # every number with its interval, the figures of the evaluation and of the trainings
        assert {"cer", "cer_low", "cer_high", "cer_lp", "wer"} <= set(evaluations[0]["candidate"])
        assert evaluations[0]["production"]["cer"] > evaluations[0]["candidate"]["cer"]
        assert {"line_precision", "line_recall", "mean_e2e_word_f1"} <= set(evaluations[1]["candidate"])
        made = {p.stem for p in Path(evaluations[0]["figures"]).glob("*.png")}
        assert {"candidate_accuracy_comparison", "production_error_types", "error_composition", "form_rows",
                "writers"} <= made
        assert "e2e_breakdown" in {p.stem for p in Path(evaluations[1]["figures"]).glob("*.png")}
        assert all("training_curves" in {p.stem for p in Path(m["figures"]).glob("*.png")} for m in models)
        # the same production model on the same test pages: its numbers are reused, not computed again
        jobs.enqueue(conn, "evaluate", "htr", model_id=models[0]["id"])
        worker.main(once=True)
        again = db.one(conn, "SELECT * FROM evaluations ORDER BY id DESC LIMIT 1")
        assert again["production"] == evaluations[0]["production"]
        assert db.one(conn, "SELECT count(*) AS n FROM job_logs WHERE line LIKE 'production numbers reused%%'")["n"] == 1
        # a model renamed in MaDoc: the training says which ones exist
        from mlops.train import base_model
        with pytest.raises(RuntimeError, match="models/htr.mlmodel"):
            base_model("htr", "models/base/htr_renamed.mlmodel")
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
        htr = settings.training(conn, "htr")                                   # params.yaml, whatever its values
        assert "batch_size" in htr and "batch_size" not in settings.training(conn, "blla")
        chosen = settings.clean("htr", {**htr, "epochs": "40", "quit": "early", "lag": 7, "batch_size": 30,
                                        "resize": "new", "learning_rate": "1e-4", "schedule": "cosine",
                                        "cos_max": 40, "kraken_augment": True, "extra_args": "--warmup 200"})
        settings.save(conn, "train.htr", chosen)
        assert settings.training(conn, "htr")["learning_rate"] == 1e-4           # saved from the app
        cmd = ketos_command("htr", settings.training(conn, "htr"), "/m.mlmodel", "t.txt", "v.txt", "out")
        text = " ".join(cmd)
        assert cmd.index("--workers") < cmd.index("train") and cmd.index("--precision") < cmd.index("train")
        assert "--format-type page" in " ".join(cmd)
        assert "--format-type binary" in " ".join(
            ketos_command("htr", settings.training(conn, "htr"), "/m.mlmodel", "t", "v", "o", binary=True))
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
        for key, value, message in (("lag", "abc", "lag: 'abc' is not a number"), ("lag", 0, "below 1"),
                                    ("epochs", "inf", "finite"), ("batch_size", 2.5, "whole number"),
                                    ("learning_rate", -1e-3, "above 0"), ("epochs", 0, "trains nothing"),
                                    ("extra_args", "--x 'open", "extra_args")):
            with pytest.raises(ValueError, match=message):
                settings.clean("htr", {**htr, key: value})
        with pytest.raises(ValueError, match="never stop"):
            settings.clean("htr", {**htr, "quit": "fixed", "epochs": -1})
        assert settings.clean("htr", {**htr, "quit": "fixed", "epochs": 30.0})["epochs"] == 30
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


def test_missing_pages(env, capsys):
    from mlops import db
    from mlops.missing import print_report, report
    with db.connection() as conn:
        pages = [("03010108_1_1_I_2004_1-1-1", (10, 10, 10)), ("03010108_1_1_I_2004_1-2-1", (20, 20, 20)),  # complete
                 ("03010108_1_1_I_2004_2-1-1", (30, 30, 30)),                                              # no page 2
                 ("03010108_1_2_I_2005_7-2-1", (40, 40, 40))]                                              # no page 1
        ingest_zip(conn, env, "m", pages)
        data = report(conn)
        assert data["03010108_1_1_I_2004"]["complete"] == 1
        assert data["03010108_1_1_I_2004"]["missing"] == {"1": [], "2": ["03010108_1_1_I_2004_2-2-1"]}
        assert data["03010108_1_2_I_2005"]["missing"] == {"1": ["03010108_1_2_I_2005_7-1-1"], "2": []}
        print_report(conn, with_names=True)
    printed = capsys.readouterr().out
    assert "03010108_1_1_I_2004_2-2-1" in printed and "03010108_1_2_I_2005_7-1-1" in printed


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


def test_the_lines_are_compiled_once(env):
    """ketos compile is run once for the same files, and again when one of them changes."""
    from mlops.train import compiled_dataset
    pages = []
    for n in range(3):
        make_page(env / "src", f"p{n}", (n * 50, 0, 0))
        pages.append(str(env / "src" / f"p{n}.xml"))
    logs = []
    first = compiled_dataset(pages, "train", {"workers": 2}, logs.append, lambda: False)
    assert first.suffix == ".arrow" and first.read_text() == "arrow of 3 pages"
    assert any("compile -f page" in line and "--workers 2" in line for line in logs)
    logs.clear()
    assert compiled_dataset(pages, "train", {"workers": 2}, logs.append, lambda: False) == first
    assert logs == [f"compiled train lines reused ({first.name})"]
    import os
    os.utime(pages[0], ns=(1, 1))                                       # a page changed (normalized)
    second = compiled_dataset(pages, "train", {}, logs.append, lambda: False)
    assert second != first and second.is_file() and not first.exists()  # the old one is deleted


def test_the_commands_of_the_deployment(env, capsys):
    """The exact command lines of .github/workflows/deploy.yml and of the README, through cli.main."""
    from mlops import cli, db, settings
    assert cli.main(["pause"]) == 0 and "paused" in capsys.readouterr().out
    with db.connection() as conn:
        assert settings.paused(conn)
    assert cli.main(["wait-idle", "--timeout", "0"]) == 0 and "no job running" in capsys.readouterr().out
    assert cli.main(["resume"]) == 0
    with db.connection() as conn:
        assert not settings.paused(conn)
    assert cli.main(["health", "ui"]) == 0
    assert cli.main(["health", "worker"]) == 1                       # no heartbeat yet
    folder = env / "to-import"
    make_page(folder, "X_2011_1-1-1", (10, 20, 30))
    assert cli.main(["import", str(folder), "--split", "train"]) == 0
    assert "ingest of" in capsys.readouterr().out
    assert cli.main(["jobs"]) == 0 and "ingest" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["import", str(env / "missing")])


def test_the_app_builds(env, monkeypatch):
    pytest.importorskip("gradio")
    import gradio as gr
    from mlops import db, ui
    assert ui.build_app() is not None
    assert [row[0] for row in ui.dataset_table()] == ["train", "val", "test"]
    assert "CPU" in ui.server_html()
    with db.connection() as conn:
        ingest_texts(conn, env, "ui", [p for w in range(3) for p in act_pages(w, 1, f"كاتب {'ابت'[w]}", (w * 60, 5, 5))])
    rows, warning, *_ = ui.writers_view()
    assert len(rows) == 3 and all(r[1].startswith("W-") for r in rows)
    assert "pages would change" in ui.normalize_preview()
    assert any(r[0] == "U+00A0" for r in ui.normalize_rules())
    table, _ = ui.on_normalize_rule("ى", "ي")
    assert any(r[0] == "U+0649" and r[4] == "app" for r in table)
    table, _ = ui.on_normalize_rule("ى", "", remove=True)
    assert not any(r[0] == "U+0649" for r in table)
    table, _ = ui.on_normalize_rule("المصصفي عبادي", "المصطفى عبادي")
    assert any(r[1] == "المصصفي عبادي" and r[3] == "المصطفى عبادي" for r in table)
    ui.on_normalize_rule("المصصفي عبادي", "", remove=True)
    # the preview zip: admins only (MLOPS_ADMINS, else the first login)
    monkeypatch.setenv("MLOPS_AUTH", "boss:x,guest:y")
    request = type("Request", (), {"username": "guest"})()
    with pytest.raises(gr.Error, match="admin"):
        ui.on_preview_download(request)
    request.username = "boss"
    with pytest.raises(gr.Error, match="Make the preview"):
        ui.on_preview_download(request)
    assert ui.experiments_table() == [] and ui.experiment_details(None)[0] == "Choose an experiment."
    assert ui.figures_of(None) == ([], [])


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

# --- writers, names, scores, figures, experiments ------------------------------------------------------------
PAGE1 = ["الصفحة رقم : 1", "PRENOM ANWAR. ....... الاسم الشخصي : أنور", "NOM EZ-ZAYANY ...... الاسم العائلي : الزياني",
         "FILS DE : ABDELLAH. والده :", "ET DE : NADIA EZAHIRI fille de MOHAMMED والدته :",
         "اختير اسم الأب طبقا لمقتضيات المادة 16"]


def page2(writer, name="مها", latin="MAHA"):
    return ["رسم ولادة رقم: 18", "في يوم : رابع ذو القعدة", f"الاسم الشخصي : {name}", f"PRENOM {latin}",
            f"من طرفنا نحن {writer} موظف جماعي", "وبعد تلاوة الرسم على المصرح"]


def act_pages(register, act, writer, color, with_page1=True):
    """The two pages of an act ("..._<act>-1-1" names, "..._<act>-2-1" details), page 2 naming its writer."""
    out = [(f"03010108_1_{register}_I_2016_{act}-2-1", color, page2(writer))]
    if with_page1:
        out.append((f"03010108_1_{register}_I_2016_{act}-1-1", tuple(min(255, c + 7) for c in color), PAGE1))
    return out


def ingest_texts(conn, tmp, name, pages, held_out=False):
    from mlops import db
    from mlops.ingest import ingest
    folder = tmp / f"src-{name}"
    for stem, color, texts in pages:
        make_page(folder, stem, color, lines=len(texts), texts=texts)
    batch = db.one(conn, "INSERT INTO batches (held_out) VALUES (%s) RETURNING id", held_out)
    return batch["id"], ingest(conn, batch["id"], make_zip(folder, tmp / f"{name}.zip"), log=lambda *_: None)


def test_names_writers_and_form_rows():
    from mlops import fields
    spans = fields.name_spans(PAGE1)
    got = {f: PAGE1[i][a:b] for f, (i, a, b) in spans.items()}
    assert got == {"latin_first": "ANWAR", "arabic_first": "أنور", "latin_family": "EZ-ZAYANY",
                   "arabic_family": "الزياني"}                       # FILS DE / ET DE / the note are never names
    assert fields.name_spans(["الاسم الشخصي : ......"]) == {"arabic_first": (0, None, None)}   # label, no name
    assert fields.writer_name(page2("لحسن اشميطي")) == "لحسن اشميطي"
    assert fields.writer_name(["من طَرفنا نحن: لحسن شميطي، موظف جماعي"]) == "لحسن شميطي"          # diacritics
    assert fields.writer_name(["وحرر لدينا من طرفنا نحن", "لحسن شميطي ضابط الحالة المدنية"]) == "لحسن شميطي"
    assert fields.writer_name(["من طرفنا لحسن شميطي ضابط الحالة المدنية"]) == "لحسن شميطي"   # without "نحن"
    assert fields.writer_name(["من طرفا نحن لحسن اشميطي موظف جماعي"]) == "لحسن اشميطي"      # the ن missed
    assert fields.writer_name(["من طرفنا نحن ضابط الحالة المدنية", "وبعد تلاوة"]) == ""        # no name written
    # hidden characters: a zero-width joiner, a direction mark, presentation forms (they look the same)
    assert fields.writer_name(["من طر\u200dفنا نحن لحسن\u200f اشميطي موظف جماعي"]) == "لحسن اشميطي"
    assert fields.writer_name(["من \ufec3\ufeae\ufed3\ufee8\ufe8e نحن لحسن اشميطي موظف جماعي"]) == "لحسن اشميطي"
    assert fields.reveal("طر\u200dفنا ڭ 12") == "طر⟨U+200D⟩فنا ڭ 12"
    assert fields.officer_candidates(["رسم ولادة", "من طرفتا نحن لحسن"]) == ["من طرفتا نحن لحسن"]   # a typo
    assert fields.close_names(fields.writer_key_text("لحسن شميطي"), fields.writer_key_text("لحسن اشميطي"))
    assert not fields.close_names("لحسن شميطي", "عبد الرحيم كلفاع")
    assert fields.form_rows(PAGE1, "1")[-1] == "Marginal notes" and fields.form_rows(page2("x"), "2")[4] == "Officer"
    # the name read in the prediction is the part aligned with the true name: a misread label does not count
    assert fields.carry_span("PRENOM ANWAR", "PRFNOM ANWAR", 7, 12) == "ANWAR"
    assert fields.name_key("latin_first", "M.A.H.A") == "MAHA" and fields.name_key("arabic_family", "الزِّياني") == "الزياني"


def test_two_cers_composition_names_and_intervals():
    from mlops.scoring import flat_metrics, score
    from mlops.textmetrics import line_errors, ratio_interval
    e = line_errors("رسم ولادة: 18", "رسم ولاده 18 x")
    assert e["errors"] == 4 and e["errors_lp"] == 2                       # without spaces and punctuation: 2
    assert e["composition"]["deletions"]["punctuation"] == 1 and e["composition"]["insertions"]["spaces"] == 1
    assert ratio_interval({}) == (None, None, None)
    pages = {"a": {"writer": "W-1", "page_type": "1", "lines": PAGE1},
             "b": {"writer": "W-2", "page_type": "2", "lines": page2("لحسن شميطي")},
             "c": {"writer": "W-3", "page_type": "2", "lines": page2("عبد الرحيم")}}
    rows = []
    for pid, meta in pages.items():
        for i, text in enumerate(meta["lines"]):
            pred = text.replace("ANWAR", "ANWAP") if pid == "a" else text.replace("مها", "مهآ") if pid == "b" else text
            rows.append({"page": pid, "line": i, "ref": text, "pred": pred})
    rows.append({"page": "c", "line": None, "ref": "", "pred": "xx"})        # a line found that is not there
    summary = score(rows, pages)
    assert summary["interval"].startswith("by page") and summary["writers_count"] == 3   # 3 < 5 writers
    assert summary["metrics"]["cer"] > 0 and summary["metrics"]["cer_lp"] > 0
    low, high = summary["ci"]["cer"]
    assert low <= summary["metrics"]["cer"] <= high
    names = summary["names"]
    assert names["latin_first|1"]["exact"] == 0.0 and names["latin_family|1"]["exact"] == 100.0
    assert names["arabic_first|2"]["names"] == 2 and names["arabic_first|2"]["exact"] == 50.0
    assert names["arabic_first|2"]["normalized"] == 100.0                  # آ = ا: the same name normalised
    assert "Officer|2" in summary["form_rows"] and set(summary["writers"]) == {"W-1", "W-2", "W-3"}
    flat = flat_metrics(summary)
    assert {"cer", "cer_low", "cer_high", "cer_lp", "names_exact"} <= set(flat)


def test_writer_independent_split_held_out_and_aliases(env):
    from mlops import db, writers
    from mlops.ingest import resplit
    names = ["لحسن شميطي", "عبد الرحيم كلفاع", "محمد العلوي", "سعيد بنعلي", "رشيد الإدريسي", "كريم الفاسي"]
    with db.connection() as conn:
        pages = []
        for w, writer in enumerate(names):
            for act in range(6 + w):
                pages += act_pages(w, act, writer, (w * 40, act * 12, 50))
        pages += act_pages(9, 1, "لحسن اشميطي", (250, 250, 1))             # another spelling of the first writer
        pages += [("03010108_1_8_I_2016_1-1-1", (1, 250, 250), PAGE1)]        # a page 1 whose page 2 is not there
        _, report = ingest_texts(conn, env, "w", pages)
        assert sum(report["added"].values()) == len(pages)
        rows = db.all_rows(conn, "SELECT p.*, o.original_name FROM pages p JOIN page_origins o ON o.page_id = p.id")
        by_writer, by_act = {}, {}
        for r in rows:
            by_act.setdefault(r["act_key"], set()).add(r["split"])
            if r["writer_key"]:
                by_writer.setdefault(r["writer_key"], set()).add(r["split"])
        assert all(len(s) == 1 for s in by_writer.values())                  # no writer in two splits
        assert all(len(s) == 1 for s in by_act.values())                     # both pages of an act together
        assert {s for v in by_writer.values() for s in v} == {"train", "val", "test"}
        orphan = next(r for r in rows if "_8_I_" in r["original_name"])
        assert orphan["writer_key"] is None and orphan["split"] == "train"   # unknown writer: train only
        assert len(by_writer) == 7 and writers.variants(conn)                 # the two spellings are flagged
        writers.add_alias(conn, "لحسن اشميطي", "لحسن شميطي")
        assert not writers.variants(conn)
        table = writers.table(conn)
        first = next(t for t in table if t["writer"] == writers.writer_key_text("لحسن شميطي"))
        assert first["pages"] == 2 * 6 + 2 and first["label"].startswith("W-")
        with pytest.raises(ValueError, match="loop"):
            writers.add_alias(conn, "لحسن شميطي", "لحسن اشميطي")
        result = resplit(conn, log=lambda *_: None)
        rows = db.all_rows(conn, "SELECT writer_key, split FROM pages WHERE writer_key IS NOT NULL")
        assert len({r["writer_key"] for r in rows}) == 6 and "writers placed again" in result["text"]
        assert all(len({r["split"] for r in rows if r["writer_key"] == w}) == 1 for w in {r["writer_key"] for r in rows})
        # the page 2 of the orphan act comes in, by a writer already placed: its page 1 leaves train for that split
        placed = db.one(conn, "SELECT writer_key, split FROM pages WHERE split = 'test' AND writer_key IS NOT NULL "
                              "LIMIT 1")
        name = next(t["writer"] for t in writers.table(conn) if t["key"] == placed["writer_key"])
        _, report = ingest_texts(conn, env, "o", [("03010108_1_8_I_2016_1-2-1", (2, 251, 249), page2(name))])
        act = {r["page_type"]: r for r in db.all_rows(conn, "SELECT p.* FROM pages p JOIN page_origins o ON "
                                                            "o.page_id = p.id WHERE o.original_name LIKE %s", "%_8_I_%")}
        assert act["1"]["writer_key"] == act["2"]["writer_key"] == placed["writer_key"]
        assert act["1"]["split"] == act["2"]["split"] == "test" and report["moved"] == 1
        # page 1s of a new writer, in train while their writer was unknown, do not tie that writer to train: it is
        # placed as a new writer, with all its pages
        ingest_texts(conn, env, "o1", [(f"03010108_1_11_I_2016_{a}-1-1", (5, 30 * a, 160), PAGE1) for a in range(3)])
        _, report = ingest_texts(conn, env, "o2", [(f"03010108_1_11_I_2016_{a}-2-1", (6, 30 * a, 170),
                                                     page2("نور الدين جديد")) for a in range(3)])
        assert report["new_groups"] == 1
        rows = db.all_rows(conn, "SELECT p.split, p.writer_key FROM pages p JOIN page_origins o ON o.page_id = p.id "
                                 "WHERE o.original_name LIKE %s", "%_11_I_%")
        assert len(rows) == 6 and len({r["split"] for r in rows}) == 1 and all(r["writer_key"] for r in rows)
        # a held-out batch: in test for good, through a new split, and reported when it shares a writer
        _, report = ingest_texts(conn, env, "held", act_pages(7, 1, "لحسن شميطي", (3, 3, 3)), held_out=True)
        assert report["held_out"] and report["added"]["test"] == 2
        result = resplit(conn, log=lambda *_: None)
        held = db.all_rows(conn, "SELECT split, pinned_split FROM pages WHERE held_out")
        assert {(r["split"], r["pinned_split"]) for r in held} == {("test", "test")}
        if db.one(conn, "SELECT split FROM pages WHERE writer_key = (SELECT writer_key FROM pages WHERE held_out "
                        "LIMIT 1) AND NOT held_out LIMIT 1")["split"] != "test":
            assert result["held_out_overlap"] > 0 and "held-out set" in result["text"]
        # the writer of the page in the browse tab, only when the original names are shown
        from mlops import browse
        options, _ = browse.choices(conn, "test", "all", show_names=True, search="03010108_1_7_I")
        assert len(options) == 2 and all("03010108_1_7_I" in label and "held out" in label for label, _ in options)
        hidden, _ = browse.choices(conn, "test", "all")
        assert all("03010108" not in label for label, _ in hidden)
        _, _, _, info = browse.show(options[0][1], True, True)
        assert "original name" in info and "writer" in info


def test_names_not_read_are_listed_and_read_again(env):
    from mlops import db, writers
    with db.connection() as conn:
        pages = act_pages(1, 1, "لحسن شميطي", (10, 10, 10))
        pages += [("03010108_1_2_I_2005_4-2-1", (20, 20, 20), ["رسم ولادة رقم: 4", "في يوم : رابع",
                                                             "الاسم الشخصي : مها", "PRENOM MAHA",
                                                             "من طرفنا نحن ضابط الحالة المدنية",
                                                             "وبعد تلاوة الرسم"])]
        ingest_texts(conn, env, "u", pages)
        unread = writers.unread(conn)
        assert list(unread) == ["03010108_1_2_I_2005"] and unread["03010108_1_2_I_2005"]["pages"] == 1
        assert unread["03010108_1_2_I_2005"]["examples"][0][1][0] == "من طرفنا نحن ضابط الحالة المدنية"
        # a page 2 read without a name is read again at the next sync (the reading may have improved)
        page = db.one(conn, "SELECT w.page_id FROM page_writers w JOIN pages p ON p.id = w.page_id "
                             "WHERE w.raw_name = '' AND p.page_type = '2'")["page_id"]
        stored = env / "data" / "pages" / f"{page}.xml"
        stored.write_text(stored.read_text(encoding="utf-8").replace("نحن ضابط", "نحن محمد العلوي ضابط"),
                          encoding="utf-8")
        writers.sync(conn)
        writers.sync(conn)
        assert not writers.unread(conn) and len(writers.table(conn)) == 2


def test_writers_put_in_a_split_by_hand(env):
    from mlops import db, ui, writers
    from mlops.ingest import resplit
    names = ["لحسن شميطي", "عبد الرحيم كلفاع", "محمد العلوي", "سعيد بنعلي", "رشيد الإدريسي"]
    with db.connection() as conn:
        pages = []
        for w, writer in enumerate(names):
            for act in range(4 + w):
                pages += act_pages(w, act, writer, (w * 40, act * 12, 50))
        ingest_texts(conn, env, "p", pages)
        label = next(t["label"] for t in writers.table(conn) if t["writer"] == writers.writer_key_text(names[1]))
        writers.pin(conn, names[0], "test")
        writers.pin(conn, label, "test")                                     # by its label too
        writers.pin(conn, names[2], "val")
        with pytest.raises(ValueError):
            writers.pin(conn, "nobody", "test")
        preview = resplit(conn, log=lambda *_: None, apply=False)
        assert "3 writers placed by hand" in preview["text"]
        result = resplit(conn, log=lambda *_: None)

        def split_of(name):
            return next(t["splits"] for t in writers.table(conn) if t["writer"] == writers.writer_key_text(name))

        assert split_of(names[0]) == {"test": 8} and split_of(names[1]) == {"test": 10}
        assert split_of(names[2]) == {"val": 12} and result["groups"]["test"] == 2 and result["groups"]["val"] == 1
        assert set(split_of(names[3])) == set(split_of(names[4])) == {"train"}   # the others: around them
        # a new batch of a writer placed by hand follows it; "auto" gives the writer back to the split
        _, report = ingest_texts(conn, env, "p2", act_pages(0, 50, names[0], (9, 99, 199)))
        assert report["added"]["test"] == 2
        writers.pin(conn, names[0], "auto")
        assert names[0] and [t["pinned"] for t in writers.table(conn)].count("test") == 1
    rows, *_, pin_choices = ui.writers_view()
    assert "chosen split" in ui.WRITER_HEADERS and any(r[8] == "val" for r in rows) and pin_choices["choices"]
    ui.on_pin(names[2], "auto")
    assert all(r[8] == "auto" or r[8] == "test" for r in ui.writers_view()[0])


def test_page_1_keys_made_with_another_secret_are_repaired(env, monkeypatch):
    import hmac
    from mlops import config, db, writers
    from mlops.ingest import place
    names = ["لحسن شميطي", "عبد الرحيم كلفاع", "محمد العلوي", "سعيد بنعلي", "رشيد الإدريسي"]
    with db.connection() as conn:
        pages = []
        for w, writer in enumerate(names):
            for act in range(4):
                pages += act_pages(w, act, writer, (w * 40, act * 12, 50))
        ingest_texts(conn, env, "k", pages)
        # what the app container did without MLOPS_SECRET: the page 1s keyed with another secret, no writer
        for row in db.all_rows(conn, "SELECT p.id, o.original_name FROM pages p JOIN page_origins o ON "
                                     "o.page_id = p.id WHERE p.page_type = '1'"):
            other = hmac.new(b"change-me", ("act:" + writers.act_template(Path(row["original_name"]).stem)).encode(),
                             "sha256").hexdigest()[:16]
            conn.execute("UPDATE pages SET act_key = %s, writer_key = NULL, split = 'train' WHERE id = %s",
                         (other, row["id"]))
        assert sum(t["pages"] for t in writers.table(conn) if t["key"] is None) == 20
        result = place(conn, log=lambda *_: None)
        assert not [t for t in writers.table(conn) if t["key"] is None]          # every page 1 has its writer
        assert all(t["page1"] == t["page2"] == 4 and t["incomplete_acts"] == 0 for t in writers.table(conn))
        acts = db.all_rows(conn, "SELECT act_key, count(DISTINCT split) AS n FROM pages GROUP BY act_key")
        assert all(a["n"] == 1 for a in acts) and result["moved"] > 0           # each act in one split
        # and no process may hash without the secret any more
        monkeypatch.delenv("MLOPS_SECRET")
        with pytest.raises(RuntimeError, match="MLOPS_SECRET"):
            config.keyed("x")


def test_delete_restore_and_purge_a_batch(env):
    from mlops import batches, datasets, db
    with db.connection() as conn:
        first, _ = ingest_texts(conn, env, "a", [(f"X_2010_{i}-1", (i * 30, 1, 1), page2("أ ب")) for i in range(4)])
        second, _ = ingest_texts(conn, env, "b", [(f"Y_2011_{i}-1", (i * 30, 99, 1), page2("ج د")) for i in range(3)])
        assert batches.delete(conn, second) == 3
        version = datasets.snapshot(conn)
        assert sum(version["counts"].values()) == 4                          # the deleted pages are out
        with pytest.raises(ValueError, match="already deleted"):
            batches.delete(conn, second)
        assert batches.restore(conn, second) == 3
        assert sum(datasets.snapshot(conn)["counts"].values()) == 7
        batches.delete(conn, second)
        with pytest.raises(ValueError, match="PURGE"):
            batches.purge(conn, second, "yes")
        # a model trained on a version holding these pages: they stay (traceable)
        used = datasets.snapshot(conn)
        db.one(conn, "SELECT 1 AS x FROM dataset_versions")
        conn.execute("UPDATE pages SET deleted_at = NULL WHERE batch_id = %s", (second,))
        with_them = datasets.snapshot(conn)
        conn.execute("UPDATE pages SET deleted_at = now() WHERE batch_id = %s", (second,))
        conn.execute("INSERT INTO models (task, path, dataset_version_id) VALUES ('htr', 'x', %s)", (with_them["id"],))
        with pytest.raises(ValueError, match="traceable"):
            batches.purge(conn, second, f"PURGE {second}")
        conn.execute("DELETE FROM models")
        files = list((env / "data" / "pages").iterdir())
        assert batches.purge(conn, second, f"PURGE {second}") == 3
        assert len(list((env / "data" / "pages").iterdir())) == len(files) - 6     # XML + image of 3 pages
        assert db.one(conn, "SELECT status FROM batches WHERE id = %s", second)["status"] == "purged"
        assert db.one(conn, "SELECT count(*) AS n FROM pages")["n"] == 4 and used
        # the same pages uploaded again after a delete come back (restored), not as duplicates
        batches.delete(conn, first)
        _, report = ingest_texts(conn, env, "a", [(f"X_2010_{i}-1", (i * 30, 1, 1), page2("أ ب")) for i in range(4)])
        assert report["restored"] == 4 and report["duplicates"] == 0


def test_missing_pages_panel_spaces_and_preview(env):
    import zipfile
    from mlops import db, missing, preview, vocabulary
    with db.connection() as conn:
        ingest_texts(conn, env, "m", act_pages(1, 1, "أ ب", (10, 10, 10)) + act_pages(1, 2, "أ ب", (90, 9, 9), False))
        rows = missing.summary_rows(conn)
        assert rows[-1] == ["total", "", 1, 1, 0] and "03010108_1_1_I_2016_2-1-1" in missing.listing(conn)
        page = vocabulary.vocabulary_html({"train": __import__("collections").Counter({" ": 8, " ": 1, "a": 3}),
                                           "val": __import__("collections").Counter(), "test": __import__("collections").Counter()})
        assert "Spaces and invisible characters" in page and "No-Break Space" in page and "U+00A0" in page
        result = preview.build(conn, log=lambda *_: None)
        file = preview.latest()
        assert file.name == result["file"] and sum(result["lines"].values()) == 3 * 6
        with zipfile.ZipFile(file) as archive:
            names = archive.namelist()
            assert "README.txt" in names and not any("03010108" in n for n in names)
            for split, n in result["lines"].items():
                if n:
                    labels = archive.read(f"{split}/labels.tsv").decode("utf-8").splitlines()
                    assert len(labels) == n and all(line.split("\t")[0].startswith(f"{split}_") for line in labels)
            dates = {i.date_time for i in archive.infolist() if i.filename.endswith(".png")}
            assert dates == {(2000, 1, 1, 0, 0, 0)}                           # no time to put the pages back
        preview.record_download(conn, "admin", file.name)
        assert db.one(conn, "SELECT username FROM downloads")["username"] == "admin"


def test_figures():
    from mlops import charts
    epochs = [{"epoch": e, "val_accuracy": 0.9 + e / 100, "val_word_accuracy": 0.7 + e / 50, "train_loss": 5 - e,
               "val_cer": 10 - e, "seconds": 60.0, "val_cer_script_Arabic": 8 - e} for e in range(1, 5)]
    assert charts.epochs_of({10: {"epoch": 0, "val_accuracy": 0.5}, 20: {"epoch": 1, "val_accuracy": 0.6}}) == [
        {"epoch": 1, "val_accuracy": 0.5}, {"epoch": 2, "val_accuracy": 0.6}]
    import tempfile
    folder = Path(tempfile.mkdtemp())
    for name, fig in {"curves": charts.training_curves([{"label": "a", "epochs": epochs}], "htr"),
                      "folds": charts.training_curves([{"label": "a", "epochs": epochs},
                                                       {"label": "b", "epochs": epochs}], "htr"),
                      "scripts": charts.script_curves(epochs), "cost": charts.cost(epochs, 9000),
                      "confusions": charts.confusion_heatmap({("ب", "ت"): 5, ("ت", "ث"): 2, (" ", ""): 9, ("", "ا"): 1}),
                      "e2e": charts.e2e_breakdown({"m": {"gt_cer": 2.0, "e2e_cer": 3.5, "precision": 0.97, "recall": 0.95}}),
                      "heat": charts.heatmap("t", ["r1", "r2"], ["m"], {("r1", "m"): 1.0})}.items():
        charts.save(fig, folder, name)
        charts.caption(folder, name, name.upper())
    assert {p.name for p in folder.glob("*.pdf")} == {f"{n}.pdf" for n in
                                                      ("curves", "folds", "scripts", "cost", "confusions", "e2e", "heat")}
    assert dict((Path(p).stem, c) for p, c in charts.gallery(folder))["e2e"] == "E2E"


def test_crossval_by_writer_and_the_full_pipeline(env, madoc_remote, monkeypatch):
    """Kraken trained per writer fold (fake ketos), the folds' test pages read (fake kraken), names and CER pooled
    with intervals by writer, figures made; then a segmentation + recognition pair on the page-2 test pages."""
    from mlops import crossval, db, jobs, worker
    for module in ("mlops.crossval", "mlops.train", "mlops.evaluate"):
        monkeypatch.setattr(f"{module}.free_gpu", lambda log=print: None, raising=False)
    monkeypatch.setattr("mlops.train.ketos_program", lambda: ["ketos"])     # the fake ketos, whatever is imported
    names = ["لحسن شميطي", "عبد الرحيم كلفاع", "محمد العلوي", "سعيد بنعلي"]
    with db.connection() as conn:
        pages = [p for w, writer in enumerate(names) for act in range(4) for p in
                 act_pages(w, act, writer, (w * 50, act * 30, 7))]
        ingest_texts(conn, env, "cv", pages)
        assert [len(f) for f in crossval.make_folds([{"writer_key": "a"}] * 6 + [{"writer_key": "b"}] * 2 +
                                                    [{"writer_key": "c"}] * 2, max_folds=10, min_pages=5)] == [1, 2]
        jobs.enqueue(conn, "crossval", None, recognizer="kraken", scheme="writers", seeds=[1, 2], segmentation=True)
        jobs.enqueue(conn, "pipeline", None, segmentation="production", recognition="production")
        worker.main(once=True)
        done = db.all_rows(conn, "SELECT kind, status, error FROM jobs ORDER BY id")
        assert [j["status"] for j in done] == ["done", "done"], done
        cv, pipe = db.all_rows(conn, "SELECT * FROM experiments ORDER BY id")
        s = cv["summary"]
        assert len(s["folds"]) == 4 and s["writers_count"] == 4 and s["interval"].startswith("by page")
        assert sum(f["test"] for f in s["folds"]) == len(pages)              # every page tested once
        assert all(not set(f["writers"]) & set(g["writers"]) for f in s["folds"] for g in s["folds"] if f is not g)
        assert s["metrics"]["cer"] == 0.0 and s["names"]["all"]["exact"] == 100.0   # the fake candidate reads all
        assert set(s["spread"]) >= {"cer"} and s["spread"]["cer"]["sd"] == 0.0       # two seeds
        assert s["pipeline"]["e2e_cer"] is not None and s["pipeline"]["recall"] == 1.0
        figures = {p.stem for p in Path(cv["figures"]).glob("*.png")}
        assert {"training_curves", "error_composition", "names_cer", "form_rows", "writers", "e2e_breakdown",
                "writers_cer"} <= figures
        # the production pair loses the last line and word of every line: errors, missed lines
        p = pipe["summary"]
        assert p["metrics"]["cer"] > 0 and p["detection"]["recall"] < 1 and p["true_lines"]["metrics"]["cer"] > 0
        assert "e2e_breakdown" in {f.stem for f in Path(pipe["figures"]).glob("*.png")}
        # no name of a writer, no file name in the logs
        logs = " ".join(r["line"] for r in db.all_rows(conn, "SELECT line FROM job_logs"))
        assert "شميطي" not in logs and "03010108" not in logs


def test_crnn_learns_on_its_own_lines(tmp_path):
    """The CRNN of the baselines container on line images (no kraken needed here): it trains, early-stops and
    gives back one row per test line."""
    torch = pytest.importorskip("torch")
    from PIL import Image, ImageDraw
    from mlops.baselines import crnn
    lines = []
    for n, text in enumerate(["ab", "ba", "aab", "bba"] * 3):
        image = Image.new("L", (24 * len(text), 32), 255)
        draw = ImageDraw.Draw(image)
        for k, char in enumerate(text):
            draw.rectangle([24 * k + 4, 6, 24 * k + (18 if char == "a" else 10), 26], fill=0)
        image.save(tmp_path / f"{n}.png")
        lines.append({"page": f"p{n % 3}", "line": n, "image": str(tmp_path / f"{n}.png"), "text": text,
                      "display": text})
    torch.set_num_threads(1)
    result = crnn.train_and_read({"train": lines, "val": lines[:4], "test": lines[:4]},
                                 {"epochs": 3, "lag": 2, "batch_size": 4, "device": "cpu"}, 1, tmp_path / "out",
                                 lambda *_: None, lambda: False)
    assert len(result["rows"]) == 4 and {"page", "line", "ref", "pred"} <= set(result["rows"][0])
    assert [e["epoch"] for e in result["epochs"]] == [1, 2, 3] and "val_cer" in result["epochs"][0]


def test_detailed_validation_curves(env):
    """Every epoch's checkpoint (fake ketos convert, fake kraken) reads the validation lines: CER with and without
    spaces and punctuation and per script, added to the epochs."""
    import wandb
    from mlops import datasets, db
    from mlops.train import detailed_curves
    with db.connection() as conn:
        ingest_texts(conn, env, "d", [p for w in range(3) for p in act_pages(w, 1, f"كاتب {'ابت'[w]}", (w * 70, 3, 3))])
        rows = datasets.pages(conn, datasets.snapshot(conn)["id"], "train")
    folder = env / "runs" / "htr" / "job-1"
    (folder / "checkpoints").mkdir(parents=True)
    for epoch in range(2):
        (folder / "checkpoints" / f"checkpoint_{epoch:02d}-0.9{epoch}.ckpt").write_text("w")
    wandb.init(project="t", mode="offline")
    try:
        epochs = detailed_curves([{"epoch": 1}, {"epoch": 2}], folder / "checkpoints", rows, folder,
                                 lambda *_: None, lambda: False)
    finally:
        wandb.finish()
    assert all({"val_cer", "val_cer_lp", "val_cer_script_Arabic"} <= set(e) for e in epochs)
    assert not list((folder / "detail").glob("*.mlmodel"))                # the converted epochs are removed


def test_git_failure_says_why(env, monkeypatch):
    from mlops import release
    monkeypatch.setenv("GH_TOKEN", "secret-token")
    monkeypatch.setenv("MLOPS_MADOC_REMOTE", str(env / "nowhere.git"))
    (env / "release" / "madoc").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError) as failure:
        release.git("fetch", str(env / "nowhere.git"), "master")
    assert "git fetch" in str(failure.value) and "secret-token" not in str(failure.value)
    assert "dubious" in release.GIT_HINTS[1][0][0] and any("401" in keys for keys, _ in release.GIT_HINTS)


def test_intervals_by_writer_need_enough_writers():
    from mlops.scoring import with_interval
    units = {f"p{i}": (i % 3, 10) for i in range(12)}
    two = {u: f"W{i % 2}" for i, u in enumerate(units)}
    six = {u: f"W{i % 6}" for i, u in enumerate(units)}
    assert with_interval(units, two) == with_interval(units, {})               # 2 writers: by page
    assert with_interval(units, six) != with_interval(units, {})               # 6 writers: by writer