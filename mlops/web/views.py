"""What the pages of the app show, as plain data (the routes of api.py turn it into JSON).

Every function takes a database connection and only reads (the changes are in api.py, through the modules of the
platform). The private values (original file names, writers' names) are only returned where the app shows them
to the person who runs the platform; nothing here is logged.
"""

import unicodedata
from collections import Counter
from pathlib import Path

from mlops import augment, datasets, db, settings as app_settings
from mlops.config import SPLITS, TASKS, params, path

TASK_NAMES = {"blla": "Segmentation (blla)", "htr": "Recognition (HTR)"}
VALIDATION_KEYS = ("val_accuracy", "val_word_accuracy", "val_bl_f1", "val_mean_iu")
LABELS = {"start_from": "Start from", "epochs": "Epochs", "quit": "Stop", "lag": "Lag", "min_epochs": "Minimum epochs",
          "batch_size": "Batch size", "resize": "Resize (alphabet)", "learning_rate": "Learning rate",
          "schedule": "Learning-rate schedule", "cos_max": "Cosine epochs",
          "kraken_augment": "kraken's line augmentation", "use_augmented": "Add the augmented train pages",
          "workers": "Data loading workers", "precision": "Precision", "compile": "Compile the lines once",
          "seed": "Seed", "detailed_curves": "Detailed validation curves", "extra_args": "Other ketos options"}


def iso(value):
    return value.isoformat() if value else None


def numbers(metrics, keys=None):
    return {k: v for k, v in (metrics or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and (keys is None or k in keys)}


# --------------------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------------------
def dataset(conn):
    """The dataset now, per split, and its digest."""
    from mlops.ingest import backfill_page_types
    backfill_page_types(conn)
    rows = db.all_rows(conn, "SELECT split, count(*) AS pages, sum(lines) AS lines, sum(characters) AS characters, "
                             "count(DISTINCT group_key) AS groups, count(DISTINCT writer_key) AS writers, "
                             "count(*) FILTER (WHERE writer_key IS NULL) AS no_writer, "
                             "count(*) FILTER (WHERE held_out) AS held_out, "
                             "count(*) FILTER (WHERE page_type = '1') AS page1, "
                             "count(*) FILTER (WHERE page_type = '2') AS page2, "
                             "count(*) FILTER (WHERE revision > 0) AS edited "
                             "FROM pages WHERE deleted_at IS NULL GROUP BY split")
    by_split = {r["split"]: r for r in rows}
    keys = ("pages", "page1", "page2", "writers", "no_writer", "held_out", "groups", "lines", "characters", "edited")
    splits = [{"split": s, **{k: int((by_split.get(s) or {}).get(k) or 0) for k in keys}} for s in SPLITS]
    total = {k: sum(s[k] for s in splits) for k in keys if k not in ("writers", "groups")}
    total["writers"] = db.one(conn, "SELECT count(DISTINCT writer_key) AS n FROM pages WHERE deleted_at IS NULL")["n"]
    total["groups"] = db.one(conn, "SELECT count(DISTINCT group_key) AS n FROM pages WHERE deleted_at IS NULL")["n"]
    latest = db.one(conn, "SELECT * FROM dataset_versions ORDER BY id DESC LIMIT 1")
    return {"splits": splits, "total": total, "digest": datasets.current_digest(conn),
            "latest_version": latest and {"id": latest["id"], "digest": latest["digest"]},
            "max_upload_mb": params()["ingest"]["max_upload_mb"]}


def versions(conn, limit=30):
    return [{"id": v["id"], "digest": v["digest"], "created": iso(v["created_at"]),
             "counts": {s: (v["counts"] or {}).get(s, 0) for s in SPLITS},
             "models": v["models"], "evaluations": v["evaluations"]}
            for v in db.all_rows(conn, "SELECT v.*, (SELECT count(*) FROM models m WHERE m.dataset_version_id = v.id) "
                                       "AS models, (SELECT count(*) FROM evaluations e WHERE e.test_version_id = v.id) "
                                       "AS evaluations FROM dataset_versions v ORDER BY id DESC LIMIT %s", limit)]


def batches(conn, limit=50):
    out = []
    for b in db.all_rows(conn, "SELECT b.*, (SELECT count(*) FROM pages p WHERE p.batch_id = b.id AND p.deleted_at "
                               "IS NULL) AS live FROM batches b ORDER BY id DESC LIMIT %s", limit):
        report = b["report"] or {}
        added = report.get("added") or {}
        out.append({"id": b["id"], "created": iso(b["created_at"]), "status": b["status"],
                    "deleted": bool(b["deleted_at"]) and b["status"] != "purged", "held_out": b["held_out"],
                    "pages": b["live"], "added": {s: added.get(s, 0) for s in SPLITS},
                    "duplicates": report.get("duplicates", 0), "rejected": report.get("rejected", 0),
                    "reasons": report.get("reasons") or {}})
    return out


def vocabulary(conn):
    """Every character with its count per split, the coverage of the splits and the page types."""
    from mlops.splitting import coverage
    from mlops.vocabulary import invisible, page_types, rare_threshold, shown, type_label, vocabulary as counts_of
    from mlops.vocabulary import vocabulary_tsv
    by_split = counts_of(conn)
    types = page_types(conn)
    every = Counter()
    for counter in by_split.values():
        every.update(counter)
    chars = [{"char": c, "shown": shown(c), "code": f"U+{ord(c):04X}",
              "name": unicodedata.name(c, "control character").title(), "total": every[c],
              "counts": {s: by_split[s].get(c, 0) for s in SPLITS}, "invisible": invisible(c)}
             for c in sorted(every)]
    report = coverage(by_split, types)
    ingest = params()["ingest"]
    return {"chars": chars, "rare": rare_threshold(), "splits": list(SPLITS),
            "coverage": {**report, "missing": {s: [{"shown": shown(c), "code": f"U+{ord(c):04X}"}
                                                   for c in report["missing"][s]] for s in SPLITS}},
            "targets": {"train": 1 - ingest["val_fraction"] - ingest["test_fraction"],
                        "val": ingest["val_fraction"], "test": ingest["test_fraction"]},
            "types": {s: dict(types[s]) for s in SPLITS},
            "type_labels": {k: type_label(k) for k in sorted(set().union(*(set(t) for t in types.values())))},
            "tsv": vocabulary_tsv(by_split, types)}


def writers(conn):
    from mlops import writers as w
    w.sync(conn)
    table = w.table(conn)
    spellings = w.raw_spellings(conn)
    return {"writers": [{**{k: row[k] for k in ("writer", "label", "pages", "acts", "page1", "page2", "incomplete_acts",
                                               "held_out")},
                         "known": bool(row["key"]), "splits": row["splits"], "pinned": row["pinned"] or "auto"}
                        for row in table],
            "variants": [{"a": a, "b": b, "pages_a": na, "pages_b": nb} for a, b, na, nb in w.variants(conn)],
            "aliases": [{"variant": v, "canonical": c} for v, c in sorted(w.aliases(conn).items())],
            "names": sorted({name for name, _, _ in spellings} | {c for _, c, _ in spellings})}


def split_settings(conn):
    value = app_settings.split(conn)
    weights = value["split_weights"]
    types = weights.get("page_types") or {}
    return {"split_by": value.get("split_by", "writer"), "val": round(100 * value["val_fraction"], 3),
            "test": round(100 * value["test_fraction"], 3), "tolerance": round(100 * value["split_tolerance"], 3),
            "missing": weights.get("missing_character", 1000), "page1": types.get("1", 1000),
            "page2": types.get("2", 1000)}


def split_from(values):
    """The form of the Split settings -> the settings (ValueError when they do not hold)."""
    def number(key, default=0):
        try:
            return float(values.get(key, default) if values.get(key, default) not in (None, "") else default)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: a number") from None
    return app_settings.check_split({
        "split_by": values.get("split_by") or "writer",
        "val_fraction": number("val") / 100, "test_fraction": number("test") / 100,
        "split_tolerance": number("tolerance") / 100,
        "split_weights": {"missing_character": number("missing"),
                          "page_types": {"1": number("page1"), "2": number("page2")}}})


def split_result(result):
    from mlops.vocabulary import shown
    report = result["coverage"]
    return {"text": result["text"].split("\n")[0], "pages": result["pages"], "units": result["groups"],
            "unit": result["unit"], "moved": result["moved"], "share": report["share"],
            "characters": report["characters"], "in_all_splits": report["in_all_splits"],
            "missing": {s: [shown(c) for c in report["missing"][s]] for s in SPLITS},
            "page_types": report.get("page_types") or {}}


def normalization(conn):
    from mlops.normalize import rules_rows
    return [dict(zip(("rule", "replaced", "what", "replaced_by", "source"), row)) for row in rules_rows(conn)]


def augmentations(conn):
    digest = datasets.current_digest(conn)
    train = db.one(conn, "SELECT count(*) AS n FROM pages WHERE deleted_at IS NULL AND split = 'train'")["n"]
    found = [{"folder": a["folder"].name, "digest": a["digest"], "copies": a["copies"], "seed": a["seed"],
              "pages": a["pages"], "current": a["digest"] == digest} for a in augment.augmentations()]
    return {"augmentations": found, "digest": digest, "train_pages": train,
            "current": any(a["current"] for a in found),
            "defaults": {"copies": params()["augment"]["copies"], "seed": params()["augment"]["seed"]}}


def missing(conn):
    from mlops.missing import summary_rows
    return [dict(zip(("register", "split", "complete", "missing1", "missing2"), row)) for row in summary_rows(conn)]


def preview():
    from mlops.preview import latest
    file = latest()
    return {"name": file.name, "mb": round(file.stat().st_size / 2 ** 20, 1)} if file else None


# --------------------------------------------------------------------------------------------------------------
# training, models, evaluations, experiments
# --------------------------------------------------------------------------------------------------------------
def training_form(conn):
    """The options of each task: their kind, help, choices and the value now (params.yaml + saved)."""
    models = app_settings.madoc_models(conn)
    out = {}
    for task in TASKS:
        values = app_settings.training(conn, task)
        fields = []
        for key in app_settings.TASK_FIELDS[task]:
            kind, help_text = app_settings.TRAIN_FIELDS[key]
            choices = (sorted(set(models) | {values[key]}) if key == "start_from"
                       else app_settings.CHOICES.get(key))
            fields.append({"key": key, "label": LABELS[key], "help": help_text, "value": values[key],
                           "kind": {bool: "bool", int: "int", float: "float"}.get(kind, "text"),
                           "choices": choices, "free": key == "start_from"})
        out[task] = {"name": TASK_NAMES[task], "fields": fields}
    return out


def models(conn, limit=50):
    return [{"id": m["id"], "task": m["task"], "status": m["status"], "artifact": m["artifact"],
             "digest": m["digest"], "validation": numbers(m["val_metrics"], VALIDATION_KEYS),
             "created": iso(m["created_at"]), "job_id": m["job_id"], "has_figures": bool(m["figures"])}
            for m in db.all_rows(conn, "SELECT m.*, v.digest FROM models m LEFT JOIN dataset_versions v "
                                       "ON v.id = m.dataset_version_id ORDER BY m.id DESC LIMIT %s", limit)]


def evaluations(conn, limit=50):
    return [{"id": e["id"], "task": e["task"], "model_id": e["model_id"], "artifact": e["artifact"],
             "passed": e["passed"], "gain": e["comparison"].get("gain"), "low": e["comparison"].get("low"),
             "high": e["comparison"].get("high"), "metric": e["comparison"].get("metric"),
             "pages": e["comparison"].get("pages"), "pr_url": e["pr_url"], "created": iso(e["created_at"])}
            for e in db.all_rows(conn, "SELECT e.*, m.artifact FROM evaluations e JOIN models m ON m.id = e.model_id "
                                       "ORDER BY e.id DESC LIMIT %s", limit)]


def interval(metrics, key):
    low, high = metrics.get(f"{key}_low"), metrics.get(f"{key}_high")
    return [low, high] if low is not None and high is not None else None


def evaluation(conn, evaluation_id):
    e = db.one(conn, "SELECT e.*, m.artifact FROM evaluations e JOIN models m ON m.id = e.model_id WHERE e.id = %s",
               evaluation_id)
    if not e:
        return None
    candidate, production = e["candidate"] or {}, e["production"] or {}
    keys = sorted(k for k, v in candidate.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                  and not k.endswith(("_low", "_high")))
    c = e["comparison"]
    return {"id": e["id"], "task": e["task"], "task_name": TASK_NAMES.get(e["task"], e["task"]),
            "artifact": e["artifact"], "model_id": e["model_id"], "passed": e["passed"], "message": e["message"],
            "pr_url": e["pr_url"], "created": iso(e["created_at"]),
            "metrics": [{"key": k, "production": production.get(k), "production_ci": interval(production, k),
                         "candidate": candidate.get(k), "candidate_ci": interval(candidate, k)} for k in keys],
            "comparison": {"metric": c.get("metric"), "pages": c.get("pages"), "gain": c.get("gain"),
                           "low": c.get("low"), "high": c.get("high"), "confidence": c.get("confidence"),
                           "interval": c.get("metric_interval") or "by page",
                           "subsets": {k: c[k] for k in ("old", "new") if c.get(k)},
                           "groups": c.get("group_gains") or {}, "unit": c.get("unit", "register-year"),
                           "excluded": {k: v for k, v in (c.get("excluded_pages") or {}).items() if v},
                           "wandb": c.get("wandb") or {}},
            "figures": figures(e["figures"])}


def experiments(conn, limit=50):
    out = []
    for e in db.all_rows(conn, "SELECT * FROM experiments ORDER BY id DESC LIMIT %s", limit):
        s = e["summary"] or {}
        metrics, ci, spread = s.get("metrics") or {}, s.get("ci") or {}, s.get("spread") or {}
        names = (s.get("names") or {}).get("all") or {}
        out.append({"id": e["id"], "kind": e["kind"], "name": e["name"], "created": iso(e["created_at"]),
                    "metrics": {k: {"value": metrics.get(k), "ci": ci.get(k), "sd": (spread.get(k) or {}).get("sd")}
                                for k in ("cer", "cer_lp", "wer") if metrics.get(k) is not None},
                    "names_exact": names.get("exact"), "names_normalized": names.get("normalized"),
                    "e2e_cer": (s.get("pipeline") or {}).get("e2e_cer"), "writers": s.get("writers_count"),
                    "folds": len(s.get("folds") or []) or None})
    return out


def experiment(conn, experiment_id):
    e = db.one(conn, "SELECT * FROM experiments WHERE id = %s", experiment_id)
    if not e:
        return None
    s = e["summary"] or {}
    pipeline = s.get("pipeline")
    if not pipeline and s.get("detection"):
        pipeline = {"e2e_cer": (s.get("metrics") or {}).get("cer"), **s["detection"],
                    "gt_cer": (s.get("true_lines") or {}).get("metrics", {}).get("cer")}
    return {"id": e["id"], "kind": e["kind"], "name": e["name"], "created": iso(e["created_at"]),
            "pages": s.get("pages", 0), "lines": s.get("lines", 0), "chars": s.get("chars", 0),
            "interval": s.get("interval", ""), "writers": s.get("writers_count", 0),
            "metrics": [{"key": k, "value": v, "ci": (s.get("ci") or {}).get(k), "spread": (s.get("spread") or {}).get(k)}
                        for k, v in (s.get("metrics") or {}).items()],
            "names": [{"key": k, **v} for k, v in (s.get("names") or {}).items()],
            "folds": s.get("folds") or [], "pipeline": pipeline, "figures": figures(e["figures"])}


def figures(folder):
    """The figures of a folder: [{"name", "caption", "pdf"}] (served one by one by api.py, never the folder)."""
    from mlops.charts import gallery
    if not folder or not Path(folder).is_dir():
        return []
    return [{"name": Path(png).name, "caption": caption, "pdf": Path(png).with_suffix(".pdf").is_file()}
            for png, caption in gallery(folder)]


def model_options(conn, task):
    return [{"value": "production", "label": "production (MaDoc)"}] + [
        {"value": str(m["id"]), "label": f'#{m["id"]} {m["artifact"] or ""}'.strip()}
        for m in db.all_rows(conn, "SELECT id, artifact FROM models WHERE task = %s ORDER BY id DESC LIMIT 30", task)]


# --------------------------------------------------------------------------------------------------------------
# jobs, workers
# --------------------------------------------------------------------------------------------------------------
def jobs(conn, limit=100, status=None, kind=None):
    where, args = [], []
    if status:
        where.append("status = %s")
        args.append(status)
    if kind:
        where.append("kind = %s")
        args.append(kind)
    rows = db.all_rows(conn, "SELECT id, kind, task, status, queue, created_at, started_at, finished_at, wandb_url, "
                             "error, cancel_requested, extract(epoch FROM coalesce(finished_at, now()) - started_at) "
                             "AS seconds FROM jobs" + (" WHERE " + " AND ".join(where) if where else "")
                       + " ORDER BY id DESC LIMIT %s", *args, limit)
    return [{"id": j["id"], "kind": j["kind"], "task": j["task"], "status": j["status"], "queue": j["queue"],
             "created": iso(j["created_at"]), "started": iso(j["started_at"]), "finished": iso(j["finished_at"]),
             "seconds": float(j["seconds"]) if j["seconds"] is not None else None, "wandb": j["wandb_url"],
             "error": (j["error"] or "")[:400], "cancel_requested": j["cancel_requested"]} for j in rows]


def job(conn, job_id):
    from mlops import jobs as queue
    row = db.one(conn, "SELECT * FROM jobs WHERE id = %s", job_id)
    if not row:
        return None
    return {"id": row["id"], "kind": row["kind"], "task": row["task"], "status": row["status"],
            "args": row["args"], "result": row["result"], "error": row["error"], "wandb": row["wandb_url"],
            "created": iso(row["created_at"]), "started": iso(row["started_at"]),
            "finished": iso(row["finished_at"]), "log": queue.tail(conn, job_id, 400)}


def workers(conn):
    return [{"name": w["name"], "job_id": w["job_id"], "age": w["age"], "alive": w["age"] < 300}
            for w in db.all_rows(conn, "SELECT name, job_id, extract(epoch FROM now() - seen_at)::int AS age "
                                       "FROM workers ORDER BY name")]


def job_stats(conn, days=30):
    """Per kind of job over the last days: runs, failures, mean and longest duration (Monitoring)."""
    return [{"kind": r["kind"], "task": r["task"], "runs": r["runs"], "failed": r["failed"],
             "cancelled": r["cancelled"], "mean": float(r["mean"]) if r["mean"] is not None else None,
             "longest": float(r["longest"]) if r["longest"] is not None else None}
            for r in db.all_rows(conn, "SELECT kind, task, count(*) AS runs, "
                                       "count(*) FILTER (WHERE status = 'failed') AS failed, "
                                       "count(*) FILTER (WHERE status = 'cancelled') AS cancelled, "
                                       "avg(extract(epoch FROM finished_at - started_at)) FILTER (WHERE status = 'done') "
                                       "AS mean, max(extract(epoch FROM finished_at - started_at)) "
                                       "FILTER (WHERE status = 'done') AS longest FROM jobs "
                                       "WHERE created_at > now() - make_interval(days => %s) "
                                       "GROUP BY kind, task ORDER BY kind, task", days)]


# --------------------------------------------------------------------------------------------------------------
# the pipeline (the graph of the home page)
# --------------------------------------------------------------------------------------------------------------
def plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def pipeline(conn):
    """The stages of the platform, each with its numbers and the state of its last job."""
    last = {}
    for row in db.all_rows(conn, "SELECT DISTINCT ON (kind, coalesce(task, '')) id, kind, task, status, created_at, "
                                 "finished_at FROM jobs ORDER BY kind, coalesce(task, ''), id DESC"):
        last[(row["kind"], row["task"])] = {"id": row["id"], "status": row["status"], "at": iso(row["finished_at"]
                                                                                               or row["created_at"])}
    active = {(r["kind"], r["task"]): r["status"] for r in db.all_rows(
        conn, "SELECT kind, task, status FROM jobs WHERE status IN ('queued', 'running') ORDER BY id")}

    def state(kind, task=None):
        if (kind, task) in active:
            return active[(kind, task)]
        found = last.get((kind, task))
        return {"done": "done", "failed": "failed", "cancelled": "idle"}.get(found["status"], "idle") if found else "idle"

    data = dataset(conn)
    total = data["total"]
    counts = {s["split"]: s["pages"] for s in data["splits"]}
    batch = db.one(conn, "SELECT count(*) FILTER (WHERE deleted_at IS NULL) AS live, max(created_at) AS last "
                         "FROM batches")
    rules = len(normalization(conn))
    split = app_settings.split(conn)
    aug = augmentations(conn)
    model_counts = {r["task"]: r for r in db.all_rows(conn, "SELECT task, count(*) AS n, max(id) AS last FROM models "
                                                            "GROUP BY task")}
    evals = {r["task"]: r for r in db.all_rows(conn, "SELECT DISTINCT ON (task) task, id, passed, pr_url FROM "
                                                     "evaluations ORDER BY task, id DESC")}
    released = db.one(conn, "SELECT count(*) AS n FROM evaluations WHERE pr_url IS NOT NULL")["n"]
    experiment_count = db.one(conn, "SELECT count(*) AS n FROM experiments")["n"]
    nodes = [
        {"id": "export", "title": "eScriptorium export", "route": "data/overview", "state": "idle",
         "lines": [plural(batch["live"] or 0, "batch").replace("batchs", "batches"), f'last {iso(batch["last"])[:10]}' if batch["last"] else "no upload yet"]},
        {"id": "ingest", "title": "Ingest", "route": "jobs", "state": state("ingest"),
         "lines": [plural(total["pages"], "page"), plural(total["lines"], "line")]},
        {"id": "normalize", "title": "Normalise", "route": "data/normalization", "state": state("normalize"),
         "lines": [plural(rules, "rule"), plural(total["characters"], "character")]},
        {"id": "edit", "title": "Edit pages", "route": "editor", "state": "idle",
         "lines": [plural(total["edited"], "page") + " edited", "revisions kept"]},
        {"id": "split", "title": f'Split by {split.get("split_by", "writer")}', "route": "data/split", "state": "idle",
         "lines": [" / ".join(str(counts.get(s, 0)) for s in SPLITS) + " pages", plural(total["writers"], "writer")]},
        {"id": "version", "title": "Dataset version", "route": "data/overview", "state": "idle",
         "lines": [data["digest"], "recorded" if data["latest_version"] and data["latest_version"]["digest"]
                   == data["digest"] else "made at the next job"]},
        {"id": "augment", "title": "Augment", "route": "data/augmentation", "state": state("augment"),
         "lines": ["current version" if aug["current"] else "not for this version",
                   plural(sum(a["pages"] for a in aug["augmentations"] if a["current"]), "page")]},
    ]
    for task in TASKS:
        m = model_counts.get(task) or {}
        nodes.append({"id": f"train_{task}", "title": f"Train {task}", "route": "training", "state": state("train", task),
                      "lines": [plural(m.get("n", 0), "model"), f'last #{m["last"]}' if m.get("last") else "none yet"]})
    for task in TASKS:
        e = evals.get(task)
        nodes.append({"id": f"evaluate_{task}", "title": f"Evaluate {task}", "route": "evaluation",
                      "state": state("evaluate", task),
                      "lines": [("passed the gate" if e["passed"] else "did not pass") if e else "no evaluation yet",
                                f'evaluation #{e["id"]}' if e else ""]})
    nodes += [
        {"id": "release", "title": "Release (pull request)", "route": "evaluation",
         "state": state("release", "htr") if ("release", "htr") in active else state("release", "blla"),
         "lines": [plural(released, "pull request"), "merged by a person"]},
        {"id": "madoc", "title": "MaDoc (production)", "route": None, "state": "idle",
         "lines": ["deployed by its CI", "after the merge"]},
        {"id": "experiments", "title": "Experiments", "route": "experiments",
         "state": state("crossval") if ("crossval", None) in active else state("pipeline"),
         "lines": [plural(experiment_count, "result"), "cross-validation, baselines"]},
    ]
    edges = [("export", "ingest"), ("ingest", "normalize"), ("normalize", "edit"), ("edit", "split"),
             ("split", "version"), ("version", "augment"), ("augment", "train_blla"), ("augment", "train_htr"),
             ("version", "train_blla"), ("version", "train_htr"), ("train_blla", "evaluate_blla"),
             ("train_htr", "evaluate_htr"), ("evaluate_blla", "release"), ("evaluate_htr", "release"),
             ("release", "madoc"), ("version", "experiments")]
    running = db.all_rows(conn, "SELECT id, kind, task, status, started_at FROM jobs WHERE status IN ('queued', "
                                "'running') ORDER BY id")
    return {"nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges],
            "active": [{"id": r["id"], "kind": r["kind"], "task": r["task"], "status": r["status"],
                        "started": iso(r["started_at"])} for r in running]}


# --------------------------------------------------------------------------------------------------------------
# the pages (Editor)
# --------------------------------------------------------------------------------------------------------------
AUGMENTED = "augmented"


def page_list(conn, split="train", kind="all", show_names=False, search="", edited=False):
    """The pages of a split (or matching a search, any split; or the augmented pages) for the Editor's list."""
    from mlops.browse import type_label
    if split == AUGMENTED:
        found = augment.augmentations()
        if not found:
            return {"pages": [], "note": "No augmented pages yet: Data, Augmentation."}
        newest = found[0]
        types = {r["id"]: r["page_type"] for r in db.all_rows(conn, "SELECT id, page_type FROM pages")}
        items = []
        for xml in sorted(newest["folder"].glob("*.xml")):
            base = xml.stem.split("_aug")[0]
            if kind != "all" and types.get(base) != kind:
                continue
            items.append({"key": f'aug:{newest["folder"].name}:{xml.stem}', "id": xml.stem, "split": AUGMENTED,
                          "type": type_label(types.get(base)), "lines": None, "revision": 0, "readonly": True})
        return {"pages": items, "note": f'Augmentation {newest["folder"].name}: {len(items)} pages (read only).'}
    where, args = ["p.deleted_at IS NULL"], []
    search = (search or "").strip()[:100]
    if search:
        where.append("(o.original_name ILIKE %s OR p.id ILIKE %s)" if show_names else "p.id ILIKE %s")
        args += [f"%{search}%", f"%{search}%"] if show_names else [f"%{search}%"]
    elif split in SPLITS:
        where.append("p.split = %s")
        args.append(split)
    if kind != "all":
        where.append("p.page_type = %s")
        args.append(kind)
    if edited:
        where.append("p.revision > 0")
    rows = db.all_rows(conn, "SELECT p.id, p.split, p.page_type, p.lines, p.held_out, p.revision, o.original_name "
                             "FROM pages p LEFT JOIN page_origins o ON o.page_id = p.id WHERE " + " AND ".join(where)
                       + " ORDER BY p.page_type, p.id LIMIT 3000", *args)
    items = [{"key": r["id"], "id": r["id"], "split": r["split"], "type": type_label(r["page_type"]),
              "lines": r["lines"], "held_out": r["held_out"], "revision": r["revision"], "readonly": False,
              "name": Path(r["original_name"] or "").stem if show_names else None} for r in rows]
    where_text = f'match "{search}"' if search else f"in {split}"
    return {"pages": items, "note": f"{len(items)} pages {where_text}"
            + ("" if kind == "all" else f" ({type_label(kind)})") + (", edited only" if edited else "") + "."}


def page_detail(conn, page_id, show_names=False):
    from mlops import editing
    from mlops.browse import type_label
    from mlops.pagexml import read_page
    from mlops.writers import label
    row = editing.page_row(conn, page_id)
    xml, image = editing.files_of(page_id)
    page = read_page(xml)
    origin = db.one(conn, "SELECT o.original_name, w.raw_name FROM pages p LEFT JOIN page_origins o "
                          "ON o.page_id = p.id LEFT JOIN page_writers w ON w.page_id = p.id WHERE p.id = %s", page_id)
    out = {"id": page_id, "split": row["split"], "type": type_label(row["page_type"]), "held_out": row["held_out"],
           "revision": row["revision"], "width": page["width"], "height": page["height"], "readonly": False,
           "image_found": image.is_file(), "writer": label(row["writer_key"]),
           "lines": [{"id": line["id"], "polygon": line["polygon"], "baseline": line["baseline_points"],
                      "text": line["text"]} for line in page["lines"]],
           "history": [{**h, "created_at": iso(h["created_at"])} for h in editing.history(conn, page_id)]}
    if image.is_file() and (not out["width"] or not out["height"]):
        from PIL import Image
        with Image.open(image) as picture:
            out["width"], out["height"] = picture.size
    if show_names:
        out["name"] = Path((origin or {}).get("original_name") or "").stem or None
        out["officer"] = (origin or {}).get("raw_name") or None
    return out


def augmented_files(key):
    """The (xml, image) of an augmented page from its key "aug:<folder>:<stem>", refusing anything else (no "/":
    the key is one segment of the address)."""
    from mlops.pagexml import read_page
    if not key.startswith("aug:"):
        raise FileNotFoundError(key)
    folder_name, _, stem = key[4:].partition(":")
    root = (path("data") / "augmented").resolve()
    xml = (root / folder_name / f"{stem}.xml").resolve()
    if xml.parent.parent != root or not xml.is_file() or "/" in stem or "\\" in stem:
        raise FileNotFoundError(key)
    image = xml.parent / Path(read_page(xml)["image"].replace("\\", "/")).name
    if image.parent != xml.parent or not image.is_file():
        raise FileNotFoundError(key)
    return xml, image


def augmented_detail(key):
    from mlops.pagexml import read_page
    xml, image = augmented_files(key)
    page = read_page(xml)
    return {"id": xml.stem, "key": key, "split": AUGMENTED, "type": "", "revision": 0, "readonly": True,
            "width": page["width"], "height": page["height"], "image_found": True, "history": [],
            "lines": [{"id": line["id"], "polygon": line["polygon"], "baseline": line["baseline_points"],
                       "text": line["text"]} for line in page["lines"]]}
