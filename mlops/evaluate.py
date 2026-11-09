"""Evaluation job: a candidate and the production model on the same test pages (old + new), live in W&B.

For each model:
  - the official ketos numbers on the whole test set (ketos test / ketos segtest);
  - a number per test page, to compare the two models page by page:
      htr   page_cer: the model reads the true lines of the page (the PageXML), CER against the transcription;
      blla  e2e_word_f1 / e2e_cer: the model segments the image, the production HTR reads its lines (as MaDoc
            does), words found and CER against the transcription, both whatever the order of the lines.
The comparison is paired (same pages): the mean gain per page and its confidence interval (bootstrap), and the
gain on the "old" test pages (those the production model was evaluated on) and on the "new" ones. The gate
(params.yaml: gate) decides; the result is saved in the evaluations table, where the app shows it.

W&B: one run per model (the candidate, the production model), in the same group, with the same keys and charts:
select both runs in W&B to compare them. The verdict is in the app (Compare & release), not in W&B.
"""

import hashlib
import json

from mlops import datasets, db, release
from mlops.eval_charts import evaluation_figures, log_details, log_model_run
from mlops.config import params, path
from mlops.kraken_output import blla_report, htr_report
from mlops.process import run
from mlops.recognize import detection_totals, meta, page_texts, pipeline, recognize
from mlops.resources import free_gpu
from mlops.scoring import flat_metrics, score
from mlops.textmetrics import bootstrap, group_bootstrap, line_cer, word_f1

# for each task: the per-page metric of the gate and whether higher is better
PAGE_METRICS = {"htr": ("page_cer", False), "blla": ("e2e_word_f1", True)}


def evaluate_model(task, model_path, test_list, rows, out, log, should_stop, htr_for_blla, page_meta):
    """-> {"metrics": aggregate + means + every number of mlops/scoring.py (flat, with intervals), "pages": per-page
    scores of the gate, "report": parsed ketos report, "summary": scoring summary, "detection": line detection}.

    htr: the model reads the true lines of every test page (kraken -f page -x); page_cer per page.
    blla: the model segments every test image and the production HTR reads its lines (as MaDoc does):
    e2e_word_f1 / e2e_cer per page, the line detection, and the scores of the recognised lines."""
    format_type = "page" if task == "htr" else "xml"
    command = ["ketos", "-d", params()["device"], "test" if task == "htr" else "segtest", "-m", model_path,
               "-f", format_type, "-e", test_list]
    text = run(command, log, should_stop)
    parsed = (htr_report if task == "htr" else blla_report)(text)
    detection = None
    if task == "htr":
        line_rows = recognize(model_path, rows, out / "lines", log, should_stop)
    else:
        line_rows, detected = pipeline(model_path, htr_for_blla, rows, out / "pipeline", log, should_stop)
        page_two = {r["id"] for r in rows if page_meta[r["id"]]["page_type"] == "2"} or {r["id"] for r in rows}
        detection = detection_totals({p: d for p, d in detected.items() if p in page_two})
    summary = score(line_rows, page_meta, task)
    if task == "htr":
        pages = {p: {"page_cer": round(v["cer"], 4)} for p, v in summary["per_page"].items()}
    else:
        pages = {}
        for page, (expected, predicted) in page_texts(line_rows).items():
            pages[page] = {"e2e_word_f1": round(word_f1(expected, predicted), 4),
                           "e2e_cer": round(line_cer(expected, predicted), 4)}
    metrics = dict(parsed["metrics"])
    for key in next(iter(pages.values()), {}):
        metrics[f"mean_{key}"] = round(sum(p[key] for p in pages.values()) / len(pages), 4)
    metrics.update(flat_metrics(summary))
    if detection:
        metrics.update({"line_precision": round(detection["precision"], 4),
                        "line_recall": round(detection["recall"], 4)})
    return {"metrics": metrics, "pages": pages, "report": parsed, "text": text, "summary": summary,
            "detection": detection}


def file_digest(file):
    digest = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_key(task, model_path, rows, htr_for_blla):
    """What an evaluation depends on: the model file, the test pages (their PageXML, so a normalisation counts),
    the HTR model of the segmentation's end-to-end score, the device and kraken. Same key: same numbers."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        kraken = version("kraken")
    except PackageNotFoundError:
        kraken = "unknown"
    parts = [task, "lines-v2", file_digest(model_path), params()["device"], kraken]
    if task == "blla":
        parts.append(file_digest(htr_for_blla))
    pages = path("data") / "pages"
    parts += [row["id"] + ":" + file_digest(pages / (row["id"] + ".xml")) for row in rows]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:24]


def cached_evaluation(key, compute, log):
    """The production model is evaluated again by every evaluation: the same model on the same pages gives the
    same numbers, so they are kept (<runs>/eval-cache/<key>.json) and reused."""
    file = path("runs") / "eval-cache" / f"{key}.json"
    if file.is_file():
        try:
            result = json.loads(file.read_text(encoding="utf-8"))
            log(f"production numbers reused (same model, same test pages: {file.name})")
            return result
        except (OSError, ValueError):
            pass
    result = compute()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def compare(task, candidate_pages, production_pages, old_ids, groups=None):
    """The paired comparison and the verdict of the gate -> (comparison, passed, message).

    groups: {page: its group (register-year)}. With enough test groups (gate.group_bootstrap_min_groups) the
    interval is computed by group; the gain of every group is reported either way."""
    settings = params()["gate"]
    rule = settings[task]
    metric, higher_is_better = PAGE_METRICS[task]
    ids = sorted(set(candidate_pages) & set(production_pages))
    gains = {i: (candidate_pages[i][metric] - production_pages[i][metric]) * (1 if higher_is_better else -1)
             for i in ids}
    groups = groups or {}
    group_gains = {}
    for i in ids:
        group_gains.setdefault(groups.get(i, i), []).append(gains[i])
    by_group = len(group_gains) >= settings.get("group_bootstrap_min_groups", 5) and bool(groups)
    if by_group:
        mean, low, high = group_bootstrap(gains, groups, settings["bootstrap_samples"], settings["confidence"])
    else:
        mean, low, high = bootstrap([gains[i] for i in ids], settings["bootstrap_samples"], settings["confidence"])
    comparison = {"metric": metric, "higher_is_better": higher_is_better, "pages": len(ids), "gain": round(mean, 4),
                  "low": round(low, 4), "high": round(high, 4), "confidence": settings["confidence"],
                  "interval": "by group" if by_group else "by page", "groups": len(group_gains)}
    if groups:   # the gain of each test group (its key is a hash: safe to show)
        comparison["group_gains"] = {g: {"pages": len(v), "gain": round(sum(v) / len(v), 4)}
                                     for g, v in sorted(group_gains.items())}
    for name, subset in (("old", [i for i in ids if i in old_ids]), ("new", [i for i in ids if i not in old_ids])):
        if subset and old_ids:
            comparison[name] = {"pages": len(subset), "gain": round(sum(gains[i] for i in subset) / len(subset), 4)}
    text = (f'{metric}: mean gain {mean:+.3f} per page, {int(100 * settings["confidence"])} % CI '
            f'[{low:+.3f}, {high:+.3f}] over {len(ids)} pages')
    if groups:
        text += (f" and {len(group_gains)} groups (interval by group)" if by_group else
                 f" ({len(group_gains)} test groups only: the interval is by page and does not cover a new register)")
    if low <= rule["min_gain"]:
        return comparison, False, f"{text}: not better with confidence (the CI must be above {rule['min_gain']})"
    if comparison.get("old") and comparison["old"]["gain"] < -rule["max_old_drop"]:
        return comparison, False, f'{text}, but worse on the old test pages ({comparison["old"]["gain"]:+.3f})'
    return comparison, True, f"{text}: better"


def model_run(task, role, name, artifact, project, group, config, compute, log):
    """One W&B run for one model (role: candidate / production): its evaluation, the charts W&B always had (mlops/
    eval_charts.py), the new numbers (intervals, error composition, names, form rows, writers) and its ketos
    report. -> (result, run url)."""
    import wandb
    settings = params()["wandb"]
    run_ = wandb.init(project=project, entity=settings["entity"], job_type="eval", group=group, name=name,
                      tags=[role, task], config={**config, "role": role, "model": artifact or "MaDoc models/"})
    try:
        if artifact:
            try:   # lineage: this run evaluated that model
                run_.use_artifact(f'{settings["entity"]}/{settings["project"][task]}/{artifact}')
            except Exception as e:
                log(f"W&B lineage skipped: {e}")
        result = compute()
        log_model_run(task, result)
        log_details(result)
        report = wandb.Artifact(f"{group}-{role}", type="evaluation",
                                metadata={"role": role, "model": artifact, **result["metrics"]})
        with report.new_file(f"{'segmentation' if task == 'blla' else 'recognition'}_evaluation_report.txt",
                             encoding="utf-8") as f:
            f.write(result.get("text", ""))
        run_.log_artifact(report)
        return result, run_.url
    finally:
        wandb.finish()


def unseen_test_pages(conn, rows, model, production_card, log):
    """The test pages neither model saw while training -> (rows, {"candidate": n, "production": n} excluded).

    The split may have changed since a model was trained (split again, groups added): a page of the test set
    now may have been a train or validation page then. Evaluated on it, the model would look better than it is,
    so it is left out (for the production model too, when its card names its dataset version)."""
    production_version = datasets.by_digest(conn, (production_card or {}).get("dataset_version"))
    seen = {"candidate": datasets.seen_page_ids(conn, model.get("dataset_version_id")),
            "production": datasets.seen_page_ids(conn, production_version["id"]) if production_version else set()}
    excluded = {role: sum(1 for r in rows if r["id"] in ids) for role, ids in seen.items()}
    kept = [r for r in rows if r["id"] not in seen["candidate"] | seen["production"]]
    for role, n in excluded.items():
        if n:
            log(f"{n} test pages left out: the {role} model trained or was validated on them (split changed since)")
    if not kept:
        raise RuntimeError("every test page was seen in training by one of the models: split the dataset again "
                           "or train the candidate on the current dataset version")
    return kept, excluded


def unit_of(conn):
    """The unit the test pages are drawn by in the intervals: the writer (split by writer) or the register-year."""
    from mlops.ingest import split_by
    from mlops import settings as app_settings
    return "writer_key" if split_by(app_settings.split(conn)) == "writer" else "group_key"


def ground_truth_cer(htr_model, rows, page_meta, out, log, should_stop):
    """The CER of the production HTR on the true lines of the page-2 test pages (the start of the end-to-end
    breakdown: what the segmentation adds comes on top of it)."""
    page_two = [r for r in rows if page_meta[r["id"]]["page_type"] == "2"] or rows
    line_rows = recognize(htr_model, page_two, out, log, should_stop)
    return score(line_rows, {r["id"]: page_meta[r["id"]] for r in page_two}, "htr")["metrics"]["cer"]


def evaluate(conn, job, log, should_stop, set_wandb_url):
    task = job["task"]
    model = db.one(conn, "SELECT * FROM models WHERE id = %s", job["args"]["model_id"])
    if model is None or model["task"] != task:
        raise RuntimeError(f"no {task} model {job['args'].get('model_id')}")
    version = datasets.snapshot(conn)
    rows = datasets.pages(conn, version["id"], "test")
    if not rows:
        raise RuntimeError("the dataset has no test page")
    release.sync()
    production = release.production(task)
    rows, excluded = unseen_test_pages(conn, rows, model, production["card"], log)
    out = path("runs") / task / f'job-{job["id"]}'
    test_list = datasets.write_list(datasets.xml_paths(rows), out / "test.txt")
    htr_for_blla = release.production("htr")["path"]
    page_meta = meta(conn, rows)
    # the test pages the production model was evaluated on (its card): the "old" pages
    old_version = datasets.by_digest(conn, production["card"].get("test_version"))
    old_ids = {r["id"] for r in datasets.pages(conn, old_version["id"], "test")} if old_version else set()
    free_gpu(log)
    # one W&B run per model, in the same group: select both in W&B to compare them
    project = params()["wandb"]["eval_project"][task]
    group = f"{task}-eval-job{job['id']}"
    config = {"task": task, "job": job["id"], "test_version": version["digest"], "test_pages": len(rows),
              "old_test_pages": len(old_ids), "excluded_test_pages": excluded}
    set_wandb_url(f'https://wandb.ai/{params()["wandb"]["entity"]}/{project}/groups/{group}')
    log("--- candidate")
    result, candidate_url = model_run(
        task, "candidate", f'{task}-model{model["id"]}', model["artifact"], project, group, config,
        lambda: evaluate_model(task, model["path"], test_list, rows, out / "candidate", log, should_stop,
                               htr_for_blla, page_meta), log)
    production_result, production_url = None, None
    unit = unit_of(conn)
    groups = {r["id"]: r.get(unit) or r["group_key"] for r in rows}
    if production["path"].is_file():
        log("--- production")
        production_result, production_url = model_run(
            task, "production", f'{task}-production-job{job["id"]}', production["card"].get("artifact"), project,
            group, config,
            lambda: cached_evaluation(
                evaluation_key(task, production["path"], rows, htr_for_blla),
                lambda: evaluate_model(task, production["path"], test_list, rows, out / "production", log,
                                       should_stop, htr_for_blla, page_meta), log), log)
        comparison, passed, message = compare(task, result["pages"], production_result["pages"], old_ids, groups)
    else:
        comparison = {"metric": PAGE_METRICS[task][0], "pages": len(rows), "gain": 0.0, "low": 0.0, "high": 0.0,
                      "confidence": params()["gate"]["confidence"]}
        passed, message = True, "no production model to compare with"
    comparison["wandb"] = {"group": group, "candidate": candidate_url, "production": production_url}
    comparison["excluded_pages"] = excluded
    comparison["unit"] = "writer" if unit == "writer_key" else "register-year"
    comparison["metric_interval"] = result["summary"]["interval"]   # how the intervals of the metrics were drawn
    gt_cer = None
    if task == "blla" and htr_for_blla.is_file():
        gt_cer = cached_evaluation(
            evaluation_key("htr", htr_for_blla, rows, htr_for_blla) + "-gt-page2",
            lambda: {"cer": ground_truth_cer(htr_for_blla, rows, page_meta, out / "gt-lines", log, should_stop)},
            log)["cer"]
    results = {"candidate": result}
    if production_result:
        results["production"] = production_result
    figures = out / "figures"
    try:
        evaluation_figures(task, results, figures, gt_cer)
    except Exception as e:   # a figure must never fail an evaluation
        log(f"figures not made: {e}")
    log(("PASS " if passed else "FAIL ") + message)
    details = {role: {k: v for k, v in r["summary"].items() if k != "per_page"} for role, r in results.items()}
    (out / "summary.json").write_text(json.dumps({"details": details, "gt_cer": gt_cer}, ensure_ascii=False),
                                      encoding="utf-8")
    evaluation = db.one(
        conn, "INSERT INTO evaluations (job_id, task, model_id, test_version_id, production_card, candidate, "
              "production, comparison, passed, message, figures) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
              "RETURNING id",
        job["id"], task, model["id"], version["id"], db.json(production["card"]), db.json(result["metrics"]),
        db.json(production_result["metrics"]) if production_result else None, db.json(comparison), passed,
        message, str(figures))
    return {"evaluation_id": evaluation["id"], "passed": passed, "message": message,
            "wandb": comparison["wandb"]}