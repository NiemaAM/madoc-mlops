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

from bidi.algorithm import get_display

from mlops import datasets, db, release
from mlops.eval_charts import log_model_run
from mlops.config import params, path
from mlops.kraken_output import blla_report, htr_report
from mlops.pagexml import read_page
from mlops.process import run
from mlops.resources import free_gpu
from mlops.textmetrics import bootstrap, cer, line_cer, word_f1

# for each task: the per-page metric of the gate and whether higher is better
PAGE_METRICS = {"htr": ("page_cer", False), "blla": ("e2e_word_f1", True)}


def truth(row):
    """The true lines of a page (its transcription, in the PageXML)."""
    return [line["text"] for line in read_page(path("data") / "pages" / f'{row["id"]}.xml')["lines"]]


def read_output(file):
    """kraken writes the lines in visual order: python-bidi gives the normal text back (as MaDoc does)."""
    if not file.is_file():
        return []
    return [get_display(line) for line in file.read_text(encoding="utf-8").splitlines()]


def per_page(task, model_path, rows, out, log, should_stop, htr_for_blla=None):
    """{page id: {metric: value}} for one model."""
    device = ["-d", params()["device"]]
    out.mkdir(parents=True, exist_ok=True)
    pages = path("data") / "pages"
    pairs = []
    for row in rows:
        source = pages / (f'{row["id"]}.xml' if task == "htr" else row["image_name"])
        pairs += ["-i", source, out / f'{row["id"]}.txt']
    if task == "htr":
        run(["kraken", *device, "-f", "page", *pairs, "ocr", "-m", model_path], log, should_stop)
    else:
        run(["kraken", *device, *pairs, "segment", "-bl", "-i", model_path, "ocr", "-m", htr_for_blla],
            log, should_stop)
    result = {}
    for row in rows:
        predicted, expected = read_output(out / f'{row["id"]}.txt'), truth(row)
        if task == "htr":
            result[row["id"]] = {"page_cer": round(cer(expected, predicted), 4)}
        else:
            result[row["id"]] = {"e2e_word_f1": round(word_f1(expected, predicted), 4),
                                 "e2e_cer": round(line_cer(expected, predicted), 4)}
    return result


def evaluate_model(task, model_path, test_list, rows, out, log, should_stop, htr_for_blla):
    """-> {"metrics": aggregate + means, "pages": {...}, "report": parsed ketos report}."""
    format_type = "page" if task == "htr" else "xml"
    command = ["ketos", "-d", params()["device"], "test" if task == "htr" else "segtest", "-m", model_path,
               "-f", format_type, "-e", test_list]
    text = run(command, log, should_stop)
    parsed = (htr_report if task == "htr" else blla_report)(text)
    pages = per_page(task, model_path, rows, out, log, should_stop, htr_for_blla)
    metrics = dict(parsed["metrics"])
    for key in next(iter(pages.values()), {}):
        metrics[f"mean_{key}"] = round(sum(p[key] for p in pages.values()) / len(pages), 4)
    return {"metrics": metrics, "pages": pages, "report": parsed, "text": text}


def compare(task, candidate_pages, production_pages, old_ids):
    """The paired comparison and the verdict of the gate -> (comparison, passed, message)."""
    settings = params()["gate"]
    rule = settings[task]
    metric, higher_is_better = PAGE_METRICS[task]
    ids = sorted(set(candidate_pages) & set(production_pages))
    gains = {i: (candidate_pages[i][metric] - production_pages[i][metric]) * (1 if higher_is_better else -1)
             for i in ids}
    mean, low, high = bootstrap([gains[i] for i in ids], settings["bootstrap_samples"], settings["confidence"])
    comparison = {"metric": metric, "higher_is_better": higher_is_better, "pages": len(ids), "gain": round(mean, 4),
                  "low": round(low, 4), "high": round(high, 4), "confidence": settings["confidence"]}
    for name, subset in (("old", [i for i in ids if i in old_ids]), ("new", [i for i in ids if i not in old_ids])):
        if subset and old_ids:
            comparison[name] = {"pages": len(subset), "gain": round(sum(gains[i] for i in subset) / len(subset), 4)}
    text = (f'{metric}: mean gain {mean:+.3f} per page, {int(100 * settings["confidence"])} % CI '
            f'[{low:+.3f}, {high:+.3f}] over {len(ids)} pages')
    if low <= rule["min_gain"]:
        return comparison, False, f"{text}: not better with confidence (the CI must be above {rule['min_gain']})"
    if comparison.get("old") and comparison["old"]["gain"] < -rule["max_old_drop"]:
        return comparison, False, f'{text}, but worse on the old test pages ({comparison["old"]["gain"]:+.3f})'
    return comparison, True, f"{text}: better"


def model_run(task, role, name, artifact, project, group, config, compute, log):
    """One W&B run for one model (role: candidate / production): its evaluation, its charts, its ketos report.
    -> (result, run url)."""
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
        report = wandb.Artifact(f"{group}-{role}", type="evaluation",
                                metadata={"role": role, "model": artifact, **result["metrics"]})
        with report.new_file(f"{'segmentation' if task == 'blla' else 'recognition'}_evaluation_report.txt",
                             encoding="utf-8") as f:
            f.write(result.get("text", ""))
        run_.log_artifact(report)
        return result, run_.url
    finally:
        wandb.finish()


def evaluate(conn, job, log, should_stop, set_wandb_url):
    task = job["task"]
    model = db.one(conn, "SELECT * FROM models WHERE id = %s", job["args"]["model_id"])
    if model is None or model["task"] != task:
        raise RuntimeError(f"no {task} model {job['args'].get('model_id')}")
    version = datasets.snapshot(conn)
    rows = datasets.pages(conn, version["id"], "test")
    if not rows:
        raise RuntimeError("the dataset has no test page")
    out = path("runs") / task / f'job-{job["id"]}'
    test_list = datasets.write_list(datasets.xml_paths(rows), out / "test.txt")
    release.sync()
    production = release.production(task)
    htr_for_blla = release.production("htr")["path"]
    # the test pages the production model was evaluated on (its card): the "old" pages
    old_version = datasets.by_digest(conn, production["card"].get("test_version"))
    old_ids = {r["id"] for r in datasets.pages(conn, old_version["id"], "test")} if old_version else set()
    free_gpu(log)
    # one W&B run per model, in the same group: select both in W&B to compare them
    project = params()["wandb"]["eval_project"][task]
    group = f"{task}-eval-job{job['id']}"
    config = {"task": task, "job": job["id"], "test_version": version["digest"], "test_pages": len(rows),
              "old_test_pages": len(old_ids)}
    set_wandb_url(f'https://wandb.ai/{params()["wandb"]["entity"]}/{project}/groups/{group}')
    log("--- candidate")
    result, candidate_url = model_run(
        task, "candidate", f'{task}-model{model["id"]}', model["artifact"], project, group, config,
        lambda: evaluate_model(task, model["path"], test_list, rows, out / "candidate", log, should_stop,
                               htr_for_blla), log)
    production_result, production_url = None, None
    if production["path"].is_file():
        log("--- production")
        production_result, production_url = model_run(
            task, "production", f'{task}-production-job{job["id"]}', production["card"].get("artifact"), project,
            group, config,
            lambda: evaluate_model(task, production["path"], test_list, rows, out / "production", log,
                                   should_stop, htr_for_blla), log)
        comparison, passed, message = compare(task, result["pages"], production_result["pages"], old_ids)
    else:
        comparison = {"metric": PAGE_METRICS[task][0], "pages": len(rows), "gain": 0.0, "low": 0.0, "high": 0.0,
                      "confidence": params()["gate"]["confidence"]}
        passed, message = True, "no production model to compare with"
    comparison["wandb"] = {"group": group, "candidate": candidate_url, "production": production_url}
    log(("PASS " if passed else "FAIL ") + message)
    evaluation = db.one(
        conn, "INSERT INTO evaluations (job_id, task, model_id, test_version_id, production_card, candidate, "
              "production, comparison, passed, message) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
              "RETURNING id",
        job["id"], task, model["id"], version["id"], db.json(production["card"]), db.json(result["metrics"]),
        db.json(production_result["metrics"]) if production_result else None, db.json(comparison), passed,
        message)
    return {"evaluation_id": evaluation["id"], "passed": passed, "message": message,
            "wandb": comparison["wandb"]}