"""Experiments: a recognizer trained and tested on writers it never saw (cross-validation), or on the dataset's own
split, scored by the same code for kraken and the other recognizers (CRNN, TrOCR-style, Qwen2.5-VL).

    scheme "writers": the folds are writers. Each fold tests the pages of its writers with a model trained on the
                      other writers; every page is tested once, by a model that never saw its hand. A writer with
                      fewer than crossval.min_writer_pages pages joins the others of its size in one fold. With
                      more writers than crossval.max_folds, the writers are grouped into that many folds of about
                      the same number of pages.
    scheme "split":   one fold: the train / val / test pages of the current dataset version (the numbers can be
                      put next to the evaluation of the production model, on the same test pages).

The validation pages of a fold come from its training writers only (whole acts, crossval.val_fraction of the
pages): never from the test writers. The pages without a known writer are always trained on; the held-out pages
are never used here (they stay for the final test). Several seeds: every fold is trained once per seed; the
results give the mean and the standard deviation over the seeds.

Per fold: the names (page 1 and page 2), the CER with and without spaces and punctuation, per writer; with
"segmentation" (kraken only), a segmentation model is trained on the same fold and the full pipeline is scored
on the page-2 test pages. The pooled result (all folds), its intervals by writer and its figures are saved in the
table experiments; the curves of every fold are in W&B (one group per experiment).
"""

import json
import random
import shutil
import statistics
from collections import defaultdict

from mlops import datasets, db, release, settings as app_settings
from mlops.config import params, path
from mlops.recognize import detection_totals, meta, pipeline, recognize
from mlops.resources import free_gpu
from mlops.scoring import cer_of, flat_metrics, score
from mlops.writers import label

KRAKEN = "kraken"
RECOGNIZERS = ("kraken", "crnn", "trocr", "qwen")
RECOGNIZER_NAMES = {"kraken": "Kraken", "crnn": "CRNN", "trocr": "TrOCR-style", "qwen": "Qwen2.5-VL-3B"}


def settings_of():
    return {"min_writer_pages": 5, "max_folds": 10, "val_fraction": 0.15, **(params().get("crossval") or {})}


def make_folds(rows, max_folds=10, min_pages=5):
    """rows: the pages with a known writer -> [[writer keys of the fold], ...]."""
    pages = defaultdict(int)
    for row in rows:
        pages[row["writer_key"]] += 1
    big = sorted((w for w, n in pages.items() if n >= min_pages), key=lambda w: (-pages[w], w))
    small = sorted(w for w, n in pages.items() if n < min_pages)
    units = [[w] for w in big] + ([small] if small else [])
    if len(units) <= max_folds:
        return units
    bins = [[] for _ in range(max_folds)]
    sizes = [0] * max_folds
    for unit in sorted(units, key=lambda u: -sum(pages[w] for w in u)):
        k = sizes.index(min(sizes))
        bins[k] += unit
        sizes[k] += sum(pages[w] for w in unit)
    return [b for b in bins if b]


def validation_split(rows, fraction, seed):
    """Whole acts of the training pages for validation -> (train rows, val rows)."""
    acts = sorted({r["act_key"] or r["id"] for r in rows})
    random.Random(seed).shuffle(acts)
    target, chosen, count = fraction * len(rows), set(), 0
    sizes = defaultdict(int)
    for r in rows:
        sizes[r["act_key"] or r["id"]] += 1
    for act in acts:
        if count >= target:
            break
        chosen.add(act)
        count += sizes[act]
    val = [r for r in rows if (r["act_key"] or r["id"]) in chosen]
    return [r for r in rows if (r["act_key"] or r["id"]) not in chosen], val


def plan(conn, scheme, version):
    """-> [{"name", "train": rows, "val": rows, "test": rows, "writers": [labels]}]."""
    if scheme == "split":
        test = [r for r in datasets.pages(conn, version["id"], "test")]
        return [{"name": "split", "train": datasets.pages(conn, version["id"], "train"),
                 "val": datasets.pages(conn, version["id"], "val"), "test": test,
                 "writers": sorted({label(r["writer_key"]) for r in test})}]
    rows = [r for s in ("train", "val", "test") for r in datasets.pages(conn, version["id"], s) if not r["held_out"]]
    known = [r for r in rows if r["writer_key"]]
    unknown = [r for r in rows if not r["writer_key"]]
    options = settings_of()
    folds = make_folds(known, options["max_folds"], options["min_writer_pages"])
    if len(folds) < 2:
        raise RuntimeError(f"{len(folds)} writer(s) found: a cross-validation by writer needs 2 or more (Data tab: "
                           "Writers; the officer is read on page 2, 'من طرفنا نحن')")
    out = []
    for k, writers in enumerate(folds, 1):
        test = [r for r in known if r["writer_key"] in set(writers)]
        pool = [r for r in known if r["writer_key"] not in set(writers)] + unknown
        train, val = validation_split(pool, options["val_fraction"], k)
        out.append({"name": f"fold{k}", "train": train, "val": val, "test": test,
                    "writers": sorted(label(w) for w in writers)})
    return out


def train_kraken(conn, task, settings, fold, version, out, log, should_stop, group, job_id, seed):
    """One kraken training of a fold (its own W&B run) -> (model path, epochs)."""
    import wandb
    from mlops.train import base_model, run_ketos, training_inputs
    fold_version = {"digest": f'{version["digest"]}-cv{job_id}-{fold["name"]}'}
    train_list, val_list, augmented, binary = training_inputs(
        task, settings, fold_version, fold["train"], fold["val"], out, log, should_stop,
        augment_root=path("data") / "work" / f"cv-{job_id}" / "augmented")
    base = base_model(task, settings.get("start_from"))
    w = params()["wandb"]
    run_ = wandb.init(project=w["project"][task], entity=w["entity"], job_type="crossval", group=group,
                      name=f'{group}-{fold["name"]}-{task}-s{seed}',
                      config={"task": task, "fold": fold["name"], "seed": seed, "settings": settings,
                              "train_pages": len(fold["train"]), "val_pages": len(fold["val"]),
                              "test_pages": len(fold["test"]), "test_writers": fold["writers"],
                              "augmented_pages": augmented, "base_model": base["relative"]})
    try:
        trained = run_ketos(task, {**settings, "seed": seed}, base["path"], train_list, val_list, out, log,
                            should_stop, binary)
        if trained["peak_memory_mib"]:
            run_.summary["peak_gpu_memory_mib"] = trained["peak_memory_mib"]
    finally:
        wandb.finish()
    shutil.rmtree(trained["checkpoints"], ignore_errors=True)
    return trained["model_path"], trained["epochs"], trained["peak_memory_mib"]


def run_fold(conn, job, fold, version, recognizer, settings, seed, out, log, should_stop, group, with_segmentation):
    """Train on the fold, read its test pages -> {"rows", "epochs", "pipeline": {...} or None}."""
    out.mkdir(parents=True, exist_ok=True)
    result = {"pipeline": None}
    if recognizer == KRAKEN:
        model, epochs, memory = train_kraken(conn, "htr", settings["htr"], fold, version, out / "htr", log,
                                             should_stop, group, job["id"], seed)
        result["rows"] = recognize(model, fold["test"], out / "lines", log, should_stop)
        result["epochs"], result["peak_memory_mib"] = epochs, memory
        if with_segmentation:
            seg, _, _ = train_kraken(conn, "blla", settings["blla"], fold, version, out / "blla", log, should_stop,
                                     group, job["id"], seed)
            page_two = [r for r in fold["test"] if r["page_type"] == "2"] or fold["test"]
            rows, detected = pipeline(seg, model, page_two, out / "pipeline", log, should_stop)
            result["pipeline"] = {"rows": rows, "detection": detection_totals(detected),
                                  "pages": [r["id"] for r in page_two]}
    else:
        import wandb
        from mlops.baselines import run_recognizer
        w = params()["wandb"]
        run_ = wandb.init(project=w["project"]["htr"], entity=w["entity"], job_type="crossval", group=group,
                          name=f'{group}-{fold["name"]}-{recognizer}-s{seed}',
                          config={"recognizer": recognizer, "fold": fold["name"], "seed": seed,
                                  "settings": settings.get(recognizer) or {}, "train_pages": len(fold["train"]),
                                  "val_pages": len(fold["val"]), "test_pages": len(fold["test"]),
                                  "test_writers": fold["writers"]})
        try:
            result.update(run_recognizer(recognizer, fold, settings.get(recognizer) or {}, seed, out, log,
                                         should_stop))
            if result.get("peak_memory_mib"):
                run_.summary["peak_gpu_memory_mib"] = result["peak_memory_mib"]
        finally:
            wandb.finish()
    return result


def crossval(conn, job, log, should_stop, set_wandb_url):
    """The "crossval" job -> {"experiment_id", "metrics"}."""
    import wandb
    args = job["args"] or {}
    recognizer = args.get("recognizer", KRAKEN)
    if recognizer not in RECOGNIZERS:
        raise RuntimeError(f"unknown recognizer {recognizer}")
    scheme = args.get("scheme", "writers")
    seeds = [int(s) for s in (args.get("seeds") or [params()["seed"]])]
    with_segmentation = bool(args.get("segmentation")) and recognizer == KRAKEN
    settings = {"htr": args.get("settings") or app_settings.training(conn, "htr"),
                "blla": args.get("blla_settings") or app_settings.training(conn, "blla"),
                recognizer: args.get("settings") or {}}
    from mlops import writers
    writers.sync(conn)
    version = datasets.snapshot(conn)
    folds = plan(conn, scheme, version)
    name = args.get("name") or f'{RECOGNIZER_NAMES[recognizer]} {"by writer" if scheme == "writers" else "on the split"}'
    log(f"{name}: {len(folds)} fold(s), seeds {seeds}: "
        + "; ".join(f'{f["name"]} test {len(f["test"])} pages ({", ".join(f["writers"])}), train {len(f["train"])}, '
                    f'val {len(f["val"])}' for f in folds))
    if recognizer == KRAKEN:
        release.sync()
    free_gpu(log)
    out = path("runs") / "experiments" / f'job-{job["id"]}'
    group = f'cv-job{job["id"]}'
    w = params()["wandb"]
    set_wandb_url(f'https://wandb.ai/{w["entity"]}/{w["project"]["htr"]}/groups/{group}')
    all_meta = meta(conn, [r for f in folds for r in f["test"]])
    per_seed, curves, pipelines, memory = {}, [], [], []
    for seed in seeds:
        rows, pipeline_rows, detections, pipeline_pages = [], [], [], []
        for fold in folds:
            log(f'--- {fold["name"]} (seed {seed})')
            result = run_fold(conn, job, fold, version, recognizer, settings, seed,
                              out / f's{seed}' / fold["name"], log, should_stop, group, with_segmentation)
            rows += result["rows"]
            if seed == seeds[0] and result.get("epochs"):
                curves.append({"label": fold["name"], "epochs": result["epochs"]})
                memory.append(result.get("peak_memory_mib") or 0)
            if result["pipeline"]:
                pipeline_rows += result["pipeline"]["rows"]
                detections.append(result["pipeline"]["detection"])
                pipeline_pages += result["pipeline"]["pages"]
        (out / f"s{seed}").mkdir(parents=True, exist_ok=True)
        (out / f"s{seed}" / "lines.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        per_seed[seed] = score(rows, all_meta, recognizer)
        if pipeline_rows:
            e2e = score(pipeline_rows, {p: all_meta[p] for p in pipeline_pages}, recognizer)
            gt = cer_of(per_seed[seed], "2")
            true = sum(d["true_lines"] for d in detections)
            predicted = sum(d["predicted_lines"] for d in detections)
            matched = sum(d["matched_lines"] for d in detections)
            pipelines.append({"seed": seed, "e2e": e2e, "gt_cer": gt,
                              "precision": matched / predicted if predicted else 0.0,
                              "recall": matched / true if true else 0.0})
    first = per_seed[seeds[0]]
    metrics = flat_metrics(first)
    spread = {}
    if len(seeds) > 1:
        for key in first["metrics"]:
            values = [s["metrics"][key] for s in per_seed.values() if s["metrics"].get(key) is not None]
            if len(values) > 1:
                spread[key] = {"mean": round(statistics.mean(values), 3), "sd": round(statistics.stdev(values), 3)}
    model_name = RECOGNIZER_NAMES[recognizer]
    summary = {k: v for k, v in first.items() if k != "per_page"}
    summary.update({"recognizer": recognizer, "scheme": scheme, "seeds": seeds, "spread": spread,
                    "folds": [{"name": f["name"], "writers": f["writers"], "train": len(f["train"]),
                               "val": len(f["val"]), "test": len(f["test"])} for f in folds],
                    "per_seed": {s: flat_metrics(v) for s, v in per_seed.items()},
                    "peak_memory_mib": max(memory) if memory else None})
    if pipelines:
        p = pipelines[0]
        summary["pipeline"] = {"e2e_cer": p["e2e"]["metrics"]["cer"], "e2e_ci": p["e2e"]["ci"]["cer"],
                               "gt_cer": p["gt_cer"], "precision": p["precision"], "recall": p["recall"],
                               "names": p["e2e"]["names"]}
    figures = out / "figures"
    try:
        from mlops.eval_charts import summary_figures
        from mlops.train import training_figures
        if curves:
            training_figures("htr", curves, figures)
        pipes = ({model_name: {"gt_cer": summary["pipeline"]["gt_cer"], "e2e_cer": summary["pipeline"]["e2e_cer"],
                               "precision": summary["pipeline"]["precision"],
                               "recall": summary["pipeline"]["recall"]}} if pipelines else None)
        summary_figures({model_name: first}, figures, pipes)
        writer_bars(first, figures)
    except Exception as e:   # a figure must never fail the experiment
        log(f"figures not made: {e}")
    wandb.init(project=w["eval_project"]["htr"], entity=w["entity"], job_type="crossval-summary", group=group,
                      name=f"{group}-summary", config={"recognizer": recognizer, "scheme": scheme, "seeds": seeds,
                                                       "folds": summary["folds"]})
    try:
        from mlops.eval_charts import log_details
        wandb.log({k: v for k, v in metrics.items()})
        log_details({"summary": first, "detection": None})
        if spread:
            wandb.log({"seed_spread": wandb.Table(columns=["metric", "mean", "sd"],
                                                  data=[[k, v["mean"], v["sd"]] for k, v in spread.items()])})
    finally:
        wandb.finish()
    experiment = db.one(conn, "INSERT INTO experiments (job_id, kind, name, summary, figures) VALUES "
                              "(%s, %s, %s, %s, %s) RETURNING id",
                        job["id"], "crossval" if scheme == "writers" else "split", name, db.json(summary),
                        str(figures))
    shutil.rmtree(path("data") / "work" / f'cv-{job["id"]}', ignore_errors=True)
    log(f"{name}: CER {metrics.get('cer')} % [{metrics.get('cer_low')}, {metrics.get('cer_high')}], "
        f"CER without spaces and punctuation {metrics.get('cer_lp')} %, names exact {metrics.get('names_exact')} %")
    return {"experiment_id": experiment["id"], "metrics": metrics}


def writer_bars(summary, folder):
    """CER per writer, with the number of pages on each bar (what to annotate)."""
    from mlops import charts
    writers = [w for w, v in summary["writers"].items() if v["cer"] is not None]
    if not writers:
        return
    values = [summary["writers"][w]["cer"] for w in writers]
    lp = [summary["writers"][w]["cer_lp"] or 0 for w in writers]
    fig = charts.grouped_bars("CER per writer (test pages of its fold)", writers,
                              {"CER": values, "CER without spaces and punctuation": lp}, "CER (%)")
    for i, w in enumerate(writers):
        fig.axes[0].annotate(f'{summary["writers"][w]["pages"]} p.', (i, 0), ha="center", va="bottom", fontsize=7,
                             xytext=(0, -24), textcoords="offset points")
    charts.save(fig, folder, "writers_cer")
    charts.caption(folder, "writers_cer", "CER per writer, with the number of test pages")


def model_file(conn, task, choice):
    """A model of the Train tab (its id) or "production" (MaDoc's) -> (path, name)."""
    if choice in (None, "", "production"):
        release.sync()
        return release.production(task)["path"], f"production {task}"
    model = db.one(conn, "SELECT * FROM models WHERE id = %s AND task = %s", int(choice), task)
    if not model:
        raise RuntimeError(f"no {task} model {choice}")
    return model["path"], f'{task} model {model["id"]}'


def pipeline_job(conn, job, log, should_stop, set_wandb_url):
    """The "pipeline" job: a segmentation model and a recognition model together on the page-2 test pages (as
    MaDoc reads them) -> {"experiment_id", "metrics"}: CER on the true lines, end-to-end CER (and what the
    segmentation adds), line detection, names, with intervals by writer."""
    import wandb
    from mlops.eval_charts import log_details, summary_figures
    args = job["args"] or {}
    seg, seg_name = model_file(conn, "blla", args.get("segmentation"))
    rec, rec_name = model_file(conn, "htr", args.get("recognition"))
    version = datasets.snapshot(conn)
    rows = [r for r in datasets.pages(conn, version["id"], "test") if r["page_type"] == "2"]
    if not rows:
        raise RuntimeError("no page 2 in the test pages")
    page_meta = meta(conn, rows)
    out = path("runs") / "experiments" / f'job-{job["id"]}'
    free_gpu(log)
    w = params()["wandb"]
    name = args.get("name") or f"{seg_name} + {rec_name}"
    run_ = wandb.init(project=w["eval_project"]["blla"], entity=w["entity"], job_type="pipeline",
                      name=f'pipeline-job{job["id"]}', config={"segmentation": seg_name, "recognition": rec_name,
                                                                "test_version": version["digest"], "pages": len(rows)})
    set_wandb_url(run_.url)
    try:
        log(f"--- {rec_name} on the true lines of {len(rows)} page-2 test pages")
        gt = score(recognize(rec, rows, out / "lines", log, should_stop), page_meta, "htr")
        log(f"--- {seg_name} + {rec_name} (end to end)")
        line_rows, detected = pipeline(seg, rec, rows, out / "pipeline", log, should_stop)
        e2e = score(line_rows, page_meta, "pipeline")
        detection = detection_totals(detected)
        metrics = {**flat_metrics(e2e, "e2e_"), **flat_metrics(gt, "gt_"),
                   "line_precision": round(detection["precision"], 4), "line_recall": round(detection["recall"], 4)}
        wandb.log(metrics)
        log_details({"summary": e2e, "detection": detection})
    finally:
        wandb.finish()
    figures = out / "figures"
    try:
        summary_figures({"end to end": e2e, "true lines": gt}, figures,
                        {name: {"gt_cer": gt["metrics"]["cer"], "e2e_cer": e2e["metrics"]["cer"],
                                "precision": detection["precision"], "recall": detection["recall"]}})
    except Exception as e:
        log(f"figures not made: {e}")
    summary = {k: v for k, v in e2e.items() if k != "per_page"}
    summary.update({"segmentation": seg_name, "recognition": rec_name, "detection": detection,
                    "true_lines": {k: v for k, v in gt.items() if k not in ("per_page", "confusions")}})
    experiment = db.one(conn, "INSERT INTO experiments (job_id, kind, name, summary, figures) VALUES "
                              "(%s, 'pipeline', %s, %s, %s) RETURNING id", job["id"], name, db.json(summary),
                        str(figures))
    log(f"end-to-end CER {metrics.get('e2e_cer')} % (true lines {metrics.get('gt_cer')} %), line precision "
        f"{metrics['line_precision']:.3f}, recall {metrics['line_recall']:.3f}")
    return {"experiment_id": experiment["id"], "metrics": metrics}