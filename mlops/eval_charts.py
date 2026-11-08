"""What an evaluation run of ONE model logs to W&B, as the evaluation runs made before the platform
(src/blla_eval.py, src/htr_eval.py). The candidate and the production model get one run each, with the same keys
and the same charts: select both runs in W&B to compare them; the app compares them page by page (the gate).

blla: precision, recall, f1 (ketos segtest), mean_e2e_word_f1, mean_e2e_cer (segmentation + production HTR) ·
      charts Segmentation Precision, Recall and F1 · Pixel Accuracy by Class · Intersection over Union ·
      tables precision_recall_f1_table, iou_table
htr:  character_accuracy, word_accuracy, cer, insertions / deletions / substitutions (ketos test), mean_page_cer ·
      charts Recognition Accuracy · Character Error Breakdown · Character Error Rate by Language ·
      tables character_error_table, language_accuracy, metrics
both: table pages (the score of every test page, the one the gate compares)
"""


def bar(wandb, title, label, value, rows):
    table = wandb.Table(columns=[label, value], data=[[str(a), float(b)] for a, b in rows])
    return wandb.plot.bar(table, label, value, title=title)


def blla_charts(wandb, result):
    report = result["report"]
    charts = {}
    names = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    rows = [(names[k], report["metrics"][k]) for k in names if k in report["metrics"]]
    if rows:
        charts["precision_recall_f1"] = bar(wandb, "Segmentation Precision, Recall and F1", "Metric", "Score", rows)
    if report["iou"]:
        charts["pixel_accuracy_plot"] = bar(wandb, "Pixel Accuracy by Class", "Class", "Pixel Accuracy",
                                            [(r["class"], r["pixel_accuracy"]) for r in report["iou"]])
        charts["iou_plot"] = bar(wandb, "Intersection over Union", "Class", "IoU",
                                 [(r["class"], r["iou"]) for r in report["iou"]])
    charts["precision_recall_f1_table"] = wandb.Table(
        columns=["class", "precision", "recall", "f1"],
        data=[[c["class"], c["precision"], c["recall"], c["f1"]] for c in report["classes"]])
    charts["iou_table"] = wandb.Table(
        columns=["category", "class", "pixel_accuracy", "iou"],
        data=[[r["category"], r["class"], r["pixel_accuracy"], r["iou"]] for r in report["iou"]])
    return charts


def htr_charts(wandb, result):
    report = result["report"]
    metrics = report["metrics"]
    charts = {}
    accuracy = [(label, metrics[key]) for key, label in (("character_accuracy", "Character Accuracy"),
                                                         ("word_accuracy", "Word Accuracy")) if key in metrics]
    if accuracy:
        charts["accuracy_comparison"] = bar(wandb, "Recognition Accuracy", "Metric", "Accuracy (%)", accuracy)
    edits = {k: metrics.get(k, 0) for k in ("insertions", "deletions", "substitutions")}
    total = sum(edits.values()) or 1
    charts["error_types"] = bar(wandb, "Character Error Breakdown", "Error Type", "Percentage",
                                [(k.capitalize(), 100 * v / total) for k, v in edits.items()])
    scripts = [s for s in report.get("scripts", []) if s["count"]]
    if scripts:
        charts["CER_by_language"] = bar(wandb, "Character Error Rate by Language", "Language", "CER (%)",
                                        [(s["script"], 100 * s["missed"] / s["count"]) for s in scripts])
    charts["language_accuracy"] = wandb.Table(
        columns=["language", "count", "missed", "right_percent"],
        data=[[s["script"], s["count"], s["missed"], s["right"]] for s in report.get("scripts", [])])
    charts["character_error_table"] = wandb.Table(columns=["count", "correct", "generated"],
                                                  data=[list(e) for e in report.get("errors", [])[:300]])
    charts["metrics"] = wandb.Table(columns=["metric", "value"],
                                    data=[[k, float(v)] for k, v in result["metrics"].items()
                                          if isinstance(v, (int, float))])
    return charts


def log_model_run(task, result):
    """Log the evaluation of one model in the current W&B run -> the keys logged."""
    import wandb
    data = {k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))}
    data.update((blla_charts if task == "blla" else htr_charts)(wandb, result))
    columns = sorted(next(iter(result["pages"].values()), {}))
    data["pages"] = wandb.Table(columns=["page", *columns],
                                data=[[page, *(values[c] for c in columns)]
                                      for page, values in sorted(result["pages"].items())])
    wandb.log(data)
    return sorted(data)

# --- the new numbers: W&B tables and the figures of the app (mlops/charts.py) ------------------------------
def log_details(result):
    """The scores of mlops/scoring.py in the current W&B run: numbers, characters and anonymous labels only."""
    import wandb
    summary = result.get("summary")
    if not summary:
        return
    from mlops.textmetrics import CATEGORIES, OPERATIONS
    data = {
        "error_composition": wandb.Table(
            columns=["operation", *CATEGORIES],
            data=[[op, *(summary["composition"][op][c] for c in CATEGORIES)] for op in OPERATIONS]),
        "confusions": wandb.Table(columns=["reference", "prediction", "count"],
                                  data=[[r or "∅", p or "∅", n] for r, p, n in summary["confusions"]]),
        "names": wandb.Table(columns=["field|page", "names", "exact", "normalized", "cer", "label_without_name"],
                             data=[[k, v["names"], v["exact"], v["normalized"], v["cer"], v["label_without_name"]]
                                   for k, v in summary["names"].items()]),
        "form_rows": wandb.Table(columns=["row|page", "cer"], data=[[k, v] for k, v in summary["form_rows"].items()]),
        "writers": wandb.Table(columns=["writer", "pages", "chars", "cer", "cer_lp"],
                               data=[[w, v["pages"], v["chars"], v["cer"], v["cer_lp"]]
                                     for w, v in summary["writers"].items()]),
        "intervals": wandb.Table(columns=["metric", "value", "low", "high", "interval"],
                                 data=[[k, v, *(summary["ci"].get(k) or [None, None]), summary["interval"]]
                                       for k, v in summary["metrics"].items()]),
    }
    if result.get("detection"):
        data["line_detection"] = wandb.Table(columns=list(result["detection"]), data=[list(result["detection"].values())])
    wandb.log(data)


def summary_figures(summaries, folder, pipelines=None, prefix=""):
    """The figures of one or more models' summaries (mlops/scoring.py) in `folder` -> names saved.
    pipelines: {model: {"gt_cer", "e2e_cer", "precision", "recall"}} for the end-to-end breakdown."""
    from mlops import charts
    saved = []

    def keep(fig, name, text):
        if fig is not None:
            charts.save(fig, folder, prefix + name)
            charts.caption(folder, prefix + name, text)
            saved.append(prefix + name)

    keep(charts.error_composition({m: {"composition": s["composition"], "chars": s["chars"]}
                                   for m, s in summaries.items()}),
         "error_composition", "Error composition: substitutions, missed and extra characters, by kind of character")
    names = {m: {k: v["cer"] for k, v in s["names"].items() if k != "all"} for m, s in summaries.items()}
    if any(names.values()):
        keep(charts.names_chart(names), "names_cer", "Name CER per field (child's names), page 1 and page 2")
    for m, s in summaries.items():
        pairs = {(r, p): n for r, p, n in s["confusions"]}
        if pairs:
            keep(charts.confusion_heatmap(pairs, title=f"Character confusions, {m}"), f"confusions_{m}",
                 f"Letter-shape confusion heatmap ({m})")
    rows = sorted({k for s in summaries.values() for k in s["form_rows"]},
                  key=lambda k: (k.split("|")[1], k))
    if rows:
        labels = [f'{k.split("|")[0]} (p.{k.split("|")[1]})' for k in rows]
        values = {(labels[i], m): s["form_rows"].get(k) for i, k in enumerate(rows) for m, s in summaries.items()}
        keep(charts.heatmap("CER per form row", labels, list(summaries), values), "form_rows",
             "Form row x model CER heatmap")
    writers = sorted({w for s in summaries.values() for w in s["writers"]})
    if writers:
        values = {(w, m): s["writers"].get(w, {}).get("cer") for w in writers for m, s in summaries.items()}
        keep(charts.heatmap("CER per writer", writers, list(summaries), values), "writers",
             "Writer x model CER heatmap (writers anonymised)")
    if pipelines:
        keep(charts.e2e_breakdown(pipelines), "e2e_breakdown",
             "End-to-end breakdown (page 2): CER on the true lines, what segmentation adds; line detection")
    return saved


def evaluation_figures(task, results, folder, gt_cer=None):
    """The figures of an evaluation (candidate and production): the charts W&B always had, per model, and the
    new ones (summary_figures)."""
    from mlops import charts
    from mlops.scoring import cer_of
    for role, result in results.items():
        report_figures = (charts.htr_report_figures if task == "htr" else charts.blla_report_figures)(result["report"])
        for name, fig in report_figures.items():
            charts.save(fig, folder, f"{role}_{name}")
            charts.caption(folder, f"{role}_{name}", f"{name.replace('_', ' ')} ({role})")
    pipelines = None
    if task == "blla":
        pipelines = {role: {"gt_cer": gt_cer, "e2e_cer": cer_of(r["summary"], "2") or cer_of(r["summary"]),
                            "precision": (r.get("detection") or {}).get("precision"),
                            "recall": (r.get("detection") or {}).get("recall")} for role, r in results.items()}
    return summary_figures({role: r["summary"] for role, r in results.items()}, folder, pipelines)