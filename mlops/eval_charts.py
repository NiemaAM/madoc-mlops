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