"""Reading what ketos prints (the regular expressions of the previous src/*_train.py and src/*_eval.py)."""

import re

TRAIN_PATTERNS = {
    "blla": {
        "train_loss_step": r"train_loss_step:\s*([0-9.]+)",
        "train_loss_epoch": r"train_loss_epoch:\s*([0-9.]+)",
        "val_accuracy": r"val_accuracy:\s*([0-9.]+)",
        "val_mean_acc": r"val_mean_acc:\s*([0-9.]+)",
        "val_mean_iu": r"val_mean_iu:\s*([0-9.]+)",
        "val_freq_iu": r"val_freq_iu:\s*([0-9.]+)",
        "val_bl_f1": r"val_bl_f1:\s*([0-9.]+)",
    },
    "htr": {
        "train_loss_step": r"train_loss_step:\s*([0-9.]+)",
        "train_loss_epoch": r"train_loss_epoch:\s*([0-9.]+)",
        "val_accuracy": r"val_accuracy:\s*([0-9.]+)",
        "val_word_accuracy": r"val_word_accuracy:\s*([0-9.]+)",
        "early_stopping": r"early_stopping:\s*(\d+)/\d+",
    },
}
STAGE = re.compile(r"stage (\d+)/")
CHECKPOINT_SCORE = re.compile(r"-(\d+\.\d+)\.ckpt$")


def train_metrics(task, line):
    """The metrics of one line of `ketos train` / `ketos segtrain` ({} when there is none)."""
    out = {}
    for name, pattern in TRAIN_PATTERNS[task].items():
        match = re.search(pattern, line)
        if match:
            out[name] = float(match.group(1))
    return out


def stage(line):
    match = STAGE.search(line)
    return int(match.group(1)) if match else None


def best_checkpoint(folder):
    """(checkpoint, score) with the best validation score in its file name, or (None, None)."""
    scored = []
    for checkpoint in folder.glob("checkpoint_*.ckpt"):
        match = CHECKPOINT_SCORE.search(checkpoint.name)
        if match:
            scored.append((checkpoint, float(match.group(1))))
    return max(scored, key=lambda x: x[1]) if scored else (None, None)


# --- ketos test (recognition) ---------------------------------------------------------------------------
HTR_METRICS = {
    "characters": r"^(\d+)\s+Characters",
    "errors": r"^(\d+)\s+Errors$",
    "character_accuracy": r"^([0-9.]+)%\s+Character Accuracy$",
    "character_accuracy_ci": r"^([0-9.]+)%\s+Character Accuracy \(Case-insensitive\)",
    "word_accuracy": r"^([0-9.]+)%\s+Word Accuracy",
    "insertions": r"^(\d+)\s+Insertions",
    "deletions": r"^(\d+)\s+Deletions",
    "substitutions": r"^(\d+)\s+Substitutions",
}
SCRIPT_ROW = re.compile(r"^(\d+)\s+(\d+)\s+([0-9.]+)%\s+(\w+)\s*$", re.MULTILINE)
ERROR_ROW = re.compile(r"^\s*(\d+)\s+\{\s*(.*?)\s*\}\s*-\s*\{\s*(.*?)\s*\}", re.MULTILINE)


def htr_report(report):
    """`ketos test` output -> {"metrics": {...}, "scripts": [...], "errors": [(count, correct, generated)]}."""
    metrics = {}
    for key, pattern in HTR_METRICS.items():
        match = re.search(pattern, report, re.MULTILINE)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if "." in value else int(value)
    if metrics.get("characters"):
        metrics["cer"] = round(100 * metrics.get("errors", 0) / metrics["characters"], 3)
    scripts = [{"script": m.group(4), "count": int(m.group(1)), "missed": int(m.group(2)),
                "right": float(m.group(3))} for m in SCRIPT_ROW.finditer(report)]
    section = report.split("Correct-Generated", 1)
    errors = [(int(m.group(1)), m.group(2), m.group(3)) for m in ERROR_ROW.finditer(section[1])] if len(section) > 1 else []
    return {"metrics": metrics, "scripts": scripts, "errors": errors}


# --- ketos segtest (segmentation) ------------------------------------------------------------------------
OVERALL = re.compile(r"│\s*Overall\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)")
CLASS_ROW = re.compile(r"│\s*([A-Za-z0-9_]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)")
IOU_ROW = re.compile(r"│\s*(\w+)\s*│\s*([A-Za-z0-9_]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)")


def blla_report(report):
    """`ketos segtest` output -> {"metrics": {precision, recall, f1}, "classes": [...], "iou": [...]}."""
    metrics = {}
    overall = OVERALL.search(report)
    if overall:
        metrics.update(precision=float(overall.group(1)), recall=float(overall.group(2)),
                       f1=float(overall.group(3)))
    # ketos prints its tables more than once when its output is not a terminal: one row per class (the last)
    classes = {m.group(1): {"class": m.group(1), "precision": float(m.group(2)), "recall": float(m.group(3)),
                            "f1": float(m.group(4))} for m in CLASS_ROW.finditer(report) if m.group(1) != "Overall"}
    iou = {(m.group(1), m.group(2)): {"category": m.group(1), "class": m.group(2),
                                      "pixel_accuracy": float(m.group(3)), "iou": float(m.group(4))}
           for m in IOU_ROW.finditer(report)}
    return {"metrics": metrics, "classes": list(classes.values()), "iou": list(iou.values())}