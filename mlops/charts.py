"""The figures of the platform, drawn with matplotlib from numbers only, saved as PNG and PDF.

The same numbers go to W&B (tables and native charts, mlops/eval_charts.py and the training runs); the app shows
these figures and offers them for download (folder of a training, an evaluation or an experiment, files
<name>.png / <name>.pdf). Nothing here sees an image or a transcription: characters, counts, rates, anonymous
writer labels (W-1a2b3c) and the printed labels of the form.

Training: validation curves (full 0-100 % range, character and word accuracy, both CERs when computed, the
training loss on its own axis, the selected checkpoint marked, the folds of a cross-validation overlaid), CER per
script per epoch, segmentation curves, time per epoch and GPU memory.
Evaluation: the charts W&B had before (accuracy, error breakdown, CER by language; precision / recall / F1,
pixel accuracy, IoU) and error composition, end-to-end breakdown, names, letter-shape confusions, form row x
model and writer x model heatmaps.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})
COLORS = ["#1f5fa8", "#d9822b", "#3a9a5b", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#b7950b"]
CATEGORY_COLORS = {"letters": "#1f5fa8", "digits": "#8e44ad", "spaces": "#d9822b", "punctuation": "#7f8c8d"}
# Arabic letters by skeleton (same shape, different dots side by side): dot errors show as blocks
SKELETON = ("ا أ إ آ ء ؤ ئ ب ت ث ن ي ى ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ه ة و ڭ گ").split()
MISSING = "∅"


def save(fig, folder, name):
    """name.png and name.pdf in folder -> the PNG path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return folder / f"{name}.png"


def gallery(folder):
    """[(png path, caption)] of a figures folder (the app's gallery), in the order they were made."""
    folder = Path(folder) if folder else None
    if not folder or not folder.is_dir():
        return []
    captions = {}
    index = folder / "captions.json"
    if index.is_file():
        try:
            captions = json.loads(index.read_text(encoding="utf-8"))
        except ValueError:
            pass
    pngs = sorted(folder.glob("*.png"), key=lambda p: p.stat().st_mtime)
    return [(str(p), captions.get(p.stem, p.stem.replace("_", " "))) for p in pngs]


def caption(folder, name, text):
    index = Path(folder) / "captions.json"
    data = {}
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data[name] = text
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _value(key, value):
    """Accuracies, IoU and F1 are fractions (0..1): shown in %; the CERs are already in %."""
    if value is None or "cer" in key:
        return value
    return 100 * value if value <= 1.0 else value


# --- training -----------------------------------------------------------------------------------------------
def epochs_of(history):
    """history {step: {tag: value}} (TensorBoard) or [{epoch, ...}] -> [{"epoch": n, tag: value}] one per epoch,
    the last value of each tag in the epoch, epochs counted from 1."""
    if isinstance(history, list):
        return sorted(history, key=lambda r: r["epoch"])
    by_epoch = {}
    for step in sorted(history, key=float):
        values = history[step]
        epoch = values.get("epoch")
        if epoch is None:
            continue
        entry = by_epoch.setdefault(int(epoch) + 1, {"epoch": int(epoch) + 1})
        entry.update({k: v for k, v in values.items() if k != "epoch"})
    return [by_epoch[e] for e in sorted(by_epoch)]


def training_curves(runs, task, selected=None):
    """runs: [{"label", "epochs": [{"epoch", val_accuracy, ...}]}] (one per fold) -> figure.
    Full 0-100 % range; the selected checkpoint (epoch with the best selection metric) marked."""
    metrics = ({"val_accuracy": "character accuracy", "val_word_accuracy": "word accuracy",
                "val_cer": "CER", "val_cer_lp": "CER without spaces and punctuation"} if task == "htr" else
               {"val_mean_iu": "mean IoU", "val_bl_f1": "baseline F1", "val_mean_acc": "mean pixel accuracy",
                "val_freq_iu": "frequency-weighted IoU"})
    select = "val_accuracy" if task == "htr" else "val_mean_iu"
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    loss_ax = ax.twinx()
    loss_ax.spines["right"].set_visible(True)
    last = 1
    for r, run in enumerate(runs):
        epochs = run["epochs"]
        if not epochs:
            continue
        last = max(last, epochs[-1]["epoch"])
        alpha = 1.0 if len(runs) == 1 else 0.75
        for m, (key, name) in enumerate(metrics.items()):
            points = [(e["epoch"], _value(key, e[key])) for e in epochs if e.get(key) is not None]
            if points:
                ax.plot(*zip(*points), color=COLORS[m % len(COLORS)], lw=1.5, alpha=alpha,
                        ls="-" if r == 0 else (0, (4, 1 + r % 3)),
                        label=name if r == 0 else None)
        losses = [(e["epoch"], e.get("train_loss_epoch", e.get("train_loss"))) for e in epochs]
        losses = [p for p in losses if p[1] is not None]
        if losses:
            loss_ax.plot(*zip(*losses), color="0.55", lw=1, alpha=0.8, ls=":", label="training loss" if r == 0 else None)
        best = [e for e in epochs if e.get(select) is not None]
        if best:
            chosen = next((e for e in best if e["epoch"] == selected), None) if selected and len(runs) == 1 else None
            chosen = chosen or max(best, key=lambda e: (e[select], -e["epoch"]))
            ax.plot([chosen["epoch"]], [_value(select, chosen[select])], "o", color="black", ms=5, zorder=5,
                    label=(f"selected checkpoint (epoch {chosen['epoch']}, {_value(select, chosen[select]):.2f} %)"
                           if len(runs) == 1 else "selected checkpoints") if r == 0 else None)
            if len(runs) == 1:
                ax.axvline(chosen["epoch"], color="0.5", ls="--", lw=0.8)
        if len(runs) > 1:
            ax.annotate(run["label"], (epochs[-1]["epoch"], _value(select, epochs[-1].get(select, 0) or 0)),
                        fontsize=7, color="0.3", xytext=(3, 0), textcoords="offset points")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, last + 1)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_locator(MultipleLocator(10 if last > 20 else 5 if last > 8 else 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation score")
    loss_ax.set_ylabel("training loss")
    loss_ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    title = "Recognition" if task == "htr" else "Segmentation"
    ax.set_title(f"{title}: validation curves" + (f" ({len(runs)} folds)" if len(runs) > 1 else ""), fontsize=10)
    handles = ax.get_legend_handles_labels()
    extra = loss_ax.get_legend_handles_labels()
    ax.legend(handles[0] + extra[0], handles[1] + extra[1], loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=8, frameon=False)
    return fig


def script_curves(epochs):
    """Validation CER per script per epoch (detailed curves) -> figure or None."""
    keys = sorted({k for e in epochs for k in e if k.startswith("val_cer_script_")})
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, key in enumerate(keys):
        points = [(e["epoch"], e[key]) for e in epochs if e.get(key) is not None]
        ax.plot(*zip(*points), color=COLORS[i % len(COLORS)], lw=1.5, label=key.replace("val_cer_script_", ""))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation CER (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Validation CER per script", fontsize=10)
    return fig


def cost(epochs, peak_memory_mib=None):
    """Time of every epoch, and the peak GPU memory -> figure or None."""
    durations = [(e["epoch"], e["seconds"]) for e in epochs if e.get("seconds")]
    if not durations:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(*zip(*[(e, s / 60) for e, s in durations]), color=COLORS[0])
    ax.set_xlabel("epoch")
    ax.set_ylabel("minutes")
    total = sum(s for _, s in durations) / 60
    title = f"Time per epoch (total {total:.0f} min)"
    if peak_memory_mib:
        title += f" · peak GPU memory {peak_memory_mib / 1024:.1f} GiB (whole card)"
    ax.set_title(title, fontsize=10)
    return fig


# --- evaluation ---------------------------------------------------------------------------------------------
def bars(title, labels, values, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(labels, values, color=COLORS[:len(labels)])
    for x, v in zip(labels, values):
        ax.annotate(f"{v:.2f}" if not percent else f"{v:.1f} %", (x, v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(values + [1]) * 1.15)
    return fig


def grouped_bars(title, groups, series, ylabel, percent=True):
    """groups: x labels; series: {name: [value per group]} -> grouped bars."""
    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * len(groups) * max(1, len(series)) / 2 + 3), 3.6))
    width = 0.8 / max(1, len(series))
    for i, (name, values) in enumerate(series.items()):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, [v if v is not None else 0 for v in values], width, label=name, color=COLORS[i % len(COLORS)])
        for x, v in zip(xs, values):
            if v is not None:
                ax.annotate(f"{v:.1f}" + (" %" if percent else ""), (x, v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(bottom=0)
    return fig


def interval_bars(title, labels, values, lows, highs, ylabel, notes=None):
    """One bar per item with its interval (writers, models)."""
    fig, ax = plt.subplots(figsize=(max(5.5, 0.6 * len(labels) + 2), 3.6))
    errors = [[max(0, v - lo) if lo is not None else 0 for v, lo in zip(values, lows)],
              [max(0, hi - v) if hi is not None else 0 for v, hi in zip(values, highs)]]
    ax.bar(labels, values, yerr=errors, capsize=3, color=COLORS[0])
    for i, note in enumerate(notes or []):
        ax.annotate(note, (i, values[i]), ha="center", va="bottom", fontsize=7, xytext=(0, 8),
                    textcoords="offset points")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(bottom=0)
    return fig


def error_composition(models):
    """models: {name: {"composition": {operation: {category: n}}, "chars": n}} -> stacked bars, in % of the
    reference characters: for every model, substitutions / deletions (missed) / insertions (extra)."""
    from mlops.textmetrics import CATEGORIES, OPERATIONS
    names = list(models)
    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(names) + 2), 4.0))
    width, x_labels, x_pos = 0.26, [], []
    for m, name in enumerate(names):
        chars = max(models[name]["chars"], 1)
        for o, op in enumerate(OPERATIONS):
            x = m + (o - 1) * width
            bottom = 0.0
            for cat in CATEGORIES:
                value = 100 * models[name]["composition"][op][cat] / chars
                ax.bar(x, value, width * 0.95, bottom=bottom, color=CATEGORY_COLORS[cat],
                       label=cat if (m == 0 and o == 0) else None)
                bottom += value
            ax.annotate({"substitutions": "sub", "deletions": "missed", "insertions": "extra"}[op], (x, bottom),
                        ha="center", va="bottom", fontsize=7)
        x_labels.append(name)
        x_pos.append(m)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("errors (% of the reference characters)")
    ax.set_title("Error composition: what the errors are", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return fig


def e2e_breakdown(models):
    """models: {name: {"gt_cer", "e2e_cer", "precision", "recall"}} (CER in %, P/R in 0..1) -> two panels:
    CER on the true lines + what segmentation adds = end-to-end CER; line detection precision and recall."""
    names = list(models)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    for i, name in enumerate(names):
        m = models[name]
        gt, e2e = m.get("gt_cer") or 0, m.get("e2e_cer") or 0
        left.bar(i, gt, 0.5, color=COLORS[0], label="CER on the true lines" if i == 0 else None)
        left.bar(i, max(e2e - gt, 0), 0.5, bottom=gt, color=COLORS[1],
                 label="added by the segmentation" if i == 0 else None)
        left.annotate(f"{e2e:.2f} %", (i, max(e2e, gt)), ha="center", va="bottom", fontsize=8)
    left.set_xticks(range(len(names)))
    left.set_xticklabels(names)
    left.set_ylabel("CER (%)")
    left.set_title("End-to-end CER", fontsize=10)
    left.legend(fontsize=8, frameon=False)
    width = 0.35
    for i, name in enumerate(names):
        m = models[name]
        for j, key in enumerate(("precision", "recall")):
            value = 100 * (m.get(key) or 0)
            right.bar(i + (j - 0.5) * width, value, width, color=COLORS[2 + j], label=key if i == 0 else None)
            right.annotate(f"{value:.1f}", (i + (j - 0.5) * width, value), ha="center", va="bottom", fontsize=7)
    right.set_xticks(range(len(names)))
    right.set_xticklabels(names)
    right.set_ylim(0, 105)
    right.set_ylabel("%")
    right.set_title("Line detection", fontsize=10)
    right.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def names_chart(models):
    """models: {model: {"<field>|<page type>": CER %}} -> grouped bars, one group per field and page."""
    from mlops.fields import FIELD_LABELS, NAME_FIELDS
    groups = [(f, p) for f in NAME_FIELDS for p in ("1", "2")]
    keep = [g for g in groups if any(models[m].get(f"{g[0]}|{g[1]}") is not None for m in models)]
    labels = [f"{FIELD_LABELS[f]}, page {p}" for f, p in keep]
    series = {m: [models[m].get(f"{f}|{p}") for f, p in keep] for m in models}
    return grouped_bars("Name CER per field and page", labels, series, "CER (%)")


def _char_order(chars):
    def key(c):
        if c == MISSING:
            return (3, 0)
        if c in SKELETON:
            return (0, SKELETON.index(c))
        if c.isalpha():
            return (1, ord(c))
        return (2, ord(c))
    return sorted(chars, key=key)


def confusion_heatmap(pairs, top=28, title="Character confusions (reference → prediction)"):
    """pairs: {(reference char or "", predicted char or ""): count} -> heatmap of the `top` most confused
    characters, Arabic letters ordered by skeleton; "∅" is a missed (column) or an extra (row) character."""
    import numpy as np
    from collections import Counter
    shown = lambda c: MISSING if c == "" else ("␣" if c == " " else c)   # noqa: E731
    involvement = Counter()
    for (r, p), n in pairs.items():
        involvement[shown(r)] += n
        involvement[shown(p)] += n
    chars = [c for c, _ in involvement.most_common() if c != MISSING][:top] + [MISSING]
    chars = _char_order(set(chars))
    index = {c: i for i, c in enumerate(chars)}
    matrix = np.zeros((len(chars), len(chars)))
    for (r, p), n in pairs.items():
        r, p = shown(r), shown(p)
        if r in index and p in index:
            matrix[index[r], index[p]] += n
    fig, ax = plt.subplots(figsize=(0.32 * len(chars) + 2.5, 0.32 * len(chars) + 1.8))
    image = ax.imshow(np.log1p(matrix), cmap="Blues")
    ax.set_xticks(range(len(chars)))
    ax.set_yticks(range(len(chars)))
    ax.set_xticklabels(chars, fontsize=9)
    ax.set_yticklabels(chars, fontsize=9)
    ax.set_xlabel("prediction (∅: missed)")
    ax.set_ylabel("reference (∅: extra)")
    for i in range(len(chars)):
        for j in range(len(chars)):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] > matrix.max() / 3 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.6, label="log(1 + count)")
    return fig


def heatmap(title, rows, cols, values, label="CER (%)"):
    """values: {(row, col): value} -> annotated heatmap (form row x model, writer x model)."""
    import numpy as np
    matrix = np.full((len(rows), len(cols)), np.nan)
    for (r, c), v in values.items():
        if r in rows and c in cols and v is not None:
            matrix[rows.index(r), cols.index(c)] = v
    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 3.5, 0.34 * len(rows) + 1.6))
    image = ax.imshow(matrix, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    top = np.nanmax(matrix) if np.isfinite(matrix).any() else 1
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > top / 2 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.8, label=label)
    return fig


# --- the charts W&B had before (eval_charts.py), as figures --------------------------------------------------
def htr_report_figures(report):
    """ketos test report -> {name: figure}: accuracy, error breakdown, CER by language."""
    metrics, out = report.get("metrics", {}), {}
    accuracy = [(label, metrics[key]) for key, label in (("character_accuracy", "Character Accuracy"),
                                                         ("word_accuracy", "Word Accuracy")) if key in metrics]
    if accuracy:
        out["accuracy_comparison"] = bars("Recognition Accuracy", [a for a, _ in accuracy], [v for _, v in accuracy],
                                          "Accuracy (%)", percent=True)
    # ketos names them the other way round: its "insertions" are characters missed, "deletions" extra ones
    edits = {"Missed (ketos: insertions)": metrics.get("insertions", 0),
             "Extra (ketos: deletions)": metrics.get("deletions", 0),
             "Substitutions": metrics.get("substitutions", 0)}
    total = sum(edits.values()) or 1
    out["error_types"] = bars("Character Error Breakdown", list(edits), [100 * v / total for v in edits.values()],
                              "Percentage", percent=True)
    scripts = [s for s in report.get("scripts", []) if s["count"]]
    if scripts:
        out["CER_by_language"] = bars("Character Error Rate by Language", [s["script"] for s in scripts],
                                      [100 * s["missed"] / s["count"] for s in scripts], "CER (%)", percent=True)
    return out


def blla_report_figures(report):
    out = {}
    names = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    rows = [(names[k], report["metrics"][k]) for k in names if k in report.get("metrics", {})]
    if rows:
        out["precision_recall_f1"] = bars("Segmentation Precision, Recall and F1", [a for a, _ in rows],
                                          [v for _, v in rows], "Score")
    if report.get("iou"):
        out["pixel_accuracy_plot"] = bars("Pixel Accuracy by Class", [r["class"] for r in report["iou"]],
                                          [r["pixel_accuracy"] for r in report["iou"]], "Pixel Accuracy")
        out["iou_plot"] = bars("Intersection over Union", [r["class"] for r in report["iou"]],
                               [r["iou"] for r in report["iou"]], "IoU")
    return out