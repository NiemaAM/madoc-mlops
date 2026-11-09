"""The control app of the platform (Gradio): data, training, evaluation, release, jobs and the server.

    python -m mlops.ui          # http://0.0.0.0:7870, login from MLOPS_AUTH ("name:password", several with commas)

The app never trains anything itself: every button adds a job to the queue (PostgreSQL), the worker runs it.
The app can restart at any time without stopping a training.
"""

import html
import os
import shutil
from pathlib import Path

import gradio as gr

from mlops import augment, browse, datasets, db, jobs, settings as app_settings
from mlops.config import SPLITS, TASKS, params, path
from mlops.resources import gpu_stats, host_stats
from mlops.vocabulary import coverage_html, page_types, vocabulary, vocabulary_html

TASK_NAMES = {"blla": "segmentation (blla)", "htr": "recognition (HTR)"}
STATUS_ICON = {"queued": "⏳", "running": "▶", "done": "✓", "failed": "✗", "cancelled": "■"}


def query(sql, *args):
    with db.connection() as conn:
        return db.all_rows(conn, sql, *args)


def execute(sql, *args):
    with db.connection() as conn:
        return db.one(conn, sql, *args)


def short(value, width=60):
    text = "" if value is None else str(value)
    return text if len(text) <= width else text[:width - 1] + "…"


def fmt_metrics(metrics, keys=None):
    if not metrics:
        return ""
    items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float)) and (keys is None or k in keys)]
    return ", ".join(f"{k} {v:.4g}" for k, v in items)


# --- data ---------------------------------------------------------------------------------------------------
def dataset_table():
    from mlops.ingest import backfill_page_types
    with db.connection() as conn:
        backfill_page_types(conn)
        rows = db.all_rows(conn, "SELECT split, count(*) AS pages, sum(lines) AS lines, sum(characters) AS characters, "
                                 "count(DISTINCT group_key) AS groups, count(DISTINCT writer_key) AS writers, "
                                 "count(*) FILTER (WHERE writer_key IS NULL) AS no_writer, "
                                 "count(*) FILTER (WHERE held_out) AS held_out, "
                                 "count(*) FILTER (WHERE page_type = '1') AS p1, "
                                 "count(*) FILTER (WHERE page_type = '2') AS p2 FROM pages WHERE deleted_at IS NULL "
                                 "GROUP BY split")
    by_split = {r["split"]: r for r in rows}
    keys = ("pages", "p1", "p2", "writers", "no_writer", "held_out", "groups", "lines", "characters")
    return [[s, *(int((by_split.get(s) or {}).get(k) or 0) for k in keys)] for s in SPLITS]


DATASET_HEADERS = ["split", "pages", "page 1 (names)", "page 2 (details)", "writers", "pages without writer",
                   "held out", "register-years", "lines", "characters"]


# --- training options and split settings (mlops/settings.py) -----------------------------------------------
LABELS = {"start_from": "Start from", "epochs": "Epochs", "quit": "Stop", "lag": "Lag", "min_epochs": "Minimum epochs",
          "batch_size": "Batch size", "resize": "Resize (alphabet)", "learning_rate": "Learning rate",
          "schedule": "Learning-rate schedule", "cos_max": "Cosine epochs",
          "kraken_augment": "kraken's line augmentation (--augment)",
          "use_augmented": "Add the augmented train pages (Data tab)", "workers": "Data loading workers",
          "precision": "Precision", "compile": "Compile the lines once (faster epochs)",
          "seed": "Seed", "detailed_curves": "Detailed validation curves",
          "extra_args": "Other ketos options"}


def training_values(task):
    with db.connection() as conn:
        values = app_settings.training(conn, task)
        models = app_settings.madoc_models(conn)
    keys = app_settings.TASK_FIELDS[task]
    out = []
    for key in keys:
        if key == "start_from":
            choices = sorted(set(models) | {values[key]})
            out.append(gr.update(choices=choices, value=values[key]))
        elif app_settings.TRAIN_FIELDS[key][0] is float:
            out.append("" if values[key] is None else f"{values[key]:g}")
        else:
            out.append(values[key])
    return out


def split_values():
    with db.connection() as conn:
        value = app_settings.split(conn)
    weights = value["split_weights"]
    types = weights.get("page_types") or {}
    return [value.get("split_by", "writer"), 100 * value["val_fraction"], 100 * value["test_fraction"],
            100 * value["split_tolerance"], weights.get("missing_character", 1000), types.get("1", 1000),
            types.get("2", 1000)]


def split_from_form(split_by, val, test, tolerance, missing, page1, page2):
    return app_settings.check_split({
        "split_by": split_by or "writer",
        "val_fraction": float(val or 0) / 100, "test_fraction": float(test or 0) / 100,
        "split_tolerance": float(tolerance or 0) / 100,
        "split_weights": {"missing_character": float(missing or 0),
                          "page_types": {"1": float(page1 or 0), "2": float(page2 or 0)}}})


def augmentations_md():
    """The augmentations on disk, and whether they belong to the dataset as it is now (after a new split or new
    pages, the dataset is a new version: its train pages are augmented again, the old folder is then deleted)."""
    found = augment.augmentations()
    if not found:
        return "No augmented pages yet."
    with db.connection() as conn:
        current = datasets.snapshot(conn)
    train = (current["counts"] or {}).get("train", 0)
    lines = []
    for a in found:
        if a["digest"] == current["digest"]:
            note = f"the current dataset version ({train} train pages): used by the trainings"
        else:
            note = ("an OLDER dataset version (before the last split or upload): not used; replaced and deleted "
                    "at the next augmentation or training")
        lines.append(f"- `{a['folder'].name}`: {a['pages']} pages ({a['copies']} copies, seed {a['seed']}): {note}")
    if not any(a["digest"] == current["digest"] for a in found):
        lines.append(f"- the current version `{current['digest']}` ({train} train pages) has no augmentation yet")
    return "\n".join(lines)


def vocabulary_view(scope="all"):
    with db.connection() as conn:
        by_split = vocabulary(conn)
        return coverage_html(by_split, page_types(conn)), vocabulary_html(by_split, scope or "all")


COPY_LABEL = "⧉"
# copies the vocabulary table (tab-separated: pastes as a table in a spreadsheet): character, code, name, total and
# count per split. The clipboard API needs https or localhost; otherwise the older execCommand("copy").
COPY_VOCABULARY_JS = """
() => {
  const text = document.querySelector('#voc-tsv')?.value || '';
  const button = [...document.querySelectorAll('button.custom-button')].find(b => b.title === '%s');
  const done = (ok) => { if (!button) return; button.title = ok ? 'Copied' : 'Not copied';
    button.style.outline = ok ? '2px solid #16a34a' : '2px solid #dc2626';
    setTimeout(() => { button.style.outline = ''; button.title = '%s'; }, 1200); };
  const fallback = () => { const area = document.createElement('textarea'); area.value = text;
    document.body.appendChild(area); area.select(); let ok = false; try { ok = document.execCommand('copy'); } catch (e) {}
    area.remove(); done(ok); };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => done(true), fallback);
  } else { fallback(); }
}
""" % (COPY_LABEL, COPY_LABEL)


def batches_table():
    out = []
    for b in query("SELECT b.*, (SELECT count(*) FROM pages p WHERE p.batch_id = b.id AND p.deleted_at IS NULL) "
                   "AS live FROM batches b ORDER BY id DESC LIMIT 50"):
        report = b["report"] or {}
        added = report.get("added") or {}
        status = b["status"] + (" · deleted" if b["deleted_at"] and b["status"] != "purged" else "")
        out.append([b["id"], f'{b["created_at"]:%Y-%m-%d %H:%M}', status, "yes" if b["held_out"] else "",
                    b["live"], added.get("train", 0), added.get("val", 0), added.get("test", 0),
                    report.get("duplicates", 0), report.get("rejected", 0),
                    ", ".join(f"{k}: {v}" for k, v in (report.get("reasons") or {}).items())])
    return out


BATCH_HEADERS = ["batch", "uploaded", "status", "held out", "pages now", "train at ingest", "val at ingest",
                 "test at ingest", "duplicates", "rejected", "reasons"]


def versions_table():
    return [[v["id"], v["digest"], f'{v["created_at"]:%Y-%m-%d %H:%M}',
             *((v["counts"] or {}).get(s, 0) for s in SPLITS)]
            for v in query("SELECT * FROM dataset_versions ORDER BY id DESC LIMIT 20")]


def on_upload(file, held_out=False):
    if not file:
        raise gr.Error("Choose the zip exported from eScriptorium (PageXML with images).")
    source = Path(file if isinstance(file, str) else file.name)
    if source.suffix.lower() != ".zip":
        raise gr.Error("A .zip file is expected.")
    if source.stat().st_size > params()["ingest"]["max_upload_mb"] * 2 ** 20:
        raise gr.Error(f'The zip is larger than {params()["ingest"]["max_upload_mb"]} MB.')
    with db.connection() as conn:
        batch = db.one(conn, "INSERT INTO batches (status, held_out) VALUES ('uploaded', %s) RETURNING id",
                       bool(held_out))
        target = path("data") / "uploads" / f'batch-{batch["id"]}.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), target)
        job = jobs.enqueue(conn, "ingest", batch_id=batch["id"], upload=str(target))
    gr.Info(f'Batch {batch["id"]} received{" (held-out set: always in test)" if held_out else ""}: '
            f'ingestion queued (job {job["id"]}).')
    return [None, batches_table(), jobs_table()]


# --- models, evaluations -----------------------------------------------------------------------------------
def models_table():
    return [[m["id"], m["task"], m["status"], m["artifact"], m["digest"] or "",
             fmt_metrics(m["val_metrics"], ("val_accuracy", "val_word_accuracy", "val_bl_f1", "val_mean_iu")),
             f'{m["created_at"]:%Y-%m-%d %H:%M}']
            for m in query("SELECT m.*, v.digest FROM models m LEFT JOIN dataset_versions v "
                           "ON v.id = m.dataset_version_id ORDER BY m.id DESC LIMIT 50")]


def model_choices():
    return [(f'#{m["id"]} {m["task"]} {m["artifact"]} ({m["status"]})', m["id"])
            for m in query("SELECT * FROM models ORDER BY id DESC LIMIT 50")]


def evaluations_table():
    return [[e["id"], e["task"], e["model_id"], "✓ better" if e["passed"] else "✗ not better",
             f'{e["comparison"]["gain"]:+.3f} [{e["comparison"]["low"]:+.3f}, {e["comparison"]["high"]:+.3f}]',
             e["comparison"]["pages"], e["pr_url"] or "", f'{e["created_at"]:%Y-%m-%d %H:%M}']
            for e in query("SELECT * FROM evaluations ORDER BY id DESC LIMIT 50")]


def evaluation_choices():
    return [(f'#{e["id"]} {e["task"]} model {e["model_id"]}: {"better" if e["passed"] else "not better"}', e["id"])
            for e in query("SELECT * FROM evaluations ORDER BY id DESC LIMIT 50")]


def with_ci(metrics, key):
    value = metrics.get(key)
    if value is None:
        return "—"
    low, high = metrics.get(f"{key}_low"), metrics.get(f"{key}_high")
    return f"{value:.4g} [{low:.4g}, {high:.4g}]" if low is not None and high is not None else f"{value:.4g}"


def metric_rows(candidate, production):
    """Table rows "| metric | production [CI] | candidate [CI] |" (the 95 % intervals by writer, or by page)."""
    keys = sorted(k for k, v in candidate.items() if isinstance(v, (int, float)) and not k.endswith(("_low", "_high")))
    return [f"| {k} | {with_ci(production, k)} | {with_ci(candidate, k)} |" for k in keys]


def evaluation_details(evaluation_id):
    if not evaluation_id:
        return "Choose an evaluation."
    e = execute("SELECT e.*, m.artifact FROM evaluations e JOIN models m ON m.id = e.model_id WHERE e.id = %s",
                evaluation_id)
    if not e:
        return "Unknown evaluation."
    c = e["comparison"]
    lines = [f'### Evaluation #{e["id"]}: {TASK_NAMES[e["task"]]}, candidate {e["artifact"]}',
             f'**{"✓ Better" if e["passed"] else "✗ Not better"}**: {html.escape(e["message"])}', "",
             "| Metric | Production | Candidate |", "|---|---|---|"]
    production = e["production"] or {}
    lines += metric_rows(e["candidate"], production)
    lines += ["", "Intervals: 95 %, " + ("by writer (the writers drawn again)" if c.get("unit") == "writer"
                                       else "by register-year or by page") + " — CER, WER in %; names in %.",
              "", f'Per page ({c["pages"]} test pages, {c["metric"]}, positive = the candidate is better): '
                  f'mean gain **{c["gain"]:+.4f}**, {int(100 * c["confidence"])} % CI [{c["low"]:+.4f}, {c["high"]:+.4f}]']
    for subset, label in (("old", "pages the production model was evaluated on"), ("new", "pages added since")):
        if c.get(subset):
            lines.append(f'- {label}: {c[subset]["pages"]} pages, mean gain {c[subset]["gain"]:+.4f}')
    if c.get("group_gains"):
        lines.append(f'- by {c.get("unit", "register-year")} ({c["groups"]} groups, interval {c.get("interval", "by page")}): '
                     + ", ".join(f'{v["gain"]:+.4f} on {v["pages"]} pages' for v in c["group_gains"].values()))
    excluded = {k: v for k, v in (c.get("excluded_pages") or {}).items() if v}
    if excluded:
        lines.append("- left out (seen in training): " + ", ".join(f"{v} by the {k}" for k, v in excluded.items()))
    runs = c.get("wandb") or {}
    links = [f"[{label}]({runs[key]})" for key, label in (("candidate", "candidate run"),
                                                          ("production", "production run")) if runs.get(key)]
    if links:
        lines.append("\nW&B (one run per model, select both to compare): " + " · ".join(links))
    if e["pr_url"]:
        lines.append(f'\nPull request: {e["pr_url"]}')
    return "\n".join(lines)


# --- jobs ---------------------------------------------------------------------------------------------------
def jobs_table():
    out = []
    for j in query("SELECT * FROM jobs ORDER BY id DESC LIMIT 50"):
        duration = ""
        if j["started_at"]:
            end = j["finished_at"]
            seconds = int(((end or execute("SELECT now() AS t")["t"]) - j["started_at"]).total_seconds())
            duration = f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"
        out.append([j["id"], j["kind"], j["task"] or "", f'{STATUS_ICON.get(j["status"], "")} {j["status"]}',
                     f'{j["created_at"]:%Y-%m-%d %H:%M}', duration, j["wandb_url"] or "", short(j["error"], 80)])
    return out


def job_log(job_id):
    if not job_id:
        return ""
    with db.connection() as conn:
        return jobs.tail(conn, int(job_id))


def queue_job(kind, task=None, **args):
    with db.connection() as conn:
        job = jobs.enqueue(conn, kind, task, **args)
    gr.Info(f'Job {job["id"]} queued: {kind} {TASK_NAMES.get(task, "")}')
    return job


# --- server -------------------------------------------------------------------------------------------------
def server_html():
    parts = []
    for gpu in gpu_stats():
        used = gpu["memory_used_mib"] / max(gpu["memory_total_mib"], 1)
        parts.append(f'<b>{html.escape(gpu["name"])}</b>: {gpu["utilization"]} % busy, memory '
                     f'{gpu["memory_used_mib"]:,} / {gpu["memory_total_mib"]:,} MiB ({used:.0%}), {gpu["temperature"]} °C')
    if not parts:
        parts.append("No GPU visible to the app (NVML).")
    host = host_stats()
    parts.append(f'CPU {host["cpu_percent"]} %, memory {host["memory_used_gib"]} / {host["memory_total_gib"]} GiB, '
                 f'data disk {host["data_disk_used_gib"]} / {host["data_disk_total_gib"]} GiB')
    for w in query("SELECT name, job_id, extract(epoch FROM now() - seen_at)::int AS age FROM workers ORDER BY seen_at DESC"):
        alive = w["age"] < 300
        parts.append(f'Worker {html.escape(w["name"])}: {"alive" if alive else "NOT SEEN"} ({w["age"]} s ago)'
                     + (f', on job {w["job_id"]}' if w["job_id"] else ", idle"))
    with db.connection() as conn:
        if app_settings.paused(conn):
            parts.append("<b>Worker paused</b> (a deployment is waiting for the running job): queued jobs start "
                         "after the restart")
    running = execute("SELECT count(*) AS n FROM jobs WHERE status = 'queued'")["n"]
    parts.append(f"{running} job(s) waiting in the queue")
    return "<div style='line-height:1.9'>" + "<br>".join(parts) + "</div>"


# --- writers, missing pages, batches, normalisation, preview (private: the app only) --------------------------
def writers_view():
    """The writer table (fewest pages first), the spellings that look alike, the aliases."""
    from mlops import writers
    with db.connection() as conn:
        writers.sync(conn)
        table = writers.table(conn)
        pairs = writers.variants(conn)
        spellings = writers.raw_spellings(conn)
        aliases = writers.aliases(conn)
    rows = [[w["writer"], w["label"], w["pages"], w["acts"], w["page1"], w["page2"], w["incomplete_acts"],
             ", ".join(f"{k} {v}" for k, v in sorted(w["splits"].items())), w["held_out"]] for w in table]
    if pairs:
        warning = ("⚠ **Spellings that may be the same officer** (merge them before splitting by writer, or the same "
                   "hand may be in train and test):\n" + "\n".join(f"- {a} ({na} pages) / {b} ({nb} pages)"
                                                                    for a, b, na, nb in pairs))
    else:
        warning = "No two writer names look alike."
    names = sorted({name for name, _, _ in spellings} | {c for _, c, _ in spellings})
    alias_rows = [[variant, canonical] for variant, canonical in sorted(aliases.items())]
    return rows, warning, gr.update(choices=names), gr.update(choices=names), alias_rows


WRITER_HEADERS = ["writer (private)", "label (W&B, figures)", "pages", "acts", "page 1", "page 2",
                  "acts missing a page", "splits", "held out"]


def on_alias(variant, canonical):
    from mlops import writers
    try:
        with db.connection() as conn:
            result = writers.add_alias(conn, variant, canonical)
    except ValueError as e:
        raise gr.Error(str(e)) from None
    gr.Info(f"{variant} → {canonical}: {result['writers']} writers now. Split again (Split settings) to use it.")
    return writers_view()


def on_unalias(variant):
    from mlops import writers
    with db.connection() as conn:
        writers.remove_alias(conn, variant)
    return writers_view()


def missing_view():
    from mlops.missing import summary_rows
    with db.connection() as conn:
        return summary_rows(conn)


def missing_download():
    """The file names of the missing pages, as a text file to download (never in a job log)."""
    import tempfile
    from mlops.missing import listing
    with db.connection() as conn:
        text = listing(conn)
    if not text:
        raise gr.Error("No act is missing a page.")
    target = Path(tempfile.mkdtemp(prefix="missing-")) / "pages_to_annotate.txt"
    target.write_text(text, encoding="utf-8")
    return str(target)


def normalize_preview():
    from mlops.ingest import normalize_dataset
    from mlops.normalize import describe, table
    with db.connection() as conn:
        result = normalize_dataset(conn, log=lambda *_: None, apply=False)
    rules = ", ".join(f"U+{ord(k):04X} → {'(deleted)' if v == '' else repr(v)}" for k, v in table().items())
    return (f"Rules (params.yaml ingest.normalize): {rules or 'none'}\n\n"
            f"**{result['pages']} pages would change**: {describe(result['replaced'])}")


def on_normalize():
    with db.connection() as conn:
        if db.one(conn, "SELECT 1 AS x FROM jobs WHERE status IN ('queued', 'running') AND kind IN ('ingest', "
                        "'normalize')"):
            raise gr.Error("An ingestion or a normalisation is already queued or running.")
    queue_job("normalize")
    return jobs_table()


def on_batch_action(batch_id, action, confirmation):
    from mlops import batches
    if not batch_id:
        raise gr.Error("Type the number of the batch.")
    try:
        with db.connection() as conn:
            if action == "delete":
                n = batches.delete(conn, int(batch_id))
                gr.Info(f"Batch {int(batch_id)}: {n} pages taken out of the dataset (restore brings them back).")
            elif action == "restore":
                n = batches.restore(conn, int(batch_id))
                gr.Info(f"Batch {int(batch_id)}: {n} pages back in the dataset.")
            else:
                n = batches.purge(conn, int(batch_id), confirmation or "")
                gr.Info(f"Batch {int(batch_id)}: {n} pages removed for good.")
    except ValueError as e:
        raise gr.Error(str(e)) from None
    return batches_table(), dataset_table()


def admins():
    names = [n.strip() for n in os.environ.get("MLOPS_ADMINS", "").split(",") if n.strip()]
    return names or [user for user, _ in logins()[:1]]


def preview_info():
    from mlops.preview import latest
    file = latest()
    if not file:
        return "No preview made yet."
    return f"Latest: `{file.name}` ({file.stat().st_size / 2 ** 20:.1f} MB)."


def on_preview_build():
    queue_job("preview")
    return jobs_table()


def on_preview_download(request: gr.Request):
    """The preview zip, for an admin only; every download is recorded (table downloads)."""
    import tempfile
    from mlops.preview import latest, record_download
    user = getattr(request, "username", None)
    if user not in admins():
        raise gr.Error("Only an admin of the app may download the preview (MLOPS_ADMINS).")
    file = latest()
    if not file:
        raise gr.Error("Make the preview first.")
    with db.connection() as conn:
        record_download(conn, user, file.name)
    target = Path(tempfile.mkdtemp(prefix="preview-")) / file.name
    shutil.copyfile(file, target)
    return str(target)


# --- figures and experiments ----------------------------------------------------------------------------------
def figures_of(folder):
    """(gallery items, files to download: every PNG and PDF of the folder). The figures are copied to a temporary
    folder first: the app serves them from there, never from the runs volume (which holds models and line images)."""
    import tempfile
    from mlops.charts import gallery
    if not folder or not Path(folder).is_dir():
        return [], []
    copy = Path(tempfile.mkdtemp(prefix="figures-"))
    files = []
    for file in sorted(Path(folder).iterdir()):
        if file.suffix in (".png", ".pdf"):
            shutil.copyfile(file, copy / file.name)
            files.append(str(copy / file.name))
    items = [(str(copy / Path(png).name), text) for png, text in gallery(folder)]
    return items, files


def model_figures(model_id):
    if not model_id:
        return [], []
    row = execute("SELECT figures FROM models WHERE id = %s", model_id)
    return figures_of(row and row["figures"])


def evaluation_figures(evaluation_id):
    if not evaluation_id:
        return [], []
    row = execute("SELECT figures FROM evaluations WHERE id = %s", evaluation_id)
    return figures_of(row and row["figures"])


def experiments_table():
    out = []
    for e in query("SELECT * FROM experiments ORDER BY id DESC LIMIT 50"):
        s = e["summary"] or {}
        m, ci = s.get("metrics") or {}, s.get("ci") or {}
        names = (s.get("names") or {}).get("all") or {}

        def cell(key):
            value = m.get(key)
            if value is None:
                return ""
            low, high = (ci.get(key) or [None, None])
            spread = (s.get("spread") or {}).get(key)
            text = f"{value:.2f}" + (f" [{low:.2f}, {high:.2f}]" if low is not None else "")
            return text + (f" (± {spread['sd']:.2f} over seeds)" if spread else "")
        e2e = (s.get("pipeline") or {}).get("e2e_cer")
        out.append([e["id"], e["kind"], e["name"], cell("cer"), cell("cer_lp"), cell("wer"),
                    f'{names.get("exact", "")}' if names else "", f'{names.get("normalized", "")}' if names else "",
                    "" if e2e is None else f"{e2e:.2f}", s.get("writers_count", ""), len(s.get("folds") or []) or "",
                    f'{e["created_at"]:%Y-%m-%d %H:%M}'])
    return out


EXPERIMENT_HEADERS = ["experiment", "kind", "name", "CER % [CI]", "CER without spaces/punct. %", "WER %",
                      "names exact %", "names normalised %", "end-to-end CER %", "writers", "folds", "created"]


def experiment_choices():
    return [(f'#{e["id"]} {e["name"]}', e["id"]) for e in query("SELECT id, name FROM experiments ORDER BY id DESC "
                                                                  "LIMIT 50")]


def experiment_details(experiment_id):
    if not experiment_id:
        return "Choose an experiment.", [], []
    e = execute("SELECT * FROM experiments WHERE id = %s", experiment_id)
    if not e:
        return "Unknown experiment.", [], []
    s = e["summary"]
    lines = [f'### {html.escape(e["name"])} ({e["kind"]})',
             f'{s.get("pages", 0)} test pages, {s.get("lines", 0)} lines, {s.get("chars", 0)} characters; '
             f'intervals {s.get("interval", "")} ({s.get("writers_count", 0)} writers)', "",
             "| metric | value | 95 % interval |" + (" mean ± sd over seeds |" if s.get("spread") else ""),
             "|---|---|---|" + ("---|" if s.get("spread") else "")]
    for key, value in (s.get("metrics") or {}).items():
        low, high = (s.get("ci") or {}).get(key) or [None, None]
        spread = (s.get("spread") or {}).get(key)
        lines.append(f"| {key} | {value} | {'' if low is None else f'[{low}, {high}]'} |"
                     + (f" {spread['mean']} ± {spread['sd']} |" if spread else ("" if not s.get("spread") else " |")))
    names = s.get("names") or {}
    if names:
        lines += ["", "| names (field\\|page) | names | exact % | normalised % | CER % | label without name |",
                  "|---|---|---|---|---|---|"]
        lines += [f'| {k} | {v["names"]} | {v["exact"]} | {v["normalized"]} | {v["cer"]} | {v["label_without_name"]} |'
                  for k, v in names.items()]
    if s.get("folds"):
        lines += ["", "Folds: " + "; ".join(f'{f["name"]} ({", ".join(f["writers"])}): test {f["test"]}, '
                                           f'train {f["train"]}, val {f["val"]}' for f in s["folds"])]
    if s.get("pipeline") or s.get("detection"):
        p = s.get("pipeline") or {"e2e_cer": s["metrics"].get("cer"), **(s.get("detection") or {}),
                                  "gt_cer": (s.get("true_lines") or {}).get("metrics", {}).get("cer")}
        lines += ["", f'End to end (page 2): CER {p.get("e2e_cer")} % (true lines {p.get("gt_cer")} %), line '
                      f'precision {p.get("precision", 0):.3f}, recall {p.get("recall", 0):.3f}']
    items, files = figures_of(e["figures"])
    return "\n".join(lines), items, files


def model_options(task):
    return [("production (MaDoc)", "production")] + [
        (f'#{m["id"]} {m["artifact"]}', str(m["id"])) for m in query("SELECT * FROM models WHERE task = %s "
                                                                     "ORDER BY id DESC LIMIT 30", task)]


def on_crossval(recognizer, scheme, seeds, segmentation, options_text):
    import json as json_
    try:
        seed_list = [int(x) for x in str(seeds or params()["seed"]).replace(" ", "").split(",") if x]
    except ValueError:
        raise gr.Error("Seeds: whole numbers separated by commas (e.g. 1,2,3).") from None
    try:
        options = json_.loads(options_text) if (options_text or "").strip() else {}
    except ValueError as e:
        raise gr.Error(f"Options: not JSON ({e})") from None
    args = {"recognizer": recognizer, "scheme": scheme, "seeds": seed_list, "segmentation": bool(segmentation)}
    if recognizer == "kraken":
        with db.connection() as conn:
            args["settings"] = {**app_settings.training(conn, "htr"), **options}
            args["blla_settings"] = app_settings.training(conn, "blla")
    else:
        args["settings"] = options
    queue_job("crossval", None, **args)
    return jobs_table()


def on_pipeline(segmentation, recognition):
    queue_job("pipeline", None, segmentation=segmentation or "production", recognition=recognition or "production")
    return jobs_table()


def build_app():
    with gr.Blocks(title="MaDoc MLOps") as app:
        gr.Markdown("## MaDoc MLOps: dataset, training and evaluation of the kraken models")
        with gr.Tab("Data"):
            with gr.Row():
                upload = gr.File(label="eScriptorium export (zip: PageXML + images)", file_types=[".zip"],
                                 type="filepath")
                with gr.Column(scale=0, min_width=220):
                    held_out = gr.Checkbox(label="Held-out set", value=False,
                                           info="always in test: never trained on, augmented or moved by a split")
                    ingest_button = gr.Button("Ingest", variant="primary")
            gr.Markdown("**Dataset now** (split by writer: the officer named on page 2 of each act; see Split settings)")
            dataset_view = gr.Dataframe(headers=DATASET_HEADERS, value=dataset_table, interactive=False)
            with gr.Accordion("Vocabulary (every character of the transcriptions and its count)", open=True):
                with gr.Row(equal_height=True):
                    vocabulary_scope = gr.Radio(["all", *SPLITS], value="all", label="Pages", scale=3)
                    vocabulary_refresh = gr.Button("🔄 Refresh", scale=0)
                gr.Markdown("The spaces and invisible characters are listed first. Red: fewer than "
                            "`ui.rare_characters` occurrences (params.yaml), too rare to be learned well. Dashed: never "
                            "in the train pages, so the model cannot learn it. Hover a character for its Unicode name "
                            "and its count in each split.")
                coverage_box = gr.HTML()
                vocabulary_copy = gr.Button(COPY_LABEL, size="sm", render=False)
                vocabulary_box = gr.HTML(label="Characters", buttons=[vocabulary_copy], container=True, show_label=True)
            with gr.Accordion("Writers (the officer of each act: what to annotate, spellings to merge)", open=False):
                gr.Markdown("Read on page 2 (\"من طرفنا نحن … موظف جماعي\"); page 1 takes the writer of page 2 of the "
                            "same act. The names stay in this app: W&B and the figures only see the labels. The "
                            "fewest pages first.")
                writers_table = gr.Dataframe(headers=WRITER_HEADERS, interactive=False)
                writers_warning = gr.Markdown()
                with gr.Row():
                    alias_variant = gr.Dropdown(label="This spelling…", choices=[], allow_custom_value=True)
                    alias_canonical = gr.Dropdown(label="…is the same officer as", choices=[], allow_custom_value=True)
                    alias_button = gr.Button("Merge", scale=0)
                    unalias_button = gr.Button("Undo (the first spelling)", scale=0)
                aliases_table = gr.Dataframe(headers=["spelling", "same officer as"], interactive=False)
                writers_refresh = gr.Button("🔄 Refresh the writers")
            with gr.Accordion("Split settings (rerun the split)", open=False):
                gr.Markdown("The writers (or the register-years) are placed to balance the characters and the page "
                            "types; held-out pages stay in test, pages without a known writer stay in train. "
                            "**Preview** computes a split with these settings without changing anything; **Apply** "
                            "saves the settings (also used by the next ingests) and splits the dataset again. "
                            "Trainings already made keep their own dataset version; evaluate them again to compare on "
                            "the new test set.")
                with gr.Row():
                    split_by_choice = gr.Radio([("writer (writer-independent test)", "writer"),
                                                ("register and year", "group")], label="Split by", value="writer")
                    split_val = gr.Number(label="Validation (% of the characters)", minimum=1, maximum=50)
                    split_test = gr.Number(label="Test (% of the characters)", minimum=1, maximum=50)
                    split_tolerance = gr.Number(label="Tolerance (% of a split's share)", minimum=0, maximum=100,
                                                info="33: a 15 % split may hold 10 % to 20 %")
                with gr.Row():
                    split_missing = gr.Number(label="Weight: a character missing from a split", minimum=0,
                                              info="x3 when missing from train")
                    split_page1 = gr.Number(label="Weight: page 1 (names) share", minimum=0)
                    split_page2 = gr.Number(label="Weight: page 2 (details) share", minimum=0)
                with gr.Row():
                    split_preview = gr.Button("Preview")
                    split_apply = gr.Button("Apply: save and split again", variant="primary")
                    split_reset = gr.Button("Reset to params.yaml")
                split_result = gr.HTML()
            with gr.Accordion("Normalisation of the transcriptions", open=False):
                gr.Markdown("Invisible or look-alike characters replaced (params.yaml `ingest.normalize`): a no-break "
                            "space looks exactly like a space. **Preview** counts what would change; **Normalize** "
                            "rewrites the pages (a backup of the old files is kept in the data folder) and makes a "
                            "new dataset version.")
                with gr.Row():
                    normalize_preview_button = gr.Button("Preview")
                    normalize_button = gr.Button("Normalize the dataset", variant="primary")
                normalize_result = gr.Markdown()
            with gr.Accordion("Augmentation of the train pages", open=False):
                gr.Markdown("Noise, lighting, shadow, blur, scribbles, rotation and perspective, the lines of the "
                            "PageXML moved with the image (script/data_augmentation_xml.py). Train pages only: "
                            "val, test and the held-out set stay real. A training that uses augmented pages takes "
                            "the newest augmentation of its dataset version.")
                with gr.Row(equal_height=True):
                    augment_copies = gr.Number(label="Copies of each train page", precision=0, minimum=1,
                                               maximum=10, value=params()["augment"]["copies"], scale=1)
                    augment_seed = gr.Number(label="Seed", precision=0, value=params()["augment"]["seed"], scale=1)
                    augment_button = gr.Button("Augment the train pages", variant="primary", scale=1)
                augmentations_view = gr.Markdown(augmentations_md)
            with gr.Accordion("Missing pages (acts with only one of their two pages annotated)", open=False):
                missing_table = gr.Dataframe(headers=["register", "split", "complete acts", "missing page 1",
                                                      "missing page 2"], interactive=False)
                with gr.Row():
                    missing_refresh = gr.Button("🔄 Refresh")
                    missing_button = gr.Button("Download the list of the pages to annotate")
                missing_file = gr.File(label="Pages to annotate (original file names)", interactive=False)
            gr.Markdown("**Batches** (one per uploaded zip). Delete takes the pages out of the dataset (restore "
                        "brings them back); Purge removes a deleted batch for good, only when no model used it.")
            batches_view = gr.Dataframe(headers=BATCH_HEADERS, value=batches_table, interactive=False)
            with gr.Row(equal_height=True):
                batch_pick = gr.Number(label="Batch", precision=0, scale=1)
                batch_delete = gr.Button("Delete", variant="stop", scale=0)
                batch_restore = gr.Button("Restore", scale=0)
                batch_confirm = gr.Textbox(label="Purge: type PURGE <batch>", scale=2)
                batch_purge = gr.Button("Purge for good", variant="stop", scale=0)
            with gr.Accordion("Preview of the dataset as shuffled line images (contains personal data)", open=False):
                gr.Markdown("Every line cut out along its polygon, shuffled across the pages of its split and renamed "
                            "by split, with its transcription. **It is not anonymised**: names, dates and the "
                            "handwriting stay. Only an admin of the app (MLOPS_ADMINS) can download it, and every "
                            "download is recorded. The trainings keep using the PageXML pages.")
                with gr.Row():
                    preview_build = gr.Button("Make the preview zip")
                    preview_download = gr.Button("Download (admin)")
                preview_status = gr.Markdown(preview_info)
                preview_file = gr.File(label="Preview zip", interactive=False)
            gr.Markdown("**Dataset versions** (made when a job needs one: the exact pages of a training)")
            versions_view = gr.Dataframe(headers=["version", "digest", "created", *SPLITS], value=versions_table,
                                         interactive=False)
        with gr.Tab("Train"):
            gr.Markdown("A training fine-tunes a model of MaDoc's repository (its production model by default) on "
                        "the current dataset, logged live in W&B. The options start from params.yaml; what you "
                        "change here is saved for the next trainings (**Reset** goes back to params.yaml), and "
                        "every job keeps the exact options it ran with.")
            train_buttons, reset_buttons, fields = {}, {}, {}
            with gr.Row():
                for task in TASKS:
                    with gr.Column():
                        gr.Markdown(f"#### {TASK_NAMES[task]}")
                        fields[task] = {}
                        for key in app_settings.TASK_FIELDS[task]:
                            kind = app_settings.TRAIN_FIELDS[key][0]
                            info = app_settings.TRAIN_FIELDS[key][1]
                            info = None if info.lower() == LABELS[key].lower() else info
                            if key == "start_from":
                                component = gr.Dropdown(label=LABELS[key], info=info, choices=[],
                                                        allow_custom_value=True)
                            elif key in app_settings.CHOICES:
                                component = gr.Dropdown(app_settings.CHOICES[key], label=LABELS[key], info=info)
                            elif kind is bool:
                                component = gr.Checkbox(label=LABELS[key], info=info)
                            elif kind is int:
                                component = gr.Number(label=LABELS[key], info=info, precision=0)
                            elif kind is float:
                                component = gr.Textbox(label=LABELS[key], info=info, placeholder="e.g. 1e-4")
                            else:
                                component = gr.Textbox(label=LABELS[key], info=info,
                                                       placeholder="e.g. --warmup 200 --weight-decay 1e-5")
                            fields[task][key] = component
                        with gr.Row():
                            train_buttons[task] = gr.Button(f"Train {TASK_NAMES[task]}", variant="primary", scale=3)
                            reset_buttons[task] = gr.Button("Reset", scale=1)
            models_view = gr.Dataframe(headers=["model", "task", "status", "W&B artifact", "dataset version",
                                                "validation", "created"], value=models_table, interactive=False)
            with gr.Accordion("Figures of a training (PNG / PDF)", open=False):
                model_figure_pick = gr.Dropdown(label="Model", choices=model_choices(), value=None)
                model_gallery = gr.Gallery(label="Figures", columns=2, height=520)
                model_files = gr.File(label="Download", file_count="multiple", interactive=False)
        with gr.Tab("Browse"):
            with gr.Row(equal_height=True):
                browse_split = gr.Radio([*SPLITS, browse.AUGMENTED], value="train", label="Pages", scale=3)
                browse_type = gr.Dropdown(browse.type_choices(), value="all", label="Page type", scale=2)
                browse_lines = gr.Checkbox(label="Draw the lines", value=True, scale=1)
                browse_names = gr.Checkbox(label="Show original names", value=False, scale=1,
                                           info="off for screenshots")
            browse_search = gr.Textbox(label="Search an original name or an id (any split)",
                                       placeholder="e.g. 03010108_1_1_I_2004_12")
            browse_note = gr.Markdown()
            browse_list = gr.State([])
            with gr.Row(equal_height=True):
                browse_previous = gr.Button("◀", scale=0, min_width=60)
                browse_page = gr.Dropdown(label="Page", choices=[], value=None, scale=6)
                browse_next = gr.Button("▶", scale=0, min_width=60)
            browse_info = gr.Markdown()
            with gr.Row():
                browse_image = gr.Image(label="Image", type="pil", interactive=False, scale=3, height=760)
                browse_text = gr.HTML(label="Transcription")
            with gr.Accordion("PageXML", open=False):
                browse_xml = gr.Code(language="html", interactive=False, lines=30, max_lines=30)
        with gr.Tab("Evaluate"):
            gr.Markdown("The candidate and the production model on the same test pages (old + new), live in W&B. "
                        "Every number has its 95 % interval (by writer).")
            with gr.Row():
                model_pick = gr.Dropdown(label="Candidate model", choices=model_choices(), value=None, scale=3)
                evaluate_button = gr.Button("Evaluate against production", variant="primary", scale=1)
            evaluations_view = gr.Dataframe(headers=["evaluation", "task", "model", "verdict", "gain [CI]", "pages",
                                                     "pull request", "created"], value=evaluations_table,
                                            interactive=False)
            with gr.Accordion("Full pipeline: a segmentation model and a recognition model together (page 2)",
                              open=False):
                gr.Markdown("The segmentation finds the lines of every page-2 test image, the recognition reads them, "
                            "as MaDoc does: CER on the true lines, end-to-end CER (what the segmentation adds), line "
                            "detection, names. The result is in the Experiments tab.")
                with gr.Row():
                    pipeline_seg = gr.Dropdown(label="Segmentation", choices=model_options("blla"), value="production")
                    pipeline_rec = gr.Dropdown(label="Recognition", choices=model_options("htr"), value="production")
                    pipeline_button = gr.Button("Evaluate the pipeline", variant="primary", scale=0)
        with gr.Tab("Experiments"):
            gr.Markdown("A recognizer trained and tested on **writers it never saw** (one fold per writer, every "
                        "page tested once), or on the dataset's own split. Kraken (the HTR options of the Train tab), "
                        "a CRNN, a TrOCR-style model or Qwen2.5-VL-3B (LoRA): the same folds, the same line "
                        "images, scored by the same code. The other recognizers run in the baselines container.")
            with gr.Row():
                cv_recognizer = gr.Radio([("Kraken", "kraken"), ("CRNN", "crnn"), ("TrOCR-style", "trocr"),
                                          ("Qwen2.5-VL-3B", "qwen")], value="kraken", label="Recognizer")
                cv_scheme = gr.Radio([("cross-validation by writer", "writers"), ("the dataset's split", "split")],
                                     value="writers", label="Pages")
            with gr.Row():
                cv_seeds = gr.Textbox(label="Seeds", value=str(params()["seed"]),
                                      info="several (1,2,3): every fold trained once per seed, mean ± sd")
                cv_segmentation = gr.Checkbox(label="Kraken: also train the segmentation per fold (end to end, page 2)")
                cv_options = gr.Textbox(label="Options (JSON, optional)",
                                        placeholder='kraken: {"lag": 10}; crnn: {"epochs": 100}; qwen: {"epochs": 2}')
                cv_button = gr.Button("Run", variant="primary", scale=0)
            experiments_view = gr.Dataframe(headers=EXPERIMENT_HEADERS, value=experiments_table, interactive=False)
            experiment_pick = gr.Dropdown(label="Experiment", choices=experiment_choices(), value=None)
            experiment_text = gr.Markdown()
            experiment_gallery = gr.Gallery(label="Figures", columns=2, height=520)
            experiment_files = gr.File(label="Download (PNG / PDF)", file_count="multiple", interactive=False)
        with gr.Tab("Compare & release"):
            evaluation_pick = gr.Dropdown(label="Evaluation", choices=evaluation_choices(), value=None)
            details = gr.Markdown("Choose an evaluation.")
            evaluation_gallery = gr.Gallery(label="Figures", columns=2, height=520)
            evaluation_files = gr.File(label="Download (PNG / PDF)", file_count="multiple", interactive=False)
            with gr.Row():
                override = gr.Checkbox(label="Open anyway (the candidate did not pass the gate)", value=False)
                release_button = gr.Button("Open a pull request on MaDoc", variant="primary")
        with gr.Tab("Jobs"):
            jobs_view = gr.Dataframe(headers=["job", "kind", "task", "status", "created", "duration", "W&B",
                                              "error"], value=jobs_table, interactive=False)
            with gr.Row():
                job_pick = gr.Number(label="Job", precision=0, scale=1)
                cancel_button = gr.Button("Cancel this job", variant="stop", scale=0)
            log_view = gr.Textbox(label="Log (live)", lines=22, max_lines=22, autoscroll=True)
        with gr.Tab("Server"):
            server_view = gr.HTML(server_html)

        # events: data
        ingest_button.click(on_upload, [upload, held_out], [upload, batches_view, jobs_view])
        vocabulary_scope.change(vocabulary_view, [vocabulary_scope], [coverage_box, vocabulary_box])
        vocabulary_refresh.click(vocabulary_view, [vocabulary_scope], [coverage_box, vocabulary_box])
        app.load(vocabulary_view, [vocabulary_scope], [coverage_box, vocabulary_box])
        vocabulary_copy.click(None, None, None, js=COPY_VOCABULARY_JS)
        writer_outputs = [writers_table, writers_warning, alias_variant, alias_canonical, aliases_table]
        writers_refresh.click(writers_view, None, writer_outputs)
        app.load(writers_view, None, writer_outputs)
        alias_button.click(on_alias, [alias_variant, alias_canonical], writer_outputs)
        unalias_button.click(on_unalias, [alias_variant], writer_outputs)
        missing_refresh.click(missing_view, None, [missing_table])
        app.load(missing_view, None, [missing_table])
        missing_button.click(missing_download, None, [missing_file])
        normalize_preview_button.click(normalize_preview, None, [normalize_result])
        normalize_button.click(on_normalize, None, [jobs_view])
        batch_delete.click(lambda b: on_batch_action(b, "delete", ""), [batch_pick], [batches_view, dataset_view])
        batch_restore.click(lambda b: on_batch_action(b, "restore", ""), [batch_pick], [batches_view, dataset_view])
        batch_purge.click(lambda b, c: on_batch_action(b, "purge", c), [batch_pick, batch_confirm],
                          [batches_view, dataset_view])
        preview_build.click(on_preview_build, None, [jobs_view])
        preview_download.click(on_preview_download, None, [preview_file])

        def on_train(task, *values):
            keys = app_settings.TASK_FIELDS[task]
            try:
                options = app_settings.clean(task, dict(zip(keys, values)))
            except ValueError as e:
                raise gr.Error(str(e)) from None
            with db.connection() as conn:
                app_settings.save(conn, f"train.{task}", options)
            queue_job("train", task, settings=options, use_augmented=bool(options.get("use_augmented")))
            return jobs_table()

        def on_reset(task):
            with db.connection() as conn:
                app_settings.reset(conn, f"train.{task}")
            gr.Info(f"{TASK_NAMES[task]}: back to the options of params.yaml")
            return training_values(task)

        for task in TASKS:
            inputs = [fields[task][k] for k in app_settings.TASK_FIELDS[task]]
            train_buttons[task].click(lambda *v, t=task: on_train(t, *v), inputs, [jobs_view])
            reset_buttons[task].click(lambda t=task: on_reset(t), None, inputs)
            app.load(lambda t=task: training_values(t), None, inputs)
        model_figure_pick.change(model_figures, [model_figure_pick], [model_gallery, model_files])

        split_inputs = [split_by_choice, split_val, split_test, split_tolerance, split_missing, split_page1,
                        split_page2]

        def split_html(result):
            from mlops.vocabulary import coverage_html
            text = html.escape(result["text"].split("\n")[0])
            return f"<p><b>{text}</b></p>" + coverage_html(result["by_split"], result["types"])

        def on_split_preview(*values):
            from mlops.ingest import resplit
            try:
                split = split_from_form(*values)
            except ValueError as e:
                raise gr.Error(str(e)) from None
            with db.connection() as conn:
                return split_html(resplit(conn, log=lambda *_: None, split=split, apply=False))

        def on_split_apply(*values):
            from mlops.ingest import resplit
            try:
                split = split_from_form(*values)
            except ValueError as e:
                raise gr.Error(str(e)) from None
            with db.connection() as conn:
                if db.one(conn, "SELECT 1 AS x FROM jobs WHERE status IN ('queued', 'running') AND kind = 'ingest'"):
                    raise gr.Error("An ingest is queued or running: split again once it is done.")
                app_settings.save(conn, "split", split)
                result = resplit(conn, log=lambda *_: None, split=split, apply=True)
            gr.Info("Split applied: " + ", ".join(f"{s} {n} pages" for s, n in result["pages"].items()))
            return split_html(result), dataset_table()

        def on_split_reset():
            with db.connection() as conn:
                app_settings.reset(conn, "split")
            gr.Info("Split settings: back to params.yaml (press Apply to split again with them)")
            return split_values()

        split_preview.click(on_split_preview, split_inputs, [split_result])
        split_apply.click(on_split_apply, split_inputs, [split_result, dataset_view])
        split_reset.click(on_split_reset, None, split_inputs)
        app.load(split_values, None, split_inputs)

        def on_augment(copies, seed):
            queue_job("augment", None, copies=int(copies or 1), seed=int(seed or 0))
            return jobs_table()

        augment_button.click(on_augment, [augment_copies, augment_seed], [jobs_view])

        # browse
        def browse_options(split, kind, names, search):
            with db.connection() as conn:
                options, note = browse.choices(conn, split, kind or "all", bool(names), search or "")
            first = options[0][1] if options else None
            return gr.update(choices=options, value=first), options, note

        browse_inputs = [browse_split, browse_type, browse_names, browse_search]
        browse_outputs = [browse_page, browse_list, browse_note]
        for component in (browse_split, browse_type, browse_names):
            component.change(browse_options, browse_inputs, browse_outputs)
        browse_search.submit(browse_options, browse_inputs, browse_outputs)
        app.load(browse_options, browse_inputs, browse_outputs)
        shown = [browse_image, browse_text, browse_xml, browse_info]
        browse_page.change(browse.show, [browse_page, browse_lines, browse_names], shown)
        browse_lines.change(browse.show, [browse_page, browse_lines, browse_names], shown)
        browse_names.change(browse.show, [browse_page, browse_lines, browse_names], shown)
        browse_previous.click(lambda options, value: browse.step(options, value, -1), [browse_list, browse_page], [browse_page])
        browse_next.click(lambda options, value: browse.step(options, value, 1), [browse_list, browse_page], [browse_page])

        # evaluate, experiments, release
        def on_evaluate(model_id):
            if not model_id:
                raise gr.Error("Choose a candidate model.")
            model = execute("SELECT task FROM models WHERE id = %s", model_id)
            queue_job("evaluate", model["task"], model_id=int(model_id))
            return jobs_table()

        evaluate_button.click(on_evaluate, [model_pick], [jobs_view])
        pipeline_button.click(on_pipeline, [pipeline_seg, pipeline_rec], [jobs_view])
        cv_button.click(on_crossval, [cv_recognizer, cv_scheme, cv_seeds, cv_segmentation, cv_options], [jobs_view])
        experiment_pick.change(experiment_details, [experiment_pick],
                               [experiment_text, experiment_gallery, experiment_files])
        evaluation_pick.change(evaluation_details, [evaluation_pick], [details])
        evaluation_pick.change(evaluation_figures, [evaluation_pick], [evaluation_gallery, evaluation_files])

        def on_release(evaluation_id, anyway):
            if not evaluation_id:
                raise gr.Error("Choose an evaluation.")
            e = execute("SELECT task, passed, pr_url FROM evaluations WHERE id = %s", evaluation_id)
            if e["pr_url"]:
                raise gr.Error(f'A pull request is already open: {e["pr_url"]}')
            if not e["passed"] and not anyway:
                raise gr.Error("The candidate is not better: tick 'Open anyway' to open the pull request regardless.")
            queue_job("release", e["task"], evaluation_id=int(evaluation_id), override=bool(anyway))
            return jobs_table()

        release_button.click(on_release, [evaluation_pick, override], [jobs_view])

        def on_cancel(job_id):
            if not job_id:
                raise gr.Error("Type the number of the job.")
            with db.connection() as conn:
                jobs.cancel(conn, int(job_id))
            gr.Info(f"Job {int(job_id)}: cancel requested.")
            return jobs_table()

        cancel_button.click(on_cancel, [job_pick], [jobs_view])

        # live refresh: the log every 3 s, the tables every 10 s, the server every 5 s
        gr.Timer(3).tick(job_log, [job_pick], [log_view])
        gr.Timer(5).tick(server_html, None, [server_view])
        gr.Timer(10).tick(lambda: [dataset_table(), batches_table(), versions_table(), models_table(),
                                   evaluations_table(), jobs_table(), augmentations_md(), experiments_table(),
                                   preview_info(),
                                   gr.update(choices=model_choices()), gr.update(choices=evaluation_choices()),
                                   gr.update(choices=model_choices()), gr.update(choices=experiment_choices()),
                                   gr.update(choices=model_options("blla")), gr.update(choices=model_options("htr"))],
                          None, [dataset_view, batches_view, versions_view, models_view, evaluations_view, jobs_view,
                                 augmentations_view, experiments_view, preview_status,
                                 model_pick, evaluation_pick, model_figure_pick, experiment_pick,
                                 pipeline_seg, pipeline_rec])
    return app


def logins():
    pairs = []
    for item in os.environ.get("MLOPS_AUTH", "").split(","):
        user, _, password = item.strip().partition(":")
        if user and password:
            pairs.append((user, password))
    return pairs


def main():
    db.init_schema()
    auth = logins()
    if not auth:
        raise SystemExit("MLOPS_AUTH is not set (.env): the app always asks for a login")
    build_app().queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("MLOPS_UI_PORT", 7870)),
                               auth=auth, auth_message="MaDoc MLOps", max_file_size=f'{params()["ingest"]["max_upload_mb"]}mb')


if __name__ == "__main__":
    main()