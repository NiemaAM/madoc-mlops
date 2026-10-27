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

from mlops import db, jobs
from mlops.config import SPLITS, TASKS, params, path
from mlops.resources import gpu_stats, host_stats

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
    rows = query("SELECT split, count(*) AS pages, sum(lines) AS lines, sum(characters) AS characters, "
                 "count(DISTINCT group_key) AS groups FROM pages GROUP BY split")
    by_split = {r["split"]: r for r in rows}
    return [[s, *(int((by_split.get(s) or {}).get(k) or 0) for k in ("pages", "groups", "lines", "characters"))]
            for s in SPLITS]


def batches_table():
    out = []
    for b in query("SELECT * FROM batches ORDER BY id DESC LIMIT 50"):
        report = b["report"] or {}
        added = report.get("added") or {}
        out.append([b["id"], f'{b["created_at"]:%Y-%m-%d %H:%M}', b["status"],
                    added.get("train", 0), added.get("val", 0), added.get("test", 0), report.get("duplicates", 0),
                    report.get("rejected", 0), ", ".join(f"{k}: {v}" for k, v in (report.get("reasons") or {}).items())])
    return out


def versions_table():
    return [[v["id"], v["digest"], f'{v["created_at"]:%Y-%m-%d %H:%M}',
             *((v["counts"] or {}).get(s, 0) for s in SPLITS)]
            for v in query("SELECT * FROM dataset_versions ORDER BY id DESC LIMIT 20")]


def on_upload(file):
    if not file:
        raise gr.Error("Choose the zip exported from eScriptorium (PageXML with images).")
    source = Path(file if isinstance(file, str) else file.name)
    if source.suffix.lower() != ".zip":
        raise gr.Error("A .zip file is expected.")
    if source.stat().st_size > params()["ingest"]["max_upload_mb"] * 2 ** 20:
        raise gr.Error(f'The zip is larger than {params()["ingest"]["max_upload_mb"]} MB.')
    with db.connection() as conn:
        batch = db.one(conn, "INSERT INTO batches (status) VALUES ('uploaded') RETURNING id")
        target = path("data") / "uploads" / f'batch-{batch["id"]}.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), target)
        job = jobs.enqueue(conn, "ingest", batch_id=batch["id"], upload=str(target))
    gr.Info(f'Batch {batch["id"]} received: ingestion queued (job {job["id"]}).')
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
    for key, value in sorted(e["candidate"].items()):
        if isinstance(value, (int, float)):
            lines.append(f"| {key} | {production.get(key, '—')} | {value} |")
    lines += ["", f'Per page ({c["pages"]} test pages, {c["metric"]}, positive = the candidate is better): '
                  f'mean gain **{c["gain"]:+.4f}**, {int(100 * c["confidence"])} % CI [{c["low"]:+.4f}, {c["high"]:+.4f}]']
    for subset, label in (("old", "pages the production model was evaluated on"), ("new", "pages added since")):
        if c.get(subset):
            lines.append(f'- {label}: {c[subset]["pages"]} pages, mean gain {c[subset]["gain"]:+.4f}')
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
    running = execute("SELECT count(*) AS n FROM jobs WHERE status = 'queued'")["n"]
    parts.append(f"{running} job(s) waiting in the queue")
    return "<div style='line-height:1.9'>" + "<br>".join(parts) + "</div>"


def build_app():
    with gr.Blocks(title="MaDoc MLOps") as app:
        gr.Markdown("## MaDoc MLOps: dataset, training and evaluation of the kraken models")
        with gr.Tab("Data"):
            with gr.Row():
                upload = gr.File(label="eScriptorium export (zip: PageXML + images)", file_types=[".zip"],
                                 type="filepath")
                ingest_button = gr.Button("Ingest", variant="primary", scale=0)
            gr.Markdown("**Dataset now** (pages are split by group: register and year)")
            dataset_view = gr.Dataframe(headers=["split", "pages", "groups", "lines", "characters"],
                                        value=dataset_table, interactive=False)
            gr.Markdown("**Batches** (one per uploaded zip)")
            batches_view = gr.Dataframe(headers=["batch", "uploaded", "status", "train", "val", "test",
                                                 "duplicates", "rejected", "reasons"],
                                        value=batches_table, interactive=False)
            gr.Markdown("**Dataset versions** (made when a job needs one: the exact pages of a training)")
            versions_view = gr.Dataframe(headers=["version", "digest", "created", *SPLITS], value=versions_table,
                                         interactive=False)
        with gr.Tab("Train"):
            gr.Markdown("A training fine-tunes MaDoc's production model on the current dataset (logged live in W&B).")
            with gr.Row():
                train_buttons = {task: gr.Button(f"Train {TASK_NAMES[task]}", variant="primary") for task in TASKS}
            models_view = gr.Dataframe(headers=["model", "task", "status", "W&B artifact", "dataset version",
                                                "validation", "created"], value=models_table, interactive=False)
        with gr.Tab("Evaluate"):
            gr.Markdown("The candidate and the production model on the same test pages (old + new), live in W&B.")
            with gr.Row():
                model_pick = gr.Dropdown(label="Candidate model", choices=model_choices(), value=None, scale=3)
                evaluate_button = gr.Button("Evaluate against production", variant="primary", scale=1)
            evaluations_view = gr.Dataframe(headers=["evaluation", "task", "model", "verdict", "gain [CI]", "pages",
                                                     "pull request", "created"], value=evaluations_table,
                                            interactive=False)
        with gr.Tab("Compare & release"):
            evaluation_pick = gr.Dropdown(label="Evaluation", choices=evaluation_choices(), value=None)
            details = gr.Markdown("Choose an evaluation.")
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

        # events
        ingest_button.click(on_upload, [upload], [upload, batches_view, jobs_view])
        def on_train(task):
            queue_job("train", task)
            return jobs_table()

        for task, button in train_buttons.items():
            button.click(lambda t=task: on_train(t), None, [jobs_view])

        def on_evaluate(model_id):
            if not model_id:
                raise gr.Error("Choose a candidate model.")
            model = execute("SELECT task FROM models WHERE id = %s", model_id)
            queue_job("evaluate", model["task"], model_id=int(model_id))
            return jobs_table()

        evaluate_button.click(on_evaluate, [model_pick], [jobs_view])
        evaluation_pick.change(evaluation_details, [evaluation_pick], [details])

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
                                   evaluations_table(), jobs_table(),
                                   gr.update(choices=model_choices()), gr.update(choices=evaluation_choices())],
                          None, [dataset_view, batches_view, versions_view, models_view, evaluations_view, jobs_view,
                                 model_pick, evaluation_pick])
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
