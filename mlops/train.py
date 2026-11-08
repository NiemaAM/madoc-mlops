"""Training job: fine-tune the production model of a task on the current dataset, live in W&B.

The dataset is frozen into a version first (the training records which pages it saw). The production model of
MaDoc is the starting point. ketos runs until early stopping; its metrics go to W&B as they come; the best
checkpoint is exported to .mlmodel and registered in W&B as a new version of the task's model (alias
"candidate"), and in the database (models).
"""

import hashlib
import shlex
import shutil
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version as package_version

from mlops import datasets, db, release, settings as app_settings
from mlops.augment import augmented_paths
from mlops.config import params, path
from mlops.kraken_output import best_checkpoint, stage, train_metrics
from mlops.process import run
from mlops.resources import free_gpu, gpu_name
from mlops.tb_bridge import TensorBoardBridge


def ketos_program():
    """ketos with a TensorBoard logger (mlops/ketos_logged.py: kraken 7.0.2 ignores --logger); plain ketos when
    kraken is not installed (the tests, with their fake ketos)."""
    try:
        import kraken.train  # noqa: F401
        return [sys.executable, "-m", "mlops.ketos_logged"]
    except ImportError:
        return ["ketos"]


def ketos_globals(settings):
    """The options of ketos itself, written before its command ("ketos -d cuda:0 --workers 4 train ..."): they
    cannot go in extra_args, which come after the command."""
    seed = settings.get("seed")
    cmd = ["-d", params()["device"], "-s", params()["seed"] if seed is None else seed]
    if settings.get("workers") is not None:
        cmd += ["--workers", settings["workers"]]
    if settings.get("precision"):
        cmd += ["--precision", settings["precision"]]
    return cmd


def ketos_command(task, settings, base, train_list, val_list, out, binary=False):
    """The ketos command of a training, from its options (mlops/settings.py: TRAIN_FIELDS). binary: the lists
    name compiled datasets (ketos compile) instead of PageXML files."""
    o = settings
    cmd = [*ketos_program(), *ketos_globals(o),
           "train" if task == "htr" else "segtrain", "--load", base, "--output", out,
           "--format-type", "binary" if binary else "page" if task == "htr" else "xml",
           "--training-data", train_list, "--evaluation-data", val_list,
           "--resize", o.get("resize") or "union", "--epochs", -1 if o.get("epochs") is None else o["epochs"],
           # early: stop when the validation has not improved for `lag` epochs (segtrain's default is "fixed": with
           # --epochs -1 it would never stop)
           "--quit", o.get("quit") or "early"]
    optional = {"lag": "--lag", "min_epochs": "--min-epochs", "learning_rate": "--lrate", "schedule": "--schedule",
                "cos_max": "--cos-max"}
    if task == "htr":
        optional["batch_size"] = "--batch-size"
    for key, flag in optional.items():
        if o.get(key) is not None and o.get(key) != "":
            cmd += [flag, o[key]]
    if o.get("kraken_augment"):
        cmd.append("--augment")
    extra = o.get("extra_args") or []
    cmd += shlex.split(extra) if isinstance(extra, str) else list(extra)
    return [str(c) for c in cmd]


def compiled_dataset(paths, name, settings, log, should_stop):
    """The PageXML pages -> one binary dataset of their lines (ketos compile), kept in <data>/compiled/ and reused
    while the same files are trained on. Without it, ketos cuts every line out of its page image again at every
    epoch. -> the .arrow file."""
    stamp = hashlib.sha256()
    for file in sorted(paths):
        info = Path(file).stat()
        stamp.update(f"{file}:{info.st_size}:{info.st_mtime_ns}\n".encode())
    folder = path("data") / "compiled"
    target = folder / f"{name}-{stamp.hexdigest()[:16]}.arrow"
    if target.is_file():
        log(f"compiled {name} lines reused ({target.name})")
        return target
    folder.mkdir(parents=True, exist_ok=True)
    manifest = datasets.write_list(paths, folder / f"{target.stem}.txt")
    partial = target.with_suffix(".part")
    globals_ = ["-d", params()["device"]] + (["--workers", settings["workers"]] if settings.get("workers") else [])
    run([*ketos_program(), *globals_, "compile", "-f", "page", "-o", partial, "-F", manifest], log, should_stop)
    partial.rename(target)
    manifest.unlink(missing_ok=True)
    for old in folder.glob(f"{name}-*.arrow"):   # the compilations of older datasets (disk)
        if old != target:
            old.unlink(missing_ok=True)
    return target


def base_model(task, start_from):
    """The model to fine-tune: a .mlmodel of the MaDoc clone (the production model by default) -> {path, card,
    production}. The card (lineage, W&B) only belongs to the production model."""
    production = release.production(task)
    relative = start_from or params()["release"]["models"][task]
    folder = release.clone_dir().resolve()
    path_ = (folder / relative).resolve()
    if folder not in path_.parents or path_.suffix != ".mlmodel":
        raise RuntimeError(f"{relative}: not a .mlmodel of MaDoc's repository")
    if not path_.is_file():
        available = sorted(str(p.relative_to(folder)) for p in folder.rglob("*.mlmodel") if ".git" not in p.parts)
        raise RuntimeError(f"no model {relative} in MaDoc's repository (renamed or removed?): choose another one "
                           f"in the Train tab ('start from'). Models there now: {', '.join(available) or 'none'}")
    is_production = path_ == production["path"].resolve()
    return {"path": path_, "card": production["card"] if is_production else {}, "production": is_production,
            "relative": relative}


def kraken_version():
    try:
        return package_version("kraken")
    except PackageNotFoundError:
        return "unknown"


def training_inputs(task, settings, version, train_rows, val_rows, out, log, should_stop, augment_root=None):
    """The manifests of a training -> (train list, val list, number of augmented pages)."""
    train_paths = datasets.xml_paths(train_rows)
    use_augmented = settings.get("use_augmented") and params()["augment"]["enabled"]
    augmented = (augmented_paths(version, train_rows, log, should_stop, root=augment_root) if use_augmented else [])
    train_paths += augmented
    val_paths = datasets.xml_paths(val_rows)
    binary = task == "htr" and bool(settings.get("compile"))
    if binary:   # the manifests then name the compiled datasets
        train_paths = [str(compiled_dataset(train_paths, "train", settings, log, should_stop))]
        val_paths = [str(compiled_dataset(val_paths, "val", settings, log, should_stop))]
    return (datasets.write_list(train_paths, out / "train.txt"), datasets.write_list(val_paths, out / "val.txt"),
            len(augmented), binary)


class MemoryWatch:
    """The peak GPU memory used during a training (the whole card, sampled every 15 s)."""

    def __init__(self, every=15.0):
        import threading
        self.peak, self.every, self.stop_event = 0, every, threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        from mlops.resources import gpu_stats
        while not self.stop_event.is_set():
            try:
                stats = gpu_stats()
                if stats:
                    self.peak = max(self.peak, stats[0]["memory_used_mib"])
            except Exception:
                pass
            self.stop_event.wait(self.every)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()


def epoch_history(bridge, printed_epochs):
    """The values of the training, one entry per epoch (TensorBoard values, else the progress bar), with the
    time each epoch took -> [{"epoch", tag: value, "seconds"}]."""
    from mlops.charts import epochs_of
    epochs = epochs_of(bridge.history) if bridge.history else []
    if not epochs:
        epochs = [{"epoch": n + 1, **values} for n, values in sorted(printed_epochs.items())]
    walls = {}
    for values in bridge.history.values():
        if "epoch" in values and "wall_time" in values:
            walls[int(values["epoch"]) + 1] = max(walls.get(int(values["epoch"]) + 1, 0), values["wall_time"])
    previous = None
    for entry in epochs:
        wall = walls.get(entry["epoch"])
        if wall and previous:
            entry["seconds"] = round(wall - previous, 1)
        previous = wall or previous
    return epochs


def run_ketos(task, settings, base_path, train_list, val_list, out, log, should_stop, binary, wandb_prefix=""):
    """ketos train / segtrain with its metrics live in the current W&B run -> {"model_path", "score", "last",
    "epochs", "peak_memory_mib", "checkpoints"}. The checkpoints are kept (the caller deletes them)."""
    import wandb
    printed, printed_epochs = {}, {}   # the metrics of the progress bar of ketos, printed once per epoch

    def on_line(line):
        metrics = train_metrics(task, line)
        printed.update(metrics)
        number = stage(line)
        if metrics and number is not None:
            printed_epochs.setdefault(number, {}).update(metrics)
            if not bridge.sent:
                # no TensorBoard values (yet): the end-of-epoch numbers of the progress bar, so W&B has curves anyway
                wandb.log({**{f"{wandb_prefix}epoch_end/{k}": v for k, v in metrics.items()},
                           f"{wandb_prefix}epoch_end/epoch": number})

    log_dir = out / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    bridge = TensorBoardBridge(log_dir, log=log).start()
    wandb.define_metric(f"{wandb_prefix}epoch_end/epoch")
    wandb.define_metric(f"{wandb_prefix}epoch_end/*", step_metric=f"{wandb_prefix}epoch_end/epoch")
    with MemoryWatch() as memory:
        try:
            # COLUMNS: the progress bar printed on one line (80 columns cut "val_bl_f1: 0.973" over two lines)
            run(ketos_command(task, settings, base_path, train_list, val_list, out / "checkpoints", binary), log,
                should_stop, on_line, env={"COLUMNS": "400", "MLOPS_TB_DIR": str(log_dir)})
        finally:
            bridge.stop()
    last = {**printed, **bridge.latest}
    log(f"W&B: {len(bridge.sent)} metrics followed ({', '.join(sorted(bridge.sent)) or 'none'})")
    checkpoint, score = best_checkpoint(out / "checkpoints")
    if checkpoint is None:
        raise RuntimeError("ketos wrote no checkpoint")
    model_path = out / f"{task}.mlmodel"
    run(["ketos", "convert", "--weights-format", "coreml", "-o", model_path, checkpoint], log, should_stop)
    epochs = epoch_history(bridge, printed_epochs)
    return {"model_path": model_path, "score": score, "last": last, "epochs": epochs,
            "peak_memory_mib": memory.peak or None, "checkpoints": out / "checkpoints"}


def detailed_curves(epochs, checkpoints, val_rows, out, log, should_stop):
    """Every epoch's checkpoint reads the validation lines: CER with and without spaces and punctuation, and per
    script, added to `epochs` and logged to W&B (val_detail/...)."""
    import wandb
    from mlops.kraken_output import CHECKPOINT_SCORE
    from mlops.recognize import meta, recognize
    from mlops.scoring import score
    page_meta = None
    by_epoch = {}
    for checkpoint in sorted(checkpoints.glob("checkpoint_*.ckpt")):
        number = checkpoint.stem.split("_", 1)[1].split("-", 1)[0]
        if not number.isdigit() or not CHECKPOINT_SCORE.search(checkpoint.name):
            continue
        model = out / "detail" / f"epoch-{int(number) + 1}.mlmodel"
        model.parent.mkdir(parents=True, exist_ok=True)
        run(["ketos", "convert", "--weights-format", "coreml", "-o", model, checkpoint], log, should_stop)
        if page_meta is None:
            with db.connection() as conn:
                page_meta = meta(conn, val_rows)
        summary = score(recognize(model, val_rows, out / "detail" / f"epoch-{int(number) + 1}", log, should_stop),
                        page_meta)
        m = summary["metrics"]
        values = {"val_cer": m.get("cer"), "val_cer_lp": m.get("cer_lp"),
                  **{f"val_cer_script_{s}": m.get(f"cer_{s.lower()}") for s in ("Arabic", "Latin", "Common")}}
        by_epoch[int(number) + 1] = {k: v for k, v in values.items() if v is not None}
        model.unlink(missing_ok=True)
    wandb.define_metric("val_detail/epoch")
    wandb.define_metric("val_detail/*", step_metric="val_detail/epoch")
    for epoch, values in sorted(by_epoch.items()):
        wandb.log({**{f"val_detail/{k}": v for k, v in values.items()}, "val_detail/epoch": epoch})
    for entry in epochs:
        entry.update(by_epoch.get(entry["epoch"], {}))
    # the CERs on the same 0-100 scale as the accuracies: shown as 100 - CER would hide them; kept as CER
    return epochs


def training_figures(task, runs, folder, peak_memory=None):
    """The figures of a training (one run) or of the folds of a cross-validation (several) -> saved names."""
    from mlops import charts
    saved = []

    def keep(fig, name, text):
        if fig is not None:
            charts.save(fig, folder, name)
            charts.caption(folder, name, text)
            saved.append(name)

    keep(charts.training_curves(runs, task), "training_curves",
         "Validation curves (full range), the selected checkpoint marked" + (", folds overlaid" if len(runs) > 1 else ""))
    if len(runs) == 1:
        keep(charts.script_curves(runs[0]["epochs"]), "validation_cer_per_script", "Validation CER per script")
        keep(charts.cost(runs[0]["epochs"], peak_memory), "cost", "Time per epoch and peak GPU memory")
    return saved


def train(conn, job, log, should_stop, set_wandb_url):
    """-> result {"model_id", "artifact", "dataset_version", "val"}."""
    import json
    import wandb

    task = job["task"]
    args = job["args"] or {}
    # the options chosen in the app when the job was queued (kept in the job), else the saved ones / params.yaml
    settings = dict(args.get("settings") or app_settings.training(conn, task))
    if "use_augmented" in args:
        settings["use_augmented"] = args["use_augmented"]
    log("options: " + ", ".join(f"{k}={v}" for k, v in settings.items() if v not in (None, "")))
    wandb_settings = params()["wandb"]
    version = datasets.snapshot(conn)
    train_rows = datasets.pages(conn, version["id"], "train")
    val_rows = datasets.pages(conn, version["id"], "val")
    if not train_rows or not val_rows:
        raise RuntimeError(f"version {version['digest']}: no train or no validation pages")
    from mlops.evaluate import unit_of
    leaking = datasets.leaking_groups(conn, version["id"], unit_of(conn))
    if leaking:
        log(f"WARNING: {len(leaking)} writers or groups have pages in more than one split "
            f"({sum(g['pages'] for g in leaking)} pages): the validation and test scores will be too optimistic. "
            "Split the dataset again (Data tab, or python -m mlops.cli resplit) for a fair evaluation.")
    out = path("runs") / task / f'job-{job["id"]}'
    out.mkdir(parents=True, exist_ok=True)
    train_list, val_list, augmented, binary = training_inputs(task, settings, version, train_rows, val_rows, out,
                                                              log, should_stop)
    app_settings.publish_models(conn, release.sync())
    base = base_model(task, settings.get("start_from"))
    log(f'starting from {base["relative"]}' + (" (production)" if base["production"] else ""))
    free_gpu(log)
    run_ = wandb.init(
        project=wandb_settings["project"][task], entity=wandb_settings["entity"], job_type="train",
        name=f'{task}-job{job["id"]}',
        config={"task": task, "job": job["id"], "dataset_version": version["digest"], "pages": version["counts"],
                "train_files": len(train_rows) + augmented, "augmented_pages": augmented,
                "base_model": base["card"].get("artifact") or base["relative"],
                "settings": settings, "augment": params()["augment"], "kraken": kraken_version(), "gpu": gpu_name(),
                "leaking_groups": len(leaking)})
    set_wandb_url(run_.url)
    if base["card"].get("artifact"):
        try:   # lineage: this candidate comes from the production model
            run_.use_artifact(f'{wandb_settings["entity"]}/{wandb_settings["project"][task]}/{base["card"]["artifact"]}')
        except Exception as e:
            log(f"W&B lineage skipped: {e}")
    try:
        trained = run_ketos(task, settings, base["path"], train_list, val_list, out, log, should_stop, binary)
        last, score, model_path = trained["last"], trained["score"], trained["model_path"]
        epochs = trained["epochs"]
        if task == "htr" and settings.get("detailed_curves"):
            log("detailed curves: every checkpoint reads the validation lines")
            epochs = detailed_curves(epochs, trained["checkpoints"], val_rows, out, log, should_stop)
        if trained["peak_memory_mib"]:
            wandb.run.summary["peak_gpu_memory_mib"] = trained["peak_memory_mib"]
        (out / "history.json").write_text(json.dumps({"epochs": epochs, "peak_memory_mib": trained["peak_memory_mib"]}),
                                          encoding="utf-8")
        figures = out / "figures"
        try:
            training_figures(task, [{"label": f"job {job['id']}", "epochs": epochs}], figures,
                             trained["peak_memory_mib"])
        except Exception as e:   # a figure must never fail a training
            log(f"figures not made: {e}")
        artifact = wandb.Artifact(wandb_settings["artifact"][task], type=f"kraken-{task}-model",
                                  metadata={"task": task, "dataset_version": version["digest"],
                                            "pages": version["counts"], "best_checkpoint_score": score, **last})
        artifact.add_file(str(model_path), name=f"{task}.mlmodel")
        logged = run_.log_artifact(artifact, aliases=["candidate", f'data-{version["digest"]}'])
        try:
            logged.wait()
            name = f'{wandb_settings["artifact"][task]}:{logged.version}'
        except Exception as e:   # W&B offline
            log(f"W&B artifact version unknown ({e})")
            name = f'{wandb_settings["artifact"][task]}:job{job["id"]}'
        model = db.one(conn, "INSERT INTO models (task, artifact, path, dataset_version_id, job_id, val_metrics, "
                             "figures) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
                       task, name, str(model_path), version["id"], job["id"], db.json({**last, "best": score}),
                       str(figures))
        shutil.rmtree(out / "checkpoints", ignore_errors=True)   # large: only the exported model is kept
        log(f"model {model['id']} registered: {name}")
        return {"model_id": model["id"], "artifact": name, "dataset_version": version["digest"], "val": last}
    finally:
        wandb.finish()