"""Training job: fine-tune the production model of a task on the current dataset, live in W&B.

The dataset is frozen into a version first (the training records which pages it saw). The production model of
MaDoc is the starting point. ketos runs until early stopping; its metrics go to W&B as they come; the best
checkpoint is exported to .mlmodel and registered in W&B as a new version of the task's model (alias
"candidate"), and in the database (models).
"""

import shlex
import shutil
import sys
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


def ketos_command(task, settings, base, train_list, val_list, out):
    """The ketos command of a training, from its options (mlops/settings.py: TRAIN_FIELDS)."""
    o = settings
    cmd = [*ketos_program(), "-d", params()["device"], "-s", params()["seed"],
           "train" if task == "htr" else "segtrain", "--load", base, "--output", out,
           "--format-type", "page" if task == "htr" else "xml",
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
        raise RuntimeError(f"no model {relative} in MaDoc's repository")
    is_production = path_ == production["path"].resolve()
    return {"path": path_, "card": production["card"] if is_production else {}, "production": is_production,
            "relative": relative}


def kraken_version():
    try:
        return package_version("kraken")
    except PackageNotFoundError:
        return "unknown"


def train(conn, job, log, should_stop, set_wandb_url):
    """-> result {"model_id", "artifact", "dataset_version", "val"}."""
    import wandb

    task = job["task"]
    args = job["args"] or {}
    # the options chosen in the app when the job was queued (kept in the job), else the saved ones / params.yaml
    settings = args.get("settings") or app_settings.training(conn, task)
    log("options: " + ", ".join(f"{k}={v}" for k, v in settings.items() if v not in (None, "")))
    wandb_settings = params()["wandb"]
    version = datasets.snapshot(conn)
    train_rows = datasets.pages(conn, version["id"], "train")
    val_rows = datasets.pages(conn, version["id"], "val")
    if not train_rows or not val_rows:
        raise RuntimeError(f"version {version['digest']}: no train or no validation pages")
    leaking = datasets.leaking_groups(conn, version["id"])
    if leaking:
        log(f"WARNING: {len(leaking)} groups have pages in more than one split "
            f"({sum(g['pages'] for g in leaking)} pages): the validation and test scores will be too optimistic. "
            "Split the dataset again (Data tab, or python -m mlops.cli resplit) for a fair evaluation.")
    out = path("runs") / task / f'job-{job["id"]}'
    out.mkdir(parents=True, exist_ok=True)
    train_paths = datasets.xml_paths(train_rows)
    use_augmented = args.get("use_augmented", settings.get("use_augmented")) and params()["augment"]["enabled"]
    augmented = augmented_paths(version, train_rows, log, should_stop) if use_augmented else []
    train_paths += augmented
    train_list = datasets.write_list(train_paths, out / "train.txt")
    val_list = datasets.write_list(datasets.xml_paths(val_rows), out / "val.txt")
    app_settings.publish_models(conn, release.sync())
    base = base_model(task, settings.get("start_from"))
    log(f'starting from {base["relative"]}' + (" (production)" if base["production"] else ""))
    free_gpu(log)
    run_ = wandb.init(
        project=wandb_settings["project"][task], entity=wandb_settings["entity"], job_type="train",
        name=f'{task}-job{job["id"]}',
        config={"task": task, "job": job["id"], "dataset_version": version["digest"], "pages": version["counts"],
                "train_files": len(train_paths), "augmented_pages": len(augmented), "base_model": base["card"].get("artifact") or base["relative"],
                "settings": settings, "augment": params()["augment"], "kraken": kraken_version(), "gpu": gpu_name(),
                "leaking_groups": len(leaking)})
    set_wandb_url(run_.url)
    if base["card"].get("artifact"):
        try:   # lineage: this candidate comes from the production model
            run_.use_artifact(f'{wandb_settings["entity"]}/{wandb_settings["project"][task]}/{base["card"]["artifact"]}')
        except Exception as e:
            log(f"W&B lineage skipped: {e}")
    printed = {}   # the metrics of the progress bar of ketos, printed once per epoch

    def on_line(line):
        metrics = train_metrics(task, line)
        printed.update(metrics)
        number = stage(line)
        if metrics and number is not None and not bridge.sent:
            # no TensorBoard values (yet): the end-of-epoch numbers of the progress bar, so W&B has curves anyway
            wandb.log({**{f"epoch_end/{k}": v for k, v in metrics.items()}, "epoch_end/epoch": number})

    log_dir = out / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    bridge = TensorBoardBridge(log_dir, log=log).start()
    wandb.define_metric("epoch_end/epoch")
    wandb.define_metric("epoch_end/*", step_metric="epoch_end/epoch")
    try:
        try:
            # COLUMNS: the progress bar printed on one line (80 columns cut "val_bl_f1: 0.973" over two lines)
            run(ketos_command(task, settings, base["path"], train_list, val_list, out / "checkpoints"), log,
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
        model = db.one(conn, "INSERT INTO models (task, artifact, path, dataset_version_id, job_id, val_metrics) "
                             "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
                       task, name, str(model_path), version["id"], job["id"], db.json({**last, "best": score}))
        shutil.rmtree(out / "checkpoints", ignore_errors=True)   # large: only the exported model is kept
        log(f"model {model['id']} registered: {name}")
        return {"model_id": model["id"], "artifact": name, "dataset_version": version["digest"], "val": last}
    finally:
        wandb.finish()