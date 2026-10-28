"""Training job: fine-tune the production model of a task on the current dataset, live in W&B.

The dataset is frozen into a version first (the training records which pages it saw). The production model of
MaDoc is the starting point. ketos runs until early stopping; its metrics go to W&B as they come; the best
checkpoint is exported to .mlmodel and registered in W&B as a new version of the task's model (alias
"candidate"), and in the database (models).
"""

import shutil
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

from mlops import datasets, db, release
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
    """The ketos command (the options of the previous src/blla_train.py and src/htr_train.py)."""
    device = ["-d", params()["device"], "-s", params()["seed"]]
    if task == "htr":
        cmd = [*ketos_program(), *device, "train", "--load", base, "--resize", settings["resize"], "--output", out,
               "--format-type", "page", "--training-data", train_list, "--evaluation-data", val_list,
               "--epochs", settings["epochs"], "--batch-size", settings["batch_size"], "--lag", settings["lag"]]
    else:
        cmd = [*ketos_program(), *device, "segtrain", "--load", base, "--resize", settings["resize"], "--output", out,
               "--format-type", "xml", "--training-data", train_list, "--evaluation-data", val_list,
               "--epochs", settings["epochs"], "--lag", settings.get("lag", 10)]
    # -q early: stop when the validation metric has not improved for --lag epochs. Needed for segtrain, whose
    # default is "fixed" (with --epochs -1 it never stops); explicit for train too.
    cmd += ["--quit", "early"]
    return [str(c) for c in cmd + list(settings.get("extra_args") or [])]


def kraken_version():
    try:
        return package_version("kraken")
    except PackageNotFoundError:
        return "unknown"


def train(conn, job, log, should_stop, set_wandb_url):
    """-> result {"model_id", "artifact", "dataset_version", "val"}."""
    import wandb

    task = job["task"]
    settings = params()[task]
    wandb_settings = params()["wandb"]
    version = datasets.snapshot(conn)
    train_rows = datasets.pages(conn, version["id"], "train")
    val_rows = datasets.pages(conn, version["id"], "val")
    if not train_rows or not val_rows:
        raise RuntimeError(f"version {version['digest']}: no train or no validation pages")
    out = path("runs") / task / f'job-{job["id"]}'
    out.mkdir(parents=True, exist_ok=True)
    train_paths = datasets.xml_paths(train_rows)
    args = job["args"] or {}
    use_augmented = args.get("use_augmented", settings.get("use_augmented")) and params()["augment"]["enabled"]
    augmented = augmented_paths(version, train_rows, log, should_stop) if use_augmented else []
    train_paths += augmented
    train_list = datasets.write_list(train_paths, out / "train.txt")
    val_list = datasets.write_list(datasets.xml_paths(val_rows), out / "val.txt")
    release.sync()
    base = release.production(task)
    if not base["path"].is_file():
        raise RuntimeError(f"no production model at {base['path']} in MaDoc's repository")
    free_gpu(log)
    run_ = wandb.init(
        project=wandb_settings["project"][task], entity=wandb_settings["entity"], job_type="train",
        name=f'{task}-job{job["id"]}',
        config={"task": task, "job": job["id"], "dataset_version": version["digest"], "pages": version["counts"],
                "train_files": len(train_paths), "augmented_pages": len(augmented), "base_model": base["card"].get("artifact", "MaDoc models/"),
                "settings": settings, "augment": params()["augment"], "kraken": kraken_version(), "gpu": gpu_name()})
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