"""ketos, with its metrics written for TensorBoard as the training goes.

    MLOPS_TB_DIR=/runs/htr/job-3/tensorboard python -m mlops.ketos_logged -d cuda:0 train ...

kraken 7.0.2 accepts "ketos segtrain/train --logger tensorboard --log-dir DIR" but never passes them to its trainer
(KrakenTrainer is built without pl_logger), so nothing is written. This wrapper gives every KrakenTrainer a
TensorBoard logger in MLOPS_TB_DIR, logging every LOG_EVERY_N_STEPS steps, then runs the normal ketos command line.
mlops/tb_bridge.py reads the files and sends the values to W&B.

The logger writes numbers only (ScalarsOnly). With a logger, kraken's recognition training also writes the first
16 validation lines as images and texts, which (1) crashes in kraken 7.0.2 (it passes a tuple to add_text:
"'tuple' object has no attribute 'encode'") and (2) would copy pieces of the certificates into the run folder.
kraken writes them only when the logger's experiment has add_image: here it has not.

Segmentation (ketos segtrain): kraken 7.0.2 keeps the checkpoints and stops early by val_mean_iu, the pixel overlap
averaged over the classes. On this data it goes the wrong way: the baselines are a few pixels wide, so a model that
finds the lines better can overlap them less (job 66: line F1 0.957 -> 0.963 while mean IU 0.178 -> 0.174), and the
model kept was the first epoch. The wrapper makes both follow val_bl_f1, the F1 of the baseline detection (what
ketos segtest reports): the checkpoints are named checkpoint_<epoch>-<F1>.ckpt, so the best one by F1 is exported.
MLOPS_SEG_MONITOR=val_mean_iu gives kraken's own choice back.
"""

import os
import sys

LOG_EVERY_N_STEPS = 10
MEDIA = ("add_image", "add_images", "add_text", "add_figure", "add_video", "add_audio", "add_embedding")


class NoMedia:
    """A TensorBoard SummaryWriter without the methods that write images or texts (numbers only)."""

    def __init__(self, writer):
        self._writer = writer

    def __getattr__(self, name):
        if name in MEDIA:
            raise AttributeError(name)
        return getattr(self._writer, name)


def scalars_logger(log_dir):
    from lightning.pytorch.loggers import TensorBoardLogger

    class ScalarsOnly(TensorBoardLogger):
        @property
        def experiment(self):
            return NoMedia(super().experiment)

    return ScalarsOnly(log_dir)


def patch():
    log_dir = os.environ.get("MLOPS_TB_DIR")
    if not log_dir:
        return
    from kraken.train import utils

    original = utils.KrakenTrainer.__init__

    def init(self, *args, **kwargs):
        if not kwargs.get("pl_logger"):
            kwargs["pl_logger"] = scalars_logger(log_dir)
            kwargs["log_dir"] = log_dir
            kwargs.setdefault("log_every_n_steps", LOG_EVERY_N_STEPS)
        original(self, *args, **kwargs)

    utils.KrakenTrainer.__init__ = init


SEG_MONITOR_DEFAULT = "val_bl_f1"


def patch_segmentation_monitor(argv=None):
    """ketos segtrain: the best checkpoint and the early stopping by MLOPS_SEG_MONITOR (default val_bl_f1)."""
    argv = sys.argv if argv is None else argv
    monitor = os.environ.get("MLOPS_SEG_MONITOR", SEG_MONITOR_DEFAULT)
    if "segtrain" not in argv or monitor in ("", "val_mean_iu"):
        return None
    from lightning.pytorch import callbacks
    from kraken.train import blla

    checkpoint_class, stopping_class = callbacks.ModelCheckpoint, blla.EarlyStopping

    class Checkpoint(checkpoint_class):
        def __init__(self, *args, **kwargs):
            if kwargs.get("monitor") == "val_metric":     # kraken's choice: the mean IU
                kwargs["monitor"] = monitor
                if kwargs.get("filename"):
                    kwargs["filename"] = kwargs["filename"].replace("val_metric", monitor)
            super().__init__(*args, **kwargs)

    class Stopping(stopping_class):
        def __init__(self, *args, **kwargs):
            if kwargs.get("monitor") == "val_mean_iu":
                kwargs["monitor"] = monitor
            super().__init__(*args, **kwargs)

    # ketos segtrain imports ModelCheckpoint when it runs (from lightning.pytorch.callbacks), blla imported
    # EarlyStopping when it was loaded: each is replaced where kraken looks it up
    callbacks.ModelCheckpoint, blla.EarlyStopping = Checkpoint, Stopping
    return monitor


def main():
    patch()
    patch_segmentation_monitor()
    from kraken.ketos import cli
    sys.argv[0] = "ketos"
    cli()


if __name__ == "__main__":
    main()