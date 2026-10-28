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


def main():
    patch()
    from kraken.ketos import cli
    sys.argv[0] = "ketos"
    cli()


if __name__ == "__main__":
    main()