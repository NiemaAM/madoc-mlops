"""ketos, with its metrics written for TensorBoard as the training goes.

    MLOPS_TB_DIR=/runs/htr/job-3/tensorboard python -m mlops.ketos_logged -d cuda:0 train ...

kraken 7.0.2 accepts "ketos segtrain/train --logger tensorboard --log-dir DIR" but never passes them to its trainer
(KrakenTrainer is built without pl_logger), so nothing is written. This wrapper gives every KrakenTrainer a
TensorBoard logger in MLOPS_TB_DIR, logging every LOG_EVERY_N_STEPS steps, then runs the normal ketos command line.
mlops/tb_bridge.py reads the files and sends the values to W&B.
"""

import os
import sys

LOG_EVERY_N_STEPS = 10


def patch():
    log_dir = os.environ.get("MLOPS_TB_DIR")
    if not log_dir:
        return
    from kraken.train import utils

    original = utils.KrakenTrainer.__init__

    def init(self, *args, **kwargs):
        if not kwargs.get("pl_logger"):
            kwargs["pl_logger"] = "tensorboard"
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