"""The metrics of a ketos training, live in W&B.

ketos shows its metrics in a progress bar that is only drawn on a terminal: run by the worker (its output goes to a
pipe), the bar is printed once, at the very end, so its metrics cannot be followed. ketos can also write them for
TensorBoard (--logger tensorboard --log-dir DIR): every metric of PyTorch Lightning (train_loss, val_accuracy,
val_bl_f1, val_mean_iu, learning rate, epoch ...) with its step, as the training goes. This bridge reads those event
files every few seconds and sends the new values to the W&B run of the job.
"""

import threading
from pathlib import Path

STEP = "trainer/global_step"


class TensorBoardBridge:
    def __init__(self, log_dir, interval=10.0, log=print):
        self.log_dir, self.interval, self.log = Path(log_dir), interval, log
        self.sent = {}            # tag -> last step sent
        self.latest = {}          # tag -> last value (the summary of the training)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        import wandb
        wandb.define_metric(STEP)
        wandb.define_metric("*", step_metric=STEP)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        """Stop, after a last reading (the values of the last epoch)."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=60)
        self.poll()

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.poll()
            except Exception as e:   # a file being written: read again at the next turn
                self.log(f"TensorBoard bridge: {e}")

    def poll(self):
        """Send the values written since the last reading -> number of values sent."""
        import wandb
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        rows = {}   # step -> {tag: value}
        for events in sorted(self.log_dir.rglob("events.out.tfevents.*")):
            accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    if event.step > self.sent.get(tag, -1):
                        rows.setdefault(event.step, {})[tag] = event.value
        count = 0
        for step in sorted(rows):
            values = rows[step]
            wandb.log({**values, STEP: step})
            for tag, value in values.items():
                self.sent[tag] = max(self.sent.get(tag, -1), step)
                self.latest[tag] = value
            count += len(values)
        return count