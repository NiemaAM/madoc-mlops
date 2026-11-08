"""What the trained recognizers share: the character codec, line images as tensors, the CER of a validation set,
the GPU memory peak and a W&B curve per epoch (numbers only)."""

import json
import random
from pathlib import Path

from rapidfuzz.distance import Levenshtein


class Codec:
    """Characters <-> integers; the first `reserved` integers are special (CTC blank, padding, start, end)."""

    def __init__(self, texts, reserved=1):
        self.chars = sorted({c for t in texts for c in t})
        self.reserved = reserved
        self.index = {c: i + reserved for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars) + self.reserved

    def encode(self, text):
        return [self.index[c] for c in text if c in self.index]

    def decode(self, ids):
        return "".join(self.chars[i - self.reserved] for i in ids if self.reserved <= i < len(self))

    def save(self, file):
        Path(file).write_text(json.dumps({"chars": self.chars, "reserved": self.reserved}, ensure_ascii=False),
                              encoding="utf-8")


def gray_line(path, height, max_width=None, augment=False):
    """A line image scaled to `height` (proportions kept), ink = 1, paper = 0, as a tensor H x W."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    image = Image.open(path).convert("L")
    if augment:   # small changes of width and slant
        image = image.resize((max(1, int(image.width * random.uniform(0.9, 1.1))), image.height))
        image = image.rotate(random.uniform(-1.5, 1.5), expand=True, fillcolor=255)
    width = max(1, round(image.width * height / image.height))
    if max_width:
        width = min(width, max_width)
    image = image.resize((width, height), Image.BILINEAR)
    return torch.from_numpy(1.0 - np.asarray(ImageOps.autocontrast(image), dtype=np.float32) / 255.0)


def cer(truths, predictions):
    """CER (%) of predicted texts against true texts."""
    chars = sum(len(t) for t in truths)
    return 100.0 * sum(Levenshtein.distance(t, p) for t, p in zip(truths, predictions)) / max(chars, 1)


class Epochs:
    """The history of a training (the figures) and its curves in the current W&B run, if any."""

    def __init__(self, prefix):
        self.prefix, self.epochs = prefix, []

    def add(self, epoch, **values):
        entry = {"epoch": epoch, **{k: v for k, v in values.items() if v is not None}}
        self.epochs.append(entry)
        try:
            import wandb
            if getattr(wandb, "run", None) is not None:
                wandb.log({**{f"{self.prefix}/{k}": v for k, v in entry.items() if k != "epoch"},
                           f"{self.prefix}/epoch": epoch})
        except ImportError:
            pass
        return entry


def peak_memory_mib():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 20)
    except ImportError:
        pass
    return None


def device_of(setting=None):
    from mlops.config import params
    import torch
    device = setting or params().get("device") or "cpu"
    return device if device.startswith("cpu") or torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)