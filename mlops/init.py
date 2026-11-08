"""The recognizers that are not kraken, trained and tested on the same pages and folds, scored by the same code.

    crnn   a convolutional-recurrent network with CTC (Shi et al. 2017), trained from scratch on our lines;
    trocr  the TrOCR image encoder with a character-level decoder trained on our lines (the recipe HATFormer uses
           for Arabic; HATFormer's own code is not used: call it "TrOCR-style" in a paper);
    qwen   Qwen2.5-VL-3B-Instruct fine-tuned with LoRA to transcribe one line image.

They run in their own container (Dockerfile.baselines: the platform's image plus transformers, peft, accelerate),
as the "baselines" queue of the jobs; they read the same line images as kraken (cut with kraken's polygon
extraction, mlops/baselines/lines.py) and give back one row per true line, in logical order. Model weights are
downloaded once into the Hugging Face cache (HF_HOME); no image or transcription leaves the machine.
"""

import importlib


def run_recognizer(kind, fold, settings, seed, out, log, should_stop):
    """Train `kind` on fold["train"] (early stopping on fold["val"]), read fold["test"] -> {"rows", "epochs",
    "peak_memory_mib"}."""
    if kind not in ("crnn", "trocr", "qwen"):
        raise ValueError(kind)
    from mlops.baselines.lines import cut
    lines = {split: cut(fold[split], out / "lines" / split, log) for split in ("train", "val", "test")}
    module = importlib.import_module(f"mlops.baselines.{kind}")
    return module.train_and_read(lines, settings, seed, out / kind, log, should_stop)