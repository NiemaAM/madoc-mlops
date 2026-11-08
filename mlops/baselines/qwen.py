"""Qwen2.5-VL-3B-Instruct, fine-tuned with LoRA to transcribe one line image, run on this machine.

The model sees the line image and an instruction, and is trained to answer with the transcription in logical
order (the order MaDoc reads). LoRA adapters (rank 16) on the attention projections of the language model; the
vision encoder is left as it is. bfloat16 weights, the adapters trained in float32, gradient checkpointing:
it fits the 16 GB card with batches of 4 lines (x 4 accumulated). At most `epochs` epochs, early stopping on the
validation CER (computed on up to `val_lines` validation lines, for speed); greedy decoding.

Licence: check the model card of the Qwen2.5-VL-3B weights before any use other than research (the 3B weights
are not under the Apache licence of the larger ones).
"""

import random
import time

from mlops.baselines.common import Epochs, cer, device_of, peak_memory_mib, seed_everything
from mlops.baselines.lines import to_rows

DEFAULTS = {"model": "Qwen/Qwen2.5-VL-3B-Instruct", "epochs": 3, "lag": 1, "batch_size": 4, "accumulate": 4,
            "learning_rate": 2e-4, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05, "max_new_tokens": 160,
            "max_pixels": 1280 * 28 * 28, "val_lines": 300, "train_lines": None}
PROMPT = ("Transcribe this handwritten line from a Moroccan birth certificate exactly as written: Arabic, with "
          "any Latin names, digits and punctuation. Answer with the transcription only.")


def messages():
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]


def load(options, device):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(options["model"], min_pixels=28 * 28 * 4,
                                              max_pixels=int(options["max_pixels"]))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(options["model"], dtype=torch.bfloat16,
                                                               attn_implementation="sdpa").to(device)
    model.config.use_cache = False          # training with gradient checkpointing; generate() asks for the cache
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=int(options["lora_rank"]), lora_alpha=int(options["lora_alpha"]),
                                             lora_dropout=float(options["lora_dropout"]), task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return processor, model


def image_of(line):
    from PIL import Image
    return Image.open(line["image"]).convert("RGB")


def training_batch(processor, batch):
    """Inputs with the labels of the answer only (the prompt and the padding are not learned)."""
    prompt = processor.apply_chat_template(messages(), add_generation_prompt=True)
    images = [image_of(line) for line in batch]
    texts = [prompt + line["text"] + "<|im_end|>" for line in batch]
    processor.tokenizer.padding_side = "right"
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    for k, image in enumerate(images):
        prompt_length = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
        labels[k, :prompt_length] = -100
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return inputs


def read(processor, model, lines, options, device):
    import torch
    prompt = processor.apply_chat_template(messages(), add_generation_prompt=True)
    processor.tokenizer.padding_side = "left"
    out = []
    model.eval()
    with torch.no_grad():
        size = max(1, int(options["batch_size"]) * 2)
        for start in range(0, len(lines), size):
            batch = lines[start:start + size]
            inputs = processor(text=[prompt] * len(batch), images=[image_of(line) for line in batch], padding=True,
                               return_tensors="pt").to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                generated = model.generate(**inputs, max_new_tokens=int(options["max_new_tokens"]), do_sample=False,
                                           use_cache=True)
            answers = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [(answer.strip().splitlines() or [""])[0].strip() for answer in answers]
    return out


def train_and_read(lines, settings, seed, out, log, should_stop):
    import torch
    from mlops.process import Cancelled
    options = {**DEFAULTS, **{k: v for k, v in settings.items() if v not in (None, "")}}
    seed_everything(seed)
    device = device_of(settings.get("device"))
    out.mkdir(parents=True, exist_ok=True)
    processor, model = load(options, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"qwen: {sum(p.numel() for p in trainable) / 1e6:.1f} M trainable parameters (LoRA)")
    optimizer = torch.optim.AdamW(trainable, lr=float(options["learning_rate"]), weight_decay=0.0)
    train = list(lines["train"])
    if options.get("train_lines"):
        train = random.Random(seed).sample(train, min(len(train), int(options["train_lines"])))
    val = lines["val"][:int(options["val_lines"])] if options.get("val_lines") else lines["val"]
    history = Epochs("qwen")
    best, best_epoch, best_state = float("inf"), 0, None
    accumulate, size = int(options["accumulate"]), int(options["batch_size"])
    for epoch in range(1, int(options["epochs"]) + 1):
        started, losses = time.time(), []
        model.train()
        random.shuffle(train)
        optimizer.zero_grad()
        for step, start in enumerate(range(0, len(train), size), 1):
            if should_stop():
                raise Cancelled("cancelled in the app")
            inputs = training_batch(processor, train[start:start + size]).to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                loss = model(**inputs).loss / accumulate
            loss.backward()
            losses.append(loss.item() * accumulate)
            if step % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        value = cer([line["text"] for line in val], read(processor, model, val, options, device))
        loss = sum(losses) / max(len(losses), 1)
        history.add(epoch, train_loss=loss, val_cer=value, val_accuracy=1 - min(value, 100) / 100,
                    seconds=round(time.time() - started, 1))
        log(f"qwen epoch {epoch}: loss {loss:.3f}, validation CER {value:.2f} % ({len(val)} lines)")
        if value < best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
            model.save_pretrained(out / "adapter")
        elif epoch - best_epoch >= int(options["lag"]):
            break
    if best_state:
        model.load_state_dict(best_state, strict=False)
    predictions = read(processor, model, lines["test"], options, device)
    log(f"qwen: best validation CER {best:.2f} % (epoch {best_epoch})")
    return {"rows": to_rows(lines["test"], predictions, logical=True), "epochs": history.epochs,
            "peak_memory_mib": peak_memory_mib()}