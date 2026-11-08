"""A TrOCR-style encoder-decoder: the pretrained TrOCR image encoder and a character-level transformer decoder
trained on our lines (TrOCR's English sub-word vocabulary would cut Arabic into bytes; a character vocabulary is
what HATFormer does for Arabic).

Settings (defaults here): init "microsoft/trocr-base-stage1" (only the encoder is kept), AdamW with learning rate
5e-5 for the encoder and 3e-4 for the decoder, linear warm-up of 500 steps, batches of 8 lines resized to the
encoder's input (384 x 384), fp16 on the GPU, at most 100 epochs, early stopping after `lag` epochs without a
better validation CER; greedy decoding. Visual order, as kraken.
"""

import random
import time

from mlops.baselines.common import Codec, Epochs, cer, device_of, peak_memory_mib, seed_everything
from mlops.baselines.lines import to_rows

DEFAULTS = {"init": "microsoft/trocr-base-stage1", "batch_size": 8, "epochs": 100, "lag": 10,
            "lr_encoder": 5e-5, "lr_decoder": 3e-4, "warmup": 500, "fp16": True}
PAD, BOS, EOS = 0, 1, 2


def build(codec_size, init, max_len):
    """init None: a small random encoder (the tests)."""
    from transformers import TrOCRConfig, TrOCRForCausalLM, ViTConfig, ViTModel, VisionEncoderDecoderModel
    if init:
        encoder = VisionEncoderDecoderModel.from_pretrained(init).encoder
        size = encoder.config.image_size
    else:
        size = 32
        encoder = ViTModel(ViTConfig(image_size=size, patch_size=16, hidden_size=32, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=64), add_pooling_layer=False)
    small = not init
    decoder = TrOCRForCausalLM(TrOCRConfig(
        vocab_size=codec_size, d_model=32 if small else 256, decoder_layers=1 if small else 6,
        decoder_attention_heads=2 if small else 8, decoder_ffn_dim=64 if small else 1024,
        cross_attention_hidden_size=encoder.config.hidden_size, max_position_embeddings=max_len + 2,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, decoder_start_token_id=BOS,
        use_learned_position_embeddings=True, layernorm_embedding=True))
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.config.decoder_start_token_id, model.config.pad_token_id = BOS, PAD
    model.config.eos_token_id, model.config.vocab_size = EOS, codec_size
    return model, size


def pixels(path, size, augment=False):
    import numpy as np
    import torch
    from PIL import Image
    image = Image.open(path).convert("RGB")
    if augment:
        image = image.rotate(random.uniform(-1.5, 1.5), expand=True, fillcolor=(255, 255, 255))
    array = np.asarray(image.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy((array - 0.5) / 0.5).permute(2, 0, 1)


def labels_of(texts, codec, max_len):
    import torch
    ids = [codec.encode(t)[:max_len] + [EOS] for t in texts]
    out = torch.full((len(ids), max(map(len, ids))), -100)
    for k, i in enumerate(ids):
        out[k, :len(i)] = torch.tensor(i)
    return out


def read(model, lines, codec, size, device, options, max_len):
    import torch
    model.eval()
    out = []
    fp16 = bool(options["fp16"]) and device.startswith("cuda")
    with torch.no_grad():
        for start in range(0, len(lines), options["batch_size"]):
            x = torch.stack([pixels(line["image"], size) for line in lines[start:start + options["batch_size"]]])
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=fp16):
                ids = model.generate(pixel_values=x.to(device), max_new_tokens=max_len + 1, num_beams=1,
                                     do_sample=False, decoder_start_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
            for sequence in ids.tolist():
                sequence = sequence[1:]
                out.append(codec.decode(sequence[:sequence.index(EOS)] if EOS in sequence else sequence))
    return out


def train_and_read(lines, settings, seed, out, log, should_stop):
    import torch
    from transformers import VisionEncoderDecoderModel
    from mlops.process import Cancelled
    options = {**DEFAULTS, **{k: v for k, v in settings.items() if v not in (None, "")}}
    seed_everything(seed)
    device = device_of(settings.get("device"))
    out.mkdir(parents=True, exist_ok=True)
    codec = Codec([line["display"] for line in lines["train"]], reserved=3)
    codec.save(out / "codec.json")
    max_len = min(256, max(len(line["display"]) for split in lines.values() for line in split))
    init = None if str(options["init"]).lower() in ("none", "random") else options["init"]   # "none": a test
    model, size = build(len(codec), init, max_len)
    model.to(device)
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": float(options["lr_encoder"])},
                                   {"params": model.decoder.parameters(), "lr": float(options["lr_decoder"])}],
                                  weight_decay=1e-4)
    warmup = int(options["warmup"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup))
    fp16 = bool(options["fp16"]) and device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=fp16)
    history = Epochs("trocr")
    best, best_epoch = float("inf"), 0
    train = list(lines["train"])
    for epoch in range(1, int(options["epochs"]) + 1):
        if should_stop():
            raise Cancelled("cancelled in the app")
        started, losses = time.time(), []
        model.train()
        random.shuffle(train)
        for start in range(0, len(train), options["batch_size"]):
            batch = train[start:start + options["batch_size"]]
            x = torch.stack([pixels(line["image"], size, augment=True) for line in batch]).to(device)
            y = labels_of([line["display"] for line in batch], codec, max_len).to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=fp16):
                loss = model(pixel_values=x, labels=y).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        value = cer([line["display"] for line in lines["val"]], read(model, lines["val"], codec, size, device,
                                                                     options, max_len))
        loss = sum(losses) / max(len(losses), 1)
        history.add(epoch, train_loss=loss, val_cer=value, val_accuracy=1 - min(value, 100) / 100,
                    seconds=round(time.time() - started, 1))
        log(f"trocr epoch {epoch}: loss {loss:.3f}, validation CER {value:.2f} %")
        if value < best:
            best, best_epoch = value, epoch
            model.save_pretrained(out / "model")
        elif epoch - best_epoch >= int(options["lag"]):
            break
    model = VisionEncoderDecoderModel.from_pretrained(out / "model").to(device)
    predictions = read(model, lines["test"], codec, size, device, options, max_len)
    log(f"trocr: best validation CER {best:.2f} % (epoch {best_epoch})")
    return {"rows": to_rows(lines["test"], predictions, logical=False), "epochs": history.epochs,
            "peak_memory_mib": peak_memory_mib()}