"""CRNN (Shi et al. 2017): convolutions, two bidirectional LSTM layers, CTC; trained from scratch on our lines.

Training (settings of the job, defaults here): AdamW, learning rate 1e-3, halved when the validation CER has not
improved for 5 epochs (once it is below 100 %), batches of 16 lines of height 64, small width / slant
augmentation, at most 200 epochs, stopped when the validation CER has not improved for `lag` epochs; the best
epoch is kept. Greedy CTC decoding. The lines are read in visual order (left to right, as the image), as kraken.
"""

import random
import time

from mlops.baselines.common import Codec, Epochs, cer, device_of, gray_line, peak_memory_mib, seed_everything
from mlops.baselines.lines import to_rows

DEFAULTS = {"height": 64, "batch_size": 16, "learning_rate": 1e-3, "epochs": 200, "lag": 20}


def network(classes, height=64, hidden=256):
    from torch import nn

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()

            def block(i, o, pool):
                layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
                return layers + ([nn.MaxPool2d(pool)] if pool else [])
            self.cnn = nn.Sequential(*block(1, 64, (2, 2)), *block(64, 128, (2, 2)), *block(128, 256, None),
                                     *block(256, 256, (2, 1)), *block(256, 512, None), *block(512, 512, (2, 1)),
                                     nn.Dropout2d(0.1))
            self.rnn = nn.LSTM(512 * (height // 16), hidden, num_layers=2, bidirectional=True, dropout=0.3,
                               batch_first=True)
            self.out = nn.Linear(2 * hidden, classes)

        def forward(self, x):                   # N x 1 x H x W -> N x W/4 x classes
            f = self.cnn(x)
            n, c, h, w = f.shape
            return self.out(self.rnn(f.permute(0, 3, 1, 2).reshape(n, w, c * h))[0])

    return CRNN()


def batch_of(lines, indices, height, augment):
    import torch
    images = [gray_line(lines[i]["image"], height, max_width=4000, augment=augment) for i in indices]
    width = max(im.shape[1] for im in images)
    x = torch.zeros((len(images), 1, height, width))
    for k, im in enumerate(images):
        x[k, 0, :, :im.shape[1]] = im
    return x, torch.tensor([im.shape[1] for im in images])


def decode(logits, widths, codec):
    best = logits.argmax(-1).cpu()
    out = []
    for k, w in enumerate(widths):
        ids, previous = [], 0
        for i in best[k, :w].tolist():
            if i != previous and i != 0:
                ids.append(i)
            previous = i
        out.append(codec.decode(ids))
    return out


def run_epoch(model, lines, codec, device, options, optimizer=None):
    import torch
    from torch import nn
    train = optimizer is not None
    model.train(train)
    order = list(range(len(lines)))
    if train:
        random.shuffle(order)
    else:
        order.sort(key=lambda i: len(lines[i]["display"]))   # similar widths together: less padding
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    predictions, losses = [None] * len(lines), []
    for start in range(0, len(order), options["batch_size"]):
        indices = order[start:start + options["batch_size"]]
        x, widths = batch_of(lines, indices, options["height"], train)
        with torch.set_grad_enabled(train):
            logits = model(x.to(device))
            out_widths = torch.clamp((widths + 3) // 4, max=logits.shape[1])
            if train:
                targets = [torch.tensor(codec.encode(lines[i]["display"]), dtype=torch.long) for i in indices]
                loss = ctc(logits.log_softmax(-1).transpose(0, 1), torch.cat(targets), out_widths,
                           torch.tensor([len(t) for t in targets]))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            else:
                for i, text in zip(indices, decode(logits, out_widths.tolist(), codec)):
                    predictions[i] = text
    return predictions, (sum(losses) / len(losses) if losses else None)


def train_and_read(lines, settings, seed, out, log, should_stop):
    import torch
    from mlops.process import Cancelled
    options = {**DEFAULTS, **{k: v for k, v in settings.items() if v not in (None, "")}}
    seed_everything(seed)
    device = device_of(settings.get("device"))
    out.mkdir(parents=True, exist_ok=True)
    codec = Codec([line["display"] for line in lines["train"]], reserved=1)
    codec.save(out / "codec.json")
    model = network(len(codec), options["height"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options["learning_rate"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    history = Epochs("crnn")
    best, best_epoch = float("inf"), 0
    for epoch in range(1, int(options["epochs"]) + 1):
        if should_stop():
            raise Cancelled("cancelled in the app")
        started = time.time()
        _, loss = run_epoch(model, lines["train"], codec, device, options, optimizer)
        predictions, _ = run_epoch(model, lines["val"], codec, device, options)
        value = cer([line["display"] for line in lines["val"]], predictions)
        if value < 100:   # not while the network still outputs only blanks (the start of every CTC training)
            scheduler.step(value)
        history.add(epoch, train_loss=loss, val_cer=value, val_accuracy=1 - min(value, 100) / 100,
                    seconds=round(time.time() - started, 1))
        log(f"crnn epoch {epoch}: loss {loss:.3f}, validation CER {value:.2f} %")
        if value < best:
            best, best_epoch = value, epoch
            torch.save(model.state_dict(), out / "model.pt")
        elif epoch - best_epoch >= int(options["lag"]) and best < 100:
            break
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    predictions, _ = run_epoch(model, lines["test"], codec, device, options)
    log(f"crnn: best validation CER {best:.2f} % (epoch {best_epoch})")
    return {"rows": to_rows(lines["test"], predictions, logical=False), "epochs": history.epochs,
            "peak_memory_mib": peak_memory_mib()}