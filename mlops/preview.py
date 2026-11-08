"""A preview of the dataset as line images: every transcribed line cut out along its polygon, the lines of all
pages shuffled and renamed by split (train/train_000001.png ...), with their transcription.

    preview-<time>.zip
        README.txt
        train/labels.tsv        file name <TAB> transcription (logical order)
        train/train_000001.png  ...
        val/ ...  test/ ...

What it removes: the page and act a line comes from (shuffled across all the pages of a split, renamed), the
original file names, the image metadata (the crops are new images), everything outside the line's polygon.
What it keeps: everything written in the lines, names, dates and places included, and the handwriting. It is
NOT anonymised: it contains personal data. The download is limited to the admins of the app and recorded (table
downloads). It is not what the trainings use: the dataset stays PageXML + images.
"""

import random
import secrets
import shutil
import zipfile
from datetime import datetime, timezone

from PIL import Image, ImageDraw

from mlops import db
from mlops.config import SPLITS, path
from mlops.pagexml import read_page

README = """Line images of the MaDoc birth certificate dataset (preview).

Every transcribed line, cut out along its polygon, shuffled across the pages of its split and renamed.
labels.tsv: file name, transcription (logical order, as typed in eScriptorium).

THIS IS NOT ANONYMISED. The lines contain names, dates, places and the handwriting of real records: personal
data under Law 09-08. Do not share or publish without the authorisation of the data controller.
"""


def crop(image, polygon, margin=4):
    """The line, white outside its polygon, on a new image (no metadata)."""
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    box = (max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin),
           min(image.width, int(max(xs)) + margin), min(image.height, int(max(ys)) + margin))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], fill=255)
    white = Image.new("RGB", image.size, "white")
    line = Image.composite(image, white, mask).crop(box)
    clean = Image.new("RGB", line.size)
    clean.paste(line)
    return clean


def build(conn, log=print, should_stop=lambda: False):
    """Write <data>/preview/preview-<time>.zip (the previous one is deleted) -> {"file", "lines"}."""
    from mlops.process import Cancelled
    folder = path("data") / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = folder / f"preview-{stamp}.zip"
    partial = target.with_suffix(".part")
    rng = random.Random(secrets.randbits(64))   # never stored: the order cannot be undone
    pages_dir = path("data") / "pages"
    counts = {}
    work = path("data") / "work" / f"preview-{stamp}"
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("README.txt", README)
            for split in SPLITS:
                rows = db.all_rows(conn, "SELECT id, image_name FROM pages WHERE split = %s AND deleted_at IS NULL "
                                         "ORDER BY id", split)
                lines = []
                for row in rows:
                    page = read_page(pages_dir / f'{row["id"]}.xml')
                    lines += [(row["image_name"], line["polygon"], line["text"]) for line in page["lines"]
                              if line["text"] and len(line["polygon"]) >= 3]
                # every line gets a random number; the crops are written page by page (one image opened at a
                # time) under that number, then put in the zip in the order of the numbers
                numbers = list(range(1, len(lines) + 1))
                rng.shuffle(numbers)
                (work / split).mkdir(parents=True, exist_ok=True)
                labels, opened, image = {}, None, None
                for k, ((image_name, polygon, text), number) in enumerate(zip(lines, numbers), 1):
                    if k % 500 == 0 and should_stop():
                        raise Cancelled("cancelled in the app")
                    if image_name != opened:
                        image, opened = Image.open(pages_dir / image_name).convert("RGB"), image_name
                    line = crop(image, polygon)
                    if line is not None:
                        line.save(work / split / f"{split}_{number:06d}.png", format="PNG")
                        labels[number] = text
                for number in sorted(labels):
                    name = f"{split}_{number:06d}.png"
                    # a fixed date: the times the crops were written (page by page) would give the pages back
                    entry = zipfile.ZipInfo(f"{split}/{name}", date_time=(2000, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(entry, (work / split / name).read_bytes())
                archive.writestr(f"{split}/labels.tsv",
                                 "".join(f"{split}_{n:06d}.png\t{labels[n]}\n" for n in sorted(labels)))
                counts[split] = len(labels)
                log(f"{split}: {len(labels)} lines")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    partial.rename(target)
    for old in folder.glob("preview-*.zip"):
        if old != target:
            old.unlink(missing_ok=True)
    return {"file": target.name, "lines": counts}


def latest():
    folder = path("data") / "preview"
    files = sorted(folder.glob("preview-*.zip")) if folder.is_dir() else []
    return files[-1] if files else None


def record_download(conn, username, what):
    conn.execute("INSERT INTO downloads (username, what) VALUES (%s, %s)", (username, what))