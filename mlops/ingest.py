"""Ingestion: an eScriptorium export (zip of PageXML + images) -> checked, pseudonymised pages in the dataset.

Every page is checked: its image exists and opens, it has its size, enough lines, a baseline and a polygon on every
line, and most lines are transcribed. A page that fails is reported (the reasons are counted in the batch
report). A page already in the dataset (same image) is skipped.

Pseudonymisation: a kept page is renamed to an opaque id (the start of the SHA-256 of its image). The original
file name, which identifies a real record, is kept only in the private table page_origins. The images and the
transcriptions still show names and dates: the data stays personal data, kept on this machine only.

Split by group (params.yaml: ingest.group_pattern): the pages of the same register and year land in the same
split, so a test page is never a near-copy of a train page. The new groups are given to the splits by their size,
to keep 70 / 15 / 15 of the pages (assign_groups); a group already in the dataset keeps its split. The group is
stored as a keyed hash (MLOPS_SECRET), never as its name.
"""

import hashlib
import hmac
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from mlops import db
from mlops.config import SPLITS, params, path
from mlops.pagexml import read_page, set_image_name

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def secret():
    return os.environ.get("MLOPS_SECRET", "").encode() or b"change-me"


def group_of(stem, pattern):
    """The group of a page from its original file name ("..._2016_162-2-1" -> "..._2016"), keyed hash."""
    match = re.match(pattern, stem) if pattern else None
    group = match.group(1) if match else stem
    return hmac.new(secret(), group.encode(), hashlib.sha256).hexdigest()[:16]


def find_image(xml_path, name):
    folder = xml_path.parent
    if name:
        candidate = folder / Path(name.replace("\\", "/")).name
        if candidate.is_file():
            return candidate
    for suffix in IMAGE_SUFFIXES + tuple(s.upper() for s in IMAGE_SUFFIXES):
        if xml_path.with_suffix(suffix).is_file():
            return xml_path.with_suffix(suffix)
    return None


def check_page(xml_path, settings):
    """-> (page, image, None) or (None, None, reason)."""
    try:
        page = read_page(xml_path)
    except ValueError as e:
        return None, None, f"not a PageXML ({e})"
    image = find_image(xml_path, page["image"])
    if image is None:
        return None, None, "image not found"
    try:
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            size = im.size
    except Exception:
        return None, None, "image unreadable"
    if not (page["width"] and page["height"]):
        return None, None, "no image size in the XML"   # kraken needs imageWidth / imageHeight
    if abs(size[0] - page["width"]) > 2 or abs(size[1] - page["height"]) > 2:
        return None, None, "image size differs from the XML"
    lines = page["lines"]
    if len(lines) < settings["min_lines"]:
        return None, None, "too few lines"
    if any(not (line["baseline"] and line["coords"]) for line in lines):
        return None, None, "line without baseline or polygon"
    if sum(1 for line in lines if line["text"]) / len(lines) < settings["min_text_ratio"]:
        return None, None, "lines not transcribed"
    return page, image, None


def extract(zip_path, target):
    """Unzip, refusing paths that leave the folder ("zip slip") and archives too large once extracted."""
    limit = params()["ingest"]["max_upload_mb"] * 1024 * 1024
    with zipfile.ZipFile(zip_path) as archive:
        if sum(member.file_size for member in archive.infolist()) > 4 * limit:
            raise ValueError("the archive is too large once extracted")
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if member.is_dir() or target.resolve() not in destination.parents:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)


def assign_groups(current, new_groups, val_fraction, test_fraction):
    """Splits of NEW groups, balanced by their number of pages -> {group: split}.

    current: pages already in each split; new_groups: {group: pages}. The largest group first, each to the split
    furthest below its share (70 / 15 / 15 by default) once the new pages are counted; equal deficits go to
    test, then val, then train, and equal sizes are taken in the order of their keys: the result only depends
    on the data. With few groups (registers), a split by hash could leave a split empty: this cannot."""
    shares = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    counts = {s: current.get(s, 0) for s in SPLITS}
    total = sum(counts.values()) + sum(new_groups.values())
    out = {}
    for group, size in sorted(new_groups.items(), key=lambda item: (-item[1], item[0])):
        split = max(("test", "val", "train"), key=lambda s: shares[s] * total - counts[s])
        out[group] = split
        counts[split] += size
    return out


def ingest(conn, batch_id, source, force_split=None, log=print):
    """Ingest a zip or a folder into the dataset -> report (counts only: safe to show and to log)."""
    settings = params()["ingest"]
    pages_dir = path("data") / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    report = {"added": {s: 0 for s in SPLITS}, "duplicates": 0, "rejected": 0, "reasons": {}}
    temporary = None
    try:
        if source.is_file():
            temporary = Path(tempfile.mkdtemp(prefix="ingest_", dir=path("data")))
            extract(source, temporary)
            folder = temporary
        else:
            folder = source
        # 1. check every page, find its group
        reasons, candidates, seen = Counter(), [], set()
        xmls = sorted(folder.rglob("*.xml"))
        log(f"{len(xmls)} PageXML files")
        for number, xml_path in enumerate(xmls, 1):
            page, image, reason = check_page(xml_path, settings)
            if reason:
                reasons[reason] += 1
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen or db.one(conn, "SELECT 1 AS x FROM pages WHERE sha256 = %s", digest):
                report["duplicates"] += 1
                continue
            seen.add(digest)
            candidates.append((xml_path, page, image, digest, group_of(xml_path.stem, settings["group_pattern"])))
            if number % 100 == 0:
                log(f"{number}/{len(xmls)} checked")
        # 2. the split of every group: a known group keeps its split, the new ones are balanced by size
        known = {}
        for row in db.all_rows(conn, "SELECT group_key, split, count(*) AS n FROM pages GROUP BY 1, 2"):
            known.setdefault(row["group_key"], Counter())[row["split"]] += row["n"]
        current = Counter()
        for counter in known.values():
            current.update(counter)
        new_groups = Counter(group for *_, group in candidates if group not in known)
        splits = {group: counter.most_common(1)[0][0] for group, counter in known.items()}
        splits.update(assign_groups(current, new_groups, settings["val_fraction"], settings["test_fraction"]))
        # 3. write the pages
        for xml_path, page, image, digest, group in candidates:
            page_id = digest[:16]
            split = force_split or splits[group]
            image_name = page_id + image.suffix.lower()
            shutil.copyfile(image, pages_dir / image_name)
            xml_text = xml_path.read_text(encoding="utf-8")
            (pages_dir / f"{page_id}.xml").write_text(set_image_name(xml_text, image_name), encoding="utf-8")
            lines = page["lines"]
            conn.execute(
                "INSERT INTO pages (id, sha256, batch_id, group_key, split, image_name, lines, characters) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (page_id, digest, batch_id, group, split, image_name, len(lines),
                 sum(len(line["text"]) for line in lines)))
            conn.execute("INSERT INTO page_origins (page_id, original_name) VALUES (%s, %s)",
                         (page_id, str(xml_path.relative_to(folder))))
            report["added"][split] += 1
        report["rejected"] = sum(reasons.values())
        report["reasons"] = dict(reasons)
        report["new_groups"] = len(new_groups)
        report["total"] = {row["split"]: row["n"] for row in
                           db.all_rows(conn, "SELECT split, count(*) AS n FROM pages GROUP BY split")}
        log(f"added {report['added']} ({len(new_groups)} new groups), {report['duplicates']} duplicates, "
            f"{report['rejected']} rejected {report['reasons'] or ''}")
        return report
    finally:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)


def resplit(conn, log=print):
    """Split every group again from scratch, balanced (assign_groups). The dataset versions already made keep
    their own splits (dataset_pages): the models trained on them stay traceable. -> {split: pages}."""
    settings = params()["ingest"]
    sizes = {row["group_key"]: row["n"] for row in
             db.all_rows(conn, "SELECT group_key, count(*) AS n FROM pages GROUP BY group_key")}
    splits = assign_groups({}, sizes, settings["val_fraction"], settings["test_fraction"])
    with conn.transaction():
        for group, split in splits.items():
            conn.execute("UPDATE pages SET split = %s WHERE group_key = %s", (split, group))
    counts = {s: sum(n for g, n in sizes.items() if splits[g] == s) for s in SPLITS}
    groups = {s: sum(1 for g in sizes if splits[g] == s) for s in SPLITS}
    log(f"{len(sizes)} groups split again: " + ", ".join(f"{s} {counts[s]} pages ({groups[s]} groups)" for s in SPLITS))
    return counts