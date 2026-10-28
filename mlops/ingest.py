"""Ingestion: an eScriptorium export (zip of PageXML + images) -> checked, pseudonymised pages in the dataset.

Every page is checked: its image exists and opens, it has its size, enough lines, a baseline and a polygon on every
line, and most lines are transcribed. A page that fails is reported (the reasons are counted in the batch
report). A page already in the dataset (same image) is skipped.

Pseudonymisation: a kept page is renamed to an opaque id (the start of the SHA-256 of its image). The original
file name, which identifies a real record, is kept only in the private table page_origins. The images and the
transcriptions still show names and dates: the data stays personal data, kept on this machine only.

Split by group (params.yaml: ingest.group_pattern): the pages of the same register and year land in the same
split, so a test page is never a near-copy of a train page. The new groups are placed to balance the CHARACTERS
(mlops/splitting.py): 70 / 15 / 15 of the characters, every character in train, val and test, the same character
distribution in every split. A group already in the dataset keeps its split. The group is stored as a keyed hash
(MLOPS_SECRET), never as its name.

Page types (ingest.page_type_pattern): "..._<act>-1-1" is page 1 of an act (the names, in Arabic and French),
"..._<act>-2-1" its page 2 (the other details). Both pages of an act are in the same group, so in the same split (no leak of the
names between train and test); the split also balances the two types.
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

from mlops import db, normalize
from mlops.config import SPLITS, params, path
from mlops.pagexml import read_page, set_image_name
from mlops.splitting import assign_groups, coverage, describe
from mlops.vocabulary import page_characters

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def secret():
    return os.environ.get("MLOPS_SECRET", "").encode() or b"change-me"


def group_of(stem, pattern):
    """The group of a page from its original file name ("..._2016_162-2-1" -> "..._2016"), keyed hash."""
    match = re.match(pattern, stem) if pattern else None
    group = match.group(1) if match else stem
    return hmac.new(secret(), group.encode(), hashlib.sha256).hexdigest()[:16]


def page_type_of(stem, pattern):
    """The type of a page from its original file name (params.yaml: ingest.page_type_pattern):
    "03010108_1_1_I_2004_1-1-1" -> "1" (page 1: the names), "03010108_1_1_I_2004_1-2-1" -> "2" (page 2: the details
    of the act). A name that does not match is "other"."""
    match = re.search(pattern, stem) if pattern else None
    return match.group(1) if match else "other"


def backfill_page_types(conn):
    """The type of the pages ingested before the page types existed, from their original name (page_origins)."""
    pattern = params()["ingest"].get("page_type_pattern")
    rows = db.all_rows(conn, "SELECT p.id, o.original_name FROM pages p LEFT JOIN page_origins o ON o.page_id = p.id "
                             "WHERE p.page_type IS NULL")
    for row in rows:
        conn.execute("UPDATE pages SET page_type = %s WHERE id = %s",
                     (page_type_of(Path(row["original_name"] or "").stem, pattern), row["id"]))
    return len(rows)


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
        reasons, candidates, seen, normalized = Counter(), [], set(), Counter()
        replacements = normalize.table()
        xmls = sorted(folder.rglob("*.xml"))
        log(f"{len(xmls)} PageXML files")
        for number, xml_path in enumerate(xmls, 1):
            page, image, reason = check_page(xml_path, settings)
            if reason:
                reasons[reason] += 1
                continue
            for line in page["lines"]:   # the texts as they will be stored (ingest.normalize)
                line["text"] = normalize.text(line["text"], replacements)[0].strip()
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen or db.one(conn, "SELECT 1 AS x FROM pages WHERE sha256 = %s", digest):
                report["duplicates"] += 1
                continue
            seen.add(digest)
            candidates.append((xml_path, page, image, digest, group_of(xml_path.stem, settings["group_pattern"]),
                               page_type_of(xml_path.stem, settings.get("page_type_pattern"))))
            if number % 100 == 0:
                log(f"{number}/{len(xmls)} checked")
        # 2. the split of every group: a known group keeps its split, the new ones are placed to balance the
        #    characters and the page types (mlops/splitting.py)
        backfill_page_types(conn)
        votes = {}
        fixed, fixed_types = {split: Counter() for split in SPLITS}, {split: Counter() for split in SPLITS}
        for row in db.all_rows(conn, "SELECT id, group_key, split, page_type FROM pages"):
            votes.setdefault(row["group_key"], Counter())[row["split"]] += 1
            fixed[row["split"]].update(page_characters(row["id"]))
            fixed_types[row["split"]][row["page_type"]] += 1
        known = {group: counter.most_common(1)[0][0] for group, counter in votes.items()}
        new_groups, new_types = {}, {}
        for xml_path, page, image, digest, group, kind in candidates:
            characters = Counter("".join(line["text"] for line in page["lines"]))
            if group in known:
                fixed[force_split or known[group]].update(characters)
                fixed_types[force_split or known[group]][kind] += 1
            else:
                new_groups.setdefault(group, Counter()).update(characters)
                new_types.setdefault(group, Counter())[kind] += 1
        splits = dict(known)
        if not force_split:
            splits.update(assign_groups(fixed, new_groups, settings["val_fraction"], settings["test_fraction"],
                                        settings.get("split_tolerance", 0.33), fixed_types, new_types,
                                        settings.get("split_weights")))
        by_split = {split: Counter(counter) for split, counter in fixed.items()}
        types_by_split = {split: Counter(counter) for split, counter in fixed_types.items()}
        for group, characters in new_groups.items():
            by_split[force_split or splits[group]].update(characters)
            types_by_split[force_split or splits[group]].update(new_types[group])
        # 3. write the pages
        for xml_path, page, image, digest, group, kind in candidates:
            page_id = digest[:16]
            split = force_split or splits[group]
            image_name = page_id + image.suffix.lower()
            shutil.copyfile(image, pages_dir / image_name)
            xml_text, found = normalize.xml(xml_path.read_text(encoding="utf-8"), replacements)
            normalized.update(found)
            (pages_dir / f"{page_id}.xml").write_text(set_image_name(xml_text, image_name), encoding="utf-8")
            lines = page["lines"]
            conn.execute(
                "INSERT INTO pages (id, sha256, batch_id, group_key, split, image_name, lines, characters, page_type) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (page_id, digest, batch_id, group, split, image_name, len(lines),
                 sum(len(line["text"]) for line in lines), kind))
            conn.execute("INSERT INTO page_origins (page_id, original_name) VALUES (%s, %s)",
                         (page_id, str(xml_path.relative_to(folder))))
            report["added"][split] += 1
        report["rejected"] = sum(reasons.values())
        report["reasons"] = dict(reasons)
        report["new_groups"] = len(new_groups)
        report["coverage"] = coverage(by_split, types_by_split)
        report["normalized"] = dict(normalized)
        report["total"] = {row["split"]: row["n"] for row in
                           db.all_rows(conn, "SELECT split, count(*) AS n FROM pages GROUP BY split")}
        log(f"added {report['added']} ({len(new_groups)} new groups), {report['duplicates']} duplicates, "
            f"{report['rejected']} rejected {report['reasons'] or ''}")
        log(describe(report["coverage"]))
        log(f"normalized: {normalize.describe(normalized)}")
        return report
    finally:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)


def normalize_dataset(conn, log=print):
    """Apply ingest.normalize to the pages already in the dataset (their XML, their character count). The
    augmented pages were made from the old texts: they are deleted (a training makes them again).
    -> Counter of what was replaced."""
    replacements = normalize.table()
    total, changed = Counter(), 0
    for row in db.all_rows(conn, "SELECT id FROM pages ORDER BY id"):
        xml_path = path("data") / "pages" / f'{row["id"]}.xml'
        found = normalize.file(xml_path, replacements)
        if found:
            changed += 1
            total.update(found)
            characters = sum(len(line["text"]) for line in read_page(xml_path)["lines"])
            conn.execute("UPDATE pages SET characters = %s WHERE id = %s", (characters, row["id"]))
    if changed:
        shutil.rmtree(path("data") / "augmented", ignore_errors=True)
    log(f"{changed} pages changed: {normalize.describe(total)}"
        + (" (augmented pages deleted: they are made again from the new texts)" if changed else ""))
    return total


def resplit(conn, log=print):
    """Split every group again from scratch, balanced on the characters and the page types (mlops/splitting.py).
    The dataset versions already made keep their own splits (dataset_pages): the models trained on them stay
    traceable. -> {"pages": {split: n}, "coverage": ...}."""
    settings = params()["ingest"]
    backfill_page_types(conn)
    groups, types, sizes = {}, {}, Counter()
    for row in db.all_rows(conn, "SELECT id, group_key, page_type FROM pages ORDER BY id"):
        groups.setdefault(row["group_key"], Counter()).update(page_characters(row["id"]))
        types.setdefault(row["group_key"], Counter())[row["page_type"]] += 1
        sizes[row["group_key"]] += 1
    splits = assign_groups({}, groups, settings["val_fraction"], settings["test_fraction"],
                           settings.get("split_tolerance", 0.33), None, types, settings.get("split_weights"))
    with conn.transaction():
        for group, split in splits.items():
            conn.execute("UPDATE pages SET split = %s WHERE group_key = %s", (split, group))
    by_split, types_by_split = {s: Counter() for s in SPLITS}, {s: Counter() for s in SPLITS}
    for group, characters in groups.items():
        by_split[splits[group]].update(characters)
        types_by_split[splits[group]].update(types[group])
    pages = {s: sum(n for g, n in sizes.items() if splits[g] == s) for s in SPLITS}
    counts = {s: sum(1 for g in groups if splits[g] == s) for s in SPLITS}
    report = coverage(by_split, types_by_split)
    log(f"{len(groups)} groups split again: " + ", ".join(f"{s} {pages[s]} pages ({counts[s]} groups)" for s in SPLITS))
    log(describe(report))
    return {"pages": pages, "coverage": report}