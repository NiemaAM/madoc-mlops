"""Ingestion: an eScriptorium export (zip of PageXML + images) -> checked, pseudonymised pages in the dataset.

Every page is checked: its image exists and opens, it has its size, enough lines, a baseline and a polygon on every
line, and most lines are transcribed. A page that fails is reported (the reasons are counted in the batch
report). A page already in the dataset (same image) is skipped.

Pseudonymisation: a kept page is renamed to an opaque id (the start of the SHA-256 of its image). The original
file name, which identifies a real record, is kept only in the private table page_origins. The images and the
transcriptions still show names and dates: the data stays personal data, kept on this machine only.

Split by writer (params.yaml: ingest.split_by, default): the officer named on page 2 of each act (mlops/writers.py)
is the unit placed, so no writer of the test pages is in train or val; page 1 takes the writer of page 2 of its
act, pages whose act has no page 2 yet stay in train. Or by group (split_by: group, or fewer than 3 writers found;
ingest.group_pattern): the pages of the same register and year land in the same split. The new writers / groups are
placed to balance the CHARACTERS (mlops/splitting.py): 70 / 15 / 15 of the characters, every character in train,
val and test, the same character distribution in every split. A writer / group already in the dataset keeps its
split. A held-out batch stays in test for good. Groups, acts and writers are stored as keyed hashes (MLOPS_SECRET),
never as their names (the officer's name is in the private table page_writers).

Page types (ingest.page_type_pattern): "..._<act>-1-1" is page 1 of an act (the names, in Arabic and French),
"..._<act>-2-1" its page 2 (the other details). Both pages of an act are in the same group, so in the same split (no leak of the
names between train and test); the split also balances the two types.
"""

import hashlib
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from mlops import db, normalize, settings as app_settings
from mlops.config import SPLITS, keyed, params, path
from mlops.pagexml import read_page, set_image_name
from mlops.splitting import assign_groups, coverage, describe
from mlops.vocabulary import page_characters

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def group_of(stem, pattern):
    """The group of a page from its original file name ("..._2016_162-2-1" -> "..._2016"), keyed hash."""
    match = re.match(pattern, stem) if pattern else None
    return keyed(match.group(1) if match else stem)


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
    if not lines or sum(1 for line in lines if line["text"]) / len(lines) < settings["min_text_ratio"]:
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
    """Ingest a zip or a folder into the dataset -> report (counts only: safe to show and to log).

    A held-out batch (batches.held_out) puts its pages in test for good: never trained on, never augmented,
    never moved by a split. force_split (mlops.cli import --split) puts the pages in that split now (an existing
    split imported as it is); a later "split again" may move them."""
    settings = params()["ingest"]
    pages_dir = path("data") / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    batch = db.one(conn, "SELECT held_out FROM batches WHERE id = %s", batch_id) or {}
    pinned = "test" if batch.get("held_out") else None
    report = {"added": {s: 0 for s in SPLITS}, "duplicates": 0, "restored": 0, "rejected": 0, "reasons": {},
              "held_out": bool(batch.get("held_out"))}
    temporary = None
    try:
        if source.is_file():
            temporary = Path(tempfile.mkdtemp(prefix="ingest_", dir=path("data")))
            extract(source, temporary)
            folder = temporary
        else:
            folder = source
        # 1. check every page
        reasons, candidates, seen, normalized = Counter(), [], set(), Counter()
        replacements = normalize.table(conn)
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
            if digest in seen:
                report["duplicates"] += 1
                continue
            seen.add(digest)
            existing = db.one(conn, "SELECT id, deleted_at FROM pages WHERE sha256 = %s", digest)
            if existing and existing["deleted_at"] is None:
                report["duplicates"] += 1
                continue
            if existing:   # a page of a deleted batch uploaded again: it comes back, in this batch
                conn.execute("UPDATE pages SET deleted_at = NULL, batch_id = %s, held_out = %s, pinned_split = %s, "
                             "split = coalesce(%s, split) WHERE id = %s",
                             (batch_id, bool(batch.get("held_out")), pinned, pinned or force_split, existing["id"]))
                report["restored"] += 1
                continue
            candidates.append((xml_path, page, image, digest, group_of(xml_path.stem, settings["group_pattern"]),
                               page_type_of(xml_path.stem, settings.get("page_type_pattern"))))
            if number % 100 == 0:
                log(f"{number}/{len(xmls)} checked")
        backfill_page_types(conn)
        # 2. write the pages, in one transaction: a batch that fails half-way adds no page (the files already
        #    copied are written again by the next try)
        with conn.transaction():
            new_ids = write_pages(conn, candidates, pinned, pinned or force_split, batch_id, folder, pages_dir,
                                  replacements, normalized, bool(batch.get("held_out")))
        # 3. the split: the writers (or groups) already in the dataset keep their split, the new ones are placed
        #    to balance the characters and the page types (mlops/splitting.py)
        # pages imported with a split keep it (they count as already placed); the others are placed
        placement = place(conn, new_ids=() if force_split else new_ids, log=lambda *_: None)
        for row in db.all_rows(conn, "SELECT split, count(*) AS n FROM pages WHERE id = ANY(%s) GROUP BY split",
                               list(new_ids)):
            report["added"][row["split"]] = row["n"]
        report["rejected"] = sum(reasons.values())
        report["reasons"] = dict(reasons)
        report["new_groups"] = placement["new_units"]
        report["moved"] = placement["moved"]
        report["coverage"] = placement["coverage"]
        report["normalized"] = dict(normalized)
        report["total"] = {row["split"]: row["n"] for row in
                           db.all_rows(conn, "SELECT split, count(*) AS n FROM pages WHERE deleted_at IS NULL "
                                             "GROUP BY split")}
        log(f"added {report['added']} ({placement['new_units']} new {placement['unit']}s), "
            f"{report['duplicates']} duplicates, {report['restored']} restored, {report['rejected']} rejected "
            f"{report['reasons'] or ''}" + (" (held-out batch: in test for good)" if report["held_out"] else ""))
        if placement["moved"]:
            log(f"{placement['moved']} pages moved to the split of their writer (their act's page 2 came in; they "
                "were in train only because their writer was unknown)")
        log(describe(report["coverage"]))
        log(f"normalized: {normalize.describe(normalized)}")
        return report
    finally:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)


def write_pages(conn, candidates, pinned, split, batch_id, folder, pages_dir, replacements, normalized, held_out):
    """Copy the checked pages into the dataset (opaque names) and record them (pages, page_origins, the officer
    named in them) -> their ids. The split is set right after, by place()."""
    from mlops import writers
    ids = []
    for xml_path, page, image, digest, group, kind in candidates:
        page_id = digest[:16]
        image_name = page_id + image.suffix.lower()
        shutil.copyfile(image, pages_dir / image_name)
        xml_text, found = normalize.xml(xml_path.read_text(encoding="utf-8"), replacements)
        normalized.update(found)
        (pages_dir / f"{page_id}.xml").write_text(set_image_name(xml_text, image_name), encoding="utf-8")
        lines = page["lines"]
        original = str(xml_path.relative_to(folder))
        conn.execute(
            "INSERT INTO pages (id, sha256, batch_id, group_key, split, image_name, lines, characters, page_type, "
            "act_key, pinned_split, held_out) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (page_id, digest, batch_id, group, split or "train", image_name, len(lines),
             sum(len(line["text"]) for line in lines), kind, writers.act_key(original), pinned, held_out))
        conn.execute("INSERT INTO page_origins (page_id, original_name) VALUES (%s, %s)", (page_id, original))
        writers.record(conn, page_id, [line["text"] for line in lines])
        ids.append(page_id)
    return ids


MIN_WRITERS = 3


def split_by(split=None):
    """"writer" (the officer of the act: writer-independent test) or "group" (register and year)."""
    mode = (split or {}).get("split_by") or params()["ingest"].get("split_by") or "writer"
    return mode if mode in ("writer", "group") else "writer"


def place(conn, split=None, new_ids=(), fresh=False, apply=True, log=print):
    """Put every page of the dataset in its split -> {"pages", "groups", "coverage", "text", "moved", ...}.

    The unit placed is the writer (split_by "writer") or the register-year (split_by "group"): all the pages of
    a unit go to the same split. A pinned page (held out, imported with --split) keeps its split. In "writer"
    mode, a page whose writer is unknown (its act has no page 2 yet) is trained on, never tested.
    fresh: every unit placed again (resplit); otherwise the units already in the dataset keep their split and
    only the new ones (pages of new_ids only) are placed. A page whose writer was unknown until now (the page 2 of
    its act just came in) was in train only for that reason: it does not vote for the split of its writer, it
    follows it (a new writer is placed freely, with all its pages). apply=False computes without changing
    anything."""
    from mlops import writers
    split = split or app_settings.split(conn)
    mode = split_by(split)
    backfill_page_types(conn)
    writers.rekey(conn)
    known_before = {r["id"] for r in db.all_rows(conn, "SELECT id FROM pages WHERE writer_key IS NOT NULL")}
    if mode == "writer":
        writers.sync(conn)
    rows = db.all_rows(conn, "SELECT id, group_key, writer_key, split, pinned_split, page_type FROM pages "
                             "WHERE deleted_at IS NULL ORDER BY id")
    new_ids = set(new_ids)
    fallback = ""
    if mode == "writer" and len({r["writer_key"] for r in rows if r["writer_key"]}) < MIN_WRITERS:
        # no officer line in the transcriptions (another kind of document, or too few writers to split): by
        # register-year rather than everything in train
        fallback = (f"fewer than {MIN_WRITERS} writers found in the transcriptions: split by register-year "
                    "instead; ")
        mode = "group"

    def unit(row):
        return row["writer_key"] if mode == "writer" else row["group_key"]

    fixed, fixed_types = {s: Counter() for s in SPLITS}, {s: Counter() for s in SPLITS}
    free, free_types, votes, forced = {}, {}, {}, {}
    for row in rows:
        characters = page_characters(row["id"])
        if row["pinned_split"] or (mode == "writer" and not row["writer_key"]):
            target = row["pinned_split"] or "train"
            forced[row["id"]] = target
            fixed[target].update(characters)
            fixed_types[target][row["page_type"]] += 1
            continue
        if not fresh and row["id"] not in new_ids and (mode != "writer" or row["id"] in known_before):
            votes.setdefault(unit(row), Counter())[row["split"]] += 1
        free.setdefault(unit(row), Counter()).update(characters)
        free_types.setdefault(unit(row), Counter())[row["page_type"]] += 1
    known = {u: counter.most_common(1)[0][0] for u, counter in votes.items()}
    for u, target in known.items():   # a unit already placed: its pages count as fixed
        fixed[target].update(free.pop(u))
        fixed_types[target].update(free_types.pop(u))
    placed = dict(known)
    placed.update(assign_groups(fixed, free, split["val_fraction"], split["test_fraction"], split["split_tolerance"],
                                fixed_types, free_types, split["split_weights"]))
    target_of = {row["id"]: forced.get(row["id"]) or placed[unit(row)] for row in rows}
    moved = sum(1 for row in rows if row["id"] not in new_ids and target_of[row["id"]] != row["split"])
    if apply:
        with conn.transaction():
            for row in rows:
                if target_of[row["id"]] != row["split"]:
                    conn.execute("UPDATE pages SET split = %s WHERE id = %s", (target_of[row["id"]], row["id"]))
    by_split, types_by_split = {s: Counter() for s in SPLITS}, {s: Counter() for s in SPLITS}
    for row in rows:
        by_split[target_of[row["id"]]].update(page_characters(row["id"]))
        types_by_split[target_of[row["id"]]][row["page_type"]] += 1
    pages = Counter(target_of.values())
    units = {s: len({unit(r) for r in rows if r["id"] not in forced and target_of[r["id"]] == s}) for s in SPLITS}
    report = coverage(by_split, types_by_split)
    unit_name = "writer" if mode == "writer" else "group"
    held_writers = {r["writer_key"] for r in rows if r["pinned_split"] == "test" and r["writer_key"]}
    overlap = sum(1 for r in rows if r["writer_key"] in held_writers and target_of[r["id"]] != "test")
    text = (fallback + f"{len(set(placed))} {unit_name}s {'placed again' if fresh else 'placed'}"
            f"{'' if apply else ' (preview, not applied)'}: "
            + ", ".join(f"{s} {pages[s]} pages ({units[s]} {unit_name}s)" for s in SPLITS)
            + (f"; {sum(1 for r in rows if r['id'] in forced and not r['pinned_split'])} pages without a known "
               "writer kept in train" if mode == "writer" else "")
            + (f"; {sum(1 for r in rows if r['pinned_split'])} pinned pages (held out or imported with a split)"
               if any(r["pinned_split"] for r in rows) else "")
            + (f"; WARNING: {overlap} train/val pages share a writer with the held-out set" if overlap else "")
            + "\n" + describe(report))
    log(text)
    return {"pages": dict(pages), "groups": units, "coverage": report, "text": text, "by_split": by_split,
            "types": types_by_split, "moved": moved, "new_units": len(set(placed) - set(known)),
            "unit": unit_name, "held_out_overlap": overlap}


def normalize_dataset(conn, log=print, apply=True):
    """Apply the normalisation rules (params.yaml ingest.normalize and the app's, mlops/normalize.py) to the pages
    already in the dataset (their XML, their character count). The
    augmented pages were made from the old texts: they are deleted (a training makes them again). Before
    changing anything, the XML files changed are copied to <data>/backup/normalize-<time>/.
    apply=False: only counts what would change (the preview of the app).
    A rule may correct the officer's name ("المصصفي عبادي" -> "المصطفى عبادي"): the officer of the pages changed is
    read again and every page follows the split of its writer (two spellings now one writer: one split).
    -> {"replaced": Counter {"U+00A0": n, "R-1a2b3c": n}, "pages": pages changed, "backup": folder or None,
        "moved": pages moved to the split of their writer}."""
    from datetime import datetime, timezone

    from mlops import writers
    replacements = normalize.table(conn)
    total, changed = Counter(), []
    for row in db.all_rows(conn, "SELECT id FROM pages WHERE deleted_at IS NULL ORDER BY id"):
        xml_path = path("data") / "pages" / f'{row["id"]}.xml'
        try:
            _, found = normalize.xml(xml_path.read_text(encoding="utf-8"), replacements)
        except OSError:
            continue
        if found:
            changed.append(row["id"])
            total.update(found)
    backup, moved = None, 0
    if apply and changed:
        backup = path("data") / "backup" / f'normalize-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}'
        backup.mkdir(parents=True, exist_ok=True)
        for page_id in changed:
            xml_path = path("data") / "pages" / f"{page_id}.xml"
            shutil.copyfile(xml_path, backup / xml_path.name)
            normalize.file(xml_path, replacements)
            characters = sum(len(line["text"]) for line in read_page(xml_path)["lines"])
            conn.execute("UPDATE pages SET characters = %s WHERE id = %s", (characters, page_id))
            writers.record(conn, page_id, [line["text"] for line in read_page(xml_path)["lines"]])
        shutil.rmtree(path("data") / "augmented", ignore_errors=True)
        shutil.rmtree(path("data") / "compiled", ignore_errors=True)
        moved = place(conn, log=lambda *_: None)["moved"]
    log(f"{len(changed)} pages {'changed' if apply else 'would change'}: {normalize.describe(total)}"
        + (f" (backup of the old files: {backup.name}; augmented pages deleted: they are made again from the new "
           "texts)" if backup else "")
        + (f"; {moved} pages moved to the split of their writer (an officer's name was corrected)" if moved else ""))
    return {"replaced": dict(total), "pages": len(changed), "backup": str(backup) if backup else None,
            "moved": moved}


def resplit(conn, log=print, split=None, apply=True):
    """Split every writer (or group) again from scratch, balanced on the characters and the page types
    (mlops/splitting.py), with `split` (split_by, val_fraction, test_fraction, split_tolerance, split_weights;
    default: the settings now). apply=False: only computes it (the preview of the app). The dataset versions
    already made keep their own splits (dataset_pages): the models trained on them stay traceable."""
    return place(conn, split=split, fresh=True, apply=apply, log=log)