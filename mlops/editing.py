"""Editing a page of the dataset (the Editor of the app): its lines (polygon, baseline, text) and its image
(rotation, crop), with a history that can be restored.

    save_lines(conn, page_id, lines, base_revision, user)   the lines as the person left them
    transform(conn, page_id, op, base_revision, user)       rotate 90 / 180 / 270, or crop to a box
    restore(conn, page_id, revision, base_revision, user)   an older revision back, as a new revision
    history(conn, page_id)                                  the revisions, the newest first

Every change is a new revision of the page: its files are copied to <data>/revisions/<page id>/ (r0 = the page
before its first edit) and pages.revision goes up, so the digest of the dataset changes (mlops/datasets.py): a
model trained after an edit is traced to the edited pages. Nothing is ever overwritten without a copy, and a
restore is itself a revision.

The PageXML is changed in place (its regions, reading order and other elements stay): only the TextLines are
moved, added, removed or given a new text. The texts typed by hand go through the normalisation rules
(mlops/normalize.py), as an ingested page does. The augmented copies of the page are deleted (they were made
from the old page; the next augmentation makes them again), and the compiled lines follow the files by
themselves (mlops/train.py).

A page is not changed while a job that reads the pages runs (a training would read half old, half new files).
"""

import re
import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from mlops import db, normalize
from mlops.config import path
from mlops.pagexml import local, read_page

READING_JOBS = ("ingest", "augment", "train", "evaluate", "crossval", "pipeline", "normalize", "preview")
MAX_LINES = 1000
MAX_POINTS = 2000
MAX_TEXT = 2000


class EditError(ValueError):
    """A change that cannot be made, with a message for the person."""


class PageNotFound(KeyError):
    """No page of the dataset has this id (or it was deleted)."""


class Conflict(EditError):
    """The page changed since it was opened, or a job reads the pages now."""


def pages_folder():
    return path("data") / "pages"


def revisions_folder(page_id):
    return path("data") / "revisions" / page_id


def page_row(conn, page_id):
    row = db.one(conn, "SELECT * FROM pages WHERE id = %s AND deleted_at IS NULL", page_id)
    if not row:
        raise PageNotFound(page_id)
    return row


def files_of(page_id):
    """(xml path, image path) of a page of the dataset."""
    xml = pages_folder() / f"{page_id}.xml"
    page = read_page(xml)
    image = xml.parent / Path(page["image"].replace("\\", "/")).name
    return xml, image


def busy_job(conn):
    return db.one(conn, "SELECT id, kind, task FROM jobs WHERE status = 'running' AND kind = ANY(%s) ORDER BY id "
                        "LIMIT 1", list(READING_JOBS))


def check(conn, page_id, base_revision):
    """The page as it is now -> its row; Conflict when it changed since base_revision or a job reads the pages."""
    row = page_row(conn, page_id)
    if base_revision is not None and int(base_revision) != row["revision"]:
        raise Conflict(f"this page changed since it was opened (revision {row['revision']} now, not "
                       f"{base_revision}): open it again")
    job = busy_job(conn)
    if job:
        raise Conflict(f"job {job['id']} ({job['kind']}{' ' + job['task'] if job['task'] else ''}) is reading the "
                       "pages: edit once it is done")
    return row


# --------------------------------------------------------------------------------------------------------------
# the PageXML
# --------------------------------------------------------------------------------------------------------------
def point_text(points):
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def parse(xml_path):
    """-> (tree, namespace uri or "")."""
    tree = ET.parse(xml_path)
    match = re.match(r"\{(.*)\}", tree.getroot().tag)
    namespace = match.group(1) if match else ""
    return tree, namespace


def write(tree, namespace, target):
    if namespace:
        ET.register_namespace("", namespace)
    temporary = target.with_suffix(".tmp")
    tree.write(temporary, encoding="UTF-8", xml_declaration=True)
    temporary.replace(target)


def find_page(root):
    page = next((el for el in root.iter() if local(el.tag) == "Page"), None)
    if page is None:
        raise EditError("no <Page> element in this PageXML")
    return page


def child(element, name, namespace, create=True, before=None):
    """The first child `name` of an element (made when missing, before the child named `before` if there is)."""
    found = next((el for el in element if local(el.tag) == name), None)
    if found is not None or not create:
        return found
    found = ET.Element(f"{{{namespace}}}{name}" if namespace else name)
    position = len(element)
    if before:
        position = next((i for i, el in enumerate(element) if local(el.tag) in before), position)
    element.insert(position, found)
    return found


def clean_lines(lines, width, height):
    """The lines sent by the app -> [{"id", "polygon", "baseline", "text"}], checked and kept inside the image."""
    if not isinstance(lines, list) or len(lines) > MAX_LINES:
        raise EditError(f"lines: a list of at most {MAX_LINES}")
    out, seen = [], set()
    for number, line in enumerate(lines, 1):
        if not isinstance(line, dict):
            raise EditError(f"line {number}: not an object")
        try:
            polygon = [(float(x), float(y)) for x, y in line.get("polygon") or []]
            baseline = [(float(x), float(y)) for x, y in line.get("baseline") or []]
        except (TypeError, ValueError):
            raise EditError(f"line {number}: its points are not numbers") from None
        if len(polygon) < 3 or len(baseline) < 2:
            raise EditError(f"line {number}: a polygon of 3 points or more and a baseline of 2 points or more")
        if len(polygon) + len(baseline) > MAX_POINTS or any(v != v for p in polygon + baseline for v in p):
            raise EditError(f"line {number}: too many points, or points that are not numbers")
        text = line.get("text", "")
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            raise EditError(f"line {number}: the text must be a text of at most {MAX_TEXT} characters")
        identifier = str(line.get("id") or "")
        if not re.fullmatch(r"[A-Za-z_][\w.-]{0,80}", identifier) or identifier in seen:
            identifier = ""          # a new line, or an id that is not an XML id: a new one is given
        seen.add(identifier)
        inside = lambda pts: [(min(max(x, 0), width), min(max(y, 0), height)) for x, y in pts]  # noqa: E731
        out.append({"id": identifier, "polygon": inside(polygon), "baseline": inside(baseline),
                    "text": " ".join(text.split())})
    return out


def rewrite_lines(xml_path, lines, replacements):
    """Write `lines` (in reading order) into the PageXML: the lines kept are updated in place, the lines removed
    are taken out, the new ones go into the region of the line before them. -> (lines, characters)."""
    tree, namespace = parse(xml_path)
    root = tree.getroot()
    page = find_page(root)
    tag = (lambda name: f"{{{namespace}}}{name}") if namespace else (lambda name: name)
    regions = [el for el in page.iter() if local(el.tag) == "TextRegion"]
    if not regions:
        region = ET.SubElement(page, tag("TextRegion"), {"id": "region_" + uuid.uuid4().hex[:8]})
        width, height = int(page.get("imageWidth") or 0), int(page.get("imageHeight") or 0)
        ET.SubElement(region, tag("Coords"), {"points": point_text([(0, 0), (width, 0), (width, height), (0, height)])})
        regions = [region]
    existing, parent_of = {}, {}
    for region in regions:
        for element in list(region):
            if local(element.tag) == "TextLine":
                existing.setdefault(element.get("id", ""), element)
                parent_of[element] = region
    used_ids = {el.get("id") for el in root.iter() if el.get("id")}
    placed = {region: [] for region in regions}
    current_region = regions[0]
    characters = 0
    for line in lines:
        element = existing.pop(line["id"], None) if line["id"] else None
        if element is None:
            identifier = line["id"] if line["id"] and line["id"] not in used_ids else ""
            while not identifier or identifier in used_ids:
                identifier = "line_" + uuid.uuid4().hex[:10]
            used_ids.add(identifier)
            element = ET.Element(tag("TextLine"), {"id": identifier})
        else:
            current_region = parent_of[element]
        child(element, "Coords", namespace, before=("Baseline", "Word", "TextEquiv", "TextStyle")).set(
            "points", point_text(line["polygon"]))
        child(element, "Baseline", namespace, before=("Word", "TextEquiv", "TextStyle")).set(
            "points", point_text(line["baseline"]))
        text, _ = normalize.text(line["text"], replacements)
        equiv = child(element, "TextEquiv", namespace, before=("TextStyle",))
        unicode = child(equiv, "Unicode", namespace)
        unicode.text = text
        characters += len(text)
        placed[current_region].append(element)
    for region in regions:            # the TextLines of each region, in the new order; the other children stay
        others = [el for el in region if local(el.tag) != "TextLine"]
        first = next((i for i, el in enumerate(region) if local(el.tag) == "TextLine"), len(others))
        head = [el for el in list(region)[:first] if local(el.tag) != "TextLine"]
        tail = [el for el in others if el not in head]
        for element in list(region):
            region.remove(element)
        for element in head + placed[region] + tail:
            region.append(element)
    write(tree, namespace, xml_path)
    return len(lines), characters


def transform_points(op, width, height):
    """-> (function (x, y) -> (x, y), new width, new height, crop box or None)."""
    kind = op.get("op")
    if kind == "rotate":
        degrees = int(op.get("degrees", 0)) % 360
        if degrees == 90:          # clockwise
            return (lambda x, y: (height - y, x)), height, width, None
        if degrees == 270:         # counter-clockwise
            return (lambda x, y: (y, width - x)), height, width, None
        if degrees == 180:
            return (lambda x, y: (width - x, height - y)), width, height, None
        raise EditError("rotate: 90, 180 or 270 degrees")
    if kind == "crop":
        try:
            x0, y0, x1, y1 = (int(round(float(v))) for v in op.get("box") or [])
        except (TypeError, ValueError):
            raise EditError("crop: a box [left, top, right, bottom]") from None
        x0, x1 = sorted((min(max(x0, 0), width), min(max(x1, 0), width)))
        y0, y1 = sorted((min(max(y0, 0), height), min(max(y1, 0), height)))
        if x1 - x0 < 32 or y1 - y0 < 32:
            raise EditError("crop: the box is too small")
        new_width, new_height = x1 - x0, y1 - y0
        return ((lambda x, y: (min(max(x - x0, 0), new_width), min(max(y - y0, 0), new_height))), new_width, new_height,
                (x0, y0, x1, y1))
    raise EditError("an image change is a rotation or a crop")


def transform_files(xml_path, image_path, op):
    """Rotate or crop the image and every point of the PageXML with it -> (lines, characters, lines removed)."""
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        move, new_width, new_height, box = transform_points(op, width, height)
        if box is None:     # PIL's ROTATE_270 turns the image clockwise by 90°
            method = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                      270: Image.Transpose.ROTATE_90}[int(op["degrees"]) % 360]
            result = image.transpose(method)
        else:
            result = image.crop(box)
        save_image(result, image_path, image.format)
    tree, namespace = parse(xml_path)
    page = find_page(tree.getroot())
    # the points of the PageXML are in its own size; a file whose size differs from the image is scaled first
    sx = width / (int(page.get("imageWidth") or 0) or width)
    sy = height / (int(page.get("imageHeight") or 0) or height)
    removed = 0
    for element in list(page.iter()):
        points = element.get("points")
        if points is None:
            continue
        moved = []
        for pair in points.split():
            x, _, y = pair.partition(",")
            try:
                moved.append(move(float(x) * sx, float(y) * sy))
            except ValueError:
                continue
        element.set("points", point_text(moved))
    if op["op"] == "crop":    # the lines left with no area (all outside the box) are taken out
        for region in [el for el in page.iter() if local(el.tag) == "TextRegion"]:
            for line in [el for el in region if local(el.tag) == "TextLine"]:
                coords = child(line, "Coords", namespace, create=False)
                points = [tuple(map(float, p.split(","))) for p in (coords.get("points") if coords is not None
                                                                     else "").split()]
                xs, ys = {p[0] for p in points}, {p[1] for p in points}
                if len(xs) < 2 or len(ys) < 2:
                    region.remove(line)
                    removed += 1
    page.set("imageWidth", str(new_width))
    page.set("imageHeight", str(new_height))
    write(tree, namespace, xml_path)
    page_now = read_page(xml_path)
    return len(page_now["lines"]), sum(len(line["text"]) for line in page_now["lines"]), removed


def save_image(image, target, original_format):
    fmt = (original_format or Path(target).suffix.lstrip(".") or "PNG").upper()
    fmt = {"JPG": "JPEG", "TIF": "TIFF"}.get(fmt, fmt)
    temporary = Path(target).with_name(Path(target).name + ".tmp")
    options = {"quality": 95, "subsampling": 0} if fmt == "JPEG" else {}
    if fmt == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(temporary, format=fmt, **options)
    temporary.replace(target)


# --------------------------------------------------------------------------------------------------------------
# revisions
# --------------------------------------------------------------------------------------------------------------
def keep(conn, page_id, revision, action, user, note, image_changed):
    """Copy the files of the page now as `revision` and record it."""
    xml, image = files_of(page_id)
    folder = revisions_folder(page_id)
    folder.mkdir(parents=True, exist_ok=True)
    xml_file = f"r{revision}.xml"
    shutil.copyfile(xml, folder / xml_file)
    previous = db.one(conn, "SELECT image_file FROM page_revisions WHERE page_id = %s ORDER BY revision DESC "
                            "LIMIT 1", page_id)
    if image_changed or not previous or not (folder / previous["image_file"]).is_file():
        image_file = f"r{revision}{image.suffix.lower()}"
        shutil.copyfile(image, folder / image_file)
    else:
        image_file = previous["image_file"]     # the same image as the revision before: not copied again
    page = read_page(xml)
    conn.execute("INSERT INTO page_revisions (page_id, revision, username, action, note, lines, characters, "
                 "xml_file, image_file) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                 (page_id, revision, user, action, note, len(page["lines"]),
                  sum(len(line["text"]) for line in page["lines"]), xml_file, image_file))


def first_copy(conn, page_id, user):
    """The page before its first edit, kept as revision 0."""
    if not db.one(conn, "SELECT 1 AS x FROM page_revisions WHERE page_id = %s AND revision = 0", page_id):
        keep(conn, page_id, 0, "original", user, "before the first edit", image_changed=True)


def after_change(conn, row, action, user, note, image_changed):
    """A new revision of the page: copied, counted, its officer read again, its augmented copies deleted."""
    from mlops import writers
    page_id = row["id"]
    xml, _ = files_of(page_id)
    page = read_page(xml)
    texts = [line["text"] for line in page["lines"]]
    revision = row["revision"] + 1
    officer_before = db.one(conn, "SELECT raw_name FROM page_writers WHERE page_id = %s", page_id)
    with conn.transaction():
        conn.execute("UPDATE pages SET lines = %s, characters = %s, revision = %s WHERE id = %s",
                     (len(texts), sum(len(t) for t in texts), revision, page_id))
        keep(conn, page_id, revision, action, user, note, image_changed)
        writers.record(conn, page_id, texts)
    for file in (path("data") / "augmented").glob(f"*/{page_id}_aug*"):
        file.unlink(missing_ok=True)
    officer_now = db.one(conn, "SELECT raw_name FROM page_writers WHERE page_id = %s", page_id)
    moved = 0
    if (officer_before or {}).get("raw_name") != (officer_now or {}).get("raw_name"):
        # another officer read on the page: its pages follow the split of their writer (as a normalisation does)
        from mlops.ingest import place
        moved = place(conn, log=lambda *_: None)["moved"]
    return {"revision": revision, "lines": len(texts), "characters": sum(len(t) for t in texts), "moved": moved}


def save_lines(conn, page_id, lines, base_revision, user=None):
    """The lines of the page as the person left them -> the new revision."""
    row = check(conn, page_id, base_revision)
    xml, image = files_of(page_id)
    with Image.open(image) as picture:
        width, height = picture.size
    cleaned = clean_lines(lines, width, height)
    first_copy(conn, page_id, user)
    before = xml.read_bytes()
    try:
        rewrite_lines(xml, cleaned, normalize.table(conn))
    except Exception:
        xml.write_bytes(before)      # the file as it was: nothing half written
        raise
    if xml.read_bytes() == before:
        return {"revision": row["revision"], "unchanged": True}
    return after_change(conn, row, "lines", user, f"{len(cleaned)} lines", image_changed=False)


def transform(conn, page_id, op, base_revision, user=None):
    """Rotate or crop the image of the page (and its lines with it) -> the new revision."""
    if not isinstance(op, dict):
        raise EditError("an image change: {\"op\": \"rotate\", \"degrees\": 90} or {\"op\": \"crop\", \"box\": [...]}")
    row = check(conn, page_id, base_revision)
    xml, image = files_of(page_id)
    first_copy(conn, page_id, user)
    saved = (xml.read_bytes(), image.read_bytes())
    try:
        _, _, removed = transform_files(xml, image, op)
    except Exception:
        xml.write_bytes(saved[0])
        image.write_bytes(saved[1])
        raise
    note = (f"rotated {int(op['degrees']) % 360}°" if op["op"] == "rotate" else
            "cropped" + (f", {removed} lines outside the box removed" if removed else ""))
    return after_change(conn, row, op["op"], user, note, image_changed=True)


def restore(conn, page_id, revision, base_revision, user=None):
    """The files of an older revision back as the page now (a new revision: nothing is lost)."""
    row = check(conn, page_id, base_revision)
    old = db.one(conn, "SELECT * FROM page_revisions WHERE page_id = %s AND revision = %s", page_id, int(revision))
    if not old:
        raise EditError(f"revision {revision} of this page does not exist")
    if old["revision"] == row["revision"]:
        raise EditError("this revision is already the page now")
    folder = revisions_folder(page_id)
    source_xml, source_image = folder / old["xml_file"], folder / old["image_file"]
    if not source_xml.is_file() or not source_image.is_file():
        raise EditError(f"the files of revision {revision} are missing")
    first_copy(conn, page_id, user)
    xml, image = files_of(page_id)
    current_image = image.read_bytes()
    shutil.copyfile(source_xml, xml)
    restored = xml.parent / Path(read_page(xml)["image"].replace("\\", "/")).name
    if restored != image:            # the PageXML names the image of the page, whatever the revision
        raise EditError("the revision names another image file")
    shutil.copyfile(source_image, image)
    return after_change(conn, row, "restore", user, f"revision {old['revision']} restored",
                        image_changed=image.read_bytes() != current_image)


def history(conn, page_id):
    return db.all_rows(conn, "SELECT revision, created_at, username, action, note, lines, characters "
                             "FROM page_revisions WHERE page_id = %s ORDER BY revision DESC", page_id)
