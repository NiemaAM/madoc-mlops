"""The Browse tab: the pages of a split (or the augmented train pages), their image with the lines drawn on it,
their transcription and their PageXML.

Pages are shown by their opaque id. With "show original names" (off by default, so a screenshot shows none), the
original file name (page_origins) is shown next to the id and can be searched: the app is only reachable by the
person who runs the platform, and the names never go to the job logs or W&B.
"""

import html
from pathlib import Path

from PIL import Image, ImageDraw

from mlops import augment, db
from mlops.config import params, path
from mlops.pagexml import read_page

AUGMENTED = "augmented"
MAX_SIDE = 1800          # the image is shown at most this large (the browser does not need the full scan)


def type_label(kind):
    labels = params()["ingest"].get("page_types") or {}
    return labels.get(kind, kind or "?")


def type_choices():
    labels = params()["ingest"].get("page_types") or {}
    return [("all types", "all"), *((label, kind) for kind, label in labels.items()), ("other", "other")]


def page_types_by_id(conn):
    return {r["id"]: r["page_type"] for r in db.all_rows(conn, "SELECT id, page_type FROM pages "
                                                               "WHERE deleted_at IS NULL")}


def choices(conn, split, kind="all", show_names=False, search=""):
    """-> ([(label, value)], note). value: the XML path inside the data folder ("pages/<id>.xml" ...).
    show_names: the original file name in the label; search: only the pages whose original name or id contains
    it (any split when searching)."""
    if split == AUGMENTED:
        found = augment.augmentations()
        if not found:
            return [], "No augmented pages yet: Data tab, \"Augment the train pages\"."
        newest = found[0]
        types = page_types_by_id(conn)
        out = []
        for xml in sorted(newest["folder"].glob("*.xml")):
            base = xml.stem.split("_aug")[0]
            if kind != "all" and types.get(base) != kind:
                continue
            out.append((f"{xml.stem} · {type_label(types.get(base))}",
                        str(xml.relative_to(path("data")))))
        note = (f"Augmentation {newest['folder'].name}: {newest['pages']} pages "
                f"({newest['copies']} copies of the train pages, seed {newest['seed']}).")
        return out, note
    search = (search or "").strip()
    where, args = ["p.deleted_at IS NULL"], []
    if search:
        where.append("(o.original_name ILIKE %s OR p.id ILIKE %s)")
        args += [f"%{search}%", f"%{search}%"]
    else:
        where.append("p.split = %s")
        args.append(split)
    if kind != "all":
        where.append("p.page_type = %s")
        args.append(kind)
    rows = db.all_rows(conn, "SELECT p.id, p.split, p.page_type, p.lines, p.held_out, o.original_name FROM pages p "
                             "LEFT JOIN page_origins o ON o.page_id = p.id WHERE " + " AND ".join(where)
                       + " ORDER BY p.page_type, p.id", *args)
    out = []
    for r in rows:
        name = f' · {Path(r["original_name"] or "").stem}' if show_names else ""
        where_ = f' · {r["split"]}' if search else ""
        held = " · held out" if r["held_out"] else ""
        out.append((f'{r["id"]}{name} · {type_label(r["page_type"])} · {r["lines"]} lines{where_}{held}',
                    f'pages/{r["id"]}.xml'))
    if search:
        return out, f'{len(out)} pages match "{search}"' + ("" if kind == "all" else f" ({type_label(kind)})") + "."
    return out, f"{len(out)} pages in {split}" + ("" if kind == "all" else f" ({type_label(kind)})") + "."


def resolve(value):
    """The XML file of a choice, refusing anything outside the data folder."""
    root = path("data").resolve()
    xml = (root / value).resolve()
    if root not in xml.parents or xml.suffix != ".xml" or not xml.is_file():
        raise FileNotFoundError(value)
    return xml


def draw(image_path, page, show_lines=True):
    image = Image.open(image_path).convert("RGB")
    sx = image.width / page["width"] if page["width"] else 1.0
    sy = image.height / page["height"] if page["height"] else 1.0
    scale = min(1.0, MAX_SIDE / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    if not show_lines:
        return image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    width = 3
    for number, line in enumerate(page["lines"], 1):
        polygon = [(x * sx * scale, y * sy * scale) for x, y in line["polygon"]]
        baseline = [(x * sx * scale, y * sy * scale) for x, y in line["baseline_points"]]
        if len(polygon) >= 3:
            pen.polygon(polygon, fill=(37, 99, 235, 40), outline=(37, 99, 235, 170))
        if len(baseline) >= 2:
            pen.line(baseline, fill=(220, 38, 38, 230), width=width)
            x, y = baseline[0]
            pen.text((x + 4, y - 14), str(number), fill=(220, 38, 38, 255))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def transcription_html(page):
    if not page["lines"]:
        return "<p>No lines.</p>"
    rows = "".join(f"<div class='br-line'><span class='br-n'>{n}</span>"
                   f"<span dir='auto'>{html.escape(line['text']) or '<i>(empty)</i>'}</span></div>"
                   for n, line in enumerate(page["lines"], 1))
    return ("<style>.br-lines{max-height:740px;overflow:auto}.br-line{display:flex;gap:12px;padding:4px 0;"
            "border-bottom:1px solid var(--border-color-primary)}.br-line span[dir]{flex:1}"
            ".br-n{color:#dc2626;min-width:2em;font-variant-numeric:tabular-nums}</style>"
            f"<div class='br-lines'>{rows}</div>")


def origin(page_id):
    """The original file name and the officer written on a page (private: only for the app, toggle on)."""
    with db.connection() as conn:
        row = db.one(conn, "SELECT o.original_name, w.raw_name, p.writer_key FROM pages p "
                           "LEFT JOIN page_origins o ON o.page_id = p.id LEFT JOIN page_writers w ON w.page_id = p.id "
                           "WHERE p.id = %s", page_id)
    return row or {}


def show(value, show_lines=True, show_names=False):
    """-> (image, transcription html, xml text, info markdown)."""
    if not value:
        return None, "", "", "Choose a page."
    try:
        xml = resolve(value)
    except FileNotFoundError:
        return None, "", "", "This page's file is not in the data folder (deleted or purged?)."
    page = read_page(xml)
    image = xml.parent / Path(page["image"].replace("\\", "/")).name
    picture = draw(image, page, show_lines) if image.is_file() else None
    words = sum(len(line["text"].split()) for line in page["lines"])
    chars = sum(len(line["text"]) for line in page["lines"])
    info = (f"**{xml.stem}** · {page['width']} × {page['height']} px · {len(page['lines'])} lines · {words} words · "
            f"{chars} characters" + ("" if picture else " · *image not found*"))
    if show_names and xml.parent.name == "pages":
        found = origin(xml.stem)
        from mlops.writers import label
        if found.get("original_name"):
            info += f" · original name `{Path(found['original_name']).stem}`"
        info += (f" · writer {found.get('raw_name') or '—'} ({label(found.get('writer_key'))})"
                 if found else "")
    return picture, transcription_html(page), xml.read_text(encoding="utf-8"), info


def step(choices_list, value, offset):
    """The previous / next page of the list."""
    values = [v for _, v in choices_list]
    if not values:
        return None
    if value not in values:
        return values[0]
    return values[(values.index(value) + offset) % len(values)]