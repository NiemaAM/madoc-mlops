"""The Browse tab: the pages of a split (or the augmented train pages), their image with the lines drawn on it,
their transcription and their PageXML.

Pages are shown by their opaque id only: the original file names (page_origins) never reach the app.
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
    return {r["id"]: r["page_type"] for r in db.all_rows(conn, "SELECT id, page_type FROM pages")}


def choices(conn, split, kind="all"):
    """-> ([(label, value)], note). value: the XML path inside the data folder ("pages/<id>.xml" ...)."""
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
    rows = db.all_rows(conn, "SELECT id, page_type, lines, characters FROM pages WHERE split = %s "
                             + ("" if kind == "all" else "AND page_type = %s ") + "ORDER BY page_type, id",
                       *((split,) if kind == "all" else (split, kind)))
    out = [(f'{r["id"]} · {type_label(r["page_type"])} · {r["lines"]} lines', f'pages/{r["id"]}.xml') for r in rows]
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


def show(value, show_lines=True):
    """-> (image, transcription html, xml text, info markdown)."""
    if not value:
        return None, "", "", "Choose a page."
    xml = resolve(value)
    page = read_page(xml)
    image = xml.parent / Path(page["image"].replace("\\", "/")).name
    picture = draw(image, page, show_lines) if image.is_file() else None
    words = sum(len(line["text"].split()) for line in page["lines"])
    chars = sum(len(line["text"]) for line in page["lines"])
    info = (f"**{xml.stem}** · {page['width']} × {page['height']} px · {len(page['lines'])} lines · {words} words · "
            f"{chars} characters" + ("" if picture else " · *image not found*"))
    return picture, transcription_html(page), xml.read_text(encoding="utf-8"), info


def step(choices_list, value, offset):
    """The previous / next page of the list."""
    values = [v for _, v in choices_list]
    if not values:
        return None
    if value not in values:
        return values[0]
    return values[(values.index(value) + offset) % len(values)]