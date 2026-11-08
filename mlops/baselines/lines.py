"""The true lines of pages, cut out of their images as kraken does (kraken.lib.segmentation.extract_polygons), so
every recognizer sees the same line images. The crops stay in the job's folder (the runs volume)."""

import dataclasses
import json

from bidi.algorithm import get_display

from mlops.config import path


def cut(rows, folder, log=print):
    """-> [{"page", "line" (index in the page), "image" (path), "text" (logical), "display" (visual order)}]."""
    from kraken.lib.segmentation import extract_polygons
    from kraken.lib.util import open_image
    from kraken.lib.xml import XMLPage
    from mlops.pagexml import read_page
    folder.mkdir(parents=True, exist_ok=True)
    index_file = folder / "index.json"
    if index_file.is_file():
        return json.loads(index_file.read_text(encoding="utf-8"))
    out, skipped = [], 0
    for row in rows:
        xml = path("data") / "pages" / f'{row["id"]}.xml'
        position = {line["id"]: i for i, line in enumerate(read_page(xml)["lines"])}
        page = XMLPage(xml)
        segmentation = page.to_container()
        lines = [line for line in segmentation.lines if line.text and line.text.strip()]
        image = open_image(str(page.imagename))
        for line in lines:
            try:
                crops = list(extract_polygons(image, dataclasses.replace(segmentation, lines=[line])))
            except ValueError:
                skipped += 1
                continue
            for crop, _ in crops:
                if not crop.width or not crop.height:
                    skipped += 1
                    continue
                name = f'{row["id"]}_{position.get(line.id, len(out))}.png'
                crop.save(folder / name)
                text = line.text.strip()
                out.append({"page": row["id"], "line": position.get(line.id), "image": str(folder / name),
                            "text": text, "display": get_display(text)})
    index_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log(f"{len(out)} lines cut from {len(rows)} pages" + (f" ({skipped} skipped)" if skipped else ""))
    return out


def to_rows(lines, predictions, logical=True):
    """Predictions (one per line, in the order of `lines`) -> the rows of mlops/scoring.py. logical=False: the
    predictions are in visual order (CTC / TrOCR trained on the display order) and are put back."""
    return [{"page": line["page"], "line": line["line"], "ref": line["text"],
             "pred": pred if logical else get_display(pred)} for line, pred in zip(lines, predictions)]