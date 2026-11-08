"""Running kraken on test pages and getting one row per line back (mlops/scoring.py reads them).

Recognition on the true lines: `kraken -f page -x -i page.xml out.xml ... ocr -m model` reads the lines of the
PageXML and writes them back with their text, under the same line ids: every true line gets its prediction.
Full pipeline: `kraken -x -i image out.xml segment -bl -i blla ocr -m htr` finds the lines itself; each true line is
paired with the predicted line closest to it in text (the closest pairs first, as textmetrics.line_cer), a line
left without a pair counts all its characters as missed (or extra). The line detection is scored apart: a true
line is found when a predicted polygon covers it (intersection over union >= 0.5).

kraken writes the text in visual order: python-bidi gives the logical order back (as MaDoc does).
"""

from bidi.algorithm import get_display
from PIL import Image, ImageDraw
from rapidfuzz.distance import Levenshtein

from mlops import db
from mlops.config import params, path
from mlops.pagexml import read_page
from mlops.process import run
from mlops.writers import label


def page_file(row):
    return path("data") / "pages" / f'{row["id"]}.xml'


def true_lines(row):
    """The transcribed lines of a page: [(index in the page, line)] (lines without text are skipped, as by
    ketos test)."""
    return [(i, line) for i, line in enumerate(read_page(page_file(row))["lines"]) if line["text"]]


def meta(conn, rows):
    """{page id: {"writer": anonymous label, "page_type", "lines": all its true texts}} (mlops/scoring.py)."""
    keys = {r["id"]: r for r in db.all_rows(conn, "SELECT id, writer_key, page_type FROM pages WHERE id = ANY(%s)",
                                            [r["id"] for r in rows])}
    out = {}
    for row in rows:
        info = keys.get(row["id"], row)
        out[row["id"]] = {"writer": label(info.get("writer_key")), "page_type": info.get("page_type") or "?",
                          "lines": [line["text"] for line in read_page(page_file(row))["lines"]]}
    return out


def kraken_device():
    return ["-d", params()["device"]]


def chunks(items, size=40):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def recognize(model_path, rows, out, log, should_stop):
    """The model reads the true lines of every page -> [{"page", "line", "ref", "pred"}]."""
    out.mkdir(parents=True, exist_ok=True)
    for part in chunks(rows):
        pairs = []
        for row in part:
            pairs += ["-i", page_file(row), out / f'{row["id"]}.xml']
        run(["kraken", *kraken_device(), "-f", "page", "-x", *pairs, "ocr", "-m", model_path], log, should_stop)
    result = []
    for row in rows:
        predicted = {}
        produced = out / f'{row["id"]}.xml'
        if produced.is_file():
            predicted = {line["id"]: get_display(line["text"]) for line in read_page(produced)["lines"]}
        for index, line in true_lines(row):
            result.append({"page": row["id"], "line": index, "ref": line["text"], "pred": predicted.get(line["id"], "")})
    return result


def polygon_iou(a, b):
    """Intersection over union of two polygons, on a small raster of their common box."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ax0, ay0, ax1, ay1 = min(x for x, _ in a), min(y for _, y in a), max(x for x, _ in a), max(y for _, y in a)
    bx0, by0, bx1, by1 = min(x for x, _ in b), min(y for _, y in b), max(x for x, _ in b), max(y for _, y in b)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return 0.0
    x0, y0, x1, y1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
    scale = 200 / max(x1 - x0, y1 - y0, 1)
    size = (max(1, int((x1 - x0) * scale) + 2), max(1, int((y1 - y0) * scale) + 2))

    def mask(points):
        image = Image.new("1", size, 0)
        ImageDraw.Draw(image).polygon([((x - x0) * scale, (y - y0) * scale) for x, y in points], fill=1)
        return image

    ma, mb = mask(a), mask(b)
    from PIL import ImageChops
    inter = ImageChops.logical_and(ma, mb).histogram()[-1]   # mode "1": the set pixels are in the last bin
    union = ImageChops.logical_or(ma, mb).histogram()[-1]
    return inter / union if union else 0.0


def detection(truth, predicted, threshold=0.5):
    """Line detection of one page -> {"true", "predicted", "matched"} (greedy matching on polygon IoU)."""
    pairs = sorted(((polygon_iou(t["polygon"], p["polygon"]), i, j) for i, t in enumerate(truth)
                    for j, p in enumerate(predicted)), reverse=True)
    used_t, used_p, matched = set(), set(), 0
    for iou, i, j in pairs:
        if iou < threshold:
            break
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matched += 1
    return {"true": len(truth), "predicted": len(predicted), "matched": matched}


def pair_lines(truth, predicted):
    """[(true index or None, predicted index or None)]: the closest texts first, each line used once."""
    pairs = sorted((Levenshtein.distance(t, p), i, j) for i, t in enumerate(truth) for j, p in enumerate(predicted))
    used_t, used_p, out = set(), set(), []
    for distance, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        if distance >= len(truth[i]) + len(predicted[j]):   # nothing in common: two unmatched lines
            continue
        used_t.add(i)
        used_p.add(j)
        out.append((i, j))
    out += [(i, None) for i in range(len(truth)) if i not in used_t]
    out += [(None, j) for j in range(len(predicted)) if j not in used_p]
    return out


def pipeline(seg_model, rec_model, rows, out, log, should_stop):
    """Segmentation then recognition of every test image -> (rows of lines, {page: detection counts})."""
    out.mkdir(parents=True, exist_ok=True)
    pages = path("data") / "pages"
    for part in chunks(rows):
        pairs = []
        for row in part:
            pairs += ["-i", pages / row["image_name"], out / f'{row["id"]}.xml']
        run(["kraken", *kraken_device(), "-x", *pairs, "segment", "-bl", "-i", seg_model, "ocr", "-m", rec_model],
            log, should_stop)
    result, detected = [], {}
    for row in rows:
        truth = true_lines(row)
        produced = out / f'{row["id"]}.xml'
        predicted = [line for line in read_page(produced)["lines"]] if produced.is_file() else []
        predicted_text = [get_display(line["text"]) for line in predicted]
        detected[row["id"]] = detection([line for _, line in truth], predicted)
        for i, j in pair_lines([line["text"] for _, line in truth], predicted_text):
            result.append({"page": row["id"], "line": truth[i][0] if i is not None else None,
                           "ref": truth[i][1]["text"] if i is not None else "",
                           "pred": predicted_text[j] if j is not None else "", "order": j})
    return result, detected


def detection_totals(detected):
    true = sum(d["true"] for d in detected.values())
    predicted = sum(d["predicted"] for d in detected.values())
    matched = sum(d["matched"] for d in detected.values())
    return {"precision": matched / predicted if predicted else 0.0, "recall": matched / true if true else 0.0,
            "true_lines": true, "predicted_lines": predicted, "matched_lines": matched}


def page_texts(line_rows):
    """{page: (true lines, predicted lines in reading order)} (the per-page scores of the gate)."""
    out = {}
    for row in line_rows:
        truth, predicted = out.setdefault(row["page"], ([], []))
        if row["ref"]:
            truth.append(row["ref"])
        if row["pred"]:
            predicted.append((row.get("order") if row.get("order") is not None else len(predicted), row["pred"]))
    return {p: (t, [text for _, text in sorted(pr)]) for p, (t, pr) in out.items()}