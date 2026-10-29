"""Reading the PageXML exported by eScriptorium (any PAGE version: the namespace is ignored)."""

import math
import re
import xml.etree.ElementTree as ET


def local(tag):
    return tag.rsplit("}", 1)[-1]


def points(text):
    """"10,20 30,40" -> [(10.0, 20.0), (30.0, 40.0)] (malformed points, "a,b" "nan,1" "inf,2", are skipped)."""
    out = []
    for pair in (text or "").split():
        x, _, y = pair.partition(",")
        try:
            point = (float(x), float(y))
        except ValueError:
            continue
        if all(math.isfinite(v) for v in point):
            out.append(point)
    return out


def read_page(path):
    """-> {"image": imageFilename, "width", "height", "lines": [{"baseline": bool, "coords": bool, "text": str,
    "baseline_points": [(x, y)], "polygon": [(x, y)]}]}.
    ValueError when the file is not a PageXML."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"not XML: {e}") from e
    page = next((el for el in root.iter() if local(el.tag) == "Page"), None)
    if page is None:
        raise ValueError("no <Page> element (not a PageXML file)")
    lines = []
    for line in (el for el in page.iter() if local(el.tag) == "TextLine"):
        children = {local(child.tag): child for child in line}
        # the line's own transcription: TextEquiv/Unicode directly under the TextLine (not the words')
        text = ""
        equiv = children.get("TextEquiv")
        if equiv is not None:
            unicode = next((el for el in equiv if local(el.tag) == "Unicode"), None)
            text = (unicode.text or "").strip() if unicode is not None else ""
        lines.append({
            "baseline": "Baseline" in children and bool(children["Baseline"].get("points", "").strip()),
            "coords": "Coords" in children and bool(children["Coords"].get("points", "").strip()),
            "text": text,
            "baseline_points": points(children["Baseline"].get("points")) if "Baseline" in children else [],
            "polygon": points(children["Coords"].get("points")) if "Coords" in children else [],
        })
    return {"image": page.get("imageFilename", ""), "width": size(page.get("imageWidth")),
            "height": size(page.get("imageHeight")), "lines": lines}


def size(value):
    """imageWidth / imageHeight -> pixels; 0 when missing or not a size ("", "abc", "inf", "-5")."""
    try:
        number = float(value or 0)
        return int(number) if number > 0 else 0
    except (ValueError, OverflowError):
        return 0


def set_image_name(xml_text, name):
    """The PageXML pointing to another image file name (the pages are renamed to their opaque id)."""
    return re.sub(r'imageFilename="[^"]*"', f'imageFilename="{name}"', xml_text, count=1)