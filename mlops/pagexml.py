"""Reading the PageXML exported by eScriptorium (any PAGE version: the namespace is ignored)."""

import re
import xml.etree.ElementTree as ET


def local(tag):
    return tag.rsplit("}", 1)[-1]


def read_page(path):
    """-> {"image": imageFilename, "width", "height", "lines": [{"baseline": bool, "coords": bool, "text": str}]}.
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
        })
    return {
        "image": page.get("imageFilename", ""),
        "width": int(float(page.get("imageWidth", 0) or 0)),
        "height": int(float(page.get("imageHeight", 0) or 0)),
        "lines": lines,
    }


def set_image_name(xml_text, name):
    """The PageXML pointing to another image file name (the pages are renamed to their opaque id)."""
    return re.sub(r'imageFilename="[^"]*"', f'imageFilename="{name}"', xml_text, count=1)
