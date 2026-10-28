"""Normalisation of the transcriptions: invisible or look-alike characters replaced (params.yaml: ingest.normalize).

A no-break space (U+00A0) looks exactly like a space in the image: the model cannot tell them apart, so every one
in the ground truth is an error it is trained to make. The table of params.yaml replaces such characters in the
<Unicode> texts of the PageXML, at ingest (new pages) and with "python -m mlops.cli normalize" (pages already in
the dataset). Nothing else of the file changes.
"""

import html
import re
from collections import Counter
from xml.sax.saxutils import escape

from mlops.config import params

UNICODE = re.compile(r"(<(?:[\w.-]+:)?Unicode(?:\s[^>]*)?>)(.*?)(</(?:[\w.-]+:)?Unicode>)", re.DOTALL)


def code(char):
    return f"U+{ord(char):04X}"


def table():
    """{character: replacement} from params.yaml (keys written "U+00A0", or the character itself)."""
    out = {}
    for key, value in ((params()["ingest"].get("normalize")) or {}).items():
        key = str(key)
        char = chr(int(key[2:], 16)) if key.upper().startswith("U+") else key
        if len(char) != 1:
            raise ValueError(f"ingest.normalize: {key!r} is not one character")
        out[char] = "" if value is None else str(value)
    return out


def text(value, replacements=None):
    """-> (normalised text, Counter {"U+00A0": n} of what was replaced)."""
    replacements = table() if replacements is None else replacements
    found = Counter(code(c) for c in value if c in replacements)
    if not found:
        return value, found
    return "".join(replacements.get(c, c) for c in value), found


def xml(xml_text, replacements=None):
    """The PageXML with its <Unicode> texts normalised -> (xml text, Counter)."""
    replacements = table() if replacements is None else replacements
    total = Counter()

    def fix(match):
        inner = html.unescape(match.group(2))
        new, found = text(inner, replacements)
        if not found:
            return match.group(0)
        total.update(found)
        return match.group(1) + escape(new) + match.group(3)

    if not replacements:
        return xml_text, total
    return UNICODE.sub(fix, xml_text), total


def file(xml_path, replacements=None):
    """Normalise a PageXML file in place (written only when something changed) -> Counter."""
    original = xml_path.read_text(encoding="utf-8")
    new, found = xml(original, replacements)
    if found:
        xml_path.write_text(new, encoding="utf-8")
    return found


def describe(found):
    return ", ".join(f"{k} × {n}" for k, n in sorted(found.items())) or "nothing to replace"