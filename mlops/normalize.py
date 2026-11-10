"""Normalisation of the transcriptions: characters, words or phrases replaced.

A no-break space (U+00A0) looks exactly like a space in the image: the model cannot tell them apart, so every one
in the ground truth is an error it is trained to make. A word misspelt again and again ("المصصفي عبادي" for
"المصطفى عبادي") is another error taught to the model. The rules replace such characters or texts in the
<Unicode> texts of the PageXML, at ingest (new pages) and with Normalize (app) or "python -m mlops.cli normalize"
(pages already in the dataset). Nothing else of the file changes.

A rule replaces a character or any text (a word, several words) wherever it is found, inside a word too (so
"والمصصفي" is fixed as well): the Preview of the app counts what a rule changes before it is applied. All the
rules are applied in one pass, the longest text first at a position: a replacement is never replaced again.

The rules are those of params.yaml (ingest.normalize: characters) and those written in the app (Data tab,
Normalisation; stored in the settings table, name "normalize": {"rules": {"U+0649": "ي", "المصصفي": "المصطفى"},
"disabled": ["U+00A0"]}; a character is stored by its code, a text as written). A rule of the app replaces the rule
of params.yaml for the same character or text; "disabled" switches a rule of params.yaml off.

A text rule may hold a name: in the logs, the job results and the reports it appears only as an anonymous label
"R-<6 characters of a keyed hash>" (the rules table of the app shows the label next to the text). A character is
shown by its code ("U+00A0").
"""

import html
import re
from collections import Counter
from functools import lru_cache
from xml.sax.saxutils import escape

from mlops.config import keyed, params

UNICODE = re.compile(r"(<(?:[\w.-]+:)?Unicode(?:\s[^>]*)?>)(.*?)(</(?:[\w.-]+:)?Unicode>)", re.DOTALL)
CODES = re.compile(r"[Uu]\+[0-9A-Fa-f]{1,6}")


def code(char):
    return f"U+{ord(char):04X}"


def label(source):
    """The name of a rule in the logs and reports: the code of a character, an anonymous label for a text."""
    return code(source) if len(source) == 1 else "R-" + keyed("normalize:" + source)[:6]


def from_codes(value):
    """"U+0020", "U+064A U+0627" -> the characters; anything else -> None."""
    parts = str(value).split()
    if parts and all(CODES.fullmatch(p) for p in parts):
        return "".join(chr(int(p[2:], 16)) for p in parts)
    return None


def source(value):
    """What a rule replaces: a character, its code ("U+00A0"), codes ("U+0627 U+0644") or any text."""
    value = "" if value is None else str(value)
    found = from_codes(value)
    if found is not None:
        return found
    if value == "":
        raise ValueError("give the character or the text to replace (or its code, e.g. U+0649)")
    return value


def character(value):
    """A source that must be one character (kept for the callers of the first version)."""
    char = source(value)
    if len(char) != 1:
        raise ValueError(f"{value!r}: give one character, or its code (U+00A0)")
    return char


def replacement(value):
    """The text a rule puts instead: "" deletes; codes ("U+0020", "U+064A U+0627") are read as characters, so a
    space or an invisible character can be typed."""
    value = "" if value is None else str(value)
    found = from_codes(value)
    return value if found is None else found


def stored(src):
    """The key of a rule in the settings: the code of a character, the text itself otherwise."""
    return code(src) if len(src) == 1 else src


def table(conn=None):
    """{character or text: replacement}: params.yaml (keys written "U+00A0", or the text itself), then the rules
    written in the app (conn given)."""
    out = {}
    for key, value in ((params()["ingest"].get("normalize")) or {}).items():
        try:
            out[source(key)] = "" if value is None else str(value)
        except ValueError:
            raise ValueError(f"ingest.normalize: {key!r} is empty") from None
    if conn is not None:
        saved = app_rules(conn)
        for key in saved["disabled"]:
            out.pop(source(key), None)
        for key, value in saved["rules"].items():
            out[source(key)] = "" if value is None else str(value)
    return out


def app_rules(conn):
    from mlops import settings as app_settings
    saved = app_settings.saved(conn, "normalize") or {}
    return {"rules": dict(saved.get("rules") or {}), "disabled": list(saved.get("disabled") or [])}


def add_rule(conn, old, new):
    """Replace `old` (a character, its code, or a text: a word, several words) by `new` (text; "" deletes it;
    codes accepted)."""
    from mlops import settings as app_settings
    old, new = source(old), replacement(new)
    if new == old:
        raise ValueError("the replacement is the same as what it replaces")
    saved = app_rules(conn)
    saved["rules"] = {k: v for k, v in saved["rules"].items() if source(k) != old}
    saved["rules"][stored(old)] = new
    saved["disabled"] = [k for k in saved["disabled"] if source(k) != old]
    app_settings.save(conn, "normalize", saved)
    return table(conn)


def remove_rule(conn, old):
    """Remove the rule of `old`: a rule written in the app is deleted; a rule of params.yaml is switched off."""
    from mlops import settings as app_settings
    old = source(old)
    saved = app_rules(conn)
    before = len(saved["rules"])
    saved["rules"] = {k: v for k, v in saved["rules"].items() if source(k) != old}
    removed = len(saved["rules"]) < before
    if old in table() and not any(source(k) == old for k in saved["disabled"]):
        saved["disabled"].append(stored(old))
        removed = True
    if not removed:
        raise ValueError(f"no rule for {old if len(old) > 1 else code(old)}")
    app_settings.save(conn, "normalize", saved)
    return table(conn)


def shown(value):
    """A text as the app shows it: a space as ␣, an invisible character by its code."""
    if value == "":
        return "(deleted)"
    if len(value) > 1 and all(c.isprintable() for c in value) and value.strip() == value:
        return value
    return " ".join("␣" if c == " " else (c if c.isprintable() and not c.isspace() else code(c)) for c in value)


def rules_rows(conn):
    """The rules for the app's table: [rule, replaced, name, replaced by, from]. Private: a text rule is shown."""
    import unicodedata
    saved = app_rules(conn)
    base = table()
    in_app = {source(k) for k in saved["rules"]}

    def name(src):
        return (unicodedata.name(src, "control character").title() if len(src) == 1
                else f"text, {len(src)} characters")

    rows = []
    for src, new in sorted(table(conn).items(), key=lambda kv: (len(kv[0]) > 1, kv[0])):
        rows.append([label(src), shown(src), name(src), shown(new), "app" if src in in_app else "params.yaml"])
    for key in saved["disabled"]:
        src = source(key)
        if src in base:
            rows.append([label(src), shown(src), name(src), shown(base[src]), "params.yaml (switched off)"])
    return rows


@lru_cache(maxsize=16)
def _pattern(sources):
    return re.compile("|".join(re.escape(s) for s in sources))


def text(value, replacements=None):
    """-> (normalised text, Counter {"U+00A0": n, "R-1a2b3c": n} of what was replaced)."""
    replacements = table() if replacements is None else replacements
    if not replacements or not value:
        return value, Counter()
    found = Counter()

    def put(match):
        found[label(match.group(0))] += 1
        return replacements[match.group(0)]

    # the longest first: at one position, "المصصفي عبادي" wins over "المصصفي"
    new = _pattern(tuple(sorted(replacements, key=lambda s: (-len(s), s)))).sub(put, value)
    return (new if found else value), found


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