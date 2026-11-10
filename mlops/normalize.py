"""Normalisation of the transcriptions: invisible or look-alike characters replaced.

A no-break space (U+00A0) looks exactly like a space in the image: the model cannot tell them apart, so every one
in the ground truth is an error it is trained to make. The rules replace such characters in the <Unicode> texts
of the PageXML, at ingest (new pages) and with Normalize (app) or "python -m mlops.cli normalize" (pages already in
the dataset). Nothing else of the file changes.

The rules are those of params.yaml (ingest.normalize) and those written in the app (Data tab, Normalisation;
stored in the settings table, name "normalize": {"rules": {"U+0649": "ي"}, "disabled": ["U+00A0"]}). A rule of the
app replaces the rule of params.yaml for the same character; "disabled" switches a rule of params.yaml off.
"""

import html
import re
from collections import Counter
from xml.sax.saxutils import escape

from mlops.config import params

UNICODE = re.compile(r"(<(?:[\w.-]+:)?Unicode(?:\s[^>]*)?>)(.*?)(</(?:[\w.-]+:)?Unicode>)", re.DOTALL)


def code(char):
    return f"U+{ord(char):04X}"


def character(value):
    """"U+00A0", "u+a0" or the character itself -> the character (ValueError if not exactly one)."""
    value = str(value)
    if re.fullmatch(r"[Uu]\+[0-9A-Fa-f]{1,6}", value.strip()):
        return chr(int(value.strip()[2:], 16))
    if len(value) != 1:
        raise ValueError(f"{value!r}: give one character, or its code (U+00A0)")
    return value


def replacement(value):
    """The text a character is replaced by: "" deletes it; codes ("U+0020", "U+064A U+0627") are read as
    characters, so a space or an invisible character can be typed."""
    value = "" if value is None else str(value)
    parts = value.split()
    if parts and all(re.fullmatch(r"[Uu]\+[0-9A-Fa-f]{1,6}", p) for p in parts):
        return "".join(chr(int(p[2:], 16)) for p in parts)
    return value


def table(conn=None):
    """{character: replacement}: params.yaml (keys written "U+00A0", or the character itself), then the rules
    written in the app (conn given)."""
    out = {}
    for key, value in ((params()["ingest"].get("normalize")) or {}).items():
        try:
            char = character(key)
        except ValueError:
            raise ValueError(f"ingest.normalize: {key!r} is not one character") from None
        out[char] = "" if value is None else str(value)
    if conn is not None:
        saved = app_rules(conn)
        for key in saved["disabled"]:
            out.pop(character(key), None)
        for key, value in saved["rules"].items():
            out[character(key)] = "" if value is None else str(value)
    return out


def app_rules(conn):
    from mlops import settings as app_settings
    saved = app_settings.saved(conn, "normalize") or {}
    return {"rules": dict(saved.get("rules") or {}), "disabled": list(saved.get("disabled") or [])}


def add_rule(conn, char, new):
    """Replace `char` (a character or its code) by `new` (text; "" deletes it; codes accepted)."""
    from mlops import settings as app_settings
    char, new = character(char), replacement(new)
    if new == char:
        raise ValueError("the replacement is the character itself")
    saved = app_rules(conn)
    saved["rules"][code(char)] = new
    saved["disabled"] = [k for k in saved["disabled"] if character(k) != char]
    app_settings.save(conn, "normalize", saved)
    return table(conn)


def remove_rule(conn, char):
    """Remove the rule of `char`: a rule written in the app is deleted; a rule of params.yaml is switched off."""
    from mlops import settings as app_settings
    char = character(char)
    saved = app_rules(conn)
    removed = saved["rules"].pop(code(char), None) is not None
    from_params = char in table()
    if from_params and code(char) not in saved["disabled"]:
        saved["disabled"].append(code(char))
        removed = True
    if not removed:
        raise ValueError(f"no rule for {code(char)}")
    app_settings.save(conn, "normalize", saved)
    return table(conn)


def rules_rows(conn):
    """The rules for the app's table: [code, character, name, replacement, from]."""
    import unicodedata
    saved = app_rules(conn)
    base = table()
    shown = lambda t: "(deleted)" if t == "" else " ".join(   # noqa: E731
        "␣" if c == " " else (c if c.isprintable() and not c.isspace() else code(c)) for c in t)
    rows = []
    for char, new in sorted(table(conn).items()):
        origin = "app" if code(char) in saved["rules"] else "params.yaml"
        rows.append([code(char), shown(char), unicodedata.name(char, "control character").title(), shown(new),
                     origin])
    for key in saved["disabled"]:
        char = character(key)
        if char in base:
            rows.append([code(char), shown(char), unicodedata.name(char, "control character").title(),
                         shown(base[char]), "params.yaml (switched off)"])
    return rows


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