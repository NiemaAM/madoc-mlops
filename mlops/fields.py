"""What the printed form says about each line of a certificate, read from its ground truth.

- the writer: the officer named after "من طرفنا نحن" (page 2), up to "موظف جماعي";
- the child's names: the Latin text after "PRENOM" (first name) and "NOM" (family name), the Arabic text after
  "الاسم الشخصي" and "الاسم العائلي", on page 1 and on page 2. "FILS DE", "ET DE" (the parents) and the marginal
  notes of page 1 are other lines: they are never read as names;
- the form row of every line (the printed label it starts with: "في يوم", "من والده" ...), for the CER per row.

Only the transcription is used (never MaDoc's parser): a name is a span of characters of one line. To score a
recognised line, the span is carried over to the prediction by aligning the two lines character by character, so a
misread label does not make the name wrong, only a misread name does.
"""

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

ALIF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
HARAKAT = re.compile("[ً-ٰٟـ]")     # diacritics, superscript alif, tatweel
NAME_FIELDS = ("latin_first", "latin_family", "arabic_first", "arabic_family")
FIELD_LABELS = {"latin_first": "First name (Latin)", "latin_family": "Family name (Latin)",
                "arabic_first": "First name (Arabic)", "arabic_family": "Family name (Arabic)"}


def ar_key(text):
    """Arabic text for comparing: no diacritics nor tatweel, one alif, ى = ي, ة = ه, single spaces."""
    text = HARAKAT.sub("", unicodedata.normalize("NFC", text or "")).translate(ALIF).replace("ة", "ه")
    return " ".join(text.split())


def is_arabic(char):
    return "؀" <= char <= "ۿ" or "ݐ" <= char <= "ݿ" or "ﭐ" <= char <= "﻿"


def is_latin(char):
    return char.isalpha() and unicodedata.name(char, "").startswith("LATIN")


# --- the writer ---------------------------------------------------------------------------------------------
# "من طرفنا نحن <name> موظف جماعي"; "نحن" may be missing, the name may be written on the next line, and the ن of
# "طرفنا" is sometimes missed in the transcription ("من طرفا نحن")
OFFICER = re.compile(r"طرف\s*ن?ا[\s،,:.\-…]*(?:نحن)?")
STOP = re.compile(r"(?:ال)?(?:موظف|ضابط)")


def _name_part(text):
    """The Arabic words of a text, up to "موظف" / "ضابط" (the officer's title)."""
    stop = STOP.search(text)
    text = re.sub(r"[.…:،,\-]+", " ", text[:stop.start()] if stop else text)
    return " ".join(w for w in text.split() if any(is_arabic(c) for c in w))


def clean_text(text):
    """A line as compared: presentation forms as plain letters (NFKC: "ﻃﺮﻓﻨﺎ" -> "طرفنا"), invisible format
    characters removed (zero-width joiner / non-joiner, direction marks: they look like nothing), no diacritics nor
    tatweel."""
    text = unicodedata.normalize("NFKC", text or "")
    return HARAKAT.sub("", "".join(c for c in text if unicodedata.category(c) != "Cf"))


def writer_name(lines):
    """The officer named in the page ("من طرفنا نحن لحسن شميطي موظف جماعي" -> "لحسن شميطي"), or "".
    Hidden characters, diacritics and tatweel are ignored (clean_text); when the line ends after "من طرفنا نحن",
    the name is read on the next line."""
    clean = [clean_text(t) for t in lines]
    for index, text in enumerate(clean):
        match = OFFICER.search(text)
        if not match:
            continue
        rest = text[match.end():]
        name = _name_part(rest)
        if not name and not STOP.search(rest) and index + 1 < len(clean) and not OFFICER.search(clean[index + 1]):
            name = _name_part(clean[index + 1])
        if name:
            return name
    return ""


def officer_lines(lines):
    """The lines that hold "طرفنا" (and the next one), for the diagnostic of the names not read."""
    clean = [clean_text(t) for t in lines]
    for index, text in enumerate(clean):
        if OFFICER.search(text):
            return clean[index:index + 2]
    return []


def reveal(text):
    """A line with every character that is not a plain Arabic or Latin letter, digit, space or common punctuation
    shown by its code: "طر⟨U+200D⟩فنا" (what makes two lines that look the same differ)."""
    plain = set(" .,:;-'\"()/،؛؟")
    out = []
    for c in text:
        ordinary = (("ء" <= c <= "ي" or c in "ڭ") or c.isascii() and (c.isalnum() or c in plain)
                    or c in plain or "٠" <= c <= "٩")
        out.append(c if ordinary else f"⟨U+{ord(c):04X}⟩")
    return "".join(out)


def officer_candidates(lines):
    """When no line holds "طرفنا": the lines that look like the officer line (close to "من طرفنا نحن", or holding
    "موظف" / "نحن"), raw, for the diagnostic."""
    from rapidfuzz import fuzz
    out = []
    for text in lines:
        clean = clean_text(text)
        if "موظف" in clean or "نحن" in clean or fuzz.partial_ratio("من طرفنا نحن", clean) >= 70:
            out.append(text)
    return out[:3]


def writer_key_text(name):
    """The writer's name as compared: ar_key, letters and spaces only."""
    return " ".join("".join(c for c in w if c.isalpha()) for w in ar_key(name).split()).strip()


def close_names(a, b):
    """Two spellings that look like the same officer: "لحسن شميطي" / "لحسن اشميطي" (a letter added at the start
    of a word), or very close (85 % similar)."""
    if a == b:
        return False
    wa, wb = a.split(), b.split()
    if len(wa) == len(wb) and all(x == y or x.lstrip("ا") == y.lstrip("ا") for x, y in zip(wa, wb)):
        return True
    return Levenshtein.normalized_similarity(a, b) >= 0.85


# --- the child's names --------------------------------------------------------------------------------------
LATIN_LABELS = {"latin_first": re.compile(r"(?<![A-Za-zÀ-ÿ])PR[EÉ]NOMS?(?![A-Za-z])", re.I),
                "latin_family": re.compile(r"(?<![A-Za-zÀ-ÿ])NOMS?(?![A-Za-z])", re.I)}
ARABIC_LABELS = {"arabic_first": re.compile(r"ال[اأإآ]?سم\s*ال?شخص[يى]"),
                 "arabic_family": re.compile(r"ال[اأإآ]?سم\s*ال?عائل[يى]")}
LATIN_VALUE = re.compile(r"[\s:.\-]*(?P<value>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'\-]*)")
ARABIC_VALUE = re.compile(r"[\s:.\-ـ…]*(?P<value>[؀-ۿ][؀-ۿ .\-]*)")
EDGE = " .:-'ـ…"


def _span(text, label, value_pattern):
    """(start, end) of the value written after the label in one line, or None."""
    match = label.search(text)
    if not match:
        return None
    value = value_pattern.match(text, match.end())
    if not value:
        return ("empty", match.end())
    start, end = value.start("value"), value.end("value")
    while end > start and text[end - 1] in EDGE:
        end -= 1
    if end <= start:
        return ("empty", match.end())
    return start, end


def name_spans(lines):
    """The child's names in the lines of a page -> {field: (line index, start, end)}; a label found with nothing
    written after it is {field: (line index, None, None)}. The first line holding a label wins."""
    out = {}
    for index, text in enumerate(lines):
        for field, label in LATIN_LABELS.items():
            if field not in out:
                span = _span(text, label, LATIN_VALUE)
                if span:
                    out[field] = (index, None, None) if span[0] == "empty" else (index, *span)
        for field, label in ARABIC_LABELS.items():
            if field not in out:
                span = _span(text, label, ARABIC_VALUE)
                if span:
                    out[field] = (index, None, None) if span[0] == "empty" else (index, *span)
    return out


def carry_span(ref, pred, start, end):
    """The part of the predicted line aligned with ref[start:end]."""
    if not pred:
        return ""
    mapping = [None] * (len(ref) + 1)   # ref position -> pred position
    i = j = 0
    for op in Levenshtein.editops(ref, pred):
        while i < op.src_pos:
            mapping[i] = j
            i, j = i + 1, j + 1
        if op.tag == "insert":
            j += 1
        elif op.tag == "delete":
            mapping[i] = j
            i += 1
        else:
            mapping[i] = j
            i, j = i + 1, j + 1
    while i < len(ref):
        mapping[i] = j
        i, j = i + 1, j + 1
    mapping[len(ref)] = len(pred)
    a, b = mapping[start], mapping[end] if end < len(ref) else len(pred)
    return pred[a:b].strip(EDGE)


def name_key(field, text):
    """The normalised form of a name: case, spaces, dots and hyphens ignored; Arabic through ar_key."""
    if field.startswith("latin"):
        return "".join(c for c in unicodedata.normalize("NFKD", text).upper() if c.isalpha() and c.isascii())
    return "".join(c for c in ar_key(text) if c.isalpha())


def exact_key(text):
    return " ".join(text.split())


# --- the form rows ------------------------------------------------------------------------------------------
# the printed labels, in the order they are tried; the first one found earliest in the line names its row
FORM_ROWS = [
    ("Act number", ["رسم ولادة رقم", "رسم الولادة رقم"]),
    ("Page number", ["الصفحة رقم"]),
    ("Marginal data header", ["البيانات الهامشية"]),
    ("Date (Hijri)", ["في يوم"]),
    ("Year", ["عام"]),
    ("Date (Gregorian)", ["موافق"]),
    ("Time", ["على الساعة", "الدقيقة"]),
    ("Place of birth", ["ولد ب", "ولدت ب", "ولد في"]),
    ("First name", ["الاسم الشخصي", "PRENOM"]),
    ("Sex", ["ذكر", "انثى"]),
    ("Family name", ["الاسم العائلي", "NOM"]),
    ("Father", ["من والده", "FILS DE", "والده"]),
    ("Mother", ["والدته", "ET DE", "والدتها"]),
    ("Nationality", ["جنسيته", "جنسيتها"]),
    ("Parent's birthplace", ["المولودة ب", "المولود ب"]),
    ("Occupation", ["حرفته", "حرفتها"]),
    ("Address", ["الساكنان", "الساكن"]),
    ("Declarant", ["بناء على"]),
    ("Drawn up on", ["وحرر لدينا", "حرر لدينا", "وحرر"]),
    ("Officer", ["من طرفنا"]),
    ("Reading", ["وبعد تلاوة", "بعد تلاوة"]),
    ("Declarant's signature", ["توقيع المصرح"]),
]
_ROW_KEYS = [(row, [ar_key(k) if any(is_arabic(c) for c in k) else k for k in keys]) for row, keys in FORM_ROWS]


def form_row(text, page_type, previous):
    """The form row of a line: the label it holds (the earliest in the line); a line without a label continues
    the row of the line before on page 2 (a date written over two lines), and is a marginal note on page 1."""
    key, upper = ar_key(text), text.upper()
    best = None
    for row, labels in _ROW_KEYS:
        for label in labels:
            position = (upper if label.isascii() else key).find(label)
            if position >= 0 and position <= 30 and (best is None or position < best[0]):
                best = (position, row)
    if best:
        return best[1]
    if page_type == "1":
        return "Marginal notes" if previous else "Other"
    return previous or "Other"


def form_rows(lines, page_type):
    out, previous = [], None
    for text in lines:
        previous = form_row(text, page_type, previous)
        out.append(previous)
    return out