"""The vocabulary of the dataset: every character of the transcriptions, with how often it appears.

The counts come from the PageXML of the pages (data/pages/<id>.xml). A page never changes once ingested, so the
characters of each page are counted once and kept in memory: a refresh only reads the pages added or changed
since (mlops.cli normalize rewrites pages).
"""

import html
import threading
import unicodedata
from collections import Counter

from mlops import db
from mlops.config import SPLITS, params, path
from mlops.pagexml import read_page
from mlops.splitting import coverage, describe

_cache = {}               # page id -> (modification time of its XML, Counter of its characters)
_lock = threading.Lock()


def page_characters(page_id):
    xml_path = path("data") / "pages" / f"{page_id}.xml"
    try:
        stamp = xml_path.stat().st_mtime_ns
    except OSError:
        return Counter()     # a missing file: counted as empty, tried again at the next refresh
    with _lock:
        cached = _cache.get(page_id)
        if cached and cached[0] == stamp:
            return cached[1]
    try:
        counts = Counter("".join(line["text"] for line in read_page(xml_path)["lines"]))
    except (OSError, ValueError):
        return Counter()
    with _lock:
        _cache[page_id] = (stamp, counts)
    return counts


def vocabulary(conn):
    """-> {split: Counter} for the pages of the dataset now."""
    by_split = {split: Counter() for split in SPLITS}
    for row in db.all_rows(conn, "SELECT id, split FROM pages ORDER BY id"):
        by_split[row["split"]].update(page_characters(row["id"]))
    return by_split


def page_types(conn):
    """-> {split: Counter {page type: pages}}."""
    out = {split: Counter() for split in SPLITS}
    for row in db.all_rows(conn, "SELECT split, page_type, count(*) AS n FROM pages GROUP BY 1, 2"):
        out[row["split"]][row["page_type"] or "?"] += row["n"]
    return out


def type_label(kind):
    labels = params()["ingest"].get("page_types") or {}
    return labels.get(kind, f"type {kind}")


def shown(char):
    """The character as it can be seen in a cell: spaces, controls and combining marks made visible."""
    category = unicodedata.category(char)
    if char == " ":
        return "␣"
    if category.startswith(("Z", "C")):
        return f"U+{ord(char):04X}"
    if category == "Mn":                 # Arabic diacritics (harakat, shadda ...): on a dotted circle
        return "◌" + char
    return char


def rare_threshold():
    return int((params().get("ui") or {}).get("rare_characters", 50))


def char_title(char, by_split):
    name = unicodedata.name(char, f"U+{ord(char):04X}")
    detail = " · ".join(f"{s} {by_split[s].get(char, 0)}" for s in SPLITS)
    missing = [s for s in SPLITS if not by_split[s].get(char)]
    return f"{name} (U+{ord(char):04X}) — {detail}" + (f" — not in {', '.join(missing)}" if missing else "")


def coverage_html(by_split, types):
    """The split summary: share of the characters, page types and characters missing, per split."""
    report = coverage(by_split, types)
    targets = {"train": 1 - params()["ingest"]["val_fraction"] - params()["ingest"]["test_fraction"],
               "val": params()["ingest"]["val_fraction"], "test": params()["ingest"]["test_fraction"]}
    kinds = sorted(set().union(*(set(t) for t in types.values())))
    head = "".join(f"<th>{html.escape(type_label(k))}</th>" for k in kinds)
    totals = Counter()
    for split in SPLITS:
        totals.update(types[split])
    rows = []
    for split in SPLITS:
        cells = "".join(f"<td>{types[split].get(k, 0)} <span class='voc-dim'>"
                        f"({100 * types[split].get(k, 0) / max(totals[k], 1):.0f} % of them)</span></td>" for k in kinds)
        missing = report["missing"][split]
        chips = "".join(f"<span class='voc-chip' dir='auto' title='{html.escape(char_title(c, by_split), quote=True)}'>"
                        f"{html.escape(shown(c))}</span>" for c in missing) or "<span class='voc-dim'>none</span>"
        rows.append(f"<tr><td><b>{split}</b></td><td>{100 * report['share'][split]:.1f} % "
                    f"<span class='voc-dim'>(target {100 * targets[split]:.0f} %)</span></td>{cells}"
                    f"<td>{len(missing)}</td><td>{chips}</td></tr>")
    return (f"<style>{VOCABULARY_CSS}</style><table class='voc-split'><thead><tr><th>split</th><th>characters</th>{head}<th>missing</th>"
            f"<th>characters missing from this split</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f"<div class='voc-line'>{html.escape(describe(report))}</div>")


def vocabulary_tsv(by_split, types=None):
    """The vocabulary as tab-separated text (spreadsheets): one character per row, its count per split."""
    lines = []
    if types:
        lines.append("# " + describe(coverage(by_split, types)))
    lines.append("\t".join(["character", "code", "name", "total", *SPLITS]))
    every = Counter()
    for counter in by_split.values():
        every.update(counter)
    for char in sorted(every):
        lines.append("\t".join([shown(char), f"U+{ord(char):04X}", unicodedata.name(char, ""), str(every[char]),
                                *(str(by_split[s].get(char, 0)) for s in SPLITS)]))
    return "\n".join(lines) + "\n"


def vocabulary_html(by_split, scope="all"):
    splits = SPLITS if scope == "all" else [scope]
    counts = Counter()
    for split in splits:
        counts.update(by_split.get(split) or {})
    if not counts:
        return "<p style='opacity:.7'>No transcribed pages in these pages yet.</p>"
    rare = rare_threshold()
    cells = []
    for char in sorted(counts):
        n = counts[char]
        missing = [s for s in SPLITS if not by_split[s].get(char)]
        classes = "voc-cell" + (" voc-rare" if n < rare else "") + (" voc-no-train" if "train" in missing else "")
        cells.append(f"<div class='{classes}' title='{html.escape(char_title(char, by_split), quote=True)}'>"
                     f"<span class='voc-char' dir='auto'>{html.escape(shown(char))}</span>"
                     f"<span class='voc-count'>{n}</span></div>")
    n_rare = sum(1 for n in counts.values() if n < rare)
    n_no_train = sum(1 for c in counts if not by_split["train"].get(c))
    summary = (f"<b>{len(counts)}</b> distinct characters, <b>{sum(counts.values())}</b> in total · "
               f"<span class='voc-rare-legend'><b>{n_rare}</b> seen fewer than {rare} times</span>")
    if n_no_train:
        summary += f" · <span class='voc-no-train-legend'><b>{n_no_train}</b> never seen in train</span>"
    tsv = html.escape(vocabulary_tsv({s: by_split.get(s) or Counter() for s in SPLITS}))
    return (f"<style>{VOCABULARY_CSS}</style><div class='voc-summary'>{summary}</div>"
            f"<div class='voc-grid'>{''.join(cells)}</div>"
            f"<textarea id='voc-tsv' hidden readonly>{tsv}</textarea>")


VOCABULARY_CSS = """
.voc-summary { margin: 4px 0 10px; opacity: .9; }
.voc-grid { column-width: 120px; column-gap: 24px; }
.voc-cell { display: flex; justify-content: space-between; align-items: baseline; break-inside: avoid;
            padding: 8px 12px; border-top: 1px solid var(--border-color-primary); }
.voc-char { font-size: 1.15em; min-width: 2.5em; text-align: left; unicode-bidi: isolate; }
.voc-count { font-variant-numeric: tabular-nums; }
.voc-rare .voc-count, .voc-rare-legend { color: #dc2626; }
.voc-no-train { outline: 2px dashed #f59e0b; outline-offset: -3px; }
.voc-no-train-legend { border-bottom: 2px dashed #f59e0b; }
.voc-split { width: 100%; border-collapse: collapse; margin: 4px 0 8px; }
.voc-split th, .voc-split td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border-color-primary); }
.voc-dim { opacity: .65; }
.voc-chip { display: inline-block; min-width: 1.6em; text-align: center; margin: 1px 3px; padding: 1px 6px;
            border: 1px solid #f59e0b; border-radius: 6px; unicode-bidi: isolate; }
.voc-line { font-family: var(--font-mono, monospace); font-size: .85em; opacity: .75; margin: 2px 0 14px;
            word-break: break-word; }
"""