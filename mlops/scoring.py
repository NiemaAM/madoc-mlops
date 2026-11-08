"""Every number of an evaluation, from the recognised lines of the test pages.

Input: one row per line, {"page": page id, "line": index of the true line (None for a predicted line that matches
no true line), "ref": true text, "pred": recognised text}, both in logical order (what MaDoc reads); and, per
page, its writer (anonymous label), its page type and its true lines.

Output (numbers, characters, anonymous labels and printed form labels only: safe for W&B and the database):
  - CER (with spaces and punctuation), CER_lp (letters and digits only), WER, line exact match, CER per script;
  - each with its 95 % interval, by writer (the writers are drawn again: pages of one hand are one piece of
    evidence) when there are 2 writers or more, else by page;
  - the error composition (substitutions / deletions = missed / insertions = extra, by letters, digits, spaces,
    punctuation) and the character confusions;
  - the child's names (page 1 and page 2): exact match, normalised match and CER per field;
  - the CER per form row and per writer (the heatmaps);
  - the CER of every page (the gate compares them page by page).
"""

from collections import Counter, defaultdict

from rapidfuzz.distance import Levenshtein

from mlops import fields
from mlops.config import params
from mlops.textmetrics import CATEGORIES, OPERATIONS, line_errors, ratio_interval


def script(char):
    if fields.is_arabic(char) and (char.isalpha() or "ً" <= char <= "ٟ"):
        return "Arabic"
    if fields.is_latin(char):
        return "Latin"
    return "Common"


def interval_settings():
    gate = params().get("gate") or {}
    return gate.get("bootstrap_samples", 2000), gate.get("confidence", 0.95)


def with_interval(units, groups):
    """(value in %, low, high) of a ratio, by writer when 2 writers or more, else by unit."""
    samples, confidence = interval_settings()
    use_groups = groups if len({groups.get(u) for u in units}) >= 2 else None
    value, low, high = ratio_interval(units, use_groups, samples, confidence)
    scale = lambda v: None if v is None else round(100 * v, 3)   # noqa: E731
    return scale(value), scale(low), scale(high)


def score(rows, pages, model="model"):
    """rows: [{"page", "line", "ref", "pred"}]; pages: {page id: {"writer", "page_type", "lines"}} -> summary."""
    totals = Counter()
    composition = {op: dict.fromkeys(CATEGORIES, 0) for op in OPERATIONS}
    confusions = Counter()
    per_page = defaultdict(Counter)
    by_script = defaultdict(Counter)       # script -> chars / errors, per page
    form = defaultdict(Counter)            # (row, page type) -> chars / errors
    rows_of_page = {pid: fields.form_rows(meta["lines"], meta["page_type"]) for pid, meta in pages.items()}
    for row in rows:
        ref, pred = row["ref"] or "", row["pred"] or ""
        e = line_errors(ref, pred)
        page = per_page[row["page"]]
        for key in ("chars", "errors", "chars_lp", "errors_lp", "words", "word_errors"):
            page[key] += e[key]
        if ref:
            page["lines"] += 1
            page["exact"] += e["exact"]
        for op in OPERATIONS:
            for cat in CATEGORIES:
                composition[op][cat] += e["composition"][op][cat]
        for op in Levenshtein.editops(ref, pred):
            if op.tag == "replace":
                confusions[(ref[op.src_pos], pred[op.dest_pos])] += 1
                by_script[(row["page"], script(ref[op.src_pos]))]["errors"] += 1
            elif op.tag == "delete":
                confusions[(ref[op.src_pos], "")] += 1
                by_script[(row["page"], script(ref[op.src_pos]))]["errors"] += 1
            else:
                confusions[("", pred[op.dest_pos])] += 1
        for char in ref:
            by_script[(row["page"], script(char))]["chars"] += 1
        meta = pages.get(row["page"])
        if meta and row.get("line") is not None and ref:
            form_row = rows_of_page[row["page"]][row["line"]]
            form[(form_row, meta["page_type"])]["chars"] += e["chars"]
            form[(form_row, meta["page_type"])]["errors"] += e["errors"]
    writer_of = {pid: meta.get("writer") or "unknown" for pid, meta in pages.items()}
    for page in per_page:
        totals.update(per_page[page])
    metrics, ci = {}, {}
    for name, num, den in (("cer", "errors", "chars"), ("cer_lp", "errors_lp", "chars_lp"),
                           ("wer", "word_errors", "words")):
        units = {p: (c[num], c[den]) for p, c in per_page.items() if c[den]}
        value, low, high = with_interval(units, writer_of)
        metrics[name], ci[name] = value, [low, high]
    units = {p: (c["lines"] - c["exact"], c["lines"]) for p, c in per_page.items() if c["lines"]}
    value, low, high = with_interval(units, writer_of)
    metrics["line_exact"] = None if value is None else round(100 - value, 3)
    ci["line_exact"] = [None if high is None else round(100 - high, 3), None if low is None else round(100 - low, 3)]
    for name in ("Arabic", "Latin", "Common"):
        units = {p: (c["errors"], c["chars"]) for (p, s), c in by_script.items() if s == name and c["chars"]}
        if units:
            value, low, high = with_interval(units, writer_of)
            metrics[f"cer_{name.lower()}"], ci[f"cer_{name.lower()}"] = value, [low, high]
    writers = defaultdict(Counter)
    for page, c in per_page.items():
        writers[writer_of.get(page, "unknown")].update(c)
        writers[writer_of.get(page, "unknown")]["pages"] += 1
    names = score_names(rows, pages, writer_of)
    return {
        "model": model, "pages": len(per_page), "lines": totals["lines"], "chars": totals["chars"],
        "metrics": metrics, "ci": ci,
        "interval": "by writer" if len(set(writer_of.get(p) for p in per_page)) >= 2 else "by page",
        "writers_count": len(set(writer_of.get(p) for p in per_page)),
        "composition": composition,
        "confusions": [[r, p, n] for (r, p), n in confusions.most_common(400)],
        "names": names,
        "form_rows": {f"{r}|{t}": round(100 * c["errors"] / c["chars"], 3) for (r, t), c in form.items() if c["chars"]},
        "writers": {w: {"pages": c["pages"], "chars": c["chars"],
                        "cer": round(100 * c["errors"] / c["chars"], 3) if c["chars"] else None,
                        "cer_lp": round(100 * c["errors_lp"] / c["chars_lp"], 3) if c["chars_lp"] else None}
                    for w, c in sorted(writers.items())},
        "per_page": {p: {"cer": round(100 * c["errors"] / c["chars"], 4) if c["chars"] else 0.0,
                         "cer_lp": round(100 * c["errors_lp"] / c["chars_lp"], 4) if c["chars_lp"] else 0.0,
                         "chars": c["chars"], "errors": c["errors"],
                         "page_type": (pages.get(p) or {}).get("page_type")} for p, c in per_page.items()},
    }


def score_names(rows, pages, writer_of):
    """The child's names on page 1 and page 2: exact, normalised, CER, per field and page type, with intervals."""
    predicted = {(r["page"], r["line"]): r["pred"] or "" for r in rows if r.get("line") is not None}
    cells = defaultdict(dict)   # "<field>|<page type>" -> {unit: (exact, normalized, errors, chars)}
    missing = Counter()
    for page_id, meta in pages.items():
        if meta["page_type"] not in ("1", "2"):
            continue
        for field, (index, start, end) in fields.name_spans(meta["lines"]).items():
            if start is None:
                missing[f"{field}|{meta['page_type']}"] += 1
                continue
            truth = meta["lines"][index][start:end]
            if (page_id, index) not in predicted:
                continue
            guess = fields.carry_span(meta["lines"][index], predicted[(page_id, index)], start, end)
            cells[f"{field}|{meta['page_type']}"][page_id] = (
                int(fields.exact_key(truth) == fields.exact_key(guess)),
                int(fields.name_key(field, truth) == fields.name_key(field, guess)),
                Levenshtein.distance(truth, guess), len(truth))
    out = {}
    pooled = defaultdict(dict)
    for key, units in sorted(cells.items()):
        exact = with_interval({u: (1 - v[0], 1) for u, v in units.items()}, writer_of)
        normalized = with_interval({u: (1 - v[1], 1) for u, v in units.items()}, writer_of)
        cer = with_interval({u: (v[2], v[3]) for u, v in units.items() if v[3]}, writer_of)
        out[key] = {"names": len(units), "exact": round(100 - exact[0], 2),
                    "exact_ci": [round(100 - exact[2], 2), round(100 - exact[1], 2)] if exact[1] is not None else None,
                    "normalized": round(100 - normalized[0], 2),
                    "normalized_ci": ([round(100 - normalized[2], 2), round(100 - normalized[1], 2)]
                                      if normalized[1] is not None else None),
                    "cer": cer[0], "cer_ci": [cer[1], cer[2]], "label_without_name": missing.get(key, 0)}
        for unit, values in units.items():
            pooled[unit + "|" + key] = values
    if pooled:
        groups = {u: writer_of.get(u.split("|")[0]) for u in pooled}
        exact = with_interval({u: (1 - v[0], 1) for u, v in pooled.items()}, groups)
        normalized = with_interval({u: (1 - v[1], 1) for u, v in pooled.items()}, groups)
        cer = with_interval({u: (v[2], v[3]) for u, v in pooled.items() if v[3]}, groups)
        out["all"] = {"names": len(pooled), "exact": round(100 - exact[0], 2), "normalized": round(100 - normalized[0], 2),
                      "cer": cer[0], "cer_ci": [cer[1], cer[2]],
                      "exact_ci": [round(100 - exact[2], 2), round(100 - exact[1], 2)] if exact[1] is not None else None,
                      "normalized_ci": ([round(100 - normalized[2], 2), round(100 - normalized[1], 2)]
                                        if normalized[1] is not None else None),
                      "label_without_name": sum(missing.values())}
    return out


def flat_metrics(summary, prefix=""):
    """The headline numbers of a summary, flat (W&B, the evaluations table, the model card)."""
    out = {}
    for key, value in (summary.get("metrics") or {}).items():
        if value is not None:
            out[prefix + key] = value
            low, high = (summary.get("ci") or {}).get(key, [None, None])
            if low is not None:
                out[prefix + key + "_low"], out[prefix + key + "_high"] = low, high
    names = (summary.get("names") or {}).get("all")
    if names:
        out[prefix + "names_exact"] = names["exact"]
        out[prefix + "names_normalized"] = names["normalized"]
        if names.get("cer") is not None:
            out[prefix + "names_cer"] = names["cer"]
    return out


def cer_of(summary, page_type=None):
    """The CER (%) of the pages of one type (all pages: None), from a summary."""
    pages = [v for v in summary["per_page"].values() if page_type is None or v.get("page_type") == page_type]
    chars = sum(v["chars"] for v in pages)
    return round(100 * sum(v["errors"] for v in pages) / chars, 3) if chars else None