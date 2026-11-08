"""Metrics per page and their comparison: character error rate, words found, paired bootstrap."""

import random
import unicodedata
from collections import Counter

from rapidfuzz.distance import Levenshtein


def normalize(lines):
    """The lines of a page as one text: one space between words, one newline between lines."""
    return "\n".join(" ".join(line.split()) for line in lines if line.strip())


def cer(truth_lines, predicted_lines):
    """Character error rate of a page, in percent (edit distance / characters of the truth)."""
    truth, predicted = normalize(truth_lines), normalize(predicted_lines)
    if not truth:
        return 0.0 if not predicted else 100.0
    return 100.0 * Levenshtein.distance(truth, predicted) / len(truth)


def line_cer(truth_lines, predicted_lines):
    """Character error rate of a page whatever the order of the lines, in percent: every true line is paired
    with the predicted line closest to it (the closest pairs first); a line left without a pair counts all its
    characters as errors. A segmentation that orders the lines differently is not penalised; one that cuts,
    merges, misses or invents lines is."""
    truth = [" ".join(line.split()) for line in truth_lines if line.strip()]
    predicted = [" ".join(line.split()) for line in predicted_lines if line.strip()]
    characters = sum(len(t) for t in truth)
    if not characters:
        return 0.0 if not predicted else 100.0
    pairs = sorted((Levenshtein.distance(t, p), i, j) for i, t in enumerate(truth) for j, p in enumerate(predicted))
    used_t, used_p, errors = set(), set(), 0
    for distance, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        errors += distance
    errors += sum(len(t) for i, t in enumerate(truth) if i not in used_t)
    errors += sum(len(p) for j, p in enumerate(predicted) if j not in used_p)
    return 100.0 * errors / characters


def word_f1(truth_lines, predicted_lines):
    """Words found, whatever the order of the lines (a segmentation may order them differently), 0..1."""
    truth = Counter(" ".join(truth_lines).split())
    predicted = Counter(" ".join(predicted_lines).split())
    common = sum((truth & predicted).values())
    if not truth or not predicted:
        return 0.0
    precision, recall = common / sum(predicted.values()), common / sum(truth.values())
    return 0.0 if common == 0 else 2 * precision * recall / (precision + recall)


def group_bootstrap(gains, groups, samples=2000, confidence=0.95, seed=0):
    """Paired bootstrap by group -> (mean, low, high). gains: {page: gain}; groups: {page: its group}. The groups
    are drawn again with replacement (all the pages of a drawn group together), so pages that look alike are
    not counted as independent evidence. The mean is the mean over the pages, as in bootstrap()."""
    if not gains:
        return 0.0, 0.0, 0.0
    by_group = {}
    for page, gain in gains.items():
        by_group.setdefault(groups.get(page, page), []).append(gain)
    totals = [(sum(values), len(values)) for values in by_group.values()]
    rng = random.Random(seed)
    k = len(totals)
    means = []
    for _ in range(samples):
        drawn = [totals[rng.randrange(k)] for _ in range(k)]
        means.append(sum(t for t, _ in drawn) / sum(n for _, n in drawn))
    means.sort()
    tail = (1 - confidence) / 2
    return (sum(gains.values()) / len(gains), means[int(tail * samples)],
            means[min(samples - 1, int((1 - tail) * samples))])


def bootstrap(gains, samples=2000, confidence=0.95, seed=0):
    """Paired bootstrap of the mean gain per page -> (mean, low, high). gains[i]: how much better the candidate
    is on page i (positive: better). The pages are drawn again with replacement `samples` times."""
    if not gains:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(gains)
    means = sorted(sum(gains[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    tail = (1 - confidence) / 2
    low = means[int(tail * samples)]
    high = means[min(samples - 1, int((1 - tail) * samples))]
    return sum(gains) / n, low, high

# --- two CERs, error composition, intervals by writer ------------------------------------------------------
CATEGORIES = ("letters", "digits", "spaces", "punctuation")
OPERATIONS = ("substitutions", "deletions", "insertions")


def category(char):
    """letters (with Arabic diacritics), digits, spaces, punctuation (and every other sign)."""
    kind = unicodedata.category(char)
    if char.isspace() or kind.startswith("Z"):
        return "spaces"
    if kind.startswith(("L", "M")):
        return "letters"
    if kind.startswith("N"):
        return "digits"
    return "punctuation"


def without_spaces_and_punctuation(text):
    """The text the second CER is computed on: letters and digits only."""
    return "".join(c for c in text if category(c) in ("letters", "digits"))


def line_errors(ref, pred):
    """One line -> {"chars", "errors", "chars_lp", "errors_lp", "composition": {operation: {category: n}},
    "words", "word_errors", "exact"}. "_lp": without spaces and punctuation. A deletion is a character of the
    reference the model missed, an insertion a character it added (the usual names; ketos test swaps them)."""
    composition = {op: dict.fromkeys(CATEGORIES, 0) for op in OPERATIONS}
    for op in Levenshtein.editops(ref, pred):
        if op.tag == "replace":
            composition["substitutions"][category(ref[op.src_pos])] += 1
        elif op.tag == "delete":
            composition["deletions"][category(ref[op.src_pos])] += 1
        else:
            composition["insertions"][category(pred[op.dest_pos])] += 1
    ref_lp, pred_lp = without_spaces_and_punctuation(ref), without_spaces_and_punctuation(pred)
    ref_words, pred_words = ref.split(), pred.split()
    return {"chars": len(ref), "errors": Levenshtein.distance(ref, pred), "chars_lp": len(ref_lp),
            "errors_lp": Levenshtein.distance(ref_lp, pred_lp), "composition": composition,
            "words": len(ref_words), "word_errors": Levenshtein.distance(ref_words, pred_words),
            "exact": int(ref == pred)}


def ratio_interval(units, groups=None, samples=2000, confidence=0.95, seed=0):
    """A ratio metric (CER: errors / characters) and its bootstrap interval -> (value, low, high).
    units: {unit: (numerator, denominator)}; groups: {unit: its group} (the writer): the GROUPS are drawn again
    with replacement, all their units together, so the pages of one writer count as one piece of evidence.
    Without groups, the units (pages, acts) are drawn."""
    if not units:
        return None, None, None
    by_group = {}
    for unit, (num, den) in units.items():
        key = (groups or {}).get(unit, unit)
        total = by_group.setdefault(key, [0.0, 0.0])
        total[0] += num
        total[1] += den
    totals = list(by_group.values())
    all_num, all_den = sum(t[0] for t in totals), sum(t[1] for t in totals)
    value = all_num / all_den if all_den else None
    rng = random.Random(seed)
    k = len(totals)
    estimates = []
    for _ in range(samples):
        drawn = [totals[rng.randrange(k)] for _ in range(k)]
        den = sum(d for _, d in drawn)
        if den:
            estimates.append(sum(n for n, _ in drawn) / den)
    if not estimates or value is None:
        return value, None, None
    estimates.sort()
    tail = (1 - confidence) / 2
    n = len(estimates)
    return value, estimates[int(tail * n)], estimates[min(n - 1, int((1 - tail) * n))]