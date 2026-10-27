"""Metrics per page and their comparison: character error rate, words found, paired bootstrap."""

import random
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


def word_f1(truth_lines, predicted_lines):
    """Words found, whatever the order of the lines (a segmentation may order them differently), 0..1."""
    truth = Counter(" ".join(truth_lines).split())
    predicted = Counter(" ".join(predicted_lines).split())
    common = sum((truth & predicted).values())
    if not truth or not predicted:
        return 0.0
    precision, recall = common / sum(predicted.values()), common / sum(truth.values())
    return 0.0 if common == 0 else 2 * precision * recall / (precision + recall)


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
