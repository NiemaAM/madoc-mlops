"""The split of the groups (register + year) into train / val / test, balanced on the characters and the page types.

A group is never cut: all its pages go to the same split (no near-copy of a train page in test; the two pages of
an act, page 1 with the names and page 2 with the details, are in the same register and year, so never split
apart). Among the ways to place the groups, the one chosen minimises, in this order of priority:

1. a split outside its share of the characters by more than the tolerance (ingest.split_tolerance: 0.33 lets a
   15 % split hold 10 % to 20 % of the characters), a split left empty, or a page type missing from a split;
2. the characters missing from a split: every character of the dataset should be in train, val AND test (one
   missing from train counts 3 times: the model cannot learn it), and each split should hold its share of the
   pages of each type (val 15 % of the page 2, test 15 % of the page 2 ...). The page 2 weighs most: MaDoc reads
   its text and structures it, so the evaluation must be on its share of them (ingest.split_weights);
3. the distance of each split to its share of the characters (70 / 15 / 15 of all the characters);
4. the difference between the character distribution of each split and the one of the whole dataset (total
   variation distance), so the rare and frequent characters are in the same proportions everywhere.

A character found in fewer groups than there are splits cannot be in all of them: the search then keeps it in
train first. The search is a local search (move one group, or swap two groups between splits, while the cost
goes down) from the size-balanced placement and from a few random placements with a fixed seed: the result
depends on the data only.
"""

from collections import Counter

import numpy as np

SPLITS = ("train", "val", "test")
MISSING_WEIGHT = {"train": 3.0, "val": 1.0, "test": 1.0}
W_OUTSIDE, W_EMPTY, W_TYPE_MISSING, W_SIZE, W_DISTRIBUTION = 1e6, 1e6, 1e5, 1e2, 1e1
W_TYPE_OUTSIDE = 10.0      # a split outside the tolerance of its share of a page type: 10 times the distance
# priority 2, set in params.yaml (ingest.split_weights): a missing character, and 1.0 of relative distance of a
# split to its share of a page type (test with 30 % of the page 2 instead of 15 %: 1.0)
DEFAULT_WEIGHTS = {"missing_character": 1000, "page_types": {"1": 1000, "2": 3000}, "other_page_type": 1000}
RANDOM_STARTS = 8


class Problem:
    def __init__(self, fixed, free, val_fraction, test_fraction, tolerance, fixed_types=None, free_types=None,
                 weights=None):
        """fixed / free: Counters of characters; fixed_types / free_types: Counters of page types (optional).
        The type counts are extra columns of the same matrices (after the C characters)."""
        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        fixed_types, free_types = fixed_types or {}, free_types or {}
        self.groups = sorted(free)
        chars = sorted(set().union(*(set(c) for c in free.values()), *(set(c) for c in fixed.values())))
        types = sorted(set().union(*(set(c) for c in free_types.values()), *(set(c) for c in fixed_types.values())))
        index = {("c", c): i for i, c in enumerate(chars)}
        index.update({("t", t): len(chars) + i for i, t in enumerate(types)})
        self.chars, self.types, self.C = chars, types, len(chars)
        self.M = np.zeros((len(self.groups), len(chars) + len(types)))
        for g, group in enumerate(self.groups):
            for char, n in free[group].items():
                self.M[g, index[("c", char)]] = n
            for kind, n in (free_types.get(group) or {}).items():
                self.M[g, index[("t", kind)]] = n
        self.F = np.zeros((3, self.M.shape[1]))
        for s, split in enumerate(SPLITS):
            for char, n in (fixed.get(split) or {}).items():
                self.F[s, index[("c", char)]] = n
            for kind, n in (fixed_types.get(split) or {}).items():
                self.F[s, index[("t", kind)]] = n
        all_counts = self.F.sum(0) + self.M.sum(0)
        chars_total, types_total = all_counts[:self.C], all_counts[self.C:]
        self.T = max(chars_total.sum(), 1.0)
        self.P = chars_total / self.T
        self.present = chars_total > 0
        self.type_totals = np.maximum(types_total, 1.0)
        self.present_types = types_total > 0
        type_weights = {str(k): float(v) for k, v in (weights.get("page_types") or {}).items()}
        self.type_weight = np.array([type_weights.get(t, float(weights["other_page_type"])) for t in types])
        self.w_missing = float(weights["missing_character"])
        self.share = np.array([1 - val_fraction - test_fraction, val_fraction, test_fraction])
        self.tolerance = tolerance
        self.weight = np.array([MISSING_WEIGHT[s] for s in SPLITS])

    def split_cost(self, rows, s):
        """Cost of split s with the counts `rows` (..., C + types) -> (...)."""
        share = self.share[s]
        chars, types = rows[..., :self.C], rows[..., self.C:]
        n = chars.sum(-1)
        fraction = n / self.T
        cost = W_SIZE * np.abs(fraction - share)
        if share > 0:
            outside = np.maximum(np.abs(fraction - share) - self.tolerance * share, 0) / share
            cost = cost + W_OUTSIDE * outside + W_EMPTY * (n == 0)
            cost = cost + self.w_missing * self.weight[s] * ((chars == 0) & self.present).sum(-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                distribution = np.where(n[..., None] > 0, chars / n[..., None], 0)
            cost = cost + W_DISTRIBUTION * 0.5 * np.abs(distribution - self.P).sum(-1) * (n > 0)
            if types.shape[-1]:
                cost = cost + W_TYPE_MISSING * ((types == 0) & self.present_types).sum(-1)
                # the share of the pages of each type held by this split, against the split's share
                distance = np.abs(types / self.type_totals - share) / share * self.present_types
                # outside the tolerance (as for the characters): 10x, so a fair share of each type comes first
                outside_types = np.maximum(distance - self.tolerance, 0)
                cost = cost + ((distance + W_TYPE_OUTSIDE * outside_types) * self.type_weight).sum(-1)
        return cost

    def totals(self, assign):
        S = self.F.copy()
        for s in range(3):
            S[s] += self.M[assign == s].sum(0)
        return S

    def cost(self, assign):
        S = self.totals(assign)
        return float(sum(self.split_cost(S[s], s) for s in range(3)))

    def improve(self, assign):
        """Local search: the best move or swap while one lowers the cost."""
        assign = assign.copy()
        S = self.totals(assign)
        current = np.array([self.split_cost(S[s], s) for s in range(3)])
        while True:
            best, action = 1e-9, None
            for a in range(3):                       # move one group from a to b
                members = np.flatnonzero(assign == a)
                if not len(members):
                    continue
                without = self.split_cost(S[a] - self.M[members], a)
                for b in range(3):
                    if b == a:
                        continue
                    gain = current[a] + current[b] - without - self.split_cost(S[b] + self.M[members], b)
                    k = int(np.argmax(gain))
                    if gain[k] > best:
                        best, action = gain[k], ("move", members[k], a, b)
            for a in range(3):                       # swap a group of a with a group of b
                for b in range(a + 1, 3):
                    ga, gb = np.flatnonzero(assign == a), np.flatnonzero(assign == b)
                    if not len(ga) or not len(gb):
                        continue
                    block = max(1, int(2e6 // (len(gb) * max(self.M.shape[1], 1))))   # bounded memory
                    for start in range(0, len(ga), block):
                        part = ga[start:start + block]
                        delta = self.M[gb][None, :, :] - self.M[part][:, None, :]      # (|part|, |gb|, C)
                        gain = (current[a] + current[b] - self.split_cost(S[a] + delta, a)
                                - self.split_cost(S[b] - delta, b))
                        i, j = np.unravel_index(int(np.argmax(gain)), gain.shape)
                        if gain[i, j] > best:
                            best, action = gain[i, j], ("swap", part[i], gb[j], a, b)
            if action is None:
                return assign
            if action[0] == "move":
                _, g, a, b = action
                assign[g] = b
                S[a] -= self.M[g]
                S[b] += self.M[g]
            else:
                _, g, h, a, b = action
                assign[g], assign[h] = b, a
                S[a] += self.M[h] - self.M[g]
                S[b] += self.M[g] - self.M[h]
            current = np.array([self.split_cost(S[s], s) for s in range(3)])

    def by_size(self):
        """The largest group first, to the split furthest below its share of the characters."""
        counts = self.F[:, :self.C].sum(1).copy()
        sizes = self.M[:, :self.C].sum(1)
        assign = np.zeros(len(self.groups), dtype=int)
        for g in sorted(range(len(self.groups)), key=lambda g: (-sizes[g], self.groups[g])):
            s = max((2, 1, 0), key=lambda s: self.share[s] * self.T - counts[s])   # ties: test, val, train
            assign[g] = s
            counts[s] += sizes[g]
        return assign

    def solve(self):
        if not self.groups:
            return np.zeros(0, dtype=int)
        rng = np.random.default_rng(0)
        starts = [self.by_size()] + [rng.choice(3, size=len(self.groups), p=self.share) for _ in range(RANDOM_STARTS)]
        results = [self.improve(start) for start in starts]
        costs = [self.cost(r) for r in results]
        return results[int(np.argmin(costs))]


def assign_groups(fixed, free, val_fraction, test_fraction, tolerance=0.33, fixed_types=None, free_types=None,
                  weights=None):
    """fixed: {split: Counter of the characters already there}; free: {group: Counter of its characters};
    fixed_types / free_types: the same with the page types (Counter {"1": pages, "2": pages}); weights:
    ingest.split_weights. -> {group: split}."""
    problem = Problem(fixed, free, val_fraction, test_fraction, tolerance, fixed_types, free_types, weights)
    assign = problem.solve()
    return {group: SPLITS[s] for group, s in zip(problem.groups, assign)}


def coverage(by_split, types_by_split=None):
    """{split: Counter of characters} (+ {split: Counter of page types}) -> counts, shares and characters missing
    per split (characters and page types are not personal data)."""
    every = Counter()
    for counter in by_split.values():
        every.update(counter)
    total = sum(every.values()) or 1
    missing = {s: "".join(sorted(c for c in every if not by_split.get(s, {}).get(c))) for s in SPLITS}
    report = {
        "characters": len(every),
        "in_all_splits": sum(1 for c in every if all(by_split.get(s, {}).get(c) for s in SPLITS)),
        "share": {s: round(sum((by_split.get(s) or {}).values()) / total, 3) for s in SPLITS},
        "missing": missing,
    }
    if types_by_split:
        report["page_types"] = {s: dict(sorted((types_by_split.get(s) or {}).items())) for s in SPLITS}
    return report


def describe(report):
    missing = ", ".join(f"{s} {len(report['missing'][s])} ({report['missing'][s]!r})" if report["missing"][s]
                        else f"{s} 0" for s in SPLITS)
    shares = " / ".join(f"{100 * report['share'][s]:.1f} %" for s in SPLITS)
    text = (f"characters: {report['in_all_splits']}/{report['characters']} in every split; "
            f"share of the characters train / val / test {shares}; missing: {missing}")
    if report.get("page_types"):
        totals = Counter()
        for s in SPLITS:
            totals.update(report["page_types"][s])
        text += "; page types: " + ", ".join(
            f"{s} " + " + ".join(f"{n} type {t} ({100 * n / max(totals[t], 1):.0f} % of them)"
                                 for t, n in report["page_types"][s].items()) for s in SPLITS)
    return text
