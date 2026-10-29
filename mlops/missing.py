"""The acts with only one of their two pages annotated: which scans to annotate next.

    python -m mlops.cli missing-pages                   # per register: complete acts, acts missing page 1 / page 2
    python -m mlops.cli missing-pages --list            # + the file names of the missing pages
    python -m mlops.cli missing-pages --list > to_annotate.txt

A certificate has two pages, "<register>_<year>_<act>-1-1" (page 1: the names) and "<register>_<year>_<act>-2-1"
(page 2: the details) (params.yaml: ingest.page_type_pattern). When both pages of the acts of a register are
annotated, the register is about half page 1 and half page 2, and so is every split.

The file names come from the private table page_origins: this command prints them on the machine only (it is not a
job: nothing goes to the job logs, the app or W&B).
"""

import re
from collections import defaultdict
from pathlib import Path

from mlops import db
from mlops.config import params

TYPES = ("1", "2")


def act_of(stem, pattern):
    """"..._2004_1-2-1" -> ("..._2004_1-{}-1", "2"); None when the name has no page type."""
    match = re.search(pattern, stem) if pattern else None
    if not match:
        return None, None
    return stem[:match.start(1)] + "{}" + stem[match.end(1):], match.group(1)


def report(conn):
    """-> {register: {"split": ..., "complete": n, "missing": {"1": [file names], "2": [...]}, "other": n}}."""
    settings = params()["ingest"]
    type_pattern, group_pattern = settings.get("page_type_pattern"), settings.get("group_pattern")
    acts = defaultdict(dict)        # act template -> {type: split}
    others = defaultdict(int)
    for row in db.all_rows(conn, "SELECT o.original_name, p.split FROM page_origins o JOIN pages p ON p.id = o.page_id"):
        stem = Path(row["original_name"]).stem
        template, kind = act_of(stem, type_pattern)
        if template is None:
            others[stem] += 1
            continue
        acts[template][kind] = row["split"]
    out = {}
    for template, present in sorted(acts.items()):
        name = template.format(next(iter(present)))
        match = re.match(group_pattern, name) if group_pattern else None
        register = match.group(1) if match else name
        entry = out.setdefault(register, {"split": set(), "complete": 0, "missing": {t: [] for t in TYPES}})
        entry["split"].update(present.values())
        absent = [t for t in TYPES if t not in present]
        if not absent:
            entry["complete"] += 1
        for kind in absent:
            entry["missing"][kind].append(template.format(kind))
    for entry in out.values():
        entry["split"] = "/".join(sorted(entry["split"]))
    return out


def print_report(conn, with_names=False):
    data = report(conn)
    labels = params()["ingest"].get("page_types") or {}
    if not data:
        print("no page with a page type in its name (ingest.page_type_pattern)")
        return data
    width = max(len(r) for r in data)
    print(f'{"register":<{width}}  split  complete  missing {labels.get("1", "page 1")}  '
          f'missing {labels.get("2", "page 2")}')
    totals = {"complete": 0, "1": 0, "2": 0}
    for register, entry in data.items():
        m1, m2 = len(entry["missing"]["1"]), len(entry["missing"]["2"])
        totals["complete"] += entry["complete"]
        totals["1"] += m1
        totals["2"] += m2
        print(f'{register:<{width}}  {entry["split"]:<5}  {entry["complete"]:>8}  {m1:>15}  {m2:>17}')
    print(f'{"total":<{width}}         {totals["complete"]:>8}  {totals["1"]:>15}  {totals["2"]:>17}')
    if with_names:
        for register, entry in data.items():
            for kind in TYPES:
                if entry["missing"][kind]:
                    print(f'\n# {register} ({entry["split"]}): {labels.get(kind, "page " + kind)} to annotate')
                    print("\n".join(entry["missing"][kind]))
    else:
        print("\n--list prints the file names of the missing pages")
    return data