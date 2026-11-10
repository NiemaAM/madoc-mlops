"""The writer of every page: the civil-status officer named in its act, the unit of the writer-independent split.

Page 2 of an act names the officer who drew it up ("من طرفنا نحن لحسن شميطي موظف جماعي"): the name is read from
its transcription (mlops/fields.py) and kept, as written, in the private table page_writers. Page 1 has no such
line: it takes the writer of page 2 of the same act, so both pages of an act always have the same writer, hence
the same split. A page whose act has no page 2 (yet) has no writer: it is trained on, never tested.

The same officer is sometimes spelt two ways ("لحسن شميطي", "لحسن اشميطي"). The aliases, written in the app,
say which spellings are one person (table writer_aliases); until then every spelling counts as its own writer,
and the app lists the spellings that look alike, because two spellings of one hand put in train and test would
leak that hand into the test.

What leaves the private tables is the keyed hash of the canonical name (pages.writer_key) and, for W&B and the
figures, an anonymous label "W-<6 characters of the hash>".
"""

from collections import Counter, defaultdict
from pathlib import Path

from mlops import db
from mlops.config import keyed, params, path
from mlops.fields import close_names, officer_lines, writer_key_text, writer_name
from mlops.pagexml import read_page


def label(writer_key):
    """The anonymous name of a writer, for W&B and the figures."""
    return f"W-{writer_key[:6]}" if writer_key else "unknown"


def act_template(stem):
    """"..._2004_1-2-1" -> "..._2004_1-{}-1": the same for both pages of the act (ingest.page_type_pattern)."""
    import re
    pattern = params()["ingest"].get("page_type_pattern")
    match = re.search(pattern, stem) if pattern else None
    return stem[:match.start(1)] + "{}" + stem[match.end(1):] if match else stem


def act_key(original_name):
    return keyed("act:" + act_template(Path(original_name).stem))


def record(conn, page_id, lines):
    """Read the officer's name in the lines of a page and keep it (an empty name: none written)."""
    conn.execute("INSERT INTO page_writers (page_id, raw_name) VALUES (%s, %s) ON CONFLICT (page_id) DO UPDATE "
                 "SET raw_name = EXCLUDED.raw_name", (page_id, writer_name(lines)))


def rekey(conn):
    """The act and the register-year of every page computed again from its file name -> pages fixed. A key made
    another way (a process that had no MLOPS_SECRET, an older version, a page ingested before acts existed) would
    keep page 1 apart from its page 2: this puts them together again."""
    from mlops.ingest import group_of
    pattern = params()["ingest"]["group_pattern"]
    fixed = 0
    for row in db.all_rows(conn, "SELECT p.id, p.act_key, p.group_key, o.original_name FROM pages p "
                                 "JOIN page_origins o ON o.page_id = p.id"):
        act, group = act_key(row["original_name"]), group_of(Path(row["original_name"]).stem, pattern)
        if (act, group) != (row["act_key"], row["group_key"]):
            conn.execute("UPDATE pages SET act_key = %s, group_key = %s WHERE id = %s", (act, group, row["id"]))
            fixed += 1
    return fixed


def backfill(conn):
    """The act and group keys checked (rekey), and the officer of the pages never read -> pages read."""
    rekey(conn)
    # never read, or read without a name (read again: the reading may have improved since)
    rows = db.all_rows(conn, "SELECT p.id FROM pages p LEFT JOIN page_writers w ON w.page_id = p.id "
                             "WHERE w.page_id IS NULL OR (w.raw_name = '' AND p.page_type = '2')")
    for row in rows:
        try:
            lines = [line["text"] for line in read_page(path("data") / "pages" / f'{row["id"]}.xml')["lines"]]
        except (OSError, ValueError):
            lines = []
        record(conn, row["id"], lines)
    return len(rows)


def aliases(conn):
    return {r["variant"]: r["canonical"] for r in db.all_rows(conn, "SELECT variant, canonical FROM writer_aliases")}


def resolve(name, alias_map):
    """The canonical spelling of a name (aliases followed, at most 10 steps: a loop cannot hang)."""
    seen = 0
    while name in alias_map and seen < 10:
        name, seen = alias_map[name], seen + 1
    return name


def page_names(conn):
    """{page id: canonical name of the officer written on it} (pages without a name are left out)."""
    alias_map = aliases(conn)
    out = {}
    for row in db.all_rows(conn, "SELECT page_id, raw_name FROM page_writers"):
        key = writer_key_text(row["raw_name"])
        if key:
            out[row["page_id"]] = resolve(key, alias_map)
    return out


def sync(conn):
    """pages.writer_key from the officer of page 2 of each act (all the pages of the act get it) -> counts."""
    backfill(conn)
    names = page_names(conn)
    pages = db.all_rows(conn, "SELECT id, act_key, page_type FROM pages WHERE deleted_at IS NULL")
    by_act = defaultdict(Counter)
    for row in pages:
        if row["id"] in names:
            # page 2 names the officer of the act; a name read on another page counts less
            by_act[row["act_key"] or row["id"]][names[row["id"]]] += 10 if row["page_type"] == "2" else 1
    changed = 0
    with conn.transaction():
        for row in pages:
            votes = by_act.get(row["act_key"] or row["id"])
            key = keyed("writer:" + votes.most_common(1)[0][0]) if votes else None
            cursor = conn.execute("UPDATE pages SET writer_key = %s WHERE id = %s AND writer_key IS DISTINCT FROM %s",
                                  (key, row["id"], key))
            changed += cursor.rowcount
    known = sum(1 for row in pages if by_act.get(row["act_key"] or row["id"]))
    return {"pages": len(pages), "with_writer": known, "writers": len({n for c in by_act.values() for n in c}),
            "changed": changed}


def add_alias(conn, variant, canonical):
    """Say that `variant` is another spelling of `canonical` (both as shown in the app)."""
    variant, canonical = writer_key_text(variant), writer_key_text(canonical)
    if not variant or not canonical:
        raise ValueError("both names are needed")
    if variant == canonical:
        raise ValueError("the two names are the same")
    if resolve(canonical, aliases(conn)) == variant:
        raise ValueError(f"{canonical} is already an alias of {variant}: that would make a loop")
    conn.execute("INSERT INTO writer_aliases (variant, canonical) VALUES (%s, %s) ON CONFLICT (variant) DO UPDATE "
                 "SET canonical = EXCLUDED.canonical", (variant, canonical))
    return sync(conn)


def remove_alias(conn, variant):
    conn.execute("DELETE FROM writer_aliases WHERE variant = %s", (writer_key_text(variant),))
    return sync(conn)


def table(conn):
    """One row per writer, the fewest pages first: what to annotate next. Private: the names are shown."""
    names = page_names(conn)
    rows = db.all_rows(conn, "SELECT id, act_key, writer_key, page_type, split, held_out FROM pages "
                             "WHERE deleted_at IS NULL")
    act_pages = defaultdict(set)
    for row in rows:
        act_pages[row["act_key"] or row["id"]].add(row["page_type"])
    by_key = defaultdict(lambda: {"pages": 0, "acts": set(), "page1": 0, "page2": 0, "splits": Counter(),
                                  "held_out": 0, "spellings": Counter(), "incomplete": set()})
    for row in rows:
        entry = by_key[row["writer_key"]]
        entry["pages"] += 1
        act = row["act_key"] or row["id"]
        entry["acts"].add(act)
        if row["page_type"] == "1":
            entry["page1"] += 1
        elif row["page_type"] == "2":
            entry["page2"] += 1
        entry["splits"][row["split"]] += 1
        entry["held_out"] += int(row["held_out"])
        if not {"1", "2"} <= act_pages[act]:
            entry["incomplete"].add(act)
        if row["id"] in names:
            entry["spellings"][names[row["id"]]] += 1
    out = []
    for key, e in by_key.items():
        out.append({"writer": e["spellings"].most_common(1)[0][0] if e["spellings"] else "(unknown)",
                    "label": label(key), "key": key, "pages": e["pages"], "acts": len(e["acts"]),
                    "page1": e["page1"], "page2": e["page2"], "incomplete_acts": len(e["incomplete"]),
                    "splits": dict(e["splits"]), "held_out": e["held_out"]})
    return sorted(out, key=lambda r: (r["key"] is None, r["pages"], r["writer"]))


def variants(conn):
    """The pairs of writer names that look like one person and are not merged yet -> [(a, b, pages a, pages b)]."""
    counts = Counter(page_names(conn).values())
    names = sorted(counts)
    return [(a, b, counts[a], counts[b]) for i, a in enumerate(names) for b in names[i + 1:] if close_names(a, b)]


def unread(conn):
    """The page 2s where no officer is read, per register-year -> {register: {"pages": n, "no_line": n,
    "examples": [officer lines]}}. Private (file names, names): printed on the machine only (cli writers --unread)."""
    import re
    group_pattern = params()["ingest"].get("group_pattern")
    out = {}
    rows = db.all_rows(conn, "SELECT p.id, o.original_name FROM pages p JOIN page_origins o ON o.page_id = p.id "
                             "JOIN page_writers w ON w.page_id = p.id WHERE p.deleted_at IS NULL "
                             "AND p.page_type = '2' AND w.raw_name = '' ORDER BY o.original_name")
    for row in rows:
        stem = Path(row["original_name"]).stem
        match = re.match(group_pattern, stem) if group_pattern else None
        entry = out.setdefault(match.group(1) if match else stem, {"pages": 0, "no_line": 0, "examples": []})
        entry["pages"] += 1
        try:
            lines = [line["text"] for line in read_page(path("data") / "pages" / f'{row["id"]}.xml')["lines"]]
        except (OSError, ValueError):
            lines = []
        found = officer_lines(lines)
        if not found:
            entry["no_line"] += 1
        elif len(entry["examples"]) < 3:
            entry["examples"].append((stem, found))
    return out


def raw_spellings(conn):
    """Every spelling as transcribed, with its canonical name and its pages (the alias editor)."""
    alias_map = aliases(conn)
    counts = Counter()
    for row in db.all_rows(conn, "SELECT w.raw_name FROM page_writers w JOIN pages p ON p.id = w.page_id "
                                 "WHERE p.deleted_at IS NULL AND w.raw_name <> ''"):
        counts[writer_key_text(row["raw_name"])] += 1
    return [(name, resolve(name, alias_map), n) for name, n in sorted(counts.items())]