"""
Extracts the CC0 catalogue metadata (designation, author, recipient, archive,
provenience, dialect) from an Oracc SAAo volume zip, and measures how many of
my test tablets it covers.

I give these catalogue facts to Domain 3 so its historical context summarises
supplied evidence instead of inventing it. Coverage depends on which volumes
are on disk, so the join merges every extracted volume. The schema is printed
from the file before anything is parsed.

Run from the project root:
    python -m cuneiform.domain3_generative.extract_catalogue
    python -m cuneiform.domain3_generative.extract_catalogue --volume saa06
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter

from cuneiform import config

# Fields I want for the historical context.
WANTED = ["designation", "ancient_author", "ancient_recipient", "archive",
          "has_date", "period", "provenience", "genre", "museum_no",
          "primary_publication", "dialect", "collection", "please_cite"]


def extract(volume: str) -> dict | None:
    """Pull catalogue.json out of saao-<volume>.zip and write it to disk."""
    zip_path = config.D3_ORACC_DIR / f"saao-{volume}.zip"
    out_dir = config.D3_ORACC_DIR / f"saao-{volume}"

    print("\n" + "-" * 70)
    print(f"1. EXTRACT  {volume}")
    print("-" * 70)

    if not zip_path.exists():
        print(f"  {zip_path} not found.")
        print(f"  Download it first:")
        print(f"    python -m cuneiform.domain3_generative.fetch_oracc_saao "
              f"--volume {volume}")
        return None

    with zipfile.ZipFile(zip_path) as zf:
        cands = [m for m in zf.namelist() if m.endswith("catalogue.json")]
        print(f"  archive members       : {len(zf.namelist())}")
        print(f"  catalogue.json found  : {cands}")
        if not cands:
            print("  !! No catalogue in this archive. STOP.")
            return None
        data = zf.read(cands[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "catalogue.json"
    target.write_bytes(data)
    print(f"  extracted -> {target}  ({len(data):,} bytes)")
    return json.loads(data.decode("utf-8"))


def members_of(cat: dict) -> dict:
    """The record dict, whatever Oracc chose to call it."""
    for k in ("members", "entries", "texts"):
        if isinstance(cat.get(k), dict):
            return cat[k]
    return {}


def inspect(cat: dict) -> dict:
    print("\n" + "-" * 70)
    print("2. STRUCTURE AND FIELD COVERAGE")
    print("-" * 70)
    print(f"  top-level keys : {list(cat)[:12]}")
    print(f"  licence        : {cat.get('license')} "
          f"({cat.get('license-url')})")

    records = members_of(cat)
    if not records:
        print("  !! No record dict found. Inspect manually before proceeding.")
        return {}
    print(f"  records        : {len(records):,}")

    n = len(records)
    populated = Counter()
    for rec in records.values():
        for k, v in rec.items():
            if v not in (None, "", []):
                populated[k] += 1

    print("\n  fields the prompt wants:")
    for f in WANTED:
        c = populated.get(f, 0)
        flag = "" if c else "   <-- ABSENT"
        print(f"      {f:<24} {c:>5}/{n}  ({100 * c / n:5.1f}%){flag}")

    return records


def load_all_catalogues() -> dict[str, dict]:
    """
    Merge every catalogue extracted under artifacts/domain3/oracc/. A key that
    appears in two volumes is reported, not silently overwritten.
    """
    merged: dict[str, dict] = {}
    clashes = 0
    found = []
    for d in sorted(config.D3_ORACC_DIR.glob("saao-*")):
        f = d / "catalogue.json"
        if not f.is_dir() and f.exists():
            recs = members_of(json.loads(f.read_text(encoding="utf-8")))
            found.append((d.name, len(recs)))
            for k, v in recs.items():
                if k in merged:
                    clashes += 1
                else:
                    merged[k] = v
    print(f"  catalogues on disk : {found}")
    if clashes:
        print(f"  !! {clashes} duplicate text ids across volumes -- unexpected.")
    return merged


def join_gate(records: dict) -> None:
    print("\n" + "-" * 70)
    print("3. JOIN GATE -- coverage of the HELD-OUT TEST SPLIT")
    print("-" * 70)

    merged = load_all_catalogues()
    print(f"  merged catalogue ids : {len(merged):,}")

    corpus = config.D2_CORPUS_JSONL
    if not corpus.exists():
        print(f"  SKIPPED -- {corpus} not found.")
        return

    all_tabs: set[str] = set()
    test_tabs: set[str] = set()
    with corpus.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            all_tabs.add(rec["tablet"])
            if rec.get("split") == "test":
                test_tabs.add(rec["tablet"])

    hit_all = all_tabs & set(merged)
    hit_test = sorted(test_tabs & set(merged))
    miss_test = sorted(test_tabs - set(merged))

    print(f"\n  whole corpus : {len(hit_all):,} / {len(all_tabs):,} "
          f"({100 * len(hit_all) / len(all_tabs):.1f}%)")
    print(f"  TEST SPLIT   : {len(hit_test)} / {len(test_tabs)} "
          f"({100 * len(hit_test) / max(1, len(test_tabs)):.1f}%)")
    print(f"\n  covered  : {hit_test}")
    print(f"  UNcovered: {miss_test}")

    gate = len(hit_test) > 0
    print(f"\n  --> {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  !! IGNORE THE PLAN ABOVE. No metadata reaches the test split;")
        print("     the prompt must run catalogue-free on the evaluation set.")

    if hit_test:
        k = hit_test[0]
        print(f"\n  WHAT THE PROMPT WOULD RECEIVE for {k}:")
        for f in WANTED:
            if merged[k].get(f):
                print(f"      {f:<24} {merged[k][f]}")

    print("\n  NOTE: metadata is OPTIONAL in the prompt either way. Uncovered")
    print("  tablets get an explicit 'no catalogue metadata available' block")
    print("  and an instruction not to speculate about author, date or")
    print("  provenance. That instruction is itself testable by the")
    print("  faithfulness arm of the evaluation.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="saa01")
    args = ap.parse_args()

    print("=" * 70)
    print(f"ORACC SAAo CATALOGUE -- {args.volume}")
    print("=" * 70)

    cat = extract(args.volume)
    if cat is None:
        return
    records = inspect(cat)
    if records:
        join_gate(records)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()