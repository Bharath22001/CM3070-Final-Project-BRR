"""
Checks whether downloading more Oracc SAAo volumes would give my test tablets
catalogue metadata, or whether those tablets aren't in the SAA catalogue at all.
It traces each test tablet back to the transliteration CSV it came from.

Run from the project root:
    python -m cuneiform.domain3_generative.inspect_catalogue_join
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import pandas as pd

from cuneiform import config


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print("CATALOGUE JOIN -- WHICH SOURCE VOLUME ARE THE TEST TABLETS FROM?")
    print("=" * 70)

    csvs = sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))
    if not csvs:
        print(f"  No CSVs under {config.TRANSLITERATIONS_DIR}")
        return

    # Check the id column really is tablet_CDLI before relying on it.
    head = pd.read_csv(csvs[0], nrows=3)
    print(f"\n  columns of {csvs[0].name}:")
    print(f"      {list(head.columns)}")
    id_col = "tablet_CDLI"
    if id_col not in head.columns:
        cand = [c for c in head.columns if "tablet" in c.lower()]
        print(f"  !! '{id_col}' absent. Candidates: {cand}. STOP.")
        return

    # tablet -> the source files it appears in
    where: dict[str, set[str]] = defaultdict(set)
    print("\n" + "-" * 70)
    print("SOURCE FILES")
    print("-" * 70)
    for p in csvs:
        tabs = set(pd.read_csv(p, usecols=[id_col])[id_col].dropna().astype(str))
        for t in tabs:
            where[t].add(p.stem)
        print(f"      {p.stem:<10} {len(tabs):>5} tablets")

    # Test tablets from the corpus
    test_tabs: set[str] = set()
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if rec.get("split") == "test":
                test_tabs.add(rec["tablet"])

    print("\n" + "-" * 70)
    print(f"THE {len(test_tabs)} TEST TABLETS, BY SOURCE FILE")
    print("-" * 70)
    by_file: dict[str, list[str]] = defaultdict(list)
    for t in sorted(test_tabs):
        srcs = where.get(t) or {"<NOT IN ANY CSV>"}
        key = "+".join(sorted(srcs))
        by_file[key].append(t)
        print(f"      {t:<14} {key}")

    print("\n  summary:")
    for k, v in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        print(f"      {k:<26} {len(v)} tablets")

    # The decision
    print("\n" + "-" * 70)
    print("DECISION")
    print("-" * 70)
    # where[] stores file stems like "transliterations_saa06", so strip the
    # prefix before testing for "saa".
    saa_vols = sorted({s.removeprefix("transliterations_")
                       for t in test_tabs for s in where.get(t, set())
                       if s.removeprefix("transliterations_").startswith("saa")})
    p_shaped = sum(1 for t in test_tabs
                   if t.startswith("P") and t[1:].isdigit())

    print(f"  test tablets with a P###### id : {p_shaped}/{len(test_tabs)}")
    print(f"  SAA volumes covering them      : {saa_vols or 'NONE'}")

    if saa_vols:
        print("\n  --> (a) DOWNLOAD. These volumes would supply metadata:")
        for v in saa_vols:
            print(f"        python -m cuneiform.domain3_generative."
                  f"fetch_oracc_saao --volume {v}")
    else:
        print("\n  --> (b) NO DOWNLOAD WILL HELP. The test tablets come from the")
        print("      CompVis annotated transliteration files, not from an SAA")
        print("      volume. Catalogue metadata cannot cover the test split.")
        print("      Treat metadata as OPTIONAL in the prompt and demonstrate")
        print("      the historical-context feature on saa01 tablets instead.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()