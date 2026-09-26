"""
Prints the arm 4 sample for me to check by hand, and writes a blank review sheet.

The automated arms can't catch a wrong gloss of a real sign (for example the
model calling URUDA, copper, "dates"), so a person has to read the prose. I
only record a verdict, never the published translation text, since that is
copyrighted.

Run from the project root:
    python -m cuneiform.domain3_generative.review_arm4
"""

from __future__ import annotations

import csv
import sys

from cuneiform import config
from cuneiform.domain3_generative.generate import load_tablet, gloss

SRC = config.D3_DIR / "domain3_qualitative.csv"
OUT = config.D3_DIR / "domain3_arm4_review.csv"

COLUMNS = [
    "tablet", "confidence",
    "reading_defensible",      # yes / partly / no
    "context_stays_in_bounds",  # yes / no / n-a
    "confidence_calibrated",   # yes / too-high / too-low
    "wrong_gloss_found",       # yes / no, only a human can spot this
    "notes",
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not SRC.exists():
        print(f"  {SRC} not found. Run evaluate_domain3 first.")
        return

    with SRC.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    print("=" * 70)
    print(f"ARM 4 -- {len(rows)} outputs to check by eye")
    print("=" * 70)

    for i, r in enumerate(rows, 1):
        tab = r["tablet"]
        print("\n" + "=" * 70)
        print(f"  [{i}/{len(rows)}]  {tab}   confidence = {r['confidence']}")
        print("=" * 70)

        # Show the input the model saw, so the output is judged against it.
        try:
            t = load_tablet(tab)
            c = t.counts
            n = sum(c.values())
            legible = n - c.get("broken", 0) - c.get("unknown", 0)
            print(f"\n  INPUT: {len(t.lines)} lines, {n} positions, "
                  f"{legible} legible ({100 * legible / max(1, n):.0f}%)")
            print("  first 3 lines as the model saw them:")
            for v in t.lines[:3]:
                print(f"    {gloss(v)[:150]}")
        except Exception as exc:
            print(f"  (could not re-derive input: {exc})")

        print(f"\n  READING:\n    {r['reading']}")
        print(f"\n  HISTORICAL CONTEXT:\n    {r['historical_context']}")
        print(f"\n  ORACC: {r['oracc_url']}")
        print(f"  (if that 404s, search  {tab}  at http://oracc.org/ )")

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r["tablet"], r["confidence"], "", "", "", "", ""])

    print("\n" + "=" * 70)
    print(f"  blank review sheet written: {OUT}")
    print("  Fill it in, then report the verdicts back.")
    print("=" * 70)


if __name__ == "__main__":
    main()