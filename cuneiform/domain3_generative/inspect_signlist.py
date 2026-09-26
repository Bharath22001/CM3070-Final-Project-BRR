"""
My second look at OGSL, still not a parser. It counts the line-initial @tags,
prints a few complete sign records from different parts of the file, and shows
the exact format of the mzl_label column in my own CSVs so the join key matches.

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.inspect_signlist
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from cuneiform import config

OSL_ROOT = config.D3_OGSL_DIR / "osl-master"
BEFORE, AFTER = 30, 12
WINDOWS_AT = [169, 1134, 2195, 2616]   # spread through the file


def _fix_console_encoding() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def list_asl_files() -> list:
    files = sorted(OSL_ROOT.rglob("*.asl"))
    print("-" * 68)
    print(f"ALL .asl FILES ({len(files)})")
    print("-" * 68)
    for p in files:
        n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        print(f"  {p.stat().st_size / 1e6:>7.2f} MB  {n:>7,} lines  "
              f"{p.relative_to(OSL_ROOT)}")
    return files


def tag_census(lines: list) -> None:
    """Count line-initial @tags to reveal how a sign record is structured."""
    tags = Counter()
    for ln in lines:
        s = ln.strip()
        if s.startswith("@"):
            tags[s.split(None, 1)[0]] += 1
    print()
    print("-" * 68)
    print(f"TAG CENSUS  ({len(tags)} distinct line-initial tags)")
    print("-" * 68)
    for tag, n in tags.most_common():
        print(f"  {tag:<20} {n:>7,}")

    blanks = sum(1 for ln in lines if not ln.strip())
    nontag = sum(1 for ln in lines
                 if ln.strip() and not ln.strip().startswith("@"))
    print(f"  {'(blank lines)':<20} {blanks:>7,}")
    print(f"  {'(non-@ content)':<20} {nontag:>7,}")


def context_windows(lines: list) -> None:
    print()
    print("-" * 68)
    print("SIGN RECORD CONTEXT WINDOWS")
    print("-" * 68)
    for hit in WINDOWS_AT:
        lo = max(0, hit - 1 - BEFORE)
        hi = min(len(lines), hit + AFTER)
        print()
        print(f"  === around line {hit} "
              f"(showing {lo + 1}-{hi}) ===")
        for i in range(lo, hi):
            marker = ">>" if i == hit - 1 else "  "
            print(f"  {marker}{i + 1:>7} | {lines[i]}")


def inspect_our_mzl_labels() -> None:
    """Print the format of mzl_label in my data, straight from the CSVs."""
    print()
    print("-" * 68)
    print("OUR mzl_label COLUMN (from transliterations/*.csv)")
    print("-" * 68)

    csvs = sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))
    print(f"  Reading {len(csvs)} files")
    frames = []
    for p in csvs:
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            print(f"  Could not read {p.name}: {exc}")
    if not frames:
        print("  No CSVs read.")
        return

    df = pd.concat(frames, ignore_index=True)
    if "mzl_label" not in df.columns:
        print(f"  No mzl_label column. Columns: {list(df.columns)}")
        return

    col = df["mzl_label"]
    print(f"  rows              : {len(col):,}")
    print(f"  dtype             : {col.dtype}")
    print(f"  nulls             : {col.isna().sum():,}")
    print(f"  distinct values   : {col.nunique():,}")
    print(f"  first 25 raw      : {list(col.dropna().unique()[:25])}")
    print(f"  repr of first 5   : {[repr(v) for v in col.dropna().head(5)]}")

    nonnum = [v for v in col.dropna().unique()
              if not str(v).replace(".0", "").isdigit()]
    print(f"  NON-NUMERIC values: {len(nonnum)} -> {nonnum[:25]}")

    if "train_label" in df.columns:
        pair = df[["train_label", "mzl_label"]].dropna().drop_duplicates()
        print(f"  distinct (train_label, mzl_label) pairs: {len(pair):,}")
        print("  sample pairs:")
        for _, r in pair.head(12).iterrows():
            print(f"    train_label {r['train_label']!r:<10} -> "
                  f"mzl_label {r['mzl_label']!r}")


def main() -> None:
    _fix_console_encoding()

    target = OSL_ROOT / "00lib" / "osl.asl"
    if not target.exists():
        raise SystemExit(f"Not found: {target}\nRun fetch_signlist.py first.")

    print("=" * 68)
    print("OGSL SECOND INSPECTION PASS")
    print("=" * 68)

    list_asl_files()

    lines = target.read_text(encoding="utf-8",
                             errors="replace").splitlines()
    print()
    print(f"  Main file: {target.relative_to(OSL_ROOT)} ({len(lines):,} lines)")

    tag_census(lines)
    context_windows(lines)
    inspect_our_mzl_labels()

    print()
    print("=" * 68)
    print("Paste this back. The tag census + windows define the parser.")
    print("=" * 68)


if __name__ == "__main__":
    main()