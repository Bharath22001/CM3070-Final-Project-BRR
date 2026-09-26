"""
Downloads the Oracc Global Sign List (OGSL) and prints what is in it, without
parsing anything, so I could read the format before writing a parser.

OGSL links MZL numbers (my `mzl_label` column) to sign names like LUGAL and
KUR, which is what lets Domain 3 get names instead of bare integers.

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.fetch_signlist
    python -m cuneiform.domain3_generative.fetch_signlist --force
"""

import argparse
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from cuneiform import config

OSL_URL = "https://github.com/oracc/osl/archive/refs/heads/master.zip"
HEAD_LINES = 60
MZL_HITS_TO_SHOW = 40
LARGEST_TO_LIST = 15


def _fix_console_encoding() -> None:
    """
    Switch the console to UTF-8, since OGSL's cuneiform characters and
    diacritics crash the default Windows codepage.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def download(force: bool = False) -> Path:
    config.ensure_dirs()
    zip_path = config.D3_OGSL_ZIP

    if zip_path.exists() and not force:
        size_mb = zip_path.stat().st_size / 1e6
        print(f"  Archive already present ({size_mb:.1f} MB): {zip_path}")
        print("  Use --force to re-download.")
        return zip_path

    print(f"  Downloading {OSL_URL}")
    req = urllib.request.Request(
        OSL_URL, headers={"User-Agent": "cuneiform-fyp/1.0"}
    )
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as fh:
        fh.write(resp.read())

    size_mb = zip_path.stat().st_size / 1e6
    print(f"  Saved {size_mb:.1f} MB to {zip_path}")
    return zip_path


def extract(zip_path: Path) -> Path:
    dest = config.D3_OGSL_DIR
    print(f"  Extracting into {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    roots = [p for p in dest.iterdir() if p.is_dir()]
    if not roots:
        raise SystemExit("  ERROR: extraction produced no directory.")
    root = max(roots, key=lambda p: len(list(p.rglob("*"))))
    print(f"  Extracted root: {root}")
    return root


def inventory(root: Path) -> list:
    files = [p for p in root.rglob("*") if p.is_file()]
    by_ext = Counter(p.suffix.lower() or "(no ext)" for p in files)

    print()
    print("-" * 68)
    print(f"FILE INVENTORY  ({len(files)} files)")
    print("-" * 68)
    for ext, n in by_ext.most_common():
        print(f"  {ext:<12} {n:>6}")

    print()
    print(f"  {LARGEST_TO_LIST} largest files:")
    largest = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    for p in largest[:LARGEST_TO_LIST]:
        rel = p.relative_to(root)
        print(f"    {p.stat().st_size / 1e6:>8.2f} MB  {rel}")

    return files


def show_file(path: Path, root: Path) -> None:
    print()
    print("=" * 68)
    print(f"FILE: {path.relative_to(root)}  ({path.stat().st_size / 1e6:.2f} MB)")
    print("=" * 68)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  Could not read: {exc}")
        return

    lines = text.splitlines()
    print(f"  Total lines: {len(lines):,}")

    print()
    print(f"  --- FIRST {HEAD_LINES} LINES ---")
    for i, line in enumerate(lines[:HEAD_LINES], start=1):
        print(f"  {i:>4} | {line}")

    hits = [(i, ln) for i, ln in enumerate(lines, start=1) if "MZL" in ln]
    print()
    print(f"  --- LINES MENTIONING 'MZL' : {len(hits):,} total ---")
    for i, line in hits[:MZL_HITS_TO_SHOW]:
        print(f"  {i:>7} | {line}")
    if len(hits) > MZL_HITS_TO_SHOW:
        print(f"  ... and {len(hits) - MZL_HITS_TO_SHOW:,} more")


def main() -> None:
    _fix_console_encoding()
    ap = argparse.ArgumentParser(description="Download and inspect OGSL.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the archive is already present.")
    args = ap.parse_args()

    config.check_environment()

    print("=" * 68)
    print("OGSL FETCH AND INSPECT")
    print("=" * 68)

    zip_path = download(force=args.force)
    root = extract(zip_path)
    files = inventory(root)

    for ext in (".asl", ".json"):
        candidates = [p for p in files if p.suffix.lower() == ext]
        if not candidates:
            print()
            print(f"  NOTE: no {ext} files found in the archive.")
            continue
        show_file(max(candidates, key=lambda p: p.stat().st_size), root)

    print()
    print("=" * 68)
    print("NEXT: paste this output back. Do not write a parser from memory of")
    print("the ASL format -- read it off the lines above.")
    print("=" * 68)


if __name__ == "__main__":
    main()