"""
Harvests extra sign names from OGSL's mzl-kwu.tsv concordance to fill gaps that
osl.asl leaves.

Rules: lines marked #CHECK are skipped (OGSL's own maintainers doubt them), a
name must look like an OGSL sign name, and existing osl.asl names always win.
The file head and column census are printed before anything is imported.

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.harvest_mzl_kwu
"""

import json
import re
import sys
from collections import Counter

import pandas as pd

from cuneiform import config

TSV_PATH = config.D3_OGSL_DIR / "osl-master" / "00etc" / "mzl-kwu.tsv"
HEAD_LINES = 25

MZL_CELL_RE = re.compile(r"^MZL\s*0*(\d+)([A-Za-z'\"]*)$")
# OGSL sign names look like A, UZ, TI@t, |UD@s×BAD|, |GI%GI|, 7(DIS).
# Lowercase and '%' are needed for variant suffixes and ligatures.
NAME_RE = re.compile(r"^[|]?[A-Za-zŠšŽžṢṭṬḪḫĜŋŊ0-9₀-₉ₓ×%.@+&()~'\-|]+$")


def _fix_console_encoding() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def show_head(lines):
    print("-" * 68)
    print(f"FILE: {TSV_PATH.name}  ({len(lines):,} lines)")
    print("-" * 68)
    for i, ln in enumerate(lines[:HEAD_LINES], 1):
        print(f"  {i:>4} | {ln}")

    widths = Counter(len(ln.split('\t')) for ln in lines if ln.strip())
    print(f"\n  Tab-separated column counts: {dict(widths)}")

    check = [ln for ln in lines if "#CHECK" in ln]
    print(f"  Lines carrying '#CHECK' (skipped): {len(check)}")
    for ln in check[:8]:
        print(f"    {ln.strip()[:110]}")


def parse(lines):
    """Return {mzl_number: name} plus diagnostics."""
    harvested, no_name, rejected, skipped_check = {}, [], [], 0

    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith("@"):
            continue
        if "#CHECK" in ln:
            skipped_check += 1
            continue

        cells = [c.strip() for c in ln.split("\t")]
        if not cells:
            continue
        m = MZL_CELL_RE.match(cells[0])
        if not m or m.group(2):        # skip lettered variants (113b etc.)
            continue
        mzl = str(int(m.group(1)))

        # The name is the last non-empty cell that isn't a KWU reference.
        name = None
        for cell in reversed(cells[1:]):
            if not cell or cell.upper().startswith("KWU"):
                continue
            name = cell
            break

        if name is None:
            no_name.append(mzl)
        elif NAME_RE.match(name):
            harvested[mzl] = name
        else:
            rejected.append((mzl, name))

    return harvested, no_name, rejected, skipped_check


def main() -> None:
    _fix_console_encoding()

    if not TSV_PATH.exists():
        raise SystemExit(f"Not found: {TSV_PATH}")

    print("=" * 68)
    print("HARVEST mzl-kwu.tsv")
    print("=" * 68)

    lines = TSV_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    show_head(lines)

    harvested, no_name, rejected, skipped = parse(lines)
    print()
    print("-" * 68)
    print("PARSE RESULT")
    print("-" * 68)
    print(f"  MZL numbers with a name   : {len(harvested):,}")
    print(f"  MZL numbers with NO name  : {len(no_name):,}")
    print(f"  names rejected by pattern : {len(rejected)}")
    for mzl, nm in rejected[:15]:
        print(f"    MZL{mzl}: {nm!r}")
    print(f"  lines skipped for #CHECK  : {skipped}")

    # Apply to the lookup, gaps only
    doc = json.loads(config.D3_SIGN_LOOKUP.read_text(encoding="utf-8"))
    lookup = doc["lookup"]

    frames = [pd.read_csv(p)
              for p in sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    tl_freq = Counter(df["train_label"].astype(int).tolist())

    print()
    print("-" * 68)
    print("FILLING GAPS (osl.asl entries are never overridden)")
    print("-" * 68)
    filled, still_missing = [], []
    for tl, rec in lookup.items():
        if rec.get("sign"):
            continue
        mzl = str(rec.get("mzl"))
        name = harvested.get(mzl)
        if name:
            rec["sign"] = name
            rec["readings"] = []
            rec["status"] = "mzl_kwu_concordance"
            rec["source"] = f"00etc/mzl-kwu.tsv (MZL{mzl})"
            filled.append((int(tl), mzl, name, tl_freq.get(int(tl), 0)))
        else:
            still_missing.append((int(tl), mzl, tl_freq.get(int(tl), 0)))

    for tl, mzl, name, n in sorted(filled, key=lambda x: -x[3]):
        print(f"  FILLED  tl {tl:>4}  MZL{mzl:<5} -> {name:<14} {n:>7,} occ")
    print()
    for tl, mzl, n in sorted(still_missing, key=lambda x: -x[2]):
        if lookup[str(tl)].get("sign") == "<BROKEN>":
            continue
        print(f"  STILL UNNAMED  tl {tl:>4}  MZL{mzl:<5} {n:>7,} occ  "
              f"({lookup[str(tl)].get('status')})")

    # Final coverage
    vocab = json.loads((config.D2_DIR / "vocab.json").read_text(encoding="utf-8"))

    def cov(label, keys):
        named = tok_named = tok_all = 0
        miss = []
        for k in keys:
            tl = int(k)
            if tl == config.BROKEN_LABEL:
                continue
            n = tl_freq.get(tl, 0)
            tok_all += n
            if lookup.get(str(tl), {}).get("sign"):
                named += 1
            else:
                miss.append((tl, n))
                continue
            tok_named += n
        print(f"\n  {label}")
        print(f"    named          : {named}/{named + len(miss)} "
              f"({named / max(named + len(miss), 1):.1%})")
        print(f"    token-weighted : {tok_named / max(tok_all, 1):.2%}")
        for tl, n in sorted(miss, key=lambda x: -x[1]):
            print(f"      unnamed tl {tl:>4}  {n:>7,} occ")

    print()
    print("-" * 68)
    print("FINAL COVERAGE (report these)")
    print("-" * 68)
    cov("Domain 2 vocabulary (282)", vocab["sign_to_id"].keys())
    cov("Domain 1 emit classes (82)", vocab["vision_emit_classes"])

    by_status = Counter(r.get("status") for r in lookup.values())
    print(f"\n  Provenance breakdown: {dict(by_status)}")

    doc["notes"]["mzl_kwu_harvest"] = {
        "file": "00etc/mzl-kwu.tsv",
        "filled": len(filled),
        "check_lines_skipped": skipped,
    }
    config.D3_SIGN_LOOKUP.write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  Updated {config.D3_SIGN_LOOKUP}")
    print("=" * 68)


if __name__ == "__main__":
    main()