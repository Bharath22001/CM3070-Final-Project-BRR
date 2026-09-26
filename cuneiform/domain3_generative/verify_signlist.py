"""
Audits the lookup build_signlist.py produced, without writing anything.

It checks whether the parser dropped any real @list MZL lines, whether the
uncovered signs are genuinely missing from OGSL (MZL numbers above 907 can't
be valid at all), and what the coverage is over the sets that matter: Domain
2's 282 vocabulary signs and Domain 1's 82 emit classes.

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.verify_signlist
"""

import json
import re
import sys
from collections import Counter

import pandas as pd

from cuneiform import config

MISSING_MZL = [851, 640, 852, 583, 729, 905, 271, 810, 378, 170, 925]
MZL_LIST_RE = re.compile(r"@list\s+MZL", re.IGNORECASE)
MZL_ANY_RE = re.compile(r"MZL", re.IGNORECASE)
MZL_DECLARED_MAX = 907          # from `@listdef MZL 1-907` in osl.asl


def _fix_console_encoding() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def enclosing_sign(lines, idx):
    """Nearest @sign / @form at or above line index idx."""
    for j in range(idx, -1, -1):
        s = lines[j].strip()
        if s.startswith("@sign") or s.startswith("@form"):
            return f"line {j + 1}: {s}"
    return "(no enclosing scope found)"


def audit_line_counts(lines):
    print("-" * 68)
    print("1. DID THE PARSER DROP ANY @list MZL LINES?")
    print("-" * 68)

    any_hits = [i for i, ln in enumerate(lines) if MZL_ANY_RE.search(ln)]
    list_hits = [i for i, ln in enumerate(lines) if MZL_LIST_RE.search(ln)]
    print(f"  lines mentioning MZL anywhere : {len(any_hits):,}")
    print(f"  lines matching '@list MZL'    : {len(list_hits):,}")
    print(f"  parser captured (build step)  : 851")
    print(f"  unaccounted for               : {len(list_hits) - 851:,}")

    other = [i for i in any_hits if i not in set(list_hits)]
    tags = Counter()
    for i in other:
        s = lines[i].strip()
        tags[s.split(None, 1)[0] if s.startswith("@") else "(prose)"] += 1
    print()
    print(f"  Non-@list MZL mentions ({len(other)}), by tag:")
    for tag, n in tags.most_common():
        print(f"    {tag:<16} {n:>5}")

    # Any @list MZL line my regex would refuse
    strict = re.compile(r"^MZL(\d+)([A-Za-z'\"]*)$")
    odd = []
    for i in list_hits:
        val = lines[i].strip().split(None, 1)[1].strip()
        if not strict.match(val):
            odd.append((i + 1, val))
    print()
    print(f"  @list MZL values our regex REJECTS: {len(odd)}")
    for ln, val in odd[:20]:
        print(f"    line {ln:>7} | {val!r}")


def audit_missing(lines):
    print()
    print("-" * 68)
    print("2. ARE THE 11 UNCOVERED SIGNS REALLY ABSENT FROM OGSL?")
    print("-" * 68)

    for mzl in MISSING_MZL:
        pat = re.compile(rf"MZL0*{mzl}\b")
        hits = [i for i, ln in enumerate(lines) if pat.search(ln)]
        flag = "  [ABOVE DECLARED MZL MAX 907]" if mzl > MZL_DECLARED_MAX else ""
        print()
        print(f"  MZL{mzl}{flag}")
        if not hits:
            print("    absent from osl.asl entirely")
            continue
        for i in hits[:4]:
            print(f"    line {i + 1:>7} | {lines[i].strip()}")
            print(f"      scope -> {enclosing_sign(lines, i)}")


def audit_collisions():
    print()
    print("-" * 68)
    print("3. MZL COLLISIONS (one number, more than one sign record)")
    print("-" * 68)

    data = json.loads(config.D3_MZL_NAMES.read_text(encoding="utf-8"))
    collisions = data["collisions"]
    variants = data["variants"]

    frames = [pd.read_csv(p)
              for p in sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    ours = set(df["mzl_label"].astype(int).tolist()) - {config.BROKEN_LABEL}
    freq = Counter(df["mzl_label"].astype(int).tolist())

    relevant = {k: v for k, v in collisions.items() if int(k) in ours}
    print(f"  total collisions          : {len(collisions)}")
    print(f"  affecting OUR corpus      : {len(relevant)}")
    for k, entries in sorted(relevant.items(),
                             key=lambda kv: -freq[int(kv[0])]):
        print(f"\n    MZL{k}  ({freq[int(k)]:,} corpus occurrences)")
        for e in entries:
            print(f"      {e['sign']:<18} scope={e['scope']:<5} "
                  f"deprecated={e['deprecated']} fake={e['fake']}")

    print(f"\n  lettered variants ({len(variants)}): {list(variants.keys())}")


def audit_vocab_coverage():
    print()
    print("-" * 68)
    print("4. COVERAGE OVER THE SETS THAT MATTER")
    print("-" * 68)

    vocab = json.loads((config.D2_DIR / "vocab.json").read_text(encoding="utf-8"))
    names = json.loads(config.D3_SIGN_NAMES.read_text(encoding="utf-8"))

    sign_to_id = vocab["sign_to_id"]
    emit = vocab["vision_emit_classes"]
    print(f"  sign_to_id sample keys : {list(sign_to_id.keys())[:8]}")
    print(f"  emit sample            : {emit[:8]}")
    print(f"  meta                   : {vocab.get('meta')}")

    frames = [pd.read_csv(p)
              for p in sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    freq = Counter(df["train_label"].astype(int).tolist())
    total_real = sum(n for k, n in freq.items() if k != config.BROKEN_LABEL)

    def report(label, keys):
        named, unnamed, tok_named, tok_all = [], [], 0, 0
        for k in keys:
            try:
                tl = int(k)
            except (TypeError, ValueError):
                continue
            if tl == config.BROKEN_LABEL:
                continue
            n = freq.get(tl, 0)
            tok_all += n
            rec = names.get(str(tl))
            if rec and rec.get("sign"):
                named.append(tl)
                tok_named += n
            else:
                unnamed.append((tl, n))
        print(f"\n  {label}")
        print(f"    signs                : {len(named) + len(unnamed)}")
        print(f"    with an OGSL name    : {len(named)} "
              f"({len(named) / max(len(named) + len(unnamed), 1):.1%})")
        print(f"    token-weighted       : "
              f"{tok_named / max(tok_all, 1):.2%} of this set's tokens "
              f"({tok_all / max(total_real, 1):.1%} of all real tokens)")
        if unnamed:
            print(f"    UNNAMED (train_label, corpus occurrences):")
            for tl, n in sorted(unnamed, key=lambda x: -x[1]):
                print(f"      {tl:>4}  {n:>7,}")

    report("Domain 2 vocabulary (282 signs)", sign_to_id.keys())
    report("Domain 1 emit classes (82)", emit)


def audit_reading_order():
    """
    Readings are listed alphabetically, not by importance, so check how often
    the sign's own name appears among them.
    """
    print()
    print("-" * 68)
    print("5. READING ORDER (is readings[0] ever the canonical value?)")
    print("-" * 68)

    names = json.loads(config.D3_SIGN_NAMES.read_text(encoding="utf-8"))
    first_is_name = same_present = total = 0
    examples = []
    for tl, rec in names.items():
        sign, rd = rec.get("sign"), rec.get("readings") or []
        if not sign or not rd:
            continue
        total += 1
        low = sign.lower().strip("|")
        if rd[0].lower() == low:
            first_is_name += 1
        if low in [r.lower() for r in rd]:
            same_present += 1
            if len(examples) < 8:
                idx = [r.lower() for r in rd].index(low)
                examples.append((sign, idx, len(rd), rd[:3]))

    print(f"  signs with readings          : {total}")
    print(f"  readings[0] == sign name     : {first_is_name} "
          f"({first_is_name / max(total, 1):.1%})")
    print(f"  sign name present anywhere   : {same_present} "
          f"({same_present / max(total, 1):.1%})")
    print("\n  Examples (name, its index in readings, list length, first 3):")
    for sign, idx, n, head in examples:
        print(f"    {sign:<14} at index {idx:>3} of {n:<4} first3={head}")


def main() -> None:
    _fix_console_encoding()
    lines = config.D3_OSL_ASL.read_text(encoding="utf-8",
                                        errors="replace").splitlines()
    print("=" * 68)
    print(f"VERIFY SIGN LIST  ({len(lines):,} lines)")
    print("=" * 68)
    audit_line_counts(lines)
    audit_missing(lines)
    audit_collisions()
    audit_vocab_coverage()
    audit_reading_order()
    print()
    print("=" * 68)


if __name__ == "__main__":
    main()