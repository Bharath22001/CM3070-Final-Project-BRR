"""
Parses OGSL into an MZL number -> sign name and readings lookup, and measures
how much of my corpus it covers.

My transliterations are bare integers, which an LLM can't use, but sign names
like "A ... AK" appear in its training data. OGSL is reference data, not a model.

A sign record can contain nested @form records with their own @list lines
(in |A.SU|, MZL016 belongs to the form SU), so I track scope with a stack.

Outputs:
    artifacts/domain3/mzl_signnames.json   MZL number -> sign record
    artifacts/domain3/sign_names.json      train_label -> sign record

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.build_signlist
"""

import json
import re
import sys
from collections import Counter

import pandas as pd

from cuneiform import config

# Matches MZL016, MZL839, MZL113b, MZL146'
MZL_RE = re.compile(r"^MZL(\d+)([A-Za-z'\"]*)$")

MAX_MISSES_SHOWN = 40


def _fix_console_encoding() -> None:
    """OGSL is full of cuneiform and subscripts, which cp1252 can't print."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _new_scope(kind: str, name: str) -> dict:
    return {
        "kind": kind,
        "name": name,
        "mzl": [],
        "readings": [],
        "withdrawn": [],
        "aka": [],
        "uname": None,
        "ucun": None,
        "oid": None,
        "deprecated": kind.endswith("-"),
        "fake": False,
        "compoundonly": False,
    }


def parse_osl(path):
    """
    Walk osl.asl and emit one record per scope (sign or form) with at least one
    MZL number. Returns (records, stats).
    """
    records = []
    stack = []
    stats = Counter()

    def close_top():
        if not stack:
            return
        scope = stack.pop()
        if scope["mzl"]:
            records.append(scope)

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s or not s.startswith("@"):
                continue

            parts = s.split(None, 1)
            tag = parts[0]
            val = parts[1].strip() if len(parts) > 1 else ""

            if tag in ("@sign", "@sign-"):
                while stack:                      # unterminated previous sign
                    close_top()
                stack.append(_new_scope(tag, val))
                stats["signs"] += 1

            elif tag in ("@form", "@form-", "@form+"):
                if stack and stack[-1]["kind"].startswith("@form"):
                    close_top()                   # a new form closes the previous form
                stack.append(_new_scope(tag, val))
                stats["forms"] += 1

            elif tag == "@@":
                if stack and stack[-1]["kind"].startswith("@form"):
                    close_top()
                    stats["form_closed_by_@@"] += 1

            elif tag == "@end":
                if val.startswith("form"):
                    if stack and stack[-1]["kind"].startswith("@form"):
                        close_top()
                        stats["form_closed_by_end"] += 1
                else:                             # @end sign
                    while stack and not stack[-1]["kind"].startswith("@sign"):
                        close_top()
                    close_top()
                    stats["sign_closed"] += 1

            elif not stack:
                continue                          # preamble: @listdef, @lit...

            elif tag == "@list":
                m = MZL_RE.match(val)
                if m:
                    stack[-1]["mzl"].append((m.group(1), m.group(2)))
                    stats["mzl_lines"] += 1
                    if stack[-1]["kind"].startswith("@form"):
                        stats["mzl_in_form_scope"] += 1

            elif tag == "@v":
                stack[-1]["readings"].append(val)
            elif tag == "@v-":
                stack[-1]["withdrawn"].append(val)
            elif tag == "@aka":
                stack[-1]["aka"].append(val)
            elif tag == "@uname":
                stack[-1]["uname"] = val
            elif tag == "@ucun":
                stack[-1]["ucun"] = val
            elif tag == "@oid":
                stack[-1]["oid"] = val
            elif tag == "@fake":
                stack[-1]["fake"] = True
            elif tag == "@compoundonly":
                stack[-1]["compoundonly"] = True

    while stack:
        close_top()
        stats["unterminated_at_eof"] += 1

    return records, stats


def build_mzl_index(records):
    """
    MZL number -> record. One number can appear under more than one sign, so
    I record collisions instead of overwriting.
    """
    index, collisions, variants = {}, {}, {}

    for rec in records:
        entry = {
            "sign": rec["name"],
            "scope": "form" if rec["kind"].startswith("@form") else "sign",
            "readings": rec["readings"],
            "uname": rec["uname"],
            "ucun": rec["ucun"],
            "oid": rec["oid"],
            "deprecated": rec["deprecated"],
            "fake": rec["fake"],
        }
        for num, suffix in rec["mzl"]:
            if suffix:
                variants.setdefault(f"{int(num)}{suffix}", []).append(entry)
                continue
            key = str(int(num))
            if key in index:
                collisions.setdefault(key, [index[key]]).append(entry)
                # Prefer a non-deprecated, non-fake entry with readings.
                incumbent = index[key]
                better = (
                    (not entry["deprecated"] and incumbent["deprecated"])
                    or (not entry["fake"] and incumbent["fake"])
                )
                if better:
                    index[key] = entry
            else:
                index[key] = entry

    return index, collisions, variants


def load_corpus_labels():
    """(train_label, mzl_label) pairs plus corpus token frequencies."""
    frames = []
    for p in sorted(config.TRANSLITERATIONS_DIR.glob("*.csv")):
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    pairs = (df[["train_label", "mzl_label"]]
             .drop_duplicates()
             .astype(int))
    freq = Counter(df["mzl_label"].astype(int).tolist())
    return pairs, freq


def inspect_vocab():
    """
    Report vocab.json's shape without assuming it (read-only). If the sign list
    isn't there I fall back to whole-corpus coverage.
    """
    path = config.D2_DIR / "vocab.json"
    if not path.exists():
        print(f"  vocab.json not found at {path}")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  vocab.json top-level type : {type(data).__name__}")
    if isinstance(data, dict):
        print(f"  top-level keys           : {list(data.keys())}")
        for k, v in data.items():
            kind = type(v).__name__
            size = len(v) if hasattr(v, "__len__") else "-"
            print(f"    {k:<28} {kind:<8} len={size}")
    return data


def main() -> None:
    _fix_console_encoding()
    config.check_environment()
    config.ensure_dirs()

    if not config.D3_OSL_ASL.exists():
        raise SystemExit(f"Not found: {config.D3_OSL_ASL}\n"
                         "Run fetch_signlist.py first.")

    print("=" * 68)
    print("BUILD MZL -> SIGN NAME LOOKUP")
    print("=" * 68)

    records, stats = parse_osl(config.D3_OSL_ASL)
    print("  Parse statistics")
    for k, v in sorted(stats.items()):
        print(f"    {k:<24} {v:>7,}")
    print(f"    {'records with MZL':<24} {len(records):>7,}")

    index, collisions, variants = build_mzl_index(records)
    print(f"    {'distinct MZL numbers':<24} {len(index):>7,}")
    print(f"    {'MZL collisions':<24} {len(collisions):>7,}")
    print(f"    {'lettered variants':<24} {len(variants):>7,}")

    # Sanity check on a mapping I already know by eye
    print()
    print("-" * 68)
    print("SANITY CHECK (osl.asl line 169: @list MZL839 inside @sign A)")
    print("-" * 68)
    got = index.get("839")
    print(f"  MZL839 -> {got['sign'] if got else 'MISSING'}   "
          f"(expected: A)")
    if got:
        print(f"    scope={got['scope']}  ucun={got['ucun']}  "
              f"readings={got['readings'][:6]}")
    got16 = index.get("16")
    print(f"  MZL016 -> {got16['sign'] if got16 else 'MISSING'}  "
          f"(expected: SU, via @form -- the nesting test)")
    if got16:
        print(f"    scope={got16['scope']}  readings={got16['readings'][:6]}")

    # Coverage over my corpus
    pairs, freq = load_corpus_labels()
    real = pairs[pairs["mzl_label"] != config.BROKEN_LABEL]
    total_real_tokens = sum(n for m, n in freq.items()
                            if m != config.BROKEN_LABEL)

    hits, misses = [], []
    for _, row in real.iterrows():
        mzl = int(row["mzl_label"])
        (hits if str(mzl) in index else misses).append(
            (int(row["train_label"]), mzl))

    covered_tokens = sum(freq[m] for _, m in hits)
    print()
    print("-" * 68)
    print("COVERAGE OVER OUR CORPUS")
    print("-" * 68)
    print(f"  real signs in corpus     : {len(real)}")
    print(f"  with an OGSL name        : {len(hits)} "
          f"({len(hits) / max(len(real), 1):.1%})")
    print(f"  MISSING                  : {len(misses)}")
    print(f"  token-weighted coverage  : "
          f"{covered_tokens / max(total_real_tokens, 1):.2%} "
          f"of {total_real_tokens:,} real tokens")

    if misses:
        print()
        print(f"  Missing (train_label, mzl_label), by corpus frequency:")
        for tl, mzl in sorted(misses, key=lambda x: -freq[x[1]])[:MAX_MISSES_SHOWN]:
            print(f"    train_label {tl:>4}  mzl {mzl:>4}  "
                  f"{freq[mzl]:>7,} occurrences")
        if len(misses) > MAX_MISSES_SHOWN:
            print(f"    ... and {len(misses) - MAX_MISSES_SHOWN} more")

    print()
    print("-" * 68)
    print("DOMAIN 2 VOCABULARY (read-only inspection)")
    print("-" * 68)
    inspect_vocab()

    # A sample of what Domain 3 will receive
    print()
    print("-" * 68)
    print("SAMPLE: first 12 corpus signs by frequency, as Domain 3 sees them")
    print("-" * 68)
    top = [m for m, _ in freq.most_common() if m != config.BROKEN_LABEL][:12]
    for mzl in top:
        e = index.get(str(mzl))
        tl = real[real["mzl_label"] == mzl]["train_label"]
        tl = int(tl.iloc[0]) if len(tl) else -1
        if e:
            print(f"  tl {tl:>4} | mzl {mzl:>4} | {e['sign']:<16} "
                  f"{e['ucun'] or '':<4} readings={e['readings'][:4]}")
        else:
            print(f"  tl {tl:>4} | mzl {mzl:>4} | *** NO OGSL ENTRY ***")

    # Write artifacts
    label_map = {}
    for _, row in pairs.iterrows():
        tl, mzl = int(row["train_label"]), int(row["mzl_label"])
        if mzl == config.BROKEN_LABEL:
            label_map[str(tl)] = {"sign": "<BROKEN>", "mzl": 0,
                                  "readings": [], "ucun": None,
                                  "note": "annotator damage placeholder"}
            continue
        e = index.get(str(mzl))
        label_map[str(tl)] = ({**e, "mzl": mzl} if e else
                              {"sign": None, "mzl": mzl, "readings": [],
                               "ucun": None, "note": "no OGSL entry"})

    config.D3_MZL_NAMES.write_text(
        json.dumps({"mzl": index, "collisions": collisions,
                    "variants": variants}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    config.D3_SIGN_NAMES.write_text(
        json.dumps(label_map, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"  Wrote {config.D3_MZL_NAMES}")
    print(f"  Wrote {config.D3_SIGN_NAMES}")
    print("=" * 68)


if __name__ == "__main__":
    main()