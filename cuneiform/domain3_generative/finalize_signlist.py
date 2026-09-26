"""
Finishes the MZL -> sign-name lookup and writes the file Domain 3 reads.

1. Searches the whole OGSL repository for MZL numbers missing from osl.asl.
2. Applies manual overrides that are stated in the source itself, with their
   citation stored alongside.
3. Resolves collisions with a written preference rule, keeping the
   alternatives (e.g. TI and TI@t).
4. Leads with the sign's own name as the reading, since OGSL lists readings
   alphabetically, not by importance.

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.finalize_signlist
"""

import json
import re
import sys
from collections import Counter

import pandas as pd

from cuneiform import config

MISSING_MZL = [851, 640, 852, 583, 729, 905, 271, 810, 378, 170, 925]
MZL_DECLARED_MAX = 907

# File types worth searching; binaries can't hold a plain-text MZL number.
TEXT_EXT = {".asl", ".lst", ".tab", ".tsv", ".txt", ".csv", ".json", ".xml",
            ".atf", ".tok", ".col", ".map", ".dump", ".q", ".html", ".log", ""}

# Sourced overrides. Each one carries its source.
MANUAL_OVERRIDES = {
    "640": {
        "sign": "KAM",
        "form": "|HI.BAD|",
        "readings": [],
        "ucun": None,
        "source": "osl.asl line 22123 (@inote under @sign KAM2): "
                  "'MZL640=KAM(|HI.BAD|)'",
    },
}


def _fix_console_encoding() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Job 1: search the repository
def search_repo_for_missing():
    root = config.D3_OGSL_DIR / "osl-master"
    files = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in TEXT_EXT]

    print("-" * 68)
    print("1. REPO-WIDE SEARCH FOR THE MISSING MZL NUMBERS")
    print("-" * 68)
    print(f"  searching {len(files):,} text files "
          f"(of 9,890 total; fonts/images/archives skipped)")

    pats = {m: re.compile(rf"MZL\s*0*{m}\b") for m in MISSING_MZL}
    found = {m: [] for m in MISSING_MZL}

    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "MZL" not in text and "mzl" not in text:
            continue
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if "MZL" not in ln:
                continue
            for m, pat in pats.items():
                if pat.search(ln):
                    found[m].append((p.relative_to(root), i + 1, ln.strip()))

    for m in MISSING_MZL:
        hits = found[m]
        flag = "  [> declared MZL max 907: invalid MZL number]" \
            if m > MZL_DECLARED_MAX else ""
        print(f"\n  MZL{m}  ({len(hits)} hits){flag}")
        seen = set()
        for rel, ln, txt in hits[:6]:
            key = (str(rel), txt[:80])
            if key in seen:
                continue
            seen.add(key)
            print(f"    {rel} : {ln}")
            print(f"      {txt[:150]}")
        if len(hits) > 6:
            print(f"    ... {len(hits) - 6} more hits")
    return found


# Job 3: collisions
def collision_rank(entry):
    """Lower is better. An explicit rule so the choice doesn't depend on dict order."""
    score = 0.0
    name = entry.get("sign") or ""
    if entry.get("deprecated"):
        score += 100          # OGSL marked it withdrawn
    if entry.get("fake"):
        score += 100          # not a genuine sign
    if entry.get("scope") == "form":
        score += 10           # prefer the top-level sign record
    if "@" in name:
        score += 20           # TI@t, DUN3@g, NAGA@180 are glyph variants
    if "|" in name:
        score += 5            # prefer an atomic sign over a compound
    score += len(name) * 0.01
    return score


def resolve(index, collisions):
    resolved, alternates = dict(index), {}
    for key, entries in collisions.items():
        ranked = sorted(entries, key=collision_rank)
        resolved[key] = ranked[0]
        others = [e["sign"] for e in ranked[1:]
                  if e["sign"] != ranked[0]["sign"]]
        if others:
            alternates[key] = sorted(set(others))
    return resolved, alternates


# Job 4: reading order
def canonical_readings(sign, readings, limit=5):
    """
    Put the reading that matches the sign's name first and keep the rest in
    source order. Returns (ordered_list, name_was_present).
    """
    if not readings:
        return [], False
    target = (sign or "").lower().strip("|")
    lowered = [r.lower() for r in readings]
    if target in lowered:
        i = lowered.index(target)
        return [readings[i]] + [r for j, r in enumerate(readings)
                                if j != i][:limit - 1], True
    return readings[:limit], False


# Build
def main() -> None:
    _fix_console_encoding()
    config.ensure_dirs()

    print("=" * 68)
    print("FINALIZE SIGN LIST")
    print("=" * 68)

    search_repo_for_missing()

    data = json.loads(config.D3_MZL_NAMES.read_text(encoding="utf-8"))
    index, collisions = data["mzl"], data["collisions"]
    resolved, alternates = resolve(index, collisions)

    print()
    print("-" * 68)
    print("2. COLLISION RESOLUTION (explicit preference rule)")
    print("-" * 68)
    frames = [pd.read_csv(p)
              for p in sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    mzl_freq = Counter(df["mzl_label"].astype(int).tolist())
    ours = set(mzl_freq) - {config.BROKEN_LABEL}

    for key in sorted((k for k in collisions if int(k) in ours),
                      key=lambda k: -mzl_freq[int(k)]):
        chosen = resolved[key]["sign"]
        alts = alternates.get(key, [])
        note = "" if alts else "   (same name in both scopes -- no ambiguity)"
        print(f"  MZL{key:<5} {mzl_freq[int(key)]:>7,} occ  -> {chosen:<16}"
              f"{'alt: ' + ', '.join(alts) if alts else ''}{note}")

    # Manual overrides
    print()
    print("-" * 68)
    print("3. SOURCED MANUAL OVERRIDES")
    print("-" * 68)
    for key, ov in MANUAL_OVERRIDES.items():
        if key in resolved:
            print(f"  MZL{key}: already present as "
                  f"{resolved[key]['sign']} -- override SKIPPED")
            continue
        resolved[key] = {"sign": ov["sign"], "scope": "manual",
                         "readings": ov["readings"], "uname": None,
                         "ucun": ov["ucun"], "oid": None,
                         "deprecated": False, "fake": False,
                         "manual_source": ov["source"]}
        print(f"  MZL{key} -> {ov['sign']}  ({ov['form']})")
        print(f"    source: {ov['source']}")

    # Final per-train_label lookup
    pairs = df[["train_label", "mzl_label"]].drop_duplicates().astype(int)
    tl_freq = Counter(df["train_label"].astype(int).tolist())
    lookup, unnamed = {}, []

    for _, row in pairs.iterrows():
        tl, mzl = int(row["train_label"]), int(row["mzl_label"])
        if mzl == config.BROKEN_LABEL:
            lookup[str(tl)] = {"sign": "<BROKEN>", "mzl": 0, "readings": [],
                               "ucun": None, "status": "damage_placeholder"}
            continue
        e = resolved.get(str(mzl))
        if not e:
            lookup[str(tl)] = {
                "sign": None, "mzl": mzl, "readings": [], "ucun": None,
                "status": ("invalid_mzl_number" if mzl > MZL_DECLARED_MAX
                           else "absent_from_ogsl"),
            }
            unnamed.append((tl, mzl, tl_freq.get(tl, 0)))
            continue
        rd, named = canonical_readings(e["sign"], e.get("readings") or [])
        lookup[str(tl)] = {
            "sign": e["sign"],
            "mzl": mzl,
            "ucun": e.get("ucun"),
            "readings": rd,
            "canonical_reading_first": named,
            "readings_unranked": not named,
            "alternates": alternates.get(str(mzl), []),
            "status": "manual" if e.get("manual_source") else "ogsl",
            **({"manual_source": e["manual_source"]}
               if e.get("manual_source") else {}),
        }

    vocab = json.loads((config.D2_DIR / "vocab.json").read_text(encoding="utf-8"))
    total_real = sum(n for k, n in tl_freq.items() if k != config.BROKEN_LABEL)

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
                tok_named += n
            else:
                miss.append((tl, n))
        print(f"\n  {label}")
        print(f"    named            : {named}/{named + len(miss)} "
              f"({named / max(named + len(miss), 1):.1%})")
        print(f"    token-weighted   : {tok_named / max(tok_all, 1):.2%}")
        for tl, n in sorted(miss, key=lambda x: -x[1]):
            print(f"      unnamed tl {tl:>4}  {n:>7,} occ")

    print()
    print("-" * 68)
    print("4. FINAL COVERAGE")
    print("-" * 68)
    cov("Domain 2 vocabulary (282)", vocab["sign_to_id"].keys())
    cov("Domain 1 emit classes (82)", vocab["vision_emit_classes"])

    # What Domain 3 will receive
    print()
    print("-" * 68)
    print("5. PREVIEW: a real corpus line as Domain 3 will see it")
    print("-" * 68)
    key_cols = ["tablet_CDLI", "view_desc", "segm_idx", "line_idx"]
    d = df.copy()
    d["view_desc"] = d["view_desc"].fillna("NA")
    grp = d.sort_values("pos_idx").groupby(key_cols, sort=False)
    for gi, (_, g) in enumerate(grp):
        if len(g) < 8 or gi > 400:
            continue
        row_ids = g["train_label"].astype(int).tolist()[:12]
        print(f"\n  raw train_labels : {row_ids}")
        print(f"  verbalised       : ", end="")
        out = []
        for tl in row_ids:
            r = lookup.get(str(tl), {})
            out.append("[damaged]" if r.get("sign") == "<BROKEN>"
                       else (r.get("sign") or f"<unnamed:{tl}>"))
        print(" ".join(out))
        break

    config.D3_SIGN_LOOKUP.write_text(
        json.dumps({"lookup": lookup, "alternates": alternates,
                    "notes": {"readings_are_alphabetical_not_ranked": True,
                              "mzl_declared_max": MZL_DECLARED_MAX}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  Wrote {config.D3_SIGN_LOOKUP}")
    print("=" * 68)


if __name__ == "__main__":
    main()