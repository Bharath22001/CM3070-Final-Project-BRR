"""
My four-arm evaluation of Domain 3. BLEU/ROUGE isn't possible because there is
no openly licensed source of the English translations.

1. Structural validity: does every call return schema-valid JSON?
2. Faithfulness, scored separately on the citation list and the free prose.
3. Robustness: the same tablets at rising mask rates.
4. A CSV sample for checking by hand.

Faithfulness compares citations against both the bare and the displayed sign
forms, and reports both strict and normalised rates. Arm 3 records how many
signs were supplied, since coverage rises just because that number shrinks.
The prose check can't see one-character names (they clash with English words
like "A" and "I"), so it reports its own blind spot each run. Each tablet runs
twice by default because the model varies a lot between runs.

About 84 calls at ~11 s each, roughly 15 minutes.

Run from the project root:
    python -m cuneiform.domain3_generative.evaluate_domain3
    python -m cuneiform.domain3_generative.evaluate_domain3 --tablets 8 --repeats 1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter

from cuneiform import config
from cuneiform.domain3_generative.generate import (
    PROSE_SIGN_RE, PROSE_STOP_WORDS, SCHEMA, TabletInput, display_tokens,
    generate, load_catalogue, load_tablet,
)

MASK_RATES = [0.0, 0.2, 0.4, 0.6, 0.8]
CONF_ORDER = ["none", "low", "medium", "high"]

# Upper-case tokens that are English or bibliographic, not signs. Kept short on
# purpose, since names like AD, BA and DI are real signs. Shared with generate.py.
STOP = PROSE_STOP_WORDS

# A sign-shaped token: a |COMPOUND| or an uppercase run of two or more
# characters (including Š, subscripts, digits, ×, @ and dots). Shared with generate.py.
SIGN_RE = re.compile(PROSE_SIGN_RE)


def norm_sign(s: str) -> str:
    """
    Canonical key for comparing a citation with the input. Strips a trailing
    " (reading)" gloss only when a space comes before the bracket, so 3(DIŠ)
    survives, and drops pipes and trailing punctuation.
    """
    s = s.strip()
    if s.endswith(")"):
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == ")":
                depth += 1
            elif s[i] == "(":
                depth -= 1
                if depth == 0:
                    if i > 0 and s[i - 1].isspace():
                        s = s[:i].rstrip()
                    break
    return s.strip().strip("|").strip(".").strip("@").strip()


def gate_can_see(token: str) -> bool:
    """Would SIGN_RE find this token in prose? Used for the blind-spot report."""
    return token in set(SIGN_RE.findall(token))


def test_tablets() -> list[str]:
    """Held-out test tablets, ordered by size so samples are reproducible."""
    sizes: Counter = Counter()
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if rec.get("split") == "test":
                sizes[rec["tablet"]] += len(rec["tokens"])
    return [t for t, _ in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))]


def prose_of(out: dict) -> str:
    return " ".join(str(out.get(k, "")) for k in
                    ("tentative_reading", "uncertainty_notes"))


def score_one(t: TabletInput, out: dict) -> dict:
    """Both faithfulness surfaces for a single response."""
    # The reference set is everything displayed, in either form.
    supplied_raw = t.all_names | t.displayed_names
    supplied = {norm_sign(x) for x in supplied_raw}

    cited = sorted({s.strip() for s in out.get("signs_used", [])
                    if s and str(s).strip()})

    # The strict comparison, kept so the size of the normalisation is visible.
    strict_bad = [s for s in cited
                  if s not in t.all_names and s.strip("|") not in t.all_names]
    cited_bad = [s for s in cited if norm_sign(s) not in supplied]
    cited_good = {norm_sign(s) for s in cited} & supplied

    tokens = {c for c in SIGN_RE.findall(prose_of(out)) if c not in STOP}
    prose_bad = sorted(c for c in tokens if norm_sign(c) not in supplied)

    # How much of this tablet's displayed vocabulary the prose check can see.
    vis = invis = 0
    for v in t.lines:
        for tok, s in zip(display_tokens(v), v.slots):
            if s.kind in ("sign", "unnamed"):
                if gate_can_see(norm_sign(tok)):
                    vis += 1
                else:
                    invis += 1

    return {
        "schema_ok": all(k in out for k in SCHEMA["required"]),
        "confidence": out.get("confidence"),
        "n_supplied": len(supplied),
        "n_positions": sum(t.counts.values()),
        "n_cited": len(cited),
        "cited": cited,
        "n_cited_bad": len(cited_bad),
        "cited_bad": cited_bad,
        "n_cited_bad_strict": len(strict_bad),
        "coverage": len(cited_good) / len(supplied) if supplied else 0.0,
        "n_prose_tokens": len(tokens),
        "n_prose_bad": len(prose_bad),
        "prose_bad": prose_bad,
        "gate_visible": vis,
        "gate_invisible": invis,
        "metadata_claim": bool(out.get("metadata_used")),
        "metadata_truth": bool(out.get("_had_metadata")),
        "reading": out.get("tentative_reading", ""),
        "context": out.get("historical_context", ""),
    }


def fail_row(exc: Exception) -> dict:
    """A scored row for a call that never returned valid JSON."""
    return {"schema_ok": False, "error": f"{type(exc).__name__}: {exc}",
            "confidence": None, "n_supplied": 0, "n_positions": 0,
            "n_cited": 0, "cited": [], "n_cited_bad": 0, "cited_bad": [],
            "n_cited_bad_strict": 0, "coverage": 0.0, "n_prose_tokens": 0,
            "n_prose_bad": 0, "prose_bad": [], "gate_visible": 0,
            "gate_invisible": 0, "metadata_claim": False,
            "metadata_truth": False, "reading": "", "context": ""}


def run_arm_12(tablets: list[str], repeats: int, catalogue: dict) -> list[dict]:
    """Arms 1 and 2: structural validity and dual-surface faithfulness."""
    print("\n" + "=" * 70)
    print("ARMS 1 & 2 -- STRUCTURAL VALIDITY AND FAITHFULNESS")
    print("=" * 70)
    print(f"  {len(tablets)} tablets x {repeats} repeats "
          f"= {len(tablets) * repeats} calls\n")

    rows = []
    for i, tab in enumerate(tablets, 1):
        t = load_tablet(tab)
        for r in range(repeats):
            t0 = time.time()
            try:
                out = generate(t, catalogue)
                s = score_one(t, out)
                s["error"] = ""
            except Exception as exc:
                s = fail_row(exc)
            s.update(arm="1-2", tablet=tab, repeat=r, mask_rate=0.0,
                     damage=t.damage_rate, secs=round(time.time() - t0, 1))
            rows.append(s)
            flag = "" if s["schema_ok"] else f"  <-- FAILED: {s['error'][:70]}"
            print(f"  [{i:>2}/{len(tablets)}] {tab} r{r} "
                  f"conf={str(s['confidence']):<6} "
                  f"cited_bad={s['n_cited_bad']:<3}"
                  f"(strict {s['n_cited_bad_strict']:<3}) "
                  f"prose_bad={s['n_prose_bad']} "
                  f"{s['secs']}s{flag}")
    return rows


def run_arm_3(tablets: list[str], repeats: int, catalogue: dict) -> list[dict]:
    """Arm 3: degradation as upstream damage rises."""
    print("\n" + "=" * 70)
    print("ARM 3 -- ROBUSTNESS TO UPSTREAM DAMAGE")
    print("=" * 70)
    print(f"  {len(tablets)} tablets x {len(MASK_RATES)} rates x {repeats} "
          f"= {len(tablets) * len(MASK_RATES) * repeats} calls\n")

    rows = []
    for rate in MASK_RATES:
        for tab in tablets:
            t = load_tablet(tab, mask_rate=rate)
            for r in range(repeats):
                t0 = time.time()
                try:
                    out = generate(t, catalogue)
                    s = score_one(t, out)
                    s["error"] = ""
                except Exception as exc:
                    s = fail_row(exc)
                    print(f"    !! {tab} r{r} mask {rate:.0%}: {s['error'][:90]}")
                # Label the arm explicitly so the audit never has to guess it.
                s.update(arm="3", tablet=tab, repeat=r, mask_rate=rate,
                         damage=t.damage_rate, secs=round(time.time() - t0, 1))
                rows.append(s)
        done = [x for x in rows if x["mask_rate"] == rate]
        ok = [x for x in done if x["schema_ok"]]
        conf = Counter(x["confidence"] for x in ok)
        sup = statistics.mean([x["n_supplied"] for x in ok]) if ok else 0.0
        print(f"  mask {rate:.0%}: schema {len(ok)}/{len(done)} | "
              f"|supplied| {sup:.0f} | conf {dict(conf)} | "
              f"prose_bad/call "
              f"{sum(x['n_prose_bad'] for x in ok)/max(1,len(ok)):.2f}")
    return rows


def _faithfulness(ok: list[dict], label: str) -> None:
    cb = sum(r["n_cited_bad"] for r in ok)
    cbs = sum(r["n_cited_bad_strict"] for r in ok)
    ct = sum(r["n_cited"] for r in ok)
    per_call = [100 * r["n_cited_bad"] / r["n_cited"] for r in ok if r["n_cited"]]

    print(f"\n  ARM 2  STRUCTURED surface (signs_used) -- {label}:")
    print(f"           citations {ct}, not in input {cb} "
          f"({100 * cb / max(1, ct):.1f}%)  <- MICRO, pooled")
    if per_call:
        print(f"           per-call rate: mean {statistics.mean(per_call):.1f}% "
              f"| median {statistics.median(per_call):.1f}% "
              f"| max {max(per_call):.1f}%")
    print(f"           calls with >=1 bad citation: "
          f"{sum(1 for r in ok if r['n_cited_bad'])}/{len(ok)}")
    print(f"           mean coverage of input: "
          f"{sum(r['coverage'] for r in ok) / len(ok):.0%} "
          f"(mean |supplied| {statistics.mean([r['n_supplied'] for r in ok]):.0f})")
    print(f"           [un-normalised comparison, Day 5 method: {cbs} bad "
          f"({100 * cbs / max(1, ct):.1f}%) -- the gap is the gloss artefact]")

    worst = max(ok, key=lambda r: r["n_cited_bad"])
    print(f"           worst call: {worst['tablet']} r{worst['repeat']} "
          f"{worst['n_cited_bad']}/{worst['n_cited']}")

    pb = sum(r["n_prose_bad"] for r in ok)
    pt = sum(r["n_prose_tokens"] for r in ok)
    gv = sum(r["gate_visible"] for r in ok)
    gi = sum(r["gate_invisible"] for r in ok)
    print(f"         PROSE surface (tentative_reading + notes):")
    print(f"           sign-shaped tokens {pt}, not in input {pb} "
          f"({100 * pb / max(1, pt):.1f}%)")
    print(f"           calls with >=1 bad prose token: "
          f"{sum(1 for r in ok if r['n_prose_bad'])}/{len(ok)}")
    bad = Counter(x for r in ok for x in r["prose_bad"])
    print(f"           most common invented: {bad.most_common(8)}")
    print(f"           GATE SELF-AUDIT: SIGN_RE can match "
          f"{100 * gv / max(1, gv + gi):.1f}% of displayed sign tokens "
          f"({gi:,} of {gv + gi:,} invisible).")
    print(f"           The prose rate is a FLOOR over that stated fraction.")


def summarise(rows12: list[dict], rows3: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    allr = rows12 + rows3
    ok12 = [r for r in rows12 if r["schema_ok"]]
    ok3 = [r for r in rows3 if r["schema_ok"]]
    ok_all = len(ok12) + len(ok3)

    print(f"\n  ARM 1  schema-conformant")
    print(f"           arms 1-2 : {len(ok12)}/{len(rows12)}")
    print(f"           arm 3    : {len(ok3)}/{len(rows3)}")
    print(f"           OVERALL  : {ok_all}/{len(allr)} "
          f"({100 * ok_all / max(1, len(allr)):.1f}%)  <- REPORT THIS FIGURE")

    errs = [r["error"] for r in allr if r.get("error")]
    if errs:
        print(f"\n         {len(errs)} failed call(s):")
        for r in allr:
            if r.get("error"):
                print(f"           {r['tablet']} r{r['repeat']} "
                      f"mask {r['mask_rate']:.0%}: {r['error'][:100]}")
    else:
        print(f"\n         no failed calls.")

    secs = [r["secs"] for r in allr if r.get("secs")]
    if secs:
        print(f"\n         wall clock: {sum(secs) / 60:.1f} min total | "
              f"per call min {min(secs):.1f}s median "
              f"{statistics.median(secs):.1f}s max {max(secs):.1f}s")

    if ok12:
        _faithfulness(ok12, "ground-truth input")

        md = [r for r in ok12 if r["metadata_claim"] != r["metadata_truth"]]
        print(f"\n         metadata_used self-report wrong: {len(md)}/{len(ok12)}")

    if rows3:
        print("\n  ARM 3  degradation with upstream damage:")
        print(f"    {'mask':>6} {'schema':>8} {'|supplied|':>11} "
              f"{'conf(med+)':>11} {'prose_bad':>10} {'coverage':>9}")
        for rate in MASK_RATES:
            sub = [r for r in rows3 if r["mask_rate"] == rate and r["schema_ok"]]
            if not sub:
                continue
            high = sum(1 for r in sub if r["confidence"] in ("medium", "high"))
            print(f"    {rate:>5.0%} {len(sub):>8} "
                  f"{statistics.mean([r['n_supplied'] for r in sub]):>11.0f} "
                  f"{high / len(sub):>10.0%} "
                  f"{sum(r['n_prose_bad'] for r in sub) / len(sub):>10.2f} "
                  f"{sum(r['coverage'] for r in sub) / len(sub):>8.0%}")
        print("\n    80% is the detector's real measured operating point.")
        print("    ** READ THE |supplied| COLUMN BEFORE THE LAST TWO. **")
        print("    coverage and prose_bad are ratios over a SHRINKING set, so")
        print("    they improve as the input empties. The CONFIDENCE column is")
        print("    the finding; the other two are denominator artefacts.")


def write_outputs(rows: list[dict]) -> None:
    config.ensure_dirs()
    out_json = config.D3_DIR / "domain3_evaluation.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  wrote {out_json}")

    qual = config.D3_DIR / "domain3_qualitative.csv"
    with qual.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tablet", "mask_rate", "confidence", "reading",
                    "historical_context", "oracc_url"])
        seen = set()
        for r in rows:
            if r["mask_rate"] == 0.0 and r["tablet"] not in seen and r["schema_ok"]:
                seen.add(r["tablet"])
                w.writerow([r["tablet"], r["mask_rate"], r["confidence"],
                            r["reading"], r["context"],
                            f"http://oracc.org/saao/{r['tablet']}/"])
    print(f"  wrote {qual}  <- ARM 4: check these by eye against Oracc")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tablets", type=int, default=12,
                    help="tablets for arms 1-2 (default 12)")
    ap.add_argument("--sweep-tablets", type=int, default=6,
                    help="tablets for the arm 3 sweep (default 6)")
    ap.add_argument("--repeats", type=int, default=2,
                    help="calls per condition; variance is large (default 2)")
    args = ap.parse_args()

    print("=" * 70)
    print("DOMAIN 3 -- FOUR-ARM EVALUATION")
    print("=" * 70)

    catalogue = load_catalogue()
    all_tabs = test_tablets()
    total = args.tablets * args.repeats + \
        args.sweep_tablets * len(MASK_RATES) * args.repeats
    print(f"  test tablets available : {len(all_tabs)}")
    print(f"  catalogue records      : {len(catalogue):,}")
    print(f"  calls planned          : {total}")

    t0 = time.time()
    rows12 = run_arm_12(all_tabs[:args.tablets], args.repeats, catalogue)
    rows3 = run_arm_3(all_tabs[:args.sweep_tablets], args.repeats, catalogue)
    summarise(rows12, rows3)
    write_outputs(rows12 + rows3)

    print(f"\n  total wall clock: {(time.time() - t0) / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()