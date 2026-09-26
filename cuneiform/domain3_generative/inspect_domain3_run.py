"""
Audits artifacts/domain3/domain3_evaluation.json after a run. Read-only, no
Ollama calls.

Rows are split by their explicit `arm` field rather than by any side effect of
the data. Sections E and F test the current code, so they turn to PASS once a
fix lands.

Run from the project root:
    python -m cuneiform.domain3_generative.inspect_domain3_run
"""

from __future__ import annotations

import hashlib
import inspect
import json
import statistics
import sys
from collections import Counter

from cuneiform import config
from cuneiform.domain3_generative.generate import (
    build_prompt, display_tokens, load_catalogue, load_tablet,
)
from cuneiform.domain3_generative.evaluate_domain3 import (
    MASK_RATES, SIGN_RE, STOP, norm_sign, test_tablets,
)

RULE = "=" * 70
DASH = "-" * 70


def head(n: str) -> None:
    print("\n" + RULE)
    print(n)
    print(RULE)


def verdict(name: str, ok: bool | None, note: str = "") -> None:
    tag = "PASS" if ok is True else ("FAIL" if ok is False else "VACUOUS")
    print(f"\n  --> {tag}: {name}")
    if note:
        print(f"      {note}")


def split_arms(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows by the explicit `arm` field, announcing any fallback for older files."""
    if any("arm" in r for r in rows):
        return ([r for r in rows if r.get("arm") == "1-2"],
                [r for r in rows if r.get("arm") == "3"])
    print("  !! NO `arm` FIELD -- this file predates the Day 6 fix.")
    print("     Falling back to the secs==0 heuristic, which is UNRELIABLE.")
    return ([r for r in rows if r.get("secs")],
            [r for r in rows if not r.get("secs")])


# A. Call accounting

def section_a(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    head("A. CALL ACCOUNTING  (keys on: the `arm` field)")

    arm12, arm3 = split_arms(rows)
    print(f"  rows in file  : {len(rows)}")
    print(f"  arms 1-2      : {len(arm12)}")
    print(f"  arm 3         : {len(arm3)}")
    print(f"  unclassified  : {len(rows) - len(arm12) - len(arm3)}")

    print("\n  arm 3 rows per mask rate (evidence beside the verdict):")
    for rate in MASK_RATES:
        sub = [r for r in arm3 if abs(r["mask_rate"] - rate) < 1e-9]
        ok = [r for r in sub if r["schema_ok"]]
        print(f"    {rate:>5.0%}  rows {len(sub):>3}  schema-ok {len(ok):>3}")

    a12_ok = sum(1 for r in arm12 if r["schema_ok"])
    a3_ok = sum(1 for r in arm3 if r["schema_ok"])
    print(f"\n  schema-conformant, arms 1-2 : {a12_ok}/{len(arm12)}")
    print(f"  schema-conformant, arm 3    : {a3_ok}/{len(arm3)}")
    print(f"  schema-conformant, OVERALL  : {a12_ok + a3_ok}/{len(rows)}")

    secs = [r["secs"] for r in rows if r.get("secs")]
    if secs:
        print(f"\n  wall clock (all arms): {sum(secs) / 60:.1f} min | "
              f"per call min {min(secs):.1f}s median "
              f"{statistics.median(secs):.1f}s max {max(secs):.1f}s")
        print(f"  calls timed: {len(secs)}/{len(rows)}")

    verdict("every row is classified into an arm",
            len(arm12) + len(arm3) == len(rows),
            f"{len(rows) - len(arm12) - len(arm3)} row(s) fell through.")
    return arm12, arm3


# B. Failed calls

def section_b(rows: list[dict]) -> None:
    head("B. FAILED CALLS")

    bad = [r for r in rows if (not r["schema_ok"]) or r.get("error")]
    print(f"  rows with schema_ok=False or a non-empty error: {len(bad)}")

    if not bad:
        verdict("failed-call classification", None, "no failures to classify.")
        return

    catalogue = load_catalogue()
    truncated = 0
    for r in bad:
        print("\n" + DASH)
        print(f"  tablet {r['tablet']}  repeat {r['repeat']}  "
              f"mask {r['mask_rate']:.0%}  arm {r.get('arm', '?')}  "
              f"{r.get('secs', 0)}s")
        err = r.get("error", "")
        print(f"  error raw   : {err!r}")
        if "TRUNCATED AT CAP" in err:
            truncated += 1
            print("  -> hit num_predict. NOT a malformed model output: the")
            print("     JSON was valid up to the point generation stopped.")
        try:
            t = load_tablet(r["tablet"], mask_rate=r["mask_rate"])
            prompt, had = build_prompt(t, catalogue)
            print(f"  positions   : {sum(t.counts.values())}  "
                  f"lines {len(t.lines)}  distinct supplied "
                  f"{len(t.all_names)}  metadata {had}")
            print(f"  prompt chars: {len(prompt):,}  "
                  f"(~{len(prompt) / 3.5:,.0f} tokens vs num_ctx 8192)")
        except Exception as exc:
            print(f"  could not re-derive input: {type(exc).__name__}: {exc}")

    print()
    verdict("no call was cut off by the generation cap",
            truncated == 0,
            f"{truncated} call(s) truncated. Compare `distinct supplied` "
            f"against the response size: if the model emitted far more "
            f"entries than the tablet has distinct signs, it looped, and the "
            f"fix is the prompt, not a bigger cap.")


# C. Faithfulness distribution

def section_c(arm12: list[dict]) -> None:
    head("C. FAITHFULNESS DISTRIBUTION  (arms 1-2 only)")

    ok = [r for r in arm12 if r["schema_ok"]]
    if not ok:
        verdict("outlier analysis", None, "no schema-ok rows.")
        return

    bads = [r["n_cited_bad"] for r in ok]
    cits = [r["n_cited"] for r in ok]
    strict = [r.get("n_cited_bad_strict", 0) for r in ok]
    per_call = [100 * b / c for b, c in zip(bads, cits) if c]

    print(f"  calls scored          : {len(ok)}")
    print(f"  bad citations         : {sum(bads)} of {sum(cits)}")
    print(f"  MICRO (pooled)        : "
          f"{100 * sum(bads) / max(1, sum(cits)):.1f}%")
    print(f"  MACRO (mean per call) : "
          f"{statistics.mean(per_call) if per_call else 0:.1f}%")
    print(f"  MEDIAN per call       : "
          f"{statistics.median(per_call) if per_call else 0:.1f}%")
    print(f"  n_cited_bad: median {statistics.median(bads):.1f} "
          f"| mean {statistics.mean(bads):.1f} | max {max(bads)}")
    print(f"  clean calls (0 bad)   : {sum(1 for b in bads if not b)}/{len(ok)}")

    print(f"\n  UN-NORMALISED (Day 5 method, gloss not stripped):")
    print(f"    bad {sum(strict)} of {sum(cits)} "
          f"({100 * sum(strict) / max(1, sum(cits)):.1f}%)")
    print(f"    the gap between this and MICRO above is the SCORER "
          f"correction,")
    print(f"    not a change in model behaviour. Say so in the report.")

    print("\n  worst five calls:")
    for r in sorted(ok, key=lambda x: -x["n_cited_bad"])[:5]:
        print(f"    {r['tablet']} r{r['repeat']}  cited {r['n_cited']:>4}  "
              f"bad {r['n_cited_bad']:>3}  (strict "
              f"{r.get('n_cited_bad_strict', 0):>3})  conf={r['confidence']}")

    worst = max(ok, key=lambda x: x["n_cited_bad"])
    if worst["n_cited_bad"]:
        print("\n" + DASH)
        print(f"  FULL DUMP -- {worst['tablet']} r{worst['repeat']}")
        print(DASH)
        print(f"  cited_bad ({len(worst['cited_bad'])}): {worst['cited_bad']}")
        print(f"\n  reading:\n    {str(worst.get('reading', ''))[:600]}")
        twin = [r for r in ok if r["tablet"] == worst["tablet"]
                and r["repeat"] != worst["repeat"]]
        if twin:
            print(f"\n  same tablet, other repeat: cited {twin[0]['n_cited']}, "
                  f"bad {twin[0]['n_cited_bad']}  <- run-to-run variance")

    # A zero median makes a ratio meaningless, so I test the absolute spread.
    med = statistics.median(bads)
    if med == 0:
        dominated = max(bads) > 0.25 * max(1, sum(bads))
        note = (f"median is 0, so a ratio is undefined. Worst call holds "
                f"{max(bads)} of {sum(bads)} total bad citations.")
    else:
        dominated = max(bads) / med >= 5
        note = f"worst call is {max(bads) / med:.0f}x the median."
    verdict("no single call dominates the pooled figure", not dominated, note)


# D. The coverage denominator

def section_d(arm3: list[dict]) -> None:
    head("D. ARM 3 COVERAGE DENOMINATOR  (uses the run's OWN saved rows)")

    print(f"    {'mask':>5} {'n':>4} {'|supplied|':>11} {'coverage':>9} "
          f"{'conf(med+)':>11}")
    sup_means, cov_means = [], []
    for rate in MASK_RATES:
        sub = [r for r in arm3
               if abs(r["mask_rate"] - rate) < 1e-9 and r["schema_ok"]]
        if not sub:
            print(f"    {rate:>4.0%}   -- no rows --")
            continue
        s = statistics.mean([r.get("n_supplied", 0) for r in sub])
        c = 100 * statistics.mean([r["coverage"] for r in sub])
        h = 100 * sum(1 for r in sub
                      if r["confidence"] in ("medium", "high")) / len(sub)
        sup_means.append(s)
        cov_means.append(c)
        print(f"    {rate:>4.0%} {len(sub):>4} {s:>11.1f} {c:>8.0f}% {h:>10.0f}%")

    falling = all(a >= b for a, b in zip(sup_means, sup_means[1:]))
    rising = len(cov_means) > 1 and cov_means[-1] > cov_means[0]
    print()
    verdict("the coverage denominator is stable across mask rates",
            not falling,
            f"|supplied| falls monotonically ({sup_means[0]:.0f} -> "
            f"{sup_means[-1]:.0f}) while coverage "
            f"{'RISES' if rising else 'moves'}. Coverage is a ratio over a "
            f"shrinking set. Report |supplied| beside it; the CONFIDENCE "
            f"column is the finding."
            if falling else "denominator holds steady.")


# E. Masking reproducibility (tests the current seeding)

def section_e(sweep: list[str]) -> None:
    head("E. IS ARM 3 MASKING REPRODUCIBLE?  (keys on: a PERSISTED fingerprint)")

    # The substring check below ignores comments and is only a hint. The real
    # test compares the mask pattern with a fingerprint saved on an earlier run,
    # since hash salting only shows up across processes.
    src = inspect.getsource(load_tablet)
    code = "\n".join(line.split("#")[0] for line in src.splitlines())
    salted = "__hash__()" in code or "hash(" in code
    print(f"  seeds from a salted builtin hash (comments stripped): {salted}")

    store = config.D3_DIR / "mask_fingerprints.json"
    prev: dict = {}
    if store.exists():
        try:
            prev = json.loads(store.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    cur: dict = {}
    print(f"\n  mask fingerprints @40% (compared against {store.name}):")
    for tab in sweep[:3]:
        t = load_tablet(tab, mask_rate=0.4)
        pat = "".join("1" if s.kind == "unknown" else "0"
                      for v in t.lines for s in v.slots)
        fp = hashlib.sha1(pat.encode()).hexdigest()[:16]
        cur[tab] = fp
        was = prev.get(tab)
        mark = ("first run" if was is None
                else ("MATCH" if was == fp else f"CHANGED (was {was})"))
        print(f"    {tab:<10} {fp}  masked {pat.count('1'):>4}/{len(pat):<4} {mark}")

    store.write_text(json.dumps(cur, indent=2), encoding="utf-8")

    compared = [k for k in cur if k in prev]
    mismatch = [k for k in compared if prev[k] != cur[k]]
    if not compared:
        verdict("masking is reproducible across processes", None,
                "baseline written; this gate tests nothing until the NEXT run. "
                "Run this script again to make it meaningful.")
    else:
        verdict("masking is reproducible across processes",
                not mismatch and not salted,
                f"{len(mismatch)} of {len(compared)} fingerprints changed "
                f"between processes."
                if mismatch else
                f"{len(compared)} fingerprint(s) reproduced exactly.")


# F. Prose gate blind spot (tests the current regex)

def section_f(tablets: list[str]) -> None:
    head("F. PROSE GATE BLIND SPOT  (keys on: the live SIGN_RE)")

    print(f"  SIGN_RE = {SIGN_RE.pattern}")
    print("  A name SIGN_RE cannot match can never be counted as invented,")
    print("  so it is invisible to the prose faithfulness figure.\n")

    seen: Counter = Counter()
    for tab in tablets:
        try:
            t = load_tablet(tab)
        except Exception:
            continue
        for v in t.lines:
            for tok, s in zip(display_tokens(v), v.slots):
                if s.kind in ("sign", "unnamed"):
                    seen[norm_sign(tok)] += 1

    invisible = {n: c for n, c in seen.items()
                 if n not in set(SIGN_RE.findall(n)) and n not in STOP}
    tot = sum(seen.values())
    inv = sum(invisible.values())
    single = sum(c for n, c in invisible.items() if len(n) == 1)

    print(f"  distinct names          : {len(seen)}")
    print(f"  invisible to SIGN_RE    : {len(invisible)}")
    print(f"  token-weighted invisible: {100 * inv / max(1, tot):.1f}%  "
          f"({inv:,} of {tot:,})")
    print(f"    of which single-character (A, E, I, U): "
          f"{100 * single / max(1, tot):.1f}%  <- UNFIXABLE ON PURPOSE:")
    print(f"    these collide with ordinary English words and matching them")
    print(f"    would corrupt the metric in the other direction.")
    print(f"    residual, fixable: "
          f"{100 * (inv - single) / max(1, tot):.1f}%")
    print("\n  most frequent invisible names:")
    for n, c in Counter(invisible).most_common(12):
        print(f"    {n:<14} {c:>6}")

    verdict("the fixable blind spot is closed",
            (inv - single) / max(1, tot) < 0.02,
            f"{100 * (inv - single) / max(1, tot):.1f}% of displayed tokens "
            f"are invisible for reasons other than being one character long. "
            f"State the total blind spot as a stated limitation: the prose "
            f"rate is a FLOOR over a KNOWN fraction of the vocabulary.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    path = config.D3_DIR / "domain3_evaluation.json"
    print(RULE)
    print("DOMAIN 3 -- POST-HOC RUN AUDIT")
    print(RULE)
    print(f"  reading {path}")
    if not path.exists():
        print("  NOT FOUND. Nothing below can run.")
        return
    rows = json.loads(path.read_text(encoding="utf-8"))

    arm12, arm3 = section_a(rows)
    section_b(rows)
    section_c(arm12)
    section_d(arm3)

    all_tabs = test_tablets()
    section_e(all_tabs[:6])
    section_f(all_tabs[:12])

    print("\n" + RULE)


if __name__ == "__main__":
    main()