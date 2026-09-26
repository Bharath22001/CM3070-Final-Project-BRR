"""
Checks whether the 0.08 posterior floor refuses the right positions.

Where Domain 2's best answer is below the floor, the pipeline shows [x] instead
of a guess. That always costs WER, so the question is whether the refused
guesses are wrong more often than the restorations that get shown.

I reuse evaluate_live's own code for everything up to the alignment, then align
the no-floor fusion sequence to the reference keeping each position's operation.
Positions fall into refused, restored and vision groups, and a position counts
as correct if it is an exact match. Both tie-breaks are run and reported.

Run from the project root:
    python -m cuneiform.orchestrator.evaluate_refusal
"""

from __future__ import annotations

import json
import sys

import numpy as np

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.orchestrator.evaluate_live import (
    LIVE_LAMBDA, LM_FLOOR, RESULTS_JSON, SENTINEL, THRESHOLD, align,
    assemble_arms, build_reference, live_slots, live_tablets, lm_pass,
    load_corpus_records, load_sign_names, load_sequence_model, score_unit,
)

OUT_JSON = config.ARTIFACTS_DIR / "refusal_floor_eval.json"
RULE = "=" * 78
SEED = 1729
N_BOOT = 20000
DEPLOYED = f"3_fusion_lam{LIVE_LAMBDA}"
FLOORED = "3f_deployed_lam1_floor"


def align_ops(pred: list[int], truth: list[int], prefer: str = "sub"):
    """
    The same DP and backtrace as evaluate_live.align, but returning the operation
    for each predicted position: "M" match, "S" substitution, "I" insertion.
    Gate R1 checks the distance matches align() on every tablet.
    """
    n, m = len(truth), len(pred)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    d[0] = list(range(m + 1))
    for i in range(1, n + 1):
        row, prev = d[i], d[i - 1]
        row[0] = i
        ti = truth[i - 1]
        for j in range(1, m + 1):
            c = prev[j - 1] + (0 if ti == pred[j - 1] else 1)
            a = prev[j] + 1
            b = row[j - 1] + 1
            row[j] = c if (c <= a and c <= b) else (a if a <= b else b)

    order = ("S", "D", "I") if prefer == "sub" else ("D", "I", "S")
    ops = [None] * m
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and truth[i - 1] == pred[j - 1]
                and d[i][j] == d[i - 1][j - 1]):
            ops[j - 1] = "M"
            i, j = i - 1, j - 1
            continue
        cand = set()
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            cand.add("S")
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            cand.add("D")
        if j > 0 and d[i][j] == d[i][j - 1] + 1:
            cand.add("I")
        op = next(o for o in order if o in cand)
        if op == "S":
            ops[j - 1] = "S"
            i, j = i - 1, j - 1
        elif op == "D":
            i -= 1
        else:
            ops[j - 1] = "I"
            j -= 1
    return d[n][m], ops


def boot_diff(per_tab: list[dict], a: str, b: str, rng) -> tuple[float, float]:
    """Tablet-cluster bootstrap CI of pooled accuracy(a) - pooled accuracy(b)."""
    ca = np.array([t[a]["M"] for t in per_tab], float)
    na = np.array([t[a]["n"] for t in per_tab], float)
    cb = np.array([t[b]["M"] for t in per_tab], float)
    nb = np.array([t[b]["n"] for t in per_tab], float)
    k = len(per_tab)
    diffs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, k, k)
        sa, sb = na[ix].sum(), nb[ix].sum()
        if sa == 0 or sb == 0:
            continue
        diffs.append(ca[ix].sum() / sa - cb[ix].sum() / sb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    config.check_environment()

    print(RULE)
    print("REFUSAL FLOOR -- does [x] withhold the guesses most likely to be wrong?")
    print(RULE)

    tok = SignTokenizer()
    names = load_sign_names()
    names[SENTINEL] = "[x]"
    tablets = live_tablets()
    refs, _ = build_reference(load_corpus_records(), tablets, tok)
    lives = {t: live_slots(t, tok=tok) for t in tablets}
    model = load_sequence_model()

    per_tab: dict[str, dict] = {p: [] for p in ("sub", "indel")}
    wer = {DEPLOYED: [], FLOORED: []}
    gate_r1 = []            # (tablet, prefer, dist_mine, dist_align)
    n_ref_total = n_res_total = 0

    for k, t in enumerate(tablets, 1):
        lt = lives[t]
        thr = lm_pass(model, tok, lt, THRESHOLD)
        fus = lm_pass(model, tok, lt, 0.0)
        arms = assemble_arms(tok, lt, thr, fus)
        pred, floored = arms[DEPLOYED], arms[FLOORED]
        truth = refs[t]
        dets = [lt.detections[si] for si in lt.scored]
        group = []
        for j, dd in enumerate(dets):
            if dd.is_informative:
                group.append("vision")
            elif floored[j] == SENTINEL:
                group.append("refused")
            else:
                group.append("restored")
        n_ref_total += group.count("refused")
        n_res_total += group.count("restored")
        for arm in (DEPLOYED, FLOORED):
            wer[arm].append(score_unit(arms[arm], truth, names)["wer"])
        for prefer in ("sub", "indel"):
            dist, ops = align_ops(pred, truth, prefer)
            gate_r1.append((t, prefer, dist, align(pred, truth, prefer)["dist"]))
            row = {g: {"n": 0, "M": 0, "S": 0, "I": 0}
                   for g in ("refused", "restored", "vision")}
            for g, op in zip(group, ops):
                row[g]["n"] += 1
                row[g][op] += 1
            row["tablet"] = t
            per_tab[prefer].append(row)
        print(f"  {k:>2}/{len(tablets)} {t}  refused {group.count('refused'):>3}  "
              f"restored {group.count('restored'):>3}", flush=True)

    # Gates, with the raw evidence beside each verdict
    pub = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["arms"]
    r1_bad = [g for g in gate_r1 if g[2] != g[3]]
    r1 = not r1_bad
    r2 = (n_ref_total, n_res_total) == (445, 914)
    mean_dep, mean_flo = float(np.mean(wer[DEPLOYED])), float(np.mean(wer[FLOORED]))
    r3 = (abs(mean_dep - pub[DEPLOYED]["wer"]) < 1e-9
          and abs(mean_flo - pub[FLOORED]["wer"]) < 1e-9)
    print("\n" + RULE)
    print("GATES")
    print(f"  R1 align_ops distance == evaluate_live.align on all "
          f"{len(gate_r1)} tablet x tie-break rows: {'PASS' if r1 else 'FAIL'}"
          f"  (mismatches {len(r1_bad)})")
    print(f"  R2 group sizes equal the service's 30-tablet counts "
          f"(declined 445, restored 914): {'PASS' if r2 else 'FAIL'}"
          f"  (got refused {n_ref_total}, restored {n_res_total})")
    print(f"  R3 reproduces the published arms' per-tablet mean WER: "
          f"{'PASS' if r3 else 'FAIL'}  ({DEPLOYED} {mean_dep:.6f} vs "
          f"{pub[DEPLOYED]['wer']:.6f}; {FLOORED} {mean_flo:.6f} vs "
          f"{pub[FLOORED]['wer']:.6f})")
    ok = r1 and r2 and r3
    print(f"  OVERALL: {'PASS' if ok else 'FAIL -- ignore every number below'}")

    # The result
    rng = np.random.default_rng(SEED)
    out = {"simulated_vision": False, "unit": "position, pooled; CI by tablet",
           "population": f"{len(tablets)} live tablets, scored positions of "
                         f"{DEPLOYED} (no floor)",
           "lm_floor": LM_FLOOR, "gates_pass": ok,
           "gates": {"R1": r1, "R2": r2, "R3": r3}, "tie_breaks": {}}
    print("\n" + RULE)
    print("RESULT  (correct = the position aligns to an identical reference sign)")
    for prefer in ("sub", "indel"):
        tot = {g: {o: sum(r[g][o] for r in per_tab[prefer])
                   for o in ("n", "M", "S", "I")}
               for g in ("refused", "restored", "vision")}
        acc = {g: tot[g]["M"] / max(1, tot[g]["n"]) for g in tot}
        lo, hi = boot_diff(per_tab[prefer], "restored", "refused", rng)
        print(f"\n  tie-break prefer={prefer}")
        for g in ("restored", "refused", "vision"):
            v = tot[g]
            print(f"    {g:9s} n={v['n']:>5}  correct {v['M']:>4} "
                  f"({acc[g]:.3f})  substituted {v['S']:>4}  inserted {v['I']:>4}")
        print(f"    restored minus refused accuracy: "
              f"{acc['restored'] - acc['refused']:+.3f}  "
              f"95% tablet-bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"    the floor withholds {tot['refused']['M']} correct guesses "
              f"and {tot['refused']['n'] - tot['refused']['M']} wrong ones")
        out["tie_breaks"][prefer] = {
            "totals": tot, "accuracy": acc,
            "restored_minus_refused": acc["restored"] - acc["refused"],
            "ci95": [lo, hi], "per_tablet": per_tab[prefer]}
    with OUT_JSON.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {OUT_JSON}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
