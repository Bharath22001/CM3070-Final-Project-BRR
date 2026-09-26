"""
Sweeps the conf threshold, cell factor and edge extrapolation together, since
they interact, and reports statistics that can actually tell configurations
apart: mean |error|, a count of tablets within the band, and over- and
under-recovery separately. It replaces inspect_segment_choice.py, which picked
a value on a median that one tablet could swing.

Run from the project root:
    python -m cuneiform.orchestrator.inspect_recovery_sweep
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from cuneiform import config
from cuneiform.orchestrator.live_adapter import (
    build_slots, grid_components, load_detections,
)

CONFS = (0.15, 0.25, 0.35, 0.50)
CELLS = (1.0, 1.5, 2.0)
RULE = "=" * 82


def head(t: str) -> None:
    print("\n" + RULE)
    print(t)
    print(RULE)


def corpus_tokens(tablet: str) -> int:
    n = 0
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            rec = json.loads(raw)
            if rec["tablet"] == tablet:
                n += sum(1 for t in rec["tokens"] if t != config.BROKEN_LABEL)
    return n


def cluster(dets, cell_factor, min_size):
    cx = np.array([d["cx"] for d in dets], float)
    cy = np.array([d["cy"] for d in dets], float)
    w = np.array([d["x2"] - d["x1"] for d in dets], float)
    cell = max(float(np.median(w)) * cell_factor, 1.0)
    labels = grid_components(cx, cy, cell)
    groups = [np.flatnonzero(labels == l) for l in np.unique(labels)]
    groups = [g for g in groups if len(g) >= min_size]
    groups.sort(key=lambda ix: (float(cy[ix].min()), float(cx[ix].min())))
    return groups


def stats(ratios: list[float]) -> dict:
    """
    Summary statistics with over- and under-recovery kept apart. 'band' counts
    tablets within +/-0.25 of 1.0, which one tablet can't swing the way it can
    swing a median.
    """
    r = np.array([x for x in ratios if not np.isnan(x)])
    if not len(r):
        return {"n": 0, "mean": float("nan"), "med": float("nan"),
                "band": 0, "n_over": 0, "n_under": 0, "worst_over": 0.0}
    err = np.abs(r - 1.0)
    return {
        "n": len(r),
        "mean": float(err.mean()),
        "med": float(np.median(err)),
        "band": int((err <= 0.25).sum()),
        "n_over": int((r > 1.25).sum()),
        "n_under": int((r < 0.75).sum()),
        "worst_over": float(r.max()),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=3,
                    help="3, not 5: at min 5 a sparse tablet yields NO cluster "
                         "at all and drops out of the sample entirely (P335597)")
    args = ap.parse_args()

    tablets = sorted(p.stem for p in config.D1_LIVE_DETECTIONS.glob("*.json")
                     if not p.stem.startswith("_"))
    toks = {t: corpus_tokens(t) for t in tablets}
    cache = {c: {t: load_detections(t, c) for t in tablets} for c in CONFS}

    head("A. THE QUESTION inspect_segment_choice.py COULD NOT ANSWER")
    print("  Does a smaller cell separate P335593's three side-by-side")
    print("  fragments? Cluster counts only -- no slot building.\n")
    print(f"  {'tablet':<12} {'conf':>6} " +
          " ".join(f"{c:>7}" for c in CELLS))
    for t in ("P335593", "P334932", "P335940", "P335941"):
        for c in (0.15, 0.25, 0.50):
            d = cache[c][t]
            counts = [len(cluster(d, cf, args.min_size)) if d else 0
                      for cf in CELLS]
            print(f"  {t:<12} {c:>6.2f} " +
                  " ".join(f"{n:>7}" for n in counts))

    head("B. JOINT SWEEP -- conf x cell x edge extrapolation")
    print("  band = tablets within +/-0.25 of ratio 1.0 (HIGHER IS BETTER).")
    print("  Selected on band, not median: a count cannot be moved by a single")
    print("  tablet crossing a rank boundary.\n")
    print(f"  {'conf':>5} {'cell':>5} {'edge':>5} {'n':>4} {'band':>5} "
          f"{'mean':>6} {'med':>6} {'over':>5} {'under':>6} {'worst':>7}")

    results = {}
    for cv in CONFS:
        for cf in CELLS:
            for edge in (False, True):
                ratios = []
                for t in tablets:
                    d = cache[cv][t]
                    g = cluster(d, cf, args.min_size) if d else []
                    if not g or not toks[t]:
                        ratios.append(float("nan"))
                        continue
                    ratios.append(
                        len(build_slots(d, g, extend_edges=edge)) / toks[t])
                s = stats(ratios)
                results[(cv, cf, edge)] = (s, ratios)
                print(f"  {cv:>5.2f} {cf:>5.1f} {'ON' if edge else 'off':>5} "
                      f"{s['n']:>4} {s['band']:>5} {s['mean']:>6.2f} "
                      f"{s['med']:>6.2f} {s['n_over']:>5} {s['n_under']:>6} "
                      f"{s['worst_over']:>7.2f}")

    best = max(results, key=lambda k: (results[k][0]["band"],
                                       -results[k][0]["mean"]))
    s, ratios = results[best]
    print(f"\n  best by band: conf {best[0]}  cell {best[1]}  "
          f"edge {'ON' if best[2] else 'off'}  "
          f"-> {s['band']}/{s['n']} in band, mean |err| {s['mean']:.2f}")

    head("C. IS THE WINNER ON A PLATEAU OR A SPIKE?")
    print("  A configuration that beats its own neighbours by 1 tablet is")
    print("  noise. Neighbours in each axis, band counts:\n")
    cv, cf, edge = best
    for label, keys in (
        ("conf", [(c, cf, edge) for c in CONFS]),
        ("cell", [(cv, c, edge) for c in CELLS]),
        ("edge", [(cv, cf, e) for e in (False, True)]),
    ):
        vals = [(k, results[k][0]["band"]) for k in keys if k in results]
        print(f"  vary {label:<5}: " +
              "  ".join(f"{k[{'conf':0,'cell':1,'edge':2}[label]]}={b}"
                        for k, b in vals))

    head("D. PER-TABLET AT THE WINNER")
    print(f"  {'tablet':<12} {'det':>5} {'toks':>6} {'clusters':>9} "
          f"{'ratio':>7}  verdict")
    for t, r in zip(tablets, ratios):
        d = cache[best[0]][t]
        g = cluster(d, best[1], args.min_size) if d else []
        if np.isnan(r):
            v = "NO CLUSTER"
        elif r > 1.25:
            v = "OVER -- duplicate view?"
        elif r < 0.75:
            v = "under -- too few anchors"
        else:
            v = "ok"
        print(f"  {t:<12} {len(d):>5} {toks[t]:>6} {len(g):>9} "
              f"{r:>7.2f}  {v}")

    print("\n" + RULE)


if __name__ == "__main__":
    main()