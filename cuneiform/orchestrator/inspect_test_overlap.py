"""
Checks whether my Domain 2 training data contains any of the 30 live
evaluation tablets.

1. Every live tablet's corpus lines are in the test split and no other.
2. How many live lines of 5+ tokens also appear word for word in training
   (shared formulae are expected and aren't a leak on their own).
3. For each live tablet, the largest share of its 5-token runs found in any
   single training tablet. A copy filed under another P-number would score near 1.0.

Run from the project root:
    python -m cuneiform.orchestrator.inspect_test_overlap
"""

from __future__ import annotations

import collections
import json

from cuneiform import config

LIVE_DIR = config.ARTIFACTS_DIR / "domain1" / "live_detections"
CORPUS = config.ARTIFACTS_DIR / "domain2" / "corpus_lines.jsonl"
N = 5
COPY_THRESHOLD = 0.9


def grams(tokens: list[int], n: int = N) -> set[tuple[int, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def main() -> None:
    live = sorted(p.stem for p in LIVE_DIR.glob("P*.json"))
    with open(CORPUS, encoding="utf8") as f:
        recs = [json.loads(line) for line in f]
    live_set = set(live)

    splits = collections.defaultdict(set)
    for r in recs:
        splits[r["tablet"]].add(r["split"])
    bad = [t for t in live if splits.get(t) != {"test"}]
    print(f"live tablets                 : {len(live)}")
    print(f"live tablets test-split only : {len(live) - len(bad)}  {bad or ''}")

    train = [r for r in recs if r["split"] == "train"]
    train_lines = {tuple(r["tokens"]) for r in train}
    live_lines = [r for r in recs if r["tablet"] in live_set]
    longer = [r for r in live_lines if len(r["tokens"]) >= N]
    verbatim = sum(tuple(r["tokens"]) in train_lines for r in longer)
    print(f"live lines of {N}+ tokens      : {len(longer)}, verbatim in train: {verbatim}")

    by_tab = collections.defaultdict(set)
    for r in train:
        by_tab[r["tablet"]] |= grams(r["tokens"])
    worst = (0.0, "", "")
    print(f"\n{'tablet':<9} {'runs':>5} {'best single train tablet':>26}")
    for t in live:
        g = set().union(*[grams(r["tokens"]) for r in live_lines if r["tablet"] == t])
        if not g:
            continue
        share, src = max((len(g & s) / len(g), k) for k, s in by_tab.items())
        print(f"{t:<9} {len(g):>5} {share:>18.2f}  {src}")
        worst = max(worst, (share, t, src))

    ok = not bad and worst[0] < COPY_THRESHOLD
    print(f"\nmax single-tablet share {worst[0]:.2f} ({worst[1]} vs {worst[2]})")
    print("PASS" if ok else "FAIL: if this failed, the no-overlap claim is void")


if __name__ == "__main__":
    main()
