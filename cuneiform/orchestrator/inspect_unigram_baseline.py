"""
Works out where the 6.2% unigram baseline comes from, by computing it on every
population it could refer to, since the choice moves the LM floor from 0.078
to 0.097. It also prints the top of the corpus frequency order, to check the
claim that my old YOLOS detector fell back on the most frequent classes
(14, 11, 26). The detector and corpus numbering only match if nothing was
remapped, so an agreement is stated with that assumption. No model is loaded.

Run from the project root:
    python -m cuneiform.orchestrator.inspect_unigram_baseline
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

from cuneiform import config
from cuneiform.domain2_sequence.dataset import build_blocks, load_corpus
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.domain3_generative.verbalise import resolve

# Same seed as check_lm_inference.py, so the "sampled" row is the same population.
SEED = 1729
CAL_BLOCKS = 120

RULE = "=" * 76


def share(counts: Counter) -> tuple[float, object, int]:
    """Top class share, the class, and the denominator."""
    if not counts:
        return (0.0, None, 0)
    top, n = counts.most_common(1)[0]
    return (n / sum(counts.values()), top, sum(counts.values()))


def name_of(train_label) -> str:
    """Display name for a train_label, via the tested resolve() path."""
    try:
        return resolve(int(train_label), strict=False).text
    except Exception:
        return f"#{train_label}"


def corpus_counts() -> dict[str, Counter]:
    """
    Raw train_label frequencies per split, straight from the corpus. The broken
    label is excluded, since it is never a prediction target.
    """
    out: dict[str, Counter] = {"all": Counter()}
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            split = rec.get("split", "?")
            out.setdefault(split, Counter())
            for t in rec["tokens"]:
                if t == config.BROKEN_LABEL:
                    continue
                out[split][t] += 1
                out["all"][t] += 1
    return out


def block_counts(tok: SignTokenizer, split: str, sampled: bool) -> Counter:
    """
    Frequencies over block token ids, which is what check_lm_inference counted.
    `sampled` repeats its shuffle and slice with the same seed.
    """
    blocks = build_blocks(load_corpus(split), tok)
    if sampled:
        rng = random.Random(SEED)
        rng.shuffle(blocks)
        blocks = blocks[:CAL_BLOCKS]
    counts = Counter()
    for body in blocks:
        for t in body:
            if tok.is_predictable(t) and t != tok.unk_id:
                counts[tok.id_to_sign[t]] += 1
    return counts


def row(label: str, counts: Counter) -> None:
    pct, top, n = share(counts)
    flag = "  <-- 6.2%" if 6.15 <= 100 * pct <= 6.25 else ""
    print(f"  {label:<34} {100 * pct:>5.2f}%   top={str(top):>4} "
          f"{name_of(top):<10} n={n:>8,}{flag}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    tok = SignTokenizer()

    print(RULE)
    print("UNIGRAM BASELINE -- WHICH POPULATION GIVES 6.2%?")
    print(RULE)
    print("  Share of the commonest sign, BROKEN_LABEL excluded throughout.\n")

    cc = corpus_counts()
    for split in ("all", "train", "val", "test"):
        if split in cc:
            row(f"raw corpus tokens, {split}", cc[split])

    print()
    for split in ("train", "val", "test"):
        try:
            row(f"block tokens, {split} (all blocks)",
                block_counts(tok, split, sampled=False))
        except Exception as exc:
            print(f"  block tokens, {split}: skipped ({exc})")

    try:
        row(f"block tokens, test ({CAL_BLOCKS} sampled)",
            block_counts(tok, "test", sampled=True))
        print("      ^ this is the exact population arm D measured")
    except Exception as exc:
        print(f"  sampled: skipped ({exc})")

    print("\n" + RULE)
    print("READING THIS")
    print(RULE)
    print("  Whichever row is flagged is where the recorded 6.2% came from.")
    print("  The floor in FIX 3 should be calibrated against the population")
    print("  the DECISION operates on -- test-split blocks -- whether or not")
    print("  that is the population the recorded figure describes. If they")
    print("  differ, report both and say which one the floor used.")
    print("  If NO row is flagged, the 6.2% describes something this script")
    print("  does not reproduce, and it should be re-derived rather than")
    print("  quoted.")

    print("\n" + RULE)
    print("DOES CORPUS FREQUENCY ORDER MATCH THE 5.3 YOLOS COLLAPSE ORDER?")
    print(RULE)
    print("  Draft report 5.3 observed, on every query token and every image:")
    print("      Class 14, then Class 11, then Class 26, behind background.")
    print("  It concluded the detector had fallen back on marginal class")
    print("  frequency. If that is right, the corpus order starts 14, 11, 26.\n")

    top = cc["all"].most_common(10)
    total = sum(cc["all"].values())
    for i, (lab, n) in enumerate(top, 1):
        print(f"  {i:>2}. train_label {lab:>4}  {name_of(lab):<12} "
              f"{n:>7,}  {100 * n / total:>5.2f}%")

    got = [lab for lab, _ in top[:3]]
    print(f"\n  corpus top three : {got}")
    print(f"  5.3 collapse order: [14, 11, 26]")
    if got == [14, 11, 26]:
        print("  --> ORDERS AGREE. State it as agreement between the detector's")
        print("      collapsed ranking and corpus frequency, and state the")
        print("      assumption that the two label spaces share a numbering.")
    elif set(got) == {14, 11, 26}:
        print("  --> SAME THREE CLASSES, DIFFERENT ORDER. Still strong support")
        print("      for the frequency-shortcut reading; report the orders as")
        print("      they are rather than claiming an exact match.")
    else:
        print("  --> ORDERS DIFFER. The 5.3 claim must be softened: the")
        print("      detector collapsed to SOME classes, not demonstrably to")
        print("      the most frequent ones. Do not leave the stronger wording")
        print("      in the rewrite.")
    print(RULE)


if __name__ == "__main__":
    main()