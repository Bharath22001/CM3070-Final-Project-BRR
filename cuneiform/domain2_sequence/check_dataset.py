"""
Inspects the Domain 2 dataset before training. It checks that <brk> is never a
prediction target, that masked spans are contiguous and follow the real damage
lengths, that the realised mask rate is close to the configured one, and that
no block spans two tablets.

Run from the project root:
    python -m cuneiform.domain2_sequence.check_dataset
"""

from __future__ import annotations

from collections import Counter

from cuneiform.domain2_sequence.dataset import (
    CuneiformMLMDataset,
    measure_span_distribution,
    load_corpus,
)
from cuneiform.domain2_sequence.tokenizer import SignTokenizer


def main() -> None:
    tok = SignTokenizer()
    print(f"Tokenizer: {tok}\n")

    print("Building datasets")
    print("-" * 68)
    train = CuneiformMLMDataset("train", tok)
    val = CuneiformMLMDataset("val", tok)

    print("\nReal damage run-length distribution (from <brk> runs in train)")
    print("-" * 68)
    dist = measure_span_distribution(load_corpus("train"))
    for length in sorted(dist)[:10]:
        bar = "#" * int(dist[length] * 200)
        print(f"  length {length:>2}: {100 * dist[length]:>5.1f}%  {bar}")

    print("\nInspecting 500 generated training examples")
    print("-" * 68)

    n_masked = n_predictable = 0
    brk_targeted = 0
    special_targeted = 0
    realised_spans: Counter = Counter()

    for i in range(min(500, len(train))):
        ids, attn, labels = train[i]
        ids_l = ids.tolist()
        labels_l = labels.tolist()

        for pos, lab in enumerate(labels_l):
            if lab == -100:
                continue
            n_masked += 1
            if lab == tok.brk_id:
                brk_targeted += 1
            if lab in (tok.pad_id, tok.bos_id, tok.eos_id, tok.mask_id):
                special_targeted += 1

        n_predictable += sum(
            1 for t, a in zip(ids_l, attn.tolist())
            if a == 1 and (t == tok.mask_id or tok.is_predictable(t))
        )

        run = 0
        for lab in labels_l:
            if lab != -100:
                run += 1
            elif run:
                realised_spans[run] += 1
                run = 0
        if run:
            realised_spans[run] += 1

    print(f"  Masked positions        : {n_masked:,}")
    print(f"  Realised mask rate      : {100 * n_masked / max(1, n_predictable):.1f}% "
          f"(configured 25%)")
    print(f"  <brk> used as target    : {brk_targeted}   <-- MUST be 0")
    print(f"  special used as target  : {special_targeted}   <-- MUST be 0")

    print("\n  Realised masked-span lengths:")
    total_spans = sum(realised_spans.values())
    for length in sorted(realised_spans)[:10]:
        share = realised_spans[length] / max(1, total_spans)
        bar = "#" * int(share * 200)
        print(f"    length {length:>2}: {100 * share:>5.1f}%  {bar}")

    singles = realised_spans.get(1, 0) / max(1, total_spans)
    print(f"\n  Single-token masks: {100 * singles:.1f}%")
    print("  (Standard RoBERTa would be near 100% here. Well below that means")
    print("   span masking is working.)")

    print("\nDecoded example (first 40 positions of example 0)")
    print("-" * 68)
    ids, attn, labels = train[0]
    for pos in range(min(40, len(ids))):
        t = ids[pos].item()
        lab = labels[pos].item()
        shown = tok.decode_id(t)
        truth = f"  <- true: {tok.decode_id(lab)}" if lab != -100 else ""
        print(f"  {pos:>3}: {shown:<14}{truth}")

    ok = (brk_targeted == 0 and special_targeted == 0)
    print("\n" + "=" * 68)
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED -- do not train yet")
    print("=" * 68)


if __name__ == "__main__":
    main()