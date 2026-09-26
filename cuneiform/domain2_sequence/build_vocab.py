"""
Builds the shared vocabulary between Domain 1 (vision) and Domain 2 (sequence).

Domain 2's vocabulary is the union of signs common enough in the text corpus
and signs the detector is allowed to emit. The detector may only emit signs
with at least 20 training boxes; everything else, plus missed or low-confidence
detections, becomes <mask>. Taking the union guarantees every class the
detector can name has a Domain 2 token, and keeping Domain 2 larger lets it
predict signs the detector can't name.

Run from the project root:
    python -m cuneiform.domain2_sequence.build_vocab
"""

from __future__ import annotations

import json
from collections import Counter

from cuneiform import config

# A sign gets a token if it appears this many times in the training text.
MIN_CORPUS_COUNT = 10

# Minimum training boxes before the detector may emit a sign. Below this its
# AP is near zero, and an honest <mask> is better than a confident wrong guess.
MIN_DETECT_BOXES = 20

VOCAB_JSON = config.D2_DIR / "vocab.json"

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<brk>"]


def count_vision_labels(split: str) -> Counter:
    """Count class ids in the YOLO label files the detector actually trained on."""
    label_dir = config.YOLO_DIR / "labels" / split
    counts = Counter()
    if not label_dir.exists():
        print(f"  WARNING: {label_dir} does not exist.")
        return counts

    n_files = 0
    for path in label_dir.glob("*.txt"):
        n_files += 1
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[int(line.split()[0])] += 1

    print(f"  {split:<6}: {n_files:>5} label files | "
          f"{sum(counts.values()):>6,} boxes | {len(counts):>4} distinct classes")
    return counts


def main() -> None:
    config.ensure_dirs()

    if not config.D2_CORPUS_STATS.exists():
        raise FileNotFoundError(
            f"{config.D2_CORPUS_STATS} not found. Run build_corpus.py first."
        )

    with open(config.D2_CORPUS_STATS, "r", encoding="utf-8") as f:
        stats = json.load(f)

    corpus_train = Counter({int(k): v for k, v in stats["counts_train"].items()})
    corpus_all = Counter({int(k): v for k, v in stats["counts_all"].items()})

    print("Reading Domain 1 label files")
    print("-" * 68)
    vis_train = count_vision_labels("train")
    count_vision_labels("val")

    # Domain 1 emit set (decided first)
    print("\nDomain 1 emit set")
    print("-" * 68)
    print(f"  {'min boxes':>10} | {'classes':>8} | {'share of train boxes':>21}")
    total_boxes = sum(vis_train.values())
    for floor in (1, 5, 10, 20, 50, 100):
        kept = [s for s, c in vis_train.items() if c >= floor]
        share = sum(vis_train[s] for s in kept) / max(1, total_boxes)
        marker = "  <-- chosen" if floor == MIN_DETECT_BOXES else ""
        print(f"  {floor:>10} | {len(kept):>8} | {100 * share:>20.1f}%{marker}")

    detectable = sorted(
        s for s, c in vis_train.items()
        if c >= MIN_DETECT_BOXES and s != config.BROKEN_LABEL
    )
    print(f"\n  Emit set: {len(detectable)} classes (excluding the broken "
          f"placeholder, which is never emitted by the detector).")

    # Domain 2 vocabulary = union
    print("\nDomain 2 vocabulary (union of text-frequent and detectable)")
    print("-" * 68)

    by_frequency = {
        s for s, c in corpus_train.items()
        if s != config.BROKEN_LABEL and c >= MIN_CORPUS_COUNT
    }
    by_detector = set(detectable)
    added_for_containment = sorted(by_detector - by_frequency)

    eligible = sorted(by_frequency | by_detector)
    # Common signs get low ids, which makes debugging output easier to read.
    eligible.sort(key=lambda s: (-corpus_train.get(s, 0), s))

    sign_to_id = {s: len(SPECIAL_TOKENS) + i for i, s in enumerate(eligible)}

    total_real = sum(v for k, v in corpus_all.items() if k != config.BROKEN_LABEL)
    kept_real = sum(corpus_all.get(s, 0) for s in eligible)

    print(f"  Frequency threshold      : >= {MIN_CORPUS_COUNT} corpus train occurrences")
    print(f"  Signs qualifying on text : {len(by_frequency)}")
    print(f"  Added for containment    : {len(added_for_containment)}")
    for s in added_for_containment:
        print(f"      train_label {s:>3}: {vis_train.get(s, 0):>4} boxes, "
              f"{corpus_train.get(s, 0):>4} corpus train occurrences")
    print(f"  Sign tokens in vocab     : {len(eligible)}")
    print(f"  Special tokens           : {len(SPECIAL_TOKENS)} "
          f"({', '.join(SPECIAL_TOKENS)})")
    print(f"  TOTAL VOCAB SIZE         : {len(SPECIAL_TOKENS) + len(eligible)}")
    print(f"  Real-token coverage      : {100 * kept_real / max(1, total_real):.2f}%")
    print(f"  Signs falling to <unk>   : "
          f"{stats['n_distinct_real_signs'] - len(eligible)}")

    # Signs with a token but almost no text to learn from; the MLM will be weak on these.
    starved = [s for s in eligible if corpus_train.get(s, 0) < MIN_CORPUS_COUNT]
    if starved:
        print(f"\n  NOTE: {len(starved)} token(s) have < {MIN_CORPUS_COUNT} corpus "
              f"training examples.")
        print("  They are representable but will be predicted weakly. This is the")
        print("  correct trade: an unrepresentable detector output is worse.")

    # Containment check
    print("\nContainment check")
    print("-" * 68)
    orphans = [s for s in detectable if s not in sign_to_id]
    if orphans:
        raise AssertionError(
            f"Containment violated after union: {orphans}. This should be "
            f"impossible -- the union guarantees it. Investigate build_vocab.py."
        )
    print(f"  PASS by construction: all {len(detectable)} detectable classes "
          f"have a Domain 2 token.")

    d1_corpus_share = sum(corpus_all.get(s, 0) for s in detectable) / max(1, total_real)
    print(f"\n  Detector can name {len(detectable)} of {len(eligible)} vocabulary signs.")
    print(f"  Those signs are {100 * d1_corpus_share:.1f}% of real corpus tokens.")
    print(f"  => roughly {100 * (1 - d1_corpus_share):.1f}% of a tablet's signs reach")
    print(f"     Domain 2 as <mask> even with a PERFECT detector on its own classes.")
    print("     This is the workload Domain 2 exists to handle.")

    # Write
    vocab = {
        "specials": {t: i for i, t in enumerate(SPECIAL_TOKENS)},
        "sign_to_id": {str(s): i for s, i in sign_to_id.items()},
        "id_to_sign": {str(i): s for s, i in sign_to_id.items()},
        "vocab_size": len(SPECIAL_TOKENS) + len(eligible),
        "vision_emit_classes": detectable,
        "meta": {
            "min_corpus_count": MIN_CORPUS_COUNT,
            "min_detect_boxes": MIN_DETECT_BOXES,
            "n_sign_tokens": len(eligible),
            "n_qualified_by_frequency": len(by_frequency),
            "n_added_for_containment": len(added_for_containment),
            "added_for_containment": added_for_containment,
            "n_starved_tokens": len(starved),
            "real_token_coverage": round(kept_real / max(1, total_real), 4),
            "n_vision_emit_classes": len(detectable),
            "vision_corpus_share": round(d1_corpus_share, 4),
            "broken_label_token": "<brk>",
        },
    }

    with open(VOCAB_JSON, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print("\n" + "=" * 68)
    print(f"VOCABULARY WRITTEN: {VOCAB_JSON}")
    print("=" * 68)


if __name__ == "__main__":
    main()