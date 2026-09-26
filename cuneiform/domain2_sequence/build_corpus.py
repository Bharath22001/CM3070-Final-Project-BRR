"""
Builds my Domain 2 training corpus from the CompVis transliteration CSVs.

Outputs:
    artifacts/domain2/corpus_lines.jsonl   one JSON object per tablet line, e.g.
        {"tablet": "P393866", "view": "Obv", "segm_idx": 192, "line_idx": 0,
         "tokens": [11, 4, 119, 31, 62, 62], "status": [1, 1, 1, 0, 0, 0],
         "mzl": [839, 110, 266, 113, 380, 380], "split": "train"}
    artifacts/domain2/corpus_stats.json    frequencies, coverage and damage-run stats

I split by tablet, not by line, and force Domain 1's held-out tablets into the
test split, so the end-to-end evaluation never tests on text the model has seen.

Run from the project root:
    python -m cuneiform.domain2_sequence.build_corpus
"""

from __future__ import annotations

import json
import random
from collections import Counter

import pandas as pd

from cuneiform import config

# Fixed seed so the split is reproducible. Changing it reshuffles train and val.
SEED = 1729
VAL_TABLET_FRACTION = 0.10

REQUIRED_COLUMNS = {
    "segm_idx", "tablet_CDLI", "view_desc", "rec_idx",
    "line_idx", "pos_idx", "train_label", "mzl_label", "status",
}

LINE_KEY = ["tablet_CDLI", "view_desc", "segm_idx", "line_idx"]
ROW_KEY = LINE_KEY + ["pos_idx"]


# Loading

def load_all_transliterations() -> pd.DataFrame:
    files = sorted(config.TRANSLITERATIONS_DIR.glob("transliterations_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No transliteration CSVs found in {config.TRANSLITERATIONS_DIR}"
        )

    print("Loading transliteration files")
    print("-" * 68)
    frames = []
    for path in files:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        df["source_file"] = path.stem
        print(f"  {path.name:<30} {len(df):>8,} rows  "
              f"{df['tablet_CDLI'].nunique():>4} tablets")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # VAT10657 has a blank view_desc on all 110 rows, and groupby silently drops
    # NaN keys, so I fill it in rather than lose a whole annotated tablet.
    n_blank_view = int(combined["view_desc"].isna().sum())
    if n_blank_view:
        blank_tablets = sorted(combined.loc[combined["view_desc"].isna(),
                                            "tablet_CDLI"].unique())
        combined["view_desc"] = combined["view_desc"].fillna("NA")
        print(f"  Filled {n_blank_view} blank view_desc value(s) with 'NA' "
              f"on tablet(s): {', '.join(map(str, blank_tablets))}")

    print("-" * 68)
    print(f"  {'TOTAL':<30} {len(combined):>8,} rows  "
          f"{combined['tablet_CDLI'].nunique():>4} tablets")
    return combined


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the sign tokens that appear in more than one of the ten files, after
    checking the duplicates agree on the sign rather than silently keeping one.
    """
    print("\nDeduplicating")
    print("-" * 68)
    n_before = len(df)

    dup_mask = df.duplicated(subset=ROW_KEY, keep=False)
    n_conflicts = 0
    if dup_mask.any():
        agree = df[dup_mask].groupby(ROW_KEY, sort=False)[["train_label", "mzl_label"]].nunique()
        n_conflicts = int((agree > 1).any(axis=1).sum())

    df = df.drop_duplicates(subset=ROW_KEY, keep="first").reset_index(drop=True)
    print(f"  Dropped {n_before - len(df):,} duplicate row(s) across source files.")
    print(f"  Rows remaining: {len(df):,}")

    if n_conflicts:
        print(f"  WARNING: {n_conflicts} position(s) had CONFLICTING sign labels "
              f"across files. Investigate before trusting this corpus.")
    else:
        print("  No conflicting labels at duplicated positions. Good.")
    return df


# Structural verification

def verify_structure(df: pd.DataFrame) -> None:
    """Check the assumptions the line reconstruction depends on."""
    print("\nVerifying structural assumptions")
    print("-" * 68)

    seg = df.groupby("segm_idx")[["tablet_CDLI", "view_desc"]].nunique()
    bad_seg = seg[(seg["tablet_CDLI"] > 1) | (seg["view_desc"] > 1)]
    if len(bad_seg):
        print(f"  NOTE: {len(bad_seg)} segm_idx value(s) span more than one "
              f"tablet/view. The line key includes tablet and view, so this is "
              f"handled, but it means segm_idx is not a global unique id.")
    else:
        print("  segm_idx maps to exactly one (tablet, view). Good.")

    n_labels_out_of_range = int(
        ((df["train_label"] < 0) | (df["train_label"] >= config.NUM_TRAIN_LABELS)).sum()
    )
    if n_labels_out_of_range:
        raise ValueError(
            f"{n_labels_out_of_range} rows have train_label outside "
            f"0..{config.NUM_TRAIN_LABELS - 1}. Raise config.NUM_TRAIN_LABELS."
        )
    print(f"  All train_label values within 0..{config.NUM_TRAIN_LABELS - 1}. Good.")
    print(f"  Highest train_label present: {int(df['train_label'].max())}")


# Line reconstruction

def build_lines(df: pd.DataFrame) -> list[dict]:
    print("\nReconstructing lines")
    print("-" * 68)

    df = df.sort_values(ROW_KEY, kind="mergesort")
    lines: list[dict] = []
    n_noncontiguous = 0

    for (tablet, view, segm, line_idx), g in df.groupby(LINE_KEY, sort=False):
        pos = g["pos_idx"].tolist()
        if pos != list(range(len(pos))):
            n_noncontiguous += 1

        lines.append({
            "tablet": str(tablet),
            "view": str(view),
            "segm_idx": int(segm),
            "line_idx": int(line_idx),
            "tokens": [int(v) for v in g["train_label"]],
            "status": [int(v) for v in g["status"]],
            "mzl": [int(v) for v in g["mzl_label"]],
        })

    total_tokens = sum(len(l["tokens"]) for l in lines)
    print(f"  Lines reconstructed : {len(lines):,}")
    print(f"  Sign tokens         : {total_tokens:,}")
    print(f"  Mean signs per line : {total_tokens / max(1, len(lines)):.2f}")
    if n_noncontiguous:
        print(f"  NOTE: {n_noncontiguous} line(s) have gaps in pos_idx. Order is "
              f"still correct (we sort by pos_idx) but positions are not 0..n-1.")
    else:
        print("  Every line has contiguous pos_idx 0..n-1. Good.")
    return lines


# Splitting

def load_domain1_test_tablets() -> set[str]:
    if not config.TEST_BBOX_CSV.exists():
        print(f"  WARNING: {config.TEST_BBOX_CSV.name} not found. Cannot force "
              f"Domain 1's held-out tablets into the Domain 2 test split.")
        return set()
    df = pd.read_csv(config.TEST_BBOX_CSV)
    return set(df["tablet_CDLI"].astype(str).unique())


def assign_splits(lines: list[dict]) -> None:
    print("\nAssigning splits (at TABLET level)")
    print("-" * 68)

    d1_test = load_domain1_test_tablets()
    all_tablets = sorted({l["tablet"] for l in lines})

    forced_test = [t for t in all_tablets if t in d1_test]
    remaining = [t for t in all_tablets if t not in d1_test]

    rng = random.Random(SEED)
    rng.shuffle(remaining)
    n_val = max(1, int(len(remaining) * VAL_TABLET_FRACTION))
    val_tablets = set(remaining[:n_val])
    train_tablets = set(remaining[n_val:])
    test_tablets = set(forced_test)

    for l in lines:
        t = l["tablet"]
        l["split"] = ("test" if t in test_tablets
                      else "val" if t in val_tablets
                      else "train")

    print(f"  Domain 1 test tablets found in corpus : {len(forced_test)} "
          f"(of {len(d1_test)} annotated)")
    for name in ("train", "val", "test"):
        subset = [l for l in lines if l["split"] == name]
        tabs = len({l['tablet'] for l in subset})
        toks = sum(len(l["tokens"]) for l in subset)
        print(f"  {name:<5} : {tabs:>4} tablets | {len(subset):>7,} lines | "
              f"{toks:>8,} tokens")


# Statistics

def compute_stats(lines: list[dict]) -> dict:
    print("\nCorpus statistics")
    print("-" * 68)

    train_lines = [l for l in lines if l["split"] == "train"]

    all_counts = Counter()
    train_counts = Counter()
    for l in lines:
        all_counts.update(l["tokens"])
    for l in train_lines:
        train_counts.update(l["tokens"])

    total_tokens = sum(all_counts.values())
    n_broken = all_counts.get(config.BROKEN_LABEL, 0)

    print(f"  Distinct train_label values : {len(all_counts)}")
    print(f"  Total sign tokens           : {total_tokens:,}")
    print(f"  Broken placeholders (id 0)  : {n_broken:,} "
          f"({100 * n_broken / max(1, total_tokens):.1f}%)")

    # Coverage over real signs only (the broken placeholder is excluded).
    real = Counter({k: v for k, v in all_counts.items() if k != config.BROKEN_LABEL})
    real_total = sum(real.values())
    ranked = [c for _, c in real.most_common()]
    running, coverage_points = 0, {}
    for i, c in enumerate(ranked, start=1):
        running += c
        frac = running / max(1, real_total)
        for target in (0.50, 0.90, 0.95, 0.99):
            key = f"{int(target * 100)}pct"
            if key not in coverage_points and frac >= target:
                coverage_points[key] = i

    print(f"  Distinct REAL signs         : {len(real)}")
    for target in (50, 90, 95, 99):
        k = f"{target}pct"
        if k in coverage_points:
            print(f"    {coverage_points[k]:>4} signs cover {target}% of real tokens")

    # How many signs clear each candidate frequency floor in the training text.
    print("\n  Corpus frequency thresholds (train split, real signs only):")
    thresholds = {}
    train_real = Counter({k: v for k, v in train_counts.items() if k != config.BROKEN_LABEL})
    train_real_total = sum(train_real.values())
    for floor in (5, 10, 20, 50, 100, 200):
        kept = [s for s, c in train_real.items() if c >= floor]
        kept_tokens = sum(train_real[s] for s in kept)
        share = kept_tokens / max(1, train_real_total)
        thresholds[str(floor)] = {
            "n_signs": len(kept),
            "token_share": round(share, 4),
        }
        print(f"    >= {floor:>3} occurrences : {len(kept):>3} signs "
              f"covering {100 * share:.1f}% of tokens")

    # Broken-run structure: the real damage Domain 2 has to repair.
    runs = []
    for l in lines:
        run = 0
        for t in l["tokens"]:
            if t == config.BROKEN_LABEL:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)

    if runs:
        runs_sorted = sorted(runs)
        mean_run = sum(runs) / len(runs)
        print(f"\n  Broken-sign runs            : {len(runs):,}")
        print(f"    mean length               : {mean_run:.2f}")
        print(f"    median length             : {runs_sorted[len(runs_sorted) // 2]}")
        print(f"    max length                : {max(runs)}")
        print(f"    longer than one sign      : "
              f"{100 * sum(1 for r in runs if r > 1) / len(runs):.0f}%")

    return {
        "n_lines": len(lines),
        "n_tokens": total_tokens,
        "n_distinct_labels": len(all_counts),
        "n_distinct_real_signs": len(real),
        "n_broken_tokens": n_broken,
        "coverage_ranks": coverage_points,
        "corpus_frequency_thresholds": thresholds,
        "broken_run_count": len(runs),
        "broken_run_mean": round(sum(runs) / len(runs), 3) if runs else 0,
        "broken_run_max": max(runs) if runs else 0,
        "counts_all": {str(k): v for k, v in sorted(all_counts.items())},
        "counts_train": {str(k): v for k, v in sorted(train_counts.items())},
        "seed": SEED,
        "val_tablet_fraction": VAL_TABLET_FRACTION,
    }



def main() -> None:
    config.ensure_dirs()

    df = load_all_transliterations()
    df = deduplicate(df)
    verify_structure(df)
    lines = build_lines(df)
    assign_splits(lines)
    stats = compute_stats(lines)

    with open(config.D2_CORPUS_JSONL, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    with open(config.D2_CORPUS_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 68)
    print("CORPUS BUILD COMPLETE")
    print(f"  {config.D2_CORPUS_JSONL}")
    print(f"  {config.D2_CORPUS_STATS}")
    print("=" * 68)


if __name__ == "__main__":
    main()