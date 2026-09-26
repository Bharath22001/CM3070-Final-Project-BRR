"""
Looks at the format of line_annotations_{train,test}.csv before I wrote any
reading-order or gap code against it.

It checks whether anchors are one per sign, how regular the sign pitch is,
whether segm_idx joins cleanly to the corpus, whether one line_idx is one
physical row, and how far lines slope. It only prints.

Run from the project root:
    python -m cuneiform.domain1_vision.inspect_line_geometry
"""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from cuneiform import config

LINE_TRAIN = config.ANNOTATIONS_DIR / "line_annotations_train.csv"
LINE_TEST = config.ANNOTATIONS_DIR / "line_annotations_test.csv"


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_lines(path):
    if not path.exists():
        print(f"  MISSING: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  {path.name}: {len(df):,} rows | columns {list(df.columns)}")
    print(f"    dtypes  : {dict(df.dtypes.astype(str))}")
    print(f"    nulls   : {df.isna().sum().to_dict()}")
    print(f"    tablets : {df['tablet_CDLI'].nunique()} | "
          f"segments {df['segm_idx'].nunique()}")
    return df


def check_segm_namespace(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check whether segm_idx is a global id and whether train and test share it."""
    rule("2. IS segm_idx GLOBAL, AND DO TRAIN/TEST SHARE ITS NAMESPACE?")

    for name, df in (("train", train), ("test", test)):
        if df.empty:
            continue
        per_segm = df.groupby("segm_idx")["tablet_CDLI"].nunique()
        bad = per_segm[per_segm > 1]
        print(f"  {name}: segm_idx range {df['segm_idx'].min()}"
              f"..{df['segm_idx'].max()}")
        print(f"    segments mapping to >1 tablet: {len(bad)} "
              f"{'(GLOBAL id -- good)' if len(bad) == 0 else '(NOT global!)'}")

    if not train.empty and not test.empty:
        st = set(train["segm_idx"])
        se = set(test["segm_idx"])
        overlap = st & se
        print(f"\n  train/test segm_idx overlap: {len(overlap)} values")
        if overlap:
            print("    ** COLLIDING. The two files use INDEPENDENT numbering, so")
            print("       segm_idx alone does NOT identify a segment. Any join")
            print("       must carry the split, or use tablet_CDLI as well.")
            sample = sorted(overlap)[:5]
            for s in sample:
                t1 = sorted(set(train[train.segm_idx == s]["tablet_CDLI"]))
                t2 = sorted(set(test[test.segm_idx == s]["tablet_CDLI"]))
                print(f"       segm {s}: train={t1} test={t2}")
        else:
            print("    disjoint -- the two files share one namespace.")


def check_corpus_join(lines: pd.DataFrame, split_name: str) -> None:
    """
    Check whether anchors match the transliteration one per sign. If they do,
    they give reading order and pitch directly; if not, pitch has to come from
    the boxes.
    """
    rule(f"3. JOIN TO THE CORPUS -- ARE ANCHORS ONE PER SIGN?  [{split_name}]")

    if not config.D2_CORPUS_JSONL.exists():
        print("  corpus_lines.jsonl missing; run build_corpus.py first.")
        return

    corpus = defaultdict(int)
    corpus_tablets = set()
    with open(config.D2_CORPUS_JSONL, "r", encoding="utf-8") as f:
        for raw in f:
            rec = json.loads(raw)
            corpus[(rec["tablet"], rec["segm_idx"], rec["line_idx"])] = \
                len(rec["tokens"])
            corpus_tablets.add(rec["tablet"])

    anchors = lines.groupby(
        ["tablet_CDLI", "segm_idx", "line_idx"]).size().to_dict()

    ann_tablets = set(lines["tablet_CDLI"])
    print(f"  anchor tablets: {len(ann_tablets)} | "
          f"also in corpus: {len(ann_tablets & corpus_tablets)}")
    missing = sorted(ann_tablets - corpus_tablets)
    if missing:
        print(f"  tablets with anchors but NOT in the corpus ({len(missing)}): "
              f"{missing[:10]}")

    # Join on the full key
    matched = [(k, n, corpus[k]) for k, n in anchors.items() if k in corpus]
    print(f"\n  lines with anchors            : {len(anchors):,}")
    print(f"  of those, joined to a corpus line: {len(matched):,} "
          f"({100 * len(matched) / max(1, len(anchors)):.1f}%)")

    if not matched:
        print("\n  ** ZERO JOIN. The (tablet, segm_idx, line_idx) key does not")
        print("     line up. Trying (tablet, line_idx) ignoring segm_idx...")
        alt_corpus = defaultdict(list)
        with open(config.D2_CORPUS_JSONL, "r", encoding="utf-8") as f:
            for raw in f:
                rec = json.loads(raw)
                alt_corpus[(rec["tablet"], rec["line_idx"])].append(
                    len(rec["tokens"]))
        alt = sum(1 for (t, s, l) in anchors if (t, l) in alt_corpus)
        print(f"     matches on (tablet, line_idx): {alt:,} of {len(anchors):,}")
        return

    diffs = Counter()
    for _, n_anchor, n_tok in matched:
        diffs[n_anchor - n_tok] += 1

    exact = diffs.get(0, 0)
    print(f"  EXACT count match             : {exact:,} "
          f"({100 * exact / len(matched):.1f}% of joined lines)")
    print("\n  (anchors - tokens) distribution, most common first:")
    for d, c in diffs.most_common(9):
        print(f"    {d:>+4} : {c:>6,} lines "
              f"({100 * c / len(matched):>5.1f}%)")

    if exact / len(matched) > 0.9:
        print("\n  => ANCHORS ARE ONE PER SIGN. They are per-sign ground truth:")
        print("     reading order and pitch can both be measured directly.")
    else:
        print("\n  => Anchors are NOT reliably one per sign. Pitch must come")
        print("     from bbox geometry, and gap COUNTING is not safe -- the")
        print("     fallback is one <mask> per detected gap.")


def check_geometry(lines: pd.DataFrame) -> None:
    """Pitch regularity, line slope, and reading direction."""
    rule("4. LINE GEOMETRY -- PITCH, SLOPE, DIRECTION")

    pitches, slopes, drifts = [], [], []
    already_sorted = n_lines = 0
    per_line_cv = []

    for (t, s, l), g in lines.groupby(["tablet_CDLI", "segm_idx", "line_idx"]):
        if len(g) < 3:
            continue
        n_lines += 1
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)

        if (np.diff(x) > 0).all():
            already_sorted += 1

        order = np.argsort(x)
        xs, ys = x[order], y[order]

        dx = np.diff(xs)
        pitches.extend(dx.tolist())
        if len(dx) >= 2 and np.mean(dx) > 0:
            per_line_cv.append(np.std(dx) / np.mean(dx))

        drifts.append(ys.max() - ys.min())
        if xs.max() > xs.min():
            slopes.append((ys[-1] - ys[0]) / (xs[-1] - xs[0]))

    p = np.array(pitches)
    print(f"  lines with >=3 anchors        : {n_lines:,}")
    print(f"  already in ascending x order  : {already_sorted:,} "
          f"({100 * already_sorted / max(1, n_lines):.1f}%)  "
          f"[low % => must sort; check for right-to-left]")
    print(f"\n  ANCHOR PITCH (x gap between neighbours), px:")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:<3} {np.percentile(p, q):>8.1f}")
    print(f"    mean {p.mean():>8.1f}  std {p.std():>8.1f}")
    print(f"    coefficient of variation, WITHIN a line: "
          f"median {np.median(per_line_cv):.2f}")
    print("    (CV well under ~0.5 => pitch regular enough to COUNT missing")
    print("     signs as round(gap / median_pitch). Above that, counting is")
    print("     guesswork and one <mask> per gap is the honest fallback.)")

    print(f"\n  VERTICAL DRIFT within a line, px:")
    d = np.array(drifts)
    for q in (50, 75, 95):
        print(f"    p{q:<3} {np.percentile(d, q):>8.1f}")
    print(f"  line slope dy/dx: median {np.median(slopes):+.4f} | "
          f"p5 {np.percentile(slopes, 5):+.4f} | "
          f"p95 {np.percentile(slopes, 95):+.4f}")
    print("  (large drift => y-banding will NOT group lines; grouping must")
    print("   follow the local slope or use the annotations directly.)")


def check_row_splitting(lines: pd.DataFrame) -> None:
    """
    Check whether one line_idx is one physical row, since some lines sit at
    the same height with separate x ranges.
    """
    rule("5. IS ONE line_idx ONE PHYSICAL ROW?")

    suspicious = total = 0
    examples = []

    for (t, s), g in lines.groupby(["tablet_CDLI", "segm_idx"]):
        by_line = {}
        for l, gl in g.groupby("line_idx"):
            by_line[l] = (gl["x"].min(), gl["x"].max(),
                          gl["y"].mean(), len(gl))
        for l in sorted(by_line):
            if l + 1 not in by_line:
                continue
            total += 1
            x1a, x2a, ya, _ = by_line[l]
            x1b, x2b, yb, _ = by_line[l + 1]
            # Same row: similar y, and the next line starts right of where this one ended.
            if abs(ya - yb) < 40 and x1b > x2a:
                suspicious += 1
                if len(examples) < 6:
                    examples.append((t, s, l, ya, yb, x2a, x1b))

    print(f"  consecutive line_idx pairs examined : {total:,}")
    print(f"  pairs that look like ONE split row  : {suspicious:,} "
          f"({100 * suspicious / max(1, total):.1f}%)")
    for t, s, l, ya, yb, x2a, x1b in examples:
        print(f"    {t} segm {s}: line {l} ends x={x2a:.0f} (y={ya:.0f}) -> "
              f"line {l+1} starts x={x1b:.0f} (y={yb:.0f})")


def check_anchor_in_box(lines: pd.DataFrame) -> None:
    """
    Check whether anchors fall inside annotated boxes, which tells me if the
    two annotation layers describe the same objects.
    """
    rule("6. DO ANCHORS SIT INSIDE THE ANNOTATED BOXES?")

    if not config.TRAIN_BBOX_CSV.exists():
        print(f"  {config.TRAIN_BBOX_CSV} missing; skipping.")
        return

    bb = pd.read_csv(config.TRAIN_BBOX_CSV).drop_duplicates()
    bb["bbox"] = bb["bbox"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v)

    shared = sorted(set(bb["tablet_CDLI"]) & set(lines["tablet_CDLI"]))
    print(f"  tablets in both files: {len(shared)}")
    if not shared:
        print("  ** No overlap -- the two files may use different id styles.")
        print(f"     bbox sample : {sorted(set(bb['tablet_CDLI']))[:5]}")
        print(f"     line sample : {sorted(set(lines['tablet_CDLI']))[:5]}")
        return

    inside = near = tested = 0
    offs_x, offs_y = [], []

    for t in shared[:15]:
        boxes = np.asarray(bb[bb.tablet_CDLI == t]["bbox"].tolist(), dtype=float)
        pts = lines[lines.tablet_CDLI == t][["x", "y"]].to_numpy(dtype=float)
        for px, py in pts:
            tested += 1
            hit = ((boxes[:, 0] <= px) & (px <= boxes[:, 2]) &
                   (boxes[:, 1] <= py) & (py <= boxes[:, 3]))
            if hit.any():
                inside += 1
                b = boxes[np.flatnonzero(hit)[0]]
                offs_x.append((px - b[0]) / max(1.0, b[2] - b[0]))
                offs_y.append((py - b[1]) / max(1.0, b[3] - b[1]))
            else:
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                if np.min(np.hypot(cx - px, cy - py)) < 100:
                    near += 1

    print(f"  anchors tested (first 15 tablets): {tested:,}")
    print(f"    inside a box              : {inside:,} "
          f"({100 * inside / max(1, tested):.1f}%)")
    print(f"    outside but within 100px  : {near:,} "
          f"({100 * near / max(1, tested):.1f}%)")
    if offs_x:
        print(f"  relative position within the box it hits "
              f"(0=left/top, 1=right/bottom):")
        print(f"    x: median {np.median(offs_x):.2f} | "
              f"y: median {np.median(offs_y):.2f}")
        print("    (~0.5/0.5 => anchors are box CENTRES; ~0/0 => top-left.)")


def main() -> None:
    rule("1. FILE INVENTORY")
    train = load_lines(LINE_TRAIN)
    print()
    test = load_lines(LINE_TEST)

    if train.empty:
        print("\nNo train line annotations; stopping.")
        return

    check_segm_namespace(train, test)
    check_corpus_join(train, "train")
    if not test.empty:
        check_corpus_join(test, "test")
    check_geometry(train)
    check_row_splitting(train)
    check_anchor_in_box(train)

    rule("WHAT DECIDES THE ALGORITHM")
    print("  section 3  -> anchors per sign? decides whether pitch is ground")
    print("                truth or must be inferred from boxes.")
    print("  section 4  -> pitch CV decides COUNTING vs one <mask> per gap.")
    print("  section 5  -> row splitting decides whether line_idx can be")
    print("                trusted as a row boundary.")
    print("  section 6  -> whether anchors and boxes are the same objects.")
    print("=" * 70)


if __name__ == "__main__":
    main()