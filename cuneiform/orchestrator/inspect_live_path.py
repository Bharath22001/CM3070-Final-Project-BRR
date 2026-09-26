"""
Answers the questions my live wiring depends on, from the data, before any
pipeline code was written. No model is loaded.

A. Which tablets have an image, bbox ground truth and are in Domain 2's test split.
B. Whether a tablet's segments can be separated geometrically.
C. How many annotated boxes fall in the strip the training tile grid never covers.
D. Whether EXIF transposition changes any image in this corpus.
E. What the reference sequence looks like per tablet, including its <brk> share.

Run from the project root:
    python -m cuneiform.orchestrator.inspect_live_path
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

from cuneiform import config
from cuneiform.domain1_vision.group_lines import load_boxes

TILE_SIZE = 512
STEP_SIZE = 384
RULE = "=" * 70
DASH = "-" * 70


def head(t: str) -> None:
    print("\n" + RULE)
    print(t)
    print(RULE)


def verdict(name: str, ok: bool | None, note: str = "") -> None:
    tag = "PASS" if ok is True else ("FAIL" if ok is False else "VACUOUS")
    print(f"\n  --> {tag}: {name}")
    if note:
        print(f"      {note}")


def corpus_by_tablet() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rec = json.loads(raw)
                out[rec["tablet"]].append(rec)
    return out


# A. Inventory

def section_a(bb, corpus) -> list[str]:
    head("A. WHAT CAN THE LIVE PATH ACTUALLY RUN ON?")

    annotated = set(bb["tablet_CDLI"].unique())
    test_split = {t for t, recs in corpus.items()
                  if recs and recs[0].get("split") == "test"}
    have_img = {t for t in annotated
                if (config.IMAGES_DIR / f"{t}.jpg").exists()}

    print(f"  tablets with bbox annotations : {len(annotated)}")
    print(f"  of those, image on disk       : {len(have_img)}")
    print(f"  Domain 2 test-split tablets   : {len(test_split)}")

    runnable = sorted(annotated & have_img & test_split)
    print(f"\n  RUNNABLE (all three)          : {len(runnable)}")
    for t in runnable:
        n_box = int((bb["tablet_CDLI"] == t).sum())
        n_tok = sum(len(r["tokens"]) for r in corpus.get(t, []))
        n_seg = bb.loc[bb["tablet_CDLI"] == t, "segm_idx"].nunique() \
            if "segm_idx" in bb.columns else -1
        print(f"    {t:<12} boxes {n_box:>5}  corpus tokens {n_tok:>5}  "
              f"segments {n_seg:>3}")

    missing_img = sorted((annotated & test_split) - have_img)
    if missing_img:
        print(f"\n  annotated + in test split but NO IMAGE ({len(missing_img)}):")
        print(f"    {missing_img}")

    verdict("enough tablets to evaluate a live run", len(runnable) >= 5,
            f"{len(runnable)} runnable tablet(s). This is the live "
            f"evaluation's n, and it is far smaller than the 60 blocks the "
            f"simulated evaluation used. State it.")
    return runnable


# B. Segment separability

def section_b(bb, runnable: list[str]) -> None:
    head("B. ARE SEGMENTS SPATIALLY SEPARABLE?  (no segm_idx exists live)")

    if "segm_idx" not in bb.columns:
        verdict("segment recovery", None, "no segm_idx column to check.")
        return

    print("  For each multi-segment tablet: do the segments' bounding regions")
    print("  overlap? Disjoint in x OR in y means a projection can split them.\n")

    multi = single = sep_x = sep_y = tangled = 0
    for t in runnable:
        g = bb[bb["tablet_CDLI"] == t]
        segs = sorted(g["segm_idx"].unique())
        if len(segs) < 2:
            single += 1
            continue
        multi += 1

        ext = {}
        for s in segs:
            gs = g[g["segm_idx"] == s]
            ext[s] = (gs["cx"].min(), gs["cx"].max(),
                      gs["cy"].min(), gs["cy"].max())

        def disjoint(axis: int) -> bool:
            iv = sorted((ext[s][axis], ext[s][axis + 1]) for s in segs)
            return all(a[1] < b[0] for a, b in zip(iv, iv[1:]))

        dx, dy = disjoint(0), disjoint(2)
        tag = ("disjoint in x" if dx else
               "disjoint in y" if dy else "OVERLAPPING -- not separable")
        sep_x += dx
        sep_y += (dy and not dx)
        tangled += (not dx and not dy)
        print(f"    {t:<12} {len(segs)} segments  {tag}")
        if not dx and not dy:
            for s in segs:
                x1, x2, y1, y2 = ext[s]
                print(f"        seg {s:>3}: x {x1:>7.0f}-{x2:<7.0f} "
                      f"y {y1:>7.0f}-{y2:<7.0f}")

    print(f"\n  single-segment tablets : {single}")
    print(f"  multi-segment tablets  : {multi}  "
          f"(x-separable {sep_x}, y-separable {sep_y}, tangled {tangled})")

    verdict("segments can be recovered geometrically at inference",
            multi == 0 or tangled == 0,
            f"{tangled} tablet(s) have interleaved segments. On those, line "
            f"grouping over the whole composite will merge rows from different "
            f"faces. Either restrict the live demo to separable tablets and say "
            f"so, or treat the composite as one segment and report the "
            f"reading-order limitation."
            if tangled else
            "a 1-D projection split recovers segments on every runnable tablet.")


# C. Tile coverage

def section_c(bb, runnable: list[str]) -> None:
    head("C. DOES THE TILING GRID COVER THE WHOLE IMAGE?")

    print(f"  Grid: range(0, dim - {TILE_SIZE} + 1, {STEP_SIZE}).")
    print("  Anything past the last tile origin + tile size is never seen.\n")

    tot_box = tot_dead = 0
    print(f"    {'tablet':<12} {'image':>12} {'covered':>14} "
          f"{'boxes':>7} {'dead':>6}")
    for t in runnable:
        p = config.IMAGES_DIR / f"{t}.jpg"
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)
                W, H = im.size
        except Exception as exc:
            print(f"    {t:<12} could not open: {exc}")
            continue

        def covered(dim: int) -> int:
            if dim < TILE_SIZE:
                return 0
            last = max(range(0, dim - TILE_SIZE + 1, STEP_SIZE))
            return last + TILE_SIZE

        cw, ch = covered(W), covered(H)
        g = bb[bb["tablet_CDLI"] == t]
        dead = int(((g["cx"] >= cw) | (g["cy"] >= ch)).sum())
        tot_box += len(g)
        tot_dead += dead
        print(f"    {t:<12} {W:>5}x{H:<6} {cw:>6}x{ch:<7} "
              f"{len(g):>7} {dead:>6}")

    print(f"\n  boxes in the dead zone: {tot_dead:,} of {tot_box:,} "
          f"({100 * tot_dead / max(1, tot_box):.1f}%)")
    verdict("the tiling grid covers every annotated box", tot_dead == 0,
            f"{tot_dead} box(es) sit where no tile is ever cut. The INFERENCE "
            f"grid must add a flush final tile at dim-{TILE_SIZE}; otherwise "
            f"those signs are undetectable by construction, not by recall.")


# D. EXIF

def section_d(runnable: list[str]) -> None:
    head("D. DOES EXIF TRANSPOSITION CHANGE ANYTHING?")

    changed = []
    for t in runnable:
        p = config.IMAGES_DIR / f"{t}.jpg"
        try:
            with Image.open(p) as im:
                raw = im.size
                tr = ImageOps.exif_transpose(im).size
        except Exception:
            continue
        if raw != tr:
            changed.append((t, raw, tr))

    print(f"  images checked          : {len(runnable)}")
    print(f"  changed by transpose    : {len(changed)}")
    for t, a, b in changed:
        print(f"    {t:<12} {a} -> {b}")

    verdict("EXIF orientation is a no-op on this corpus", not changed,
            "some images rotate. The inference path MUST apply "
            "ImageOps.exif_transpose exactly as generate_tiles.py does, or "
            "every detection coordinate is wrong on those tablets."
            if changed else
            "still call exif_transpose in the inference path for safety, but "
            "this is not a live risk on the current images.")


# E. Scoring reference

def section_e(bb, corpus, runnable: list[str]) -> None:
    head("E. WHAT DOES THE LIVE PATH GET SCORED AGAINST?")

    print("  The reconstructed sequence does not align to corpus token")
    print("  indices, so scoring is sequence-level edit distance. <brk> has no")
    print("  ground truth and must be stripped from the reference -- which")
    print("  changes the reference relative to the simulated evaluation.\n")

    print(f"    {'tablet':<12} {'tokens':>7} {'<brk>':>7} {'real':>7} "
          f"{'boxes':>7} {'box/real':>9}")
    tot_tok = tot_brk = tot_box = 0
    ratios = []
    for t in runnable:
        recs = corpus.get(t, [])
        toks = [x for r in recs for x in r["tokens"]]
        brk = sum(1 for x in toks if x == config.BROKEN_LABEL)
        real = len(toks) - brk
        nbox = int((bb["tablet_CDLI"] == t).sum())
        tot_tok += len(toks)
        tot_brk += brk
        tot_box += nbox
        r = nbox / max(1, real)
        ratios.append(r)
        print(f"    {t:<12} {len(toks):>7} {brk:>7} {real:>7} "
              f"{nbox:>7} {r:>9.2f}")

    print(f"\n  totals: {tot_tok:,} tokens | {tot_brk:,} <brk> "
          f"({100 * tot_brk / max(1, tot_tok):.1f}%) | {tot_box:,} boxes")
    if ratios:
        print(f"  boxes per real token: median {np.median(ratios):.2f} "
              f"(min {min(ratios):.2f}, max {max(ratios):.2f})")

    ok = bool(ratios) and 0.8 <= float(np.median(ratios)) <= 1.25
    verdict("annotated boxes and corpus tokens describe the same signs", ok,
            "the two sources do not line up 1:1, so a perfect detector would "
            "still not reproduce the corpus sequence. That is an upper bound "
            "on live accuracy and belongs in the report as a stated ceiling."
            if not ok else
            "box and token counts agree, so the reference is a fair target.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(RULE)
    print("LIVE WIRING -- PRE-FLIGHT INSPECTION")
    print(RULE)

    bb = load_boxes()
    corpus = corpus_by_tablet()
    print(f"  {len(bb):,} boxes | {len(corpus):,} tablets in corpus")

    runnable = section_a(bb, corpus)
    if not runnable:
        print("\n  No runnable tablets. Everything below would be vacuous.")
        return
    section_b(bb, runnable)
    section_c(bb, runnable)
    section_d(runnable)
    section_e(bb, corpus, runnable)

    print("\n" + RULE)


if __name__ == "__main__":
    main()