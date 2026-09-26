"""
My live evaluation: CER and WER on real Domain 1 detector output, scored per
tablet against a reference built from the Domain 2 corpus. Nothing here is
simulated, and it writes artifacts/live_fusion_results.json with
"simulated_vision": false. Every constant and metric is imported from the
module that owns it (live_pipeline, evaluate_fusion, live_adapter, fusion),
never redeclared, so this measures exactly the configuration that ships.

Spurious detections aren't removed, so WER can go above 1.0 and every arm gets
a substitution / deletion / insertion breakdown. The headline is always the
paired difference between arms, never the absolute level: each tablet's WER
floor is |slots/real - 1|, so the level mostly reflects detector recall. Live
absolute figures must never share a table column with simulated ones.

The unit is the tablet (n = 30), and a minimum-detectable-effect table is
printed beside the p-values. The reference is sorted explicitly by (tablet,
view, segm_idx, line_idx), with <brk> and out-of-vocabulary tokens removed:
5,242 corpus tokens, 949 <brk>, 4,283 scored. Only 54% of the reference is
both covered by an annotated box and nameable by the detector.

The 0a arms are annotation-coverage oracles, not ceilings, since an
insertion-heavy arm can beat them without reading better. Five tablets have a
face-level defect (P335940's faces merge; on four others the reverse sits above
the obverse), so gate G13 prints the face-order audit and the sensitivity
analysis leaves out P335940 alone and all five.

The aligner is Wagner-Fischer with rows as the reference and columns as the
prediction. Only S + D + I is well defined, so I run it under both tie-breaks
and publish both splits. CER uses evaluate_fusion.levenshtein unchanged.

Run from the project root:
    python -m cuneiform.orchestrator.evaluate_live --gates-only
    python -m cuneiform.orchestrator.evaluate_live --self-test
    python -m cuneiform.orchestrator.evaluate_live
    python -m cuneiform.orchestrator.evaluate_live --conf-sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.orchestrator.evaluate_fusion import (
    LAMBDAS, THRESHOLD, context_density, levenshtein, load_sign_names, score,
    vision_context,
)
from cuneiform.orchestrator.fusion import (
    Detection, argmax_labels, fuse, lm_posterior, load_sequence_model,
    support_ids, vision_posterior,
)
from cuneiform.orchestrator.live_adapter import build_slots, load_detections
from cuneiform.orchestrator.live_pipeline import (
    LIVE_CELL, LIVE_CONF, LIVE_EDGE_EXTEND, LIVE_LAMBDA, LIVE_MIN_SIZE,
    LM_FLOOR, MAX_LEN, MIN_BLOCK, build_live_blocks, segment, to_detections,
)

RESULTS_JSON = config.ARTIFACTS_DIR / "live_fusion_results.json"
RULE = "=" * 78
SEED = 1729

# Marks a position where the pipeline emits no name. It never equals a real
# train_label, so the alignment can't accidentally match it.
SENTINEL = -1

CONF_SWEEP = [0.10, 0.15, 0.25, 0.35, 0.50]

# Arms that stay silent where vision can't name a sign, and so emit fewer tokens
# than the reference. Their absolute WER is only comparable with each other;
# 1b_vision_only_padded is the length-matched version for fair comparison.
DROP_UNINFORMATIVE_ARMS = {
    "1_vision_only",
    "0a_oracle_geo", "0a_oracle_view", "0a_oracle_flat",
}

# Expected reference totals for these 30 tablets (gate G3a). If the corpus is
# rebuilt the gate should fail rather than quietly accept new numbers.
REF_EXPECT = {
    "corpus_tokens": 5242,
    "brk": 949,
    "real": 4293,
    "unk": 10,
    "scored": 4283,
}
REF_EXPECT_UNK_LABELS = {175, 317, 318, 348}

# The four tablets whose reverse face is photographed above the obverse.
FACE_ORDER_EXPECT = {"P335937", "P335960", "P335980", "P336009"}
FACE_MERGE_KNOWN = "P335940"
CONFOUNDED = sorted(FACE_ORDER_EXPECT | {FACE_MERGE_KNOWN})

# Simulated context densities, for the mask-share comparison.
SIM_CTX_THRESHOLD = 0.119      # simulated arm 2: 88.1% mask share
SIM_CTX_FUSION = 0.194         # simulated arm 3: 80.6% mask share


# Population

def live_tablets() -> list[str]:
    """The 30 tablets with saved detections, sorted (the summary file is excluded)."""
    out = []
    for p in sorted(config.D1_LIVE_DETECTIONS.glob("*.json")):
        if p.stem == "_summary":
            continue
        out.append(p.stem)
    return out


def load_corpus_records() -> list[dict]:
    """Every record in the Domain 2 corpus artifact, in file order."""
    recs = []
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                recs.append(json.loads(raw))
    return recs


# The reference sequence

def build_reference(records: list[dict], tablets: list[str],
                    tok: SignTokenizer,
                    drop_unk: bool = True) -> tuple[dict[str, list[int]], dict]:
    """
    Corpus records to one reference train_label sequence per tablet.

    I sort explicitly instead of trusting file order (gate G4 checks this on a
    shuffled copy). Views sort alphabetically, so Obv comes before Rev. I remove
    <brk> (no ground truth) and tokens the Domain 2 vocabulary has no id for.
    With drop_unk=False the second removal is skipped, for gate G3b.
    Returns (refs, diagnostics).
    """
    wanted = set(tablets)
    used = [r for r in records
            if r["tablet"] in wanted and r.get("split") == "test"]

    used.sort(key=lambda r: (r["tablet"], r["view"],
                             int(r["segm_idx"]), int(r["line_idx"])))

    refs: dict[str, list[int]] = {t: [] for t in tablets}
    per_tablet_corpus: dict[str, int] = {t: 0 for t in tablets}
    per_tablet_brk: dict[str, int] = {t: 0 for t in tablets}
    per_tablet_real: dict[str, int] = {t: 0 for t in tablets}
    unk_labels: dict[int, int] = {}
    n_corpus = n_brk = n_real = n_unk = 0
    n_emittable = 0
    views: set[str] = set()
    splits: set[str] = set()
    groups: set[tuple[str, str, int]] = set()

    for r in used:
        t = r["tablet"]
        views.add(r["view"])
        splits.add(r.get("split"))
        groups.add((t, r["view"], int(r["segm_idx"])))
        for tokv in r["tokens"]:
            tokv = int(tokv)
            n_corpus += 1
            per_tablet_corpus[t] += 1
            if tokv == config.BROKEN_LABEL:
                n_brk += 1
                per_tablet_brk[t] += 1
                continue
            n_real += 1
            per_tablet_real[t] += 1
            if tokv in tok.vision_emit_classes:
                n_emittable += 1
            if tokv not in tok.sign_to_id:
                n_unk += 1
                unk_labels[tokv] = unk_labels.get(tokv, 0) + 1
                if drop_unk:
                    continue
            refs[t].append(tokv)

    diag = {
        "n_records": len(used),
        "corpus_tokens": n_corpus,
        "brk": n_brk,
        "real": n_real,
        "unk": n_unk,
        "unk_labels": dict(sorted(unk_labels.items())),
        "scored": sum(len(v) for v in refs.values()),
        "emittable_real": n_emittable,
        "views": sorted(views),
        "splits": sorted(s for s in splits if s is not None),
        "n_corpus_view_pairs": len(groups),
        "per_tablet_corpus": per_tablet_corpus,
        "per_tablet_brk": per_tablet_brk,
        "per_tablet_real": per_tablet_real,
    }
    return refs, diag


# The live slot sequence

@dataclass
class LiveTablet:
    """One tablet's live geometry, computed once and reused by every arm."""
    tablet: str
    dets: list = field(repr=False)
    groups: list = field(repr=False)
    slots: list = field(repr=False)
    detections: list = field(repr=False)
    blocks: list = field(repr=False)
    scored: list = field(repr=False)       # slot indices inside a block, sorted
    source: dict = field(repr=False)       # slot index -> source detection dict

    @property
    def n_slots(self) -> int:
        return len(self.slots)


def _slot_sources(dets: list[dict], groups: list,
                  slots: list) -> dict[int, dict]:
    """
    Map each detected slot back to the detection it came from, matching on
    (x, label, confidence) within the slot's own segment and using each
    detection once. The oracle arms need each slot's box; gate G5b checks every
    slot resolves.
    """
    pools: dict[int, dict] = {}
    for seg_i, idx in enumerate(groups):
        pool: dict[tuple, list[int]] = {}
        for i in idx:
            d = dets[int(i)]
            pool.setdefault(
                (float(d["cx"]), int(d["train_label"]), float(d["conf"])),
                []).append(int(i))
        pools[seg_i] = pool

    out: dict[int, dict] = {}
    for si, (seg_i, s) in enumerate(slots):
        if s.kind != "detected" or s.label is None:
            continue
        key = (float(s.x), int(s.label), float(s.confidence))
        bucket = pools[seg_i].get(key)
        if bucket:
            out[si] = dets[bucket.pop(0)]
    return out


def live_slots(tablet: str, conf: float = LIVE_CONF, cell: float = LIVE_CELL,
               min_size: int = LIVE_MIN_SIZE, edge: bool = LIVE_EDGE_EXTEND,
               tok: SignTokenizer | None = None) -> LiveTablet:
    """
    Detections to segments, slots and blocks, exactly as live_pipeline does.
    It touches no annotation. `scored` holds the slot indices inside a block;
    every arm is scored on that same set.
    """
    tok = tok or SignTokenizer()
    dets = load_detections(tablet, conf)
    groups = segment(dets, cell, min_size)
    slots = build_slots(dets, groups, extend_edges=edge)
    detections = to_detections(slots, tok)
    blocks = build_live_blocks(slots)
    scored = sorted(i for b in blocks for i in b)
    return LiveTablet(tablet=tablet, dets=dets, groups=groups, slots=slots,
                      detections=detections, blocks=blocks, scored=scored,
                      source=_slot_sources(dets, groups, slots))


# Ground truth: everything below here uses the annotation

def load_gt_boxes(tablets: list[str]):
    """Annotated boxes for the 30 tablets (uses ground truth)."""
    from cuneiform.domain1_vision.group_lines import load_boxes

    bb = load_boxes()
    return bb[bb["tablet_CDLI"].isin(tablets)].copy()


def _gt_detlike(sub) -> list[dict]:
    """Annotated boxes as detection-shaped dicts, so build_slots can use them."""
    out = []
    for r in sub.itertuples(index=False):
        out.append({
            "cx": float(r.cx), "cy": float(r.cy),
            "x1": float(r.cx - r.w / 2), "x2": float(r.cx + r.w / 2),
            "y1": float(r.cy - r.h / 2), "y2": float(r.cy + r.h / 2),
            "train_label": int(r.train_label), "conf": 1.0,
            "view_desc": str(r.view_desc), "segm_idx": int(r.segm_idx),
        })
    return out


def ceiling_slots(bb, tablet: str, mode: str,
                  tok: SignTokenizer) -> list[int]:
    """
    The annotation-coverage oracle sequence for one tablet, in one of three orderings:

      "flat"  true faces in true order, lines from the pipeline's grouper, so
              the only loss is annotation coverage.
      "view"  the same faces ordered by cy.min(), the pipeline's convention;
              flat -> view shows the cost of face ordering.
      "geo"   the live segment() clustering on the annotated boxes;
              view -> geo shows the cost of clustering.

    Inserted slots and labels outside the 82 emit classes are dropped.
    """
    sub = bb[bb["tablet_CDLI"] == tablet]
    if len(sub) == 0:
        return []
    dets = _gt_detlike(sub)
    cy = np.array([d["cy"] for d in dets], float)
    cx = np.array([d["cx"] for d in dets], float)

    if mode == "geo":
        groups = segment(dets, LIVE_CELL, LIVE_MIN_SIZE)
    else:
        by_key: dict[tuple, list[int]] = {}
        for i, d in enumerate(dets):
            by_key.setdefault((d["view_desc"], d["segm_idx"]), []).append(i)
        keys = list(by_key)
        if mode == "view":
            keys.sort(key=lambda k: (float(cy[by_key[k]].min()),
                                     float(cx[by_key[k]].min())))
        elif mode == "flat":
            keys.sort()          # by view_desc, then segm_idx: the true face order
        else:
            raise ValueError(f"unknown ceiling mode {mode!r}")
        groups = [np.array(by_key[k], dtype=int) for k in keys]

    slots = build_slots(dets, groups, extend_edges=False)
    return [int(s.label) for _, s in slots
            if s.kind == "detected" and s.label is not None
            and int(s.label) in tok.vision_emit_classes]


def oracle_map(lt: LiveTablet, bb, tok: SignTokenizer,
               radius_ratio: float = 0.5) -> tuple[dict[int, int], dict]:
    """
    Live slot index to the train_label of the annotated box it sits on, for
    arm 0b. Each slot matches the nearest box centre within radius_ratio times
    the median box width, one box per slot. It uses ground truth, so it runs
    after the live predictions are hashed. The match rate is returned too.
    """
    sub = bb[bb["tablet_CDLI"] == lt.tablet]
    out: dict[int, int] = {}
    diag = {"slots_with_box": len(lt.source), "gt_boxes": int(len(sub)),
            "matched": 0, "matched_emittable": 0}
    if len(sub) == 0 or not lt.source:
        return out, diag

    gx = sub["cx"].to_numpy(float)
    gy = sub["cy"].to_numpy(float)
    gl = sub["train_label"].to_numpy(int)
    radius = float(np.median(sub["w"].to_numpy(float))) * radius_ratio

    pairs = []
    for si, d in lt.source.items():
        dist = np.hypot(gx - float(d["cx"]), gy - float(d["cy"]))
        j = int(dist.argmin())
        if dist[j] <= radius:
            pairs.append((float(dist[j]), si, j))
    pairs.sort()

    used_box: set[int] = set()
    for _dist, si, j in pairs:
        if j in used_box:
            continue
        used_box.add(j)
        diag["matched"] += 1
        if int(gl[j]) in tok.vision_emit_classes:
            out[si] = int(gl[j])
            diag["matched_emittable"] += 1
    return out, diag


# The aligner

def align(pred: list[int], truth: list[int], prefer: str = "sub") -> dict:
    """
    Wagner-Fischer with a backtrace, returning {"dist", "S", "D", "I"}. Rows are
    the reference and columns the prediction, so D is a reference sign nothing
    produced and I is a predicted slot with no reference sign.

    The two tie-breaks:
        prefer="sub"   match > sub > del > ins   (usually fewer insertions)
        prefer="indel" match > del > ins > sub   (usually more insertions)
    They are local preferences, so neither is guaranteed to be an extreme; on the
    saved results their order flips in 1 of 570 tablet-arm rows. Only S + D + I is
    well defined, so both splits are published. Gate G7 checks the identities
    dist == levenshtein, S + D + I == dist and I - D == len(pred) - len(truth).
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
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and truth[i - 1] == pred[j - 1]
                and d[i][j] == d[i - 1][j - 1]):
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
            S += 1
            i, j = i - 1, j - 1
        elif op == "D":
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return {"dist": d[n][m], "S": S, "D": D, "I": I}


def score_unit(pred: list[int], truth: list[int], names: dict) -> dict:
    """
    One tablet, one arm. WER and CER come from evaluate_fusion.score unchanged,
    with the S/D/I breakdown added for both tie-breaks.
    """
    base = score(pred, truth, names)
    a_sub = align(pred, truth, "sub")
    a_ind = align(pred, truth, "indel")
    # Recover the raw character distance from score() instead of running the DP again.
    t_txt = " ".join(names.get(s, f"?{s}") for s in truth)
    cer_len = max(1, len(t_txt))
    return {
        "wer": base["wer"], "cer": base["cer"],
        "dist": a_sub["dist"],
        "len_pred": len(pred), "len_truth": len(truth),
        "ratio": len(pred) / max(1, len(truth)),
        "sdi_sub": {k: a_sub[k] for k in ("S", "D", "I")},
        "sdi_indel": {k: a_ind[k] for k in ("S", "D", "I")},
        "cer_dist": int(round(base["cer"] * cer_len)),
        "cer_len": cer_len,
    }


# The language-model passes

def lm_pass(model, tok: SignTokenizer, lt: LiveTablet, threshold: float,
            oracle: dict[int, int] | None = None, verbose: bool = False) -> dict:
    """
    One leave-one-out LM pass over every block of one tablet.

    `threshold` sets the context: THRESHOLD (0.50) is arm 2's, where only
    detections at or above it are written in; 0.0 is arm 3's, with every named
    detection as context. `oracle` (arm 0b) swaps in the annotated label at
    matched slots. The two contexts get separate passes (gate G9), and the block
    starts as all <mask> so no true token can leak in.

    Returns {"p_lm", "lm_conf", "density", "n_ctx", "n_pos"}.
    """
    sup = support_ids(tok)
    p_lm: dict[int, torch.Tensor] = {}
    lm_conf: dict[int, float] = {}
    n_ctx = n_pos = 0

    for block in lt.blocks:
        n = len(block)
        ids = torch.tensor([tok.bos_id] + [tok.mask_id] * n + [tok.eos_id],
                           dtype=torch.long)
        dets_map = {}
        for pos, si in enumerate(block, start=1):
            if oracle is None:
                dets_map[pos] = lt.detections[si]
                continue
            # For 0b an unmatched slot has no oracle evidence, so it becomes
            # <mask>, never the detector's own guess.
            lbl = oracle.get(si)
            dets_map[pos] = (Detection("detected", train_label=int(lbl),
                                       confidence=1.0)
                             if lbl is not None else Detection("missed"))

        inp = vision_context(ids, dets_map, tok, threshold)
        positions = sorted(dets_map)
        n_pos += len(positions)
        n_ctx += sum(1 for p in positions if inp[p] != tok.mask_id)

        targets = ([p for p in positions if inp[p] == tok.mask_id]
                   if threshold > 0.0 or oracle is not None else positions)
        if not targets:
            continue
        rows = lm_posterior(model, inp, targets, tok)
        for r, p in enumerate(targets):
            si = block[p - 1]
            p_lm[si] = rows[r]
            tot = float(rows[r][sup].sum())
            lm_conf[si] = float(rows[r][sup].max()) / tot if tot > 0 else 0.0

    return {"p_lm": p_lm, "lm_conf": lm_conf, "n_ctx": n_ctx, "n_pos": n_pos,
            "density": n_ctx / max(1, n_pos)}


# The arms

def assemble_arms(tok: SignTokenizer, lt: LiveTablet, thr_pass: dict,
                  fus_pass: dict) -> dict[str, list[int]]:
    """
    Every live arm's prediction sequence for one tablet (no ground truth used).

      1_vision_only          silent where vision can't name a sign
      1b_vision_only_padded  length-matched, SENTINEL where vision can't name;
                             1b vs 3_fusion_lam1.0 is the central hypothesis
      2_threshold_mask       conf >= 0.50 kept, the rest filled by the LM
      3_fusion_lam{lam}      ten lambdas from one cached pass
      3b_lmfill_fusion_ctx   the LM alone with arm 3's context, which splits
                             the fusion gain into richer context vs keeping
                             the vision vote
      3f_deployed_lam1_floor lambda 1.0 with the posterior floor: what ships

    Only the lambda arms may start with "3_fusion_lam", because
    significance.selection_corrected parses lambda from those keys.
    """
    scored = lt.scored
    dets = [lt.detections[si] for si in scored]
    arms: dict[str, list[int]] = {}

    # A sparse tablet can produce no block at higher conf. Every arm then comes
    # back empty (not missing), so the per-tablet series stay aligned, and it
    # scores WER 1.0 by pure deletion.
    if not scored:
        empty: list[int] = []
        for name in (["1_vision_only", "1b_vision_only_padded",
                      "2_threshold_mask", "3b_lmfill_fusion_ctx",
                      "3f_deployed_lam1_floor"]
                     + [f"3_fusion_lam{lam}" for lam in LAMBDAS]):
            arms[name] = list(empty)
        return arms

    arms["1_vision_only"] = [int(d.train_label) for d in dets
                             if d.is_informative]
    arms["1b_vision_only_padded"] = [
        int(d.train_label) if d.is_informative else SENTINEL for d in dets]

    # Arm 2
    uniform_row = torch.full((tok.vocab_size,), 1.0 / tok.vocab_size,
                             dtype=torch.float64)
    pred2 = []
    thr_rows, thr_idx = [], []
    for k, si in enumerate(scored):
        d = lt.detections[si]
        if d.is_informative and d.confidence >= THRESHOLD:
            pred2.append(int(d.train_label))
        else:
            pred2.append(None)
            row = thr_pass["p_lm"].get(si)
            if row is not None:
                thr_rows.append(row)
                thr_idx.append(k)
    if thr_rows:
        stack = torch.stack(thr_rows)
        filled = argmax_labels(
            fuse(uniform_row.expand_as(stack), stack, 1.0, tok), tok)
        for k, lbl in zip(thr_idx, filled):
            pred2[k] = int(lbl)
    arms["2_threshold_mask"] = [SENTINEL if p is None else p for p in pred2]

    # Arms 3, 3b and 3f from one cached fusion pass
    rows = torch.stack([fus_pass["p_lm"][si] for si in scored])
    p_v = vision_posterior(dets, tok)
    for lam in LAMBDAS:
        arms[f"3_fusion_lam{lam}"] = [
            int(x) for x in argmax_labels(fuse(p_v, rows, lam, tok), tok)]
    arms["3b_lmfill_fusion_ctx"] = [
        int(x) for x in
        argmax_labels(fuse(uniform_row.expand_as(rows), rows, 1.0, tok), tok)]

    deployed = arms[f"3_fusion_lam{LIVE_LAMBDA}"]
    arms["3f_deployed_lam1_floor"] = [
        lbl if (dets[k].is_informative
                or fus_pass["lm_conf"].get(scored[k], 0.0) >= LM_FLOOR)
        else SENTINEL
        for k, lbl in enumerate(deployed)]
    return arms


def oracle_lmfill_arm(tok: SignTokenizer, lt: LiveTablet,
                      oracle: dict[int, int], ora_pass: dict) -> list[int]:
    """
    Arm 0b: the arm 2/3 structure with vision replaced by the annotation. It is
    length-matched, so it is the only oracle that bounds arms 2 and 3.
    """
    uniform_row = torch.full((tok.vocab_size,), 1.0 / tok.vocab_size,
                             dtype=torch.float64)
    out: list[int] = []
    fill_rows, fill_idx = [], []
    for k, si in enumerate(lt.scored):
        lbl = oracle.get(si)
        if lbl is not None:
            out.append(int(lbl))
            continue
        out.append(SENTINEL)
        row = ora_pass["p_lm"].get(si)
        if row is not None:
            fill_rows.append(row)
            fill_idx.append(k)
    if fill_rows:
        stack = torch.stack(fill_rows)
        filled = argmax_labels(
            fuse(uniform_row.expand_as(stack), stack, 1.0, tok), tok)
        for k, lbl in zip(fill_idx, filled):
            out[k] = int(lbl)
    return out


# Validation gates

def _gate(gid: str, desc: str, passed: bool, evidence: list[str]) -> dict:
    return {"id": gid, "desc": desc, "pass": bool(passed), "evidence": evidence}


def gate_g1(tablets: list[str], records: list[dict]) -> dict:
    """30 detection files, each with a test-split corpus record, and no strays."""
    have_recs = {r["tablet"] for r in records if r.get("split") == "test"}
    missing = sorted(t for t in tablets if t not in have_recs)
    return _gate("G1", "30 detection files, all joinable to the corpus",
                 len(tablets) == 30 and not missing,
                 [f"detection files (excluding _summary) : {len(tablets)}",
                  f"test-split tablets in corpus         : {len(have_recs)}",
                  f"detection tablets with no record     : {missing or 'none'}",
                  f"corpus test tablets with no detection: "
                  f"{sorted(have_recs - set(tablets))}"])


def gate_g2(diag: dict) -> dict:
    """Every record used is split == 'test'; views are a subset of {Obv, Rev}."""
    ok = diag["splits"] == ["test"] and set(diag["views"]) <= {"Obv", "Rev"}
    return _gate("G2", "all records test-split; views subset of {Obv, Rev}", ok,
                 [f"splits of records used : {diag['splits']}",
                  f"views of records used  : {diag['views']}",
                  f"records used           : {diag['n_records']}",
                  f"CORPUS (tablet, view) pairs : "
                  f"{diag['n_corpus_view_pairs']}",
                  "the ANNOTATION has 51 -- three tablets carry reverse-face",
                  "text with no reverse-face boxes. Do not conflate the two.",
                  "views must be a subset of {Obv, Rev}: the alphabetical sort",
                  "is a reading-order CONVENTION for these two views only"])


def gate_g3a(diag: dict) -> dict:
    """Regression against the expected reference totals."""
    got = {k: diag[k] for k in REF_EXPECT}
    ok = got == REF_EXPECT and set(diag["unk_labels"]) == REF_EXPECT_UNK_LABELS
    ev = [f"{k:<14} got {got[k]:>6}  expected {REF_EXPECT[k]:>6}  "
          f"{'OK' if got[k] == REF_EXPECT[k] else 'MISMATCH'}"
          for k in ("corpus_tokens", "brk", "real", "unk", "scored")]
    ev += [f"out-of-vocabulary train_labels : {diag['unk_labels']}",
           f"expected                       : {sorted(REF_EXPECT_UNK_LABELS)}",
           f"<brk> share of corpus tokens   : "
           f"{100 * diag['brk'] / max(1, diag['corpus_tokens']):.1f}%",
           f"real tokens inside the 82 emit classes : "
           f"{diag['emittable_real']} "
           f"({100 * diag['emittable_real'] / max(1, diag['real']):.1f}%)"]
    return _gate("G3a", "reference totals match the measured population", ok, ev)


def gate_g3b(records: list[dict], tablets: list[str],
             tok: SignTokenizer) -> dict:
    """
    Per tablet, len(reference) == count of non-<brk> corpus tokens, derived
    independently rather than from the totals in G3a.
    """
    refs_all, diag_all = build_reference(records, tablets, tok, drop_unk=False)
    bad = [(t, len(refs_all[t]), diag_all["per_tablet_real"][t])
           for t in tablets
           if len(refs_all[t]) != diag_all["per_tablet_real"][t]]
    return _gate("G3b", "per-tablet reference length equals non-<brk> count",
                 not bad,
                 [f"tablets checked : {len(tablets)}",
                  f"mismatches      : {len(bad)}  {bad[:6]}",
                  f"total non-<brk> corpus tokens : {diag_all['real']}",
                  f"total reference length (drop_unk=False) : "
                  f"{sum(len(v) for v in refs_all.values())}"])


def gate_g4(records: list[dict], tablets: list[str], tok: SignTokenizer,
            seed: int = SEED) -> dict:
    """The reference builder must sort, not inherit file order: shuffle the records and compare."""
    ref_a, _ = build_reference(records, tablets, tok)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    ref_b, _ = build_reference(shuffled, tablets, tok)
    ha = hashlib.sha256(
        json.dumps({k: ref_a[k] for k in sorted(ref_a)}).encode()).hexdigest()
    hb = hashlib.sha256(
        json.dumps({k: ref_b[k] for k in sorted(ref_b)}).encode()).hexdigest()
    differing = sorted(t for t in tablets if ref_a[t] != ref_b[t])
    return _gate("G4", "build_reference is order-independent (it sorts)",
                 ha == hb,
                 [f"sha256 from file order     : {ha}",
                  f"sha256 from shuffled order : {hb}",
                  f"records shuffled with seed {seed}; tablets differing: "
                  f"{differing or 'none'}"])


def gate_g5(tablets: list[str], lives: dict, arms: dict,
            matched_arms: list[str], per_unit: dict) -> dict:
    """
    Every length-matched arm really is length-matched on the same scored set,
    and every arm's per-tablet series is in the same tablet order (the paired
    tests pair by index).
    """
    bad_len, bad_order = [], []
    for t in tablets:
        n = len(lives[t].scored)
        for a in matched_arms:
            if len(arms[t][a]) != n:
                bad_len.append((t, a, len(arms[t][a]), n))
    for a in per_unit:
        if [r["tablet"] for r in per_unit[a]] != tablets:
            bad_order.append(a)
    return _gate("G5", "length-matched arms share one scored set and one order",
                 not bad_len and not bad_order,
                 [f"length-matched arms : {len(matched_arms)}",
                  f"length mismatches   : {len(bad_len)}  {bad_len[:4]}",
                  f"per-tablet series with a different tablet order : "
                  f"{bad_order or 'none'}",
                  f"scored slots total  : "
                  f"{sum(len(lives[t].scored) for t in tablets)}"])


def gate_g5b(tablets: list[str], lives: dict) -> dict:
    """Every detected slot resolves to exactly one source detection."""
    bad = []
    for t in tablets:
        lt = lives[t]
        want = sum(1 for _, s in lt.slots
                   if s.kind == "detected" and s.label is not None)
        if len(lt.source) != want:
            bad.append((t, len(lt.source), want))
    return _gate("G5b", "every detected slot maps back to one detection",
                 not bad,
                 [f"tablets with unresolved slots : {len(bad)}  {bad[:6]}",
                  f"detected slots resolved : "
                  f"{sum(len(lives[t].source) for t in tablets)}"])


def gate_g6a(hash_before: str, hash_after: str) -> dict:
    """The live predictions are byte-identical before and after ground truth."""
    return _gate("G6a", "live predictions unchanged across the GT phase",
                 hash_before == hash_after,
                 [f"sha256 before ground truth loaded : {hash_before}",
                  f"sha256 after  oracle arms built   : {hash_after}"])


def gate_g6b(tablets: list[str], lives: dict, arms: dict,
             oracles: dict, ceilings: dict) -> dict:
    """
    Provenance leak gate: every label in 1_vision_only must appear in that
    tablet's detections, and every label in 0a_* in its annotated boxes. This
    catches ground truth reaching the live arms before the snapshot.
    """
    bad_live, bad_or = [], []
    for t in tablets:
        det_labels = {int(d["train_label"]) for d in lives[t].dets}
        stray = {l for l in arms[t]["1_vision_only"] if l not in det_labels}
        if stray:
            bad_live.append((t, sorted(stray)[:5]))
        gt_labels = set(oracles[t]["labels"])
        for mode in ("geo", "view", "flat"):
            s2 = {l for l in ceilings[t][mode] if l not in gt_labels}
            if s2:
                bad_or.append((t, mode, sorted(s2)[:5]))
    return _gate("G6b", "no label appears that its own source never produced",
                 not bad_live and not bad_or,
                 [f"live arm labels absent from the detection JSON : "
                  f"{len(bad_live)}  {bad_live[:4]}",
                  f"oracle arm labels absent from the GT boxes     : "
                  f"{len(bad_or)}  {bad_or[:4]}"])


def gate_g7(tablets: list[str], refs: dict, arms: dict) -> dict:
    """
    Aligner identities on all 30 tablets and every arm:
        dist == evaluate_fusion.levenshtein
        S + D + I == dist
        I - D == len(pred) - len(truth)
        dist(sub) == dist(indel)
    """
    fails = []
    checked = 0
    for t in tablets:
        truth = refs[t]
        for a, pred in arms[t].items():
            checked += 1
            ref_dist = levenshtein(pred, truth)
            s = align(pred, truth, "sub")
            n = align(pred, truth, "indel")
            if s["dist"] != ref_dist or n["dist"] != ref_dist:
                fails.append((t, a, "dist", s["dist"], n["dist"], ref_dist))
            elif s["S"] + s["D"] + s["I"] != s["dist"]:
                fails.append((t, a, "sum_sub"))
            elif n["S"] + n["D"] + n["I"] != n["dist"]:
                fails.append((t, a, "sum_indel"))
            elif s["I"] - s["D"] != len(pred) - len(truth):
                fails.append((t, a, "delta_sub"))
            elif n["I"] - n["D"] != len(pred) - len(truth):
                fails.append((t, a, "delta_indel"))
    return _gate("G7", "aligner identities hold on every tablet and arm",
                 not fails,
                 [f"(tablet, arm) pairs checked : {checked}",
                  f"failures : {len(fails)}  {fails[:4]}",
                  "dist == levenshtein | S+D+I == dist | I-D == delta_len | "
                  "dist(sub) == dist(indel)"])


def gate_g8(tablets: list[str], ceilings: dict, oracles: dict,
            tok: SignTokenizer) -> dict:
    """Every label an 0a_* arm emits is a real annotated label for that tablet and inside the 82 emit classes."""
    bad = []
    for t in tablets:
        gt = set(oracles[t]["labels"])
        for mode in ("geo", "view", "flat"):
            for l in ceilings[t][mode]:
                if l not in gt or l not in tok.vision_emit_classes:
                    bad.append((t, mode, l))
                    break
    return _gate("G8", "oracle arms emit only annotated, emittable labels",
                 not bad,
                 [f"violations : {len(bad)}  {bad[:6]}",
                  f"oracle sequence lengths (geo/view/flat) : "
                  f"{sum(len(ceilings[t]['geo']) for t in tablets)} / "
                  f"{sum(len(ceilings[t]['view']) for t in tablets)} / "
                  f"{sum(len(ceilings[t]['flat']) for t in tablets)}"])


def gate_g9(tablets: list[str], lives: dict, thr: dict, fus: dict) -> dict:
    """
    Arm 2 and arm 3 must not share an LM cache: their contexts differ, so their
    posteriors must too. Checked on every tablet with a sub-threshold detection.
    """
    same, tested = [], 0
    for t in tablets:
        lt = lives[t]
        has_sub = any(d.is_informative and d.confidence < THRESHOLD
                      for d in (lt.detections[si] for si in lt.scored))
        if not has_sub:
            continue
        tested += 1
        shared = set(thr[t]["p_lm"]) & set(fus[t]["p_lm"])
        if not shared or all(
                torch.equal(thr[t]["p_lm"][si], fus[t]["p_lm"][si])
                for si in shared):
            same.append(t)
    d2 = sum(p["n_ctx"] for p in thr.values()) / max(
        1, sum(p["n_pos"] for p in thr.values()))
    d3 = sum(p["n_ctx"] for p in fus.values()) / max(
        1, sum(p["n_pos"] for p in fus.values()))
    return _gate("G9", "the two LM passes are distinct and ordered correctly",
                 not same and tested > 0 and d2 < d3,
                 [f"tablets with a sub-{THRESHOLD} detection : {tested}",
                  f"of those, tablets whose two passes agree everywhere : "
                  f"{same or 'none'}   (any is a cache bug)",
                  f"context density arm 2 : {d2:.3f}  "
                  f"(mask share {100 * (1 - d2):.1f}%)",
                  f"context density arm 3 : {d3:.3f}  "
                  f"(mask share {100 * (1 - d3):.1f}%)",
                  f"SIMULATED, for comparison: arm 2 {SIM_CTX_THRESHOLD:.3f} "
                  f"(mask 88.1%) | arm 3 {SIM_CTX_FUSION:.3f} (mask 80.6%)",
                  "live gives the LM FEWER mask positions at arm 3, not more"])


def gate_g10(tablets: list[str], lives: dict, arms: dict) -> dict:
    """
    3_fusion_lam0.0 must equal 1b_vision_only_padded at every informative slot,
    since at lambda 0 the score is vision's alone.
    """
    bad, checked = [], 0
    a0 = f"3_fusion_lam{LAMBDAS[0]}"
    for t in tablets:
        lt = lives[t]
        for k, si in enumerate(lt.scored):
            if not lt.detections[si].is_informative:
                continue
            checked += 1
            if arms[t][a0][k] != arms[t]["1b_vision_only_padded"][k]:
                bad.append((t, si, arms[t][a0][k],
                            arms[t]["1b_vision_only_padded"][k]))
    return _gate("G10", f"{a0} equals vision at every informative slot",
                 not bad,
                 [f"informative slots checked : {checked}",
                  f"disagreements : {len(bad)}  {bad[:4]}"])


def gate_g11(tablets: list[str], lives: dict, bb) -> dict:
    """
    Coordinate frame check: per tablet, the annotated box centres must overlap
    the detections' extent on both axes.
    """
    bad, rows = [], []
    for t in tablets:
        sub = bb[bb["tablet_CDLI"] == t]
        dx = np.array([d["cx"] for d in lives[t].dets], float)
        dy = np.array([d["cy"] for d in lives[t].dets], float)
        if len(sub) == 0 or len(dx) == 0:
            continue
        gx, gy = sub["cx"].to_numpy(float), sub["cy"].to_numpy(float)
        ox = min(gx.max(), dx.max()) - max(gx.min(), dx.min())
        oy = min(gy.max(), dy.max()) - max(gy.min(), dy.min())
        if ox <= 0 or oy <= 0:
            bad.append(t)
        rows.append(f"    {t:<10} gt x[{gx.min():.0f},{gx.max():.0f}] "
                    f"y[{gy.min():.0f},{gy.max():.0f}]   "
                    f"det x[{dx.min():.0f},{dx.max():.0f}] "
                    f"y[{dy.min():.0f},{dy.max():.0f}]")
    return _gate("G11", "GT and detection coordinate frames overlap",
                 not bad, [f"tablets with no overlap : {bad or 'none'}"]
                 + rows)


def gate_g12(lives: dict) -> dict:
    """
    Slots outside every block are excluded and counted, and every arm uses the
    same scored set.
    """
    tot = sum(lt.n_slots for lt in lives.values())
    inb = sum(len(lt.scored) for lt in lives.values())
    dup = [t for t, lt in lives.items() if len(set(lt.scored)) != len(lt.scored)]
    oob = [t for t, lt in lives.items()
           if lt.scored and (lt.scored[0] < 0 or lt.scored[-1] >= lt.n_slots)]
    return _gate("G12", "one shared scored population; remnants excluded",
                 not dup and not oob and inb > 0,
                 [f"slots total          : {tot}",
                  f"slots inside a block : {inb}",
                  f"slots dropped in under-{MIN_BLOCK} remnants : {tot - inb}"
                  f"  ({100 * (tot - inb) / max(1, tot):.2f}%)",
                  f"tablets with duplicate scored indices : {dup or 'none'}",
                  f"tablets with out-of-range indices     : {oob or 'none'}"])


def gate_g13(tablets: list[str], bb) -> dict:
    """
    Face-order audit: prints each tablet's per-view mean annotated cy so the
    reversed faces can be seen, and fails if the set differs from the expected one.
    """
    rows, flagged = [], []
    for t in tablets:
        sub = bb[bb["tablet_CDLI"] == t]
        means = {v: float(g["cy"].mean()) for v, g in sub.groupby("view_desc")}
        counts = {v: int(len(g)) for v, g in sub.groupby("view_desc")}
        obv, rev = means.get("Obv"), means.get("Rev")
        flag = ""
        if obv is not None and rev is not None and rev < obv:
            flag = "REV ABOVE OBV"
            flagged.append(t)
        rows.append(
            f"    {t:<10} "
            f"Obv cy {('%8.1f' % obv) if obv is not None else '      --'}"
            f" n={counts.get('Obv', 0):<4} "
            f"Rev cy {('%8.1f' % rev) if rev is not None else '      --'}"
            f" n={counts.get('Rev', 0):<4} {flag}")
    got = set(flagged)
    ev = ["per-tablet mean annotated box cy by view (segment() orders by "
          "cy.min(), so Rev above Obv transposes two whole faces):"] + rows
    ev += [f"  reverse-above-obverse : {sorted(got)}",
           f"  expected (2026-09-18) : {sorted(FACE_ORDER_EXPECT)}",
           f"  plus the known face MERGE on {FACE_MERGE_KNOWN}, a distinct "
           f"defect, giving {len(got | {FACE_MERGE_KNOWN})} of {len(tablets)} "
           f"tablets ({100 * len(got | {FACE_MERGE_KNOWN}) / len(tablets):.0f}%)"]
    return _gate("G13", "face-order audit reproduces the measured population",
                 got == FACE_ORDER_EXPECT, ev)


# Reporting

def print_gate_panel(gates: list[dict]) -> bool:
    """One explicit PASS/FAIL line at the top, then the evidence beneath it."""
    all_ok = all(g["pass"] for g in gates)
    print("\n" + RULE)
    print(f"VALIDATION GATES : {'PASS' if all_ok else 'FAIL'}   "
          f"({sum(g['pass'] for g in gates)}/{len(gates)})")
    if not all_ok:
        print("IF THIS FAILED, IGNORE EVERY NUMBER BELOW.")
    print(RULE)
    for g in gates:
        print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {g['id']:<5} {g['desc']}")
    print()
    for g in gates:
        print("-" * 78)
        print(f"  {g['id']}  {g['desc']}   -> {'PASS' if g['pass'] else 'FAIL'}")
        for line in g["evidence"]:
            print(line if line.startswith("    ") else f"    {line}")
    print("-" * 78)
    return all_ok


def hash_arms(tablets: list[str], arms: dict) -> str:
    """sha256 of every live arm on every tablet, in a fixed order."""
    blob = json.dumps(
        {t: {a: arms[t][a] for a in sorted(arms[t])} for t in tablets},
        sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def mde_table(n: int = 30) -> list[str]:
    """
    Minimum detectable effect at n tablets, printed beside the p-value: the
    critical standardised effect is 1.96/sqrt(n), and 80% power needs
    (1.96 + 0.84)/sqrt(n).
    """
    root = n ** 0.5
    return [f"n = {n} tablets, paired, two-sided alpha = 0.05",
            f"  detectable at p < 0.05 : d >= {1.96 / root:.3f}",
            f"  80% power at           : d  = {(1.96 + 0.8416) / root:.3f}",
            f"  the SIMULATED WER effect was d = 0.516 -- exactly on that",
            f"  boundary, so an interval spanning zero here is weak evidence",
            f"  of nothing, not evidence of no effect"]


def summarise(per_unit: dict, arm: str) -> dict:
    rows = per_unit[arm]
    n = max(1, len(rows))
    return {
        "wer": sum(r["wer"] for r in rows) / n,
        "cer": sum(r["cer"] for r in rows) / n,
        "wer_micro": sum(r["dist"] for r in rows)
                     / max(1, sum(r["len_truth"] for r in rows)),
        "cer_micro": sum(r["cer_dist"] for r in rows)
                     / max(1, sum(r["cer_len"] for r in rows)),
        "S": sum(r["sdi_sub"]["S"] for r in rows),
        "D": sum(r["sdi_sub"]["D"] for r in rows),
        # These key names are kept for compatibility with the saved artifact,
        # even though neither tie-break is a true extreme.
        "I_min": sum(r["sdi_sub"]["I"] for r in rows),
        "I_max": sum(r["sdi_indel"]["I"] for r in rows),
        "len_pred": sum(r["len_pred"] for r in rows),
        "len_truth": sum(r["len_truth"] for r in rows),
    }


# Main

def run_conf_sweep(tok, names, refs, tablets, args) -> dict:
    """
    The conf sweep, published as evidence for not selecting conf on WER: the
    WER floor moves with conf, so a WER sweep only finds the length-matching point.
    """
    model = load_sequence_model()
    out = {}
    for conf in CONF_SWEEP:
        t0 = time.time()
        lives = {t: live_slots(t, conf, args.cell, args.min_size, args.edge,
                               tok) for t in tablets}
        ratios = sorted(lives[t].n_slots / max(1, len(refs[t]))
                        for t in tablets)
        rows = {"1b_vision_only_padded": [], "2_threshold_mask": [],
                f"3_fusion_lam{LIVE_LAMBDA}": []}
        for t in tablets:
            lt = lives[t]
            thr = lm_pass(model, tok, lt, THRESHOLD)
            fus = lm_pass(model, tok, lt, 0.0)
            arms = assemble_arms(tok, lt, thr, fus)
            for a in rows:
                rows[a].append(score_unit(arms[a], refs[t], names))
        contributing = sum(1 for lt in lives.values() if lt.scored)
        out[f"{conf}"] = {
            "median_slots_per_real": float(np.median(ratios)),
            "slots": sum(lt.n_slots for lt in lives.values()),
            "scored_slots": sum(len(lt.scored) for lt in lives.values()),
            "tablets_contributing": contributing,
            "tablets_total": len(tablets),
            "arms": {a: summarise(rows, a) for a in rows},
        }
        print(f"    conf {conf:<5} median slots/real "
              f"{out[str(conf)]['median_slots_per_real']:.2f}  "
              f"tablets {contributing}/{len(tablets)}  "
              f"WER 1b {out[str(conf)]['arms']['1b_vision_only_padded']['wer']:.4f}"
              f"  fusion "
              f"{out[str(conf)]['arms'][f'3_fusion_lam{LIVE_LAMBDA}']['wer']:.4f}"
              f"   [{time.time() - t0:.0f}s]", flush=True)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true",
                    help="phases 0-2 plus G1-G4, G12 and the G13 audit. "
                         "Seconds, no model, no scoring.")
    ap.add_argument("--self-test", action="store_true",
                    help="the aligner against evaluate_fusion.levenshtein on "
                         "all 30 tablets (G7), with no model loaded.")
    ap.add_argument("--conf-sweep", action="store_true",
                    help="the descriptive appendix for the conf refusal. Slow.")
    ap.add_argument("--conf", type=float, default=LIVE_CONF)
    ap.add_argument("--cell", type=float, default=LIVE_CELL)
    ap.add_argument("--min-size", type=int, default=LIVE_MIN_SIZE)
    ap.add_argument("--edge", action="store_true", default=LIVE_EDGE_EXTEND)
    args = ap.parse_args()

    config.check_environment()

    print(RULE)
    print("LIVE FUSION EVALUATION -- real detector output, scored per tablet")
    print(RULE)
    print(f"  conf {args.conf}  cell {args.cell}  min_size {args.min_size}  "
          f"edge {'ON' if args.edge else 'off'}")
    print(f"  lambda sweep {LAMBDAS} | threshold arm T={THRESHOLD} | "
          f"LM floor {LM_FLOOR}")
    print("  NOTHING HERE IS SIMULATED. artifacts/fusion_results.json stays "
          "the simulated artifact.")

    tok = SignTokenizer()
    names = load_sign_names()
    names[SENTINEL] = "[x]"          # a slot with no name

    # Phase 0: population
    tablets = live_tablets()
    records = load_corpus_records()
    print(f"\n  phase 0  population : {len(tablets)} tablets with detections, "
          f"{len(records)} corpus records")

    # Phase 1: the reference (no ground truth)
    refs, diag = build_reference(records, tablets, tok)
    print(f"  phase 1  reference  : {diag['corpus_tokens']} corpus tokens -> "
          f"{diag['real']} real -> {diag['scored']} scored")

    # Phase 2: the live slot sequence (no ground truth)
    lives = {t: live_slots(t, args.conf, args.cell, args.min_size, args.edge,
                           tok) for t in tablets}
    n_slots = sum(lt.n_slots for lt in lives.values())
    n_det = sum(sum(1 for d in lt.detections if d.kind == "detected")
                for lt in lives.values())
    n_unkd = sum(sum(1 for d in lt.detections if d.kind == "unknown")
                 for lt in lives.values())
    n_mis = sum(sum(1 for d in lt.detections if d.kind == "missed")
                for lt in lives.values())
    n_scored_slots = sum(len(lt.scored) for lt in lives.values())
    print(f"  phase 2  live slots : {n_slots} slots "
          f"(detected {n_det} | unknown {n_unkd} | inserted {n_mis}) in "
          f"{sum(len(lt.groups) for lt in lives.values())} segments, "
          f"{sum(len(lt.blocks) for lt in lives.values())} blocks, "
          f"{n_scored_slots} slots scored")

    ratios = sorted(lives[t].n_slots / max(1, len(refs[t])) for t in tablets)
    median_ratio = float(np.median(ratios))

    if args.gates_only:
        bb = load_gt_boxes(tablets)
        gates = [gate_g1(tablets, records), gate_g2(diag), gate_g3a(diag),
                 gate_g3b(records, tablets, tok), gate_g4(records, tablets, tok),
                 gate_g5b(tablets, lives), gate_g12(lives),
                 gate_g13(tablets, bb)]
        ok = print_gate_panel(gates)
        print_length_regime(tablets, refs, lives, ratios, median_ratio)
        print("\n" + RULE)
        print("--gates-only: phases 0-2 complete. Nothing scored, no model "
              "loaded. Ground truth")
        print("is touched only by G11/G13, which run last by design.")
        print(RULE)
        sys.exit(0 if ok else 1)

    if args.self_test:
        print("\n  --self-test: aligner identities against "
              "evaluate_fusion.levenshtein")
        stub = {t: {"1b_vision_only_padded":
                    [int(d.train_label) if d.is_informative else SENTINEL
                     for d in (lives[t].detections[si]
                               for si in lives[t].scored)],
                    "1_vision_only":
                    [int(lives[t].detections[si].train_label)
                     for si in lives[t].scored
                     if lives[t].detections[si].is_informative]}
                for t in tablets}
        ok = print_gate_panel([gate_g7(tablets, refs, stub)])
        sys.exit(0 if ok else 1)

    if args.conf_sweep:
        print("\n  --conf-sweep: the descriptive appendix for the conf "
              "refusal (slow)")
        sweep = run_conf_sweep(tok, names, refs, tablets, args)
        path = config.ARTIFACTS_DIR / "live_conf_sweep.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"simulated_vision": False, "unit": "tablet",
                       "cell": args.cell, "min_size": args.min_size,
                       "edge_extend": bool(args.edge), "sweep": sweep}, fh,
                      indent=2)
        print(f"\n  written: {path}")
        print("\n  THE SWEEP IS PUBLISHED AND DECLINED AS A SELECTION RULE.")
        print("  Reason 1: each tablet's WER floor is |slots/real - 1| and that")
        print("  ratio is monotone in conf, so WER selects the length-matching")
        print("  point, not the best-reading point.")
        print("  Reason 2, FOUND BY RUNNING IT: the SCORED POPULATION SHRINKS")
        print("  as conf rises. Above the frozen value, sparse tablets lose")
        print("  enough detections that no cluster reaches the minimum size and")
        print("  they contribute no scored slots at all -- see the tablets")
        print("  column above. WER at two different conf values is therefore")
        print("  not measured on the same evidence, and comparing them selects")
        print("  partly on which tablets survived. conf stays 0.15; this")
        print("  sweep is the evidence FOR not selecting on WER.")
        print(RULE)
        return

    # Phase 3: the two LM passes
    print(f"\n  phase 3  loading Domain 2 checkpoint...")
    t0 = time.time()
    model = load_sequence_model()
    print(f"           loaded in {time.time() - t0:.1f}s; two leave-one-out "
          f"passes over {n_scored_slots} positions (CPU, ~43 ms each)")
    thr_passes, fus_passes = {}, {}
    t0 = time.time()
    for k, t in enumerate(tablets, 1):
        thr_passes[t] = lm_pass(model, tok, lives[t], THRESHOLD)
        fus_passes[t] = lm_pass(model, tok, lives[t], 0.0)
        print(f"    {k:>2}/{len(tablets)}  {t}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    # Phase 4: the live arms (still no ground truth)
    arms = {t: assemble_arms(tok, lives[t], thr_passes[t], fus_passes[t])
            for t in tablets}
    print(f"  phase 4  live arms  : {len(next(iter(arms.values())))} arms")

    # Phase 5: snapshot
    hash_before = hash_arms(tablets, arms)
    print(f"  phase 5  live hash  : {hash_before[:32]}...")

    # Phase 6: ground truth enters here and nowhere earlier
    print("  phase 6  loading annotation (oracle arms)...")
    bb = load_gt_boxes(tablets)
    ceilings, oracles, ora_arm = {}, {}, {}
    for t in tablets:
        lt = lives[t]
        ceilings[t] = {m: ceiling_slots(bb, t, m, tok)
                       for m in ("geo", "view", "flat")}
        omap, odiag = oracle_map(lt, bb, tok)
        sub = bb[bb["tablet_CDLI"] == t]
        oracles[t] = {"map": omap, "diag": odiag,
                      "labels": sorted({int(x) for x in sub["train_label"]})}
        ora_arm[t] = oracle_lmfill_arm(
            tok, lt, omap, lm_pass(model, tok, lt, 0.0, oracle=omap))

    # Phase 7: re-hash, the leak gate
    hash_after = hash_arms(tablets, arms)

    # Phase 8: score every arm, per tablet
    print("  phase 8  scoring...")
    all_arms = {t: dict(arms[t]) for t in tablets}
    for t in tablets:
        for m in ("geo", "view", "flat"):
            all_arms[t][f"0a_oracle_{m}"] = ceilings[t][m]
        all_arms[t]["0b_oracle_lmfill"] = ora_arm[t]

    arm_names = sorted(next(iter(all_arms.values())))
    per_unit = {a: [dict(score_unit(all_arms[t][a], refs[t], names),
                         tablet=t) for t in tablets] for a in arm_names}

    matched = [a for a in arm_names if a not in DROP_UNINFORMATIVE_ARMS]

    # Phase 9: gates, panel, artifact
    gates = [
        gate_g1(tablets, records), gate_g2(diag), gate_g3a(diag),
        gate_g3b(records, tablets, tok), gate_g4(records, tablets, tok),
        gate_g5(tablets, lives, all_arms, matched, per_unit),
        gate_g5b(tablets, lives),
        gate_g6a(hash_before, hash_after),
        gate_g6b(tablets, lives, all_arms, oracles, ceilings),
        gate_g7(tablets, refs, all_arms),
        gate_g8(tablets, ceilings, oracles, tok),
        gate_g9(tablets, lives, thr_passes, fus_passes),
        gate_g10(tablets, lives, all_arms),
        gate_g11(tablets, lives, bb),
        gate_g12(lives),
        gate_g13(tablets, bb),
    ]
    ok = print_gate_panel(gates)

    print_length_regime(tablets, refs, lives, ratios, median_ratio)

    # The results panel
    print("\n" + RULE)
    print("LIVE RESULTS  (REAL detector output. Absolute levels are NOT "
          "accuracies --")
    print("see the length regime above. The headline is the PAIRED "
          "DIFFERENCE between arms.)")
    print(RULE)
    print(f"  {'arm':<26} {'CER':>7} {'WER':>7} {'S':>6} {'D':>6} "
          f"{'I_sub':>6} {'I_ind':>6} {'len':>6}  regime")
    print("  " + "-" * 82)
    for a in arm_names:
        s = summarise(per_unit, a)
        tag = "  SILENT-WHERE-UNSURE" if a in DROP_UNINFORMATIVE_ARMS else ""
        print(f"  {a:<26} {s['cer']:>7.4f} {s['wer']:>7.4f} {s['S']:>6} "
              f"{s['D']:>6} {s['I_min']:>6} {s['I_max']:>6} "
              f"{s['len_pred']:>6}{tag}")
    print("  " + "-" * 82)
    print(f"  reference length (scored) : {diag['scored']}")
    print()
    print("  !! THE TWO REGIMES ARE NOT COMPARABLE ON AN ABSOLUTE LEVEL !!")
    print("  Rows tagged SILENT-WHERE-UNSURE emit NOTHING where vision could")
    print("  not name a sign, so they never pay for the ~1,100 insertions the")
    print("  other rows pay for answering at every position. They score better")
    print("  by DECLINING TO ANSWER, not by reading better. In particular,")
    print("  1_vision_only's WER sitting below the fusion arms' is an artefact")
    print("  of length and says nothing about reading quality -- which is")
    print("  exactly why 1b_vision_only_padded exists. Compare within a regime:")
    print("  0a_* with 1, and 0b / 1b / 2 / 3* with each other.")
    print()
    print("  0a_* are ANNOTATION-COVERAGE ORACLES, not ceilings: they are")
    print("  deletion-dominated where arms 2/3 are insertion-dominated, so an")
    print("  arm CAN exceed one. That means a longer slot sequence, not a")
    print("  better reading.")
    print("  There is NO live analogue of the simulated arm 4 language")
    print("  ceiling: it conditions on correct ground-truth context, which")
    print("  does not exist live. Stated, not faked.")

    # The paired headline
    print("\n  THE PAIRED DIFFERENCES (positive = the second arm is better)")
    pairs = [("2_threshold_mask", f"3_fusion_lam{LIVE_LAMBDA}"),
             ("2_threshold_mask", "3b_lmfill_fusion_ctx"),
             ("3b_lmfill_fusion_ctx", f"3_fusion_lam{LIVE_LAMBDA}"),
             ("1b_vision_only_padded", f"3_fusion_lam{LIVE_LAMBDA}"),
             ("1b_vision_only_padded", "3f_deployed_lam1_floor"),
             ("0b_oracle_lmfill", f"3_fusion_lam{LIVE_LAMBDA}")]
    for a, b in pairs:
        for metric in ("wer", "cer"):
            da = np.array([r[metric] for r in per_unit[a]])
            db = np.array([r[metric] for r in per_unit[b]])
            d = da - db
            sd = d.std(ddof=1)
            print(f"    {metric.upper()}  {b} vs {a}: "
                  f"{d.mean():+.4f}  d = {d.mean() / sd if sd else 0:+.3f}  "
                  f"(better/worse/tied "
                  f"{int((d > 0).sum())}/{int((d < 0).sum())}/"
                  f"{int((d == 0).sum())})")
    print()
    for line in mde_table(len(tablets)):
        print(f"    {line}")
    print("    p-values: python -m cuneiform.orchestrator.significance_live")

    # Sensitivity
    sens = {}
    for name, drop in (("drop_P335940", [FACE_MERGE_KNOWN]),
                       ("drop_all_five", CONFOUNDED)):
        keep = [i for i, t in enumerate(tablets) if t not in set(drop)]
        sens[name] = {"dropped": drop, "n": len(keep), "arms": {}}
        for a in arm_names:
            rows = [per_unit[a][i] for i in keep]
            sens[name]["arms"][a] = {
                "wer": sum(r["wer"] for r in rows) / max(1, len(rows)),
                "cer": sum(r["cer"] for r in rows) / max(1, len(rows))}
    print("\n  SENSITIVITY -- both pre-specified leave-one-out sets")
    print(f"    P335940 alone (its two faces merge), and all")
    print(f"    five face-defect tablets {CONFOUNDED}.")
    dep = f"3_fusion_lam{LIVE_LAMBDA}"
    for name, s in sens.items():
        print(f"    {name:<14} n={s['n']:<3} "
              f"{dep} WER {s['arms'][dep]['wer']:.4f} | "
              f"2_threshold_mask WER "
              f"{s['arms']['2_threshold_mask']['wer']:.4f}")

    # The artifact
    ceil_boxcov = sum(len(ceilings[t]["flat"]) for t in tablets)
    payload = {
        "simulated_vision": False,
        "unit": "tablet",
        "n_units": len(tablets),
        "tablet_order": tablets,
        "config": {"conf": args.conf, "cell": args.cell,
                   "min_size": args.min_size, "edge_extend": bool(args.edge),
                   "lambda_deployed": LIVE_LAMBDA, "lm_floor": LM_FLOOR,
                   "threshold": THRESHOLD, "lambdas": LAMBDAS,
                   "max_len": MAX_LEN, "min_block": MIN_BLOCK,
                   "sentinel": SENTINEL},
        "population": {
            "corpus_tokens": diag["corpus_tokens"], "brk": diag["brk"],
            "real": diag["real"], "unk": diag["unk"],
            "unk_labels": diag["unk_labels"], "scored_reference": diag["scored"],
            "emittable_real": diag["emittable_real"],
            "corpus_view_pairs": diag["n_corpus_view_pairs"],
            "gt_boxes": int(len(bb)),
            "slots": n_slots, "slots_detected": n_det,
            "slots_unknown": n_unkd, "slots_inserted": n_mis,
            "slots_scored": n_scored_slots,
            "segments": sum(len(lt.groups) for lt in lives.values()),
            "blocks": sum(len(lt.blocks) for lt in lives.values()),
            "median_slots_per_real": median_ratio,
            "slots_per_real_span": [ratios[0], ratios[-1]],
        },
        "ctx_density": {
            "live_arm2": sum(p["n_ctx"] for p in thr_passes.values())
                         / max(1, sum(p["n_pos"] for p in thr_passes.values())),
            "live_arm3": sum(p["n_ctx"] for p in fus_passes.values())
                         / max(1, sum(p["n_pos"] for p in fus_passes.values())),
            "simulated_arm2": SIM_CTX_THRESHOLD,
            "simulated_arm3": SIM_CTX_FUSION,
            "note": "quote all four with their denominators; live gives the "
                    "LM FEWER mask positions at arm 3 than the simulator did",
        },
        "ceilings": {
            "vision_nameable": 0.541,
            "vision_nameable_note": "2,323 of 4,293 real reference tokens are "
                                    "both box-covered and inside the 82 emit "
                                    "classes. NOT 0.673, which assumes a "
                                    "vocabulary the pipeline cannot emit from.",
            "oracle_flat_tokens": ceil_boxcov,
            "oracle_note": "0a_* are annotation-coverage oracles, not upper "
                           "bounds on arms 2/3.",
        },
        "confounds": {
            "face_order_reversed": sorted(FACE_ORDER_EXPECT),
            "face_merged": [FACE_MERGE_KNOWN],
            "all": CONFOUNDED,
            "share": len(CONFOUNDED) / len(tablets),
        },
        "oracle_match": {t: oracles[t]["diag"] for t in tablets},
        "arms": {a: summarise(per_unit, a) for a in arm_names},
        "per_unit": per_unit,
        "per_block": {a: [{"wer": r["wer"], "cer": r["cer"]}
                          for r in per_unit[a]] for a in arm_names},
        "sensitivity": sens,
        "mde": {"n": len(tablets), "d_detectable": 1.96 / len(tablets) ** 0.5,
                "d_80pct_power": (1.96 + 0.8416) / len(tablets) ** 0.5},
        "hashes": {"live_before_gt": hash_before, "live_after_gt": hash_after},
        "gates": [{k: g[k] for k in ("id", "desc", "pass")} for g in gates],
        "gates_pass": ok,
    }
    with RESULTS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  written: {RESULTS_JSON}")
    print(RULE)
    if not ok:
        sys.exit(1)


def print_length_regime(tablets, refs, lives, ratios, median_ratio) -> None:
    """The length regime, printed beside every absolute figure by design."""
    print("\n  PER-TABLET LENGTH REGIME  (each tablet's WER floor is "
          "|slots/real - 1|)")
    print(f"    {'tablet':<10} {'real':>6} {'slots':>6} {'scored':>7} "
          f"{'slots/real':>11}  {'floor':>6}")
    for t in tablets:
        real = len(refs[t])
        s = lives[t].n_slots
        r = s / max(1, real)
        mark = "  <- face defect" if t in CONFOUNDED else ""
        print(f"    {t:<10} {real:>6} {s:>6} {len(lives[t].scored):>7} "
              f"{r:>11.2f}  {abs(r - 1):>6.2f}{mark}")
    print(f"    span {ratios[0]:.2f} to {ratios[-1]:.2f}, median "
          f"{median_ratio:.2f}")
    print("    The absolute WER level reads out detector recall and annotation")
    print("    coverage, NOT reading accuracy. Never quote it bare, and never")
    print("    put it in the same column as the simulated 0.8266.")


if __name__ == "__main__":
    main()
