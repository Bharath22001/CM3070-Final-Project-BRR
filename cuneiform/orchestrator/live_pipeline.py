"""
My whole pipeline on real detector output, one tablet at a time: detections,
segments, lines, slots, blocks, fusion, the posterior floor, verbalisation and
Domain 3.

Frozen configuration: conf 0.15, cell 2.0, edge extrapolation off, min cluster
3, lambda 1.0, LM posterior floor 0.08. conf 0.15 is the middle of a plateau in
the recovery sweep. cell 2.0 was fixed rather than selected, because a length
metric rewards shattering a surface into pieces and so can't choose it. Edge
extrapolation stays off as the weaker assumption.

The floor: at a restored position vision is uniform, so the decision is the
LM's own posterior over the 282 signs. Below 0.08 I show [x] instead of a name.
0.08 is where accuracy below the line falls to the test-split unigram baseline
(5.3%); it is only accurate to about one bucket (roughly 0.06 to 0.10).

Blocks never span two segments, since two segments are two physical surfaces.
Fusion's change counts are not accuracy figures, because no live position has
a known answer. One known issue: on P335940 the two faces merge into one segment.

Run from the project root:
    python -m cuneiform.orchestrator.live_pipeline --tablet P335940
    python -m cuneiform.orchestrator.live_pipeline --tablet P335940 --generate
    python -m cuneiform.orchestrator.live_pipeline --tablet P335940 --lm-floor 0
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
import torch

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.domain3_generative.generate import (
    TabletInput, build_prompt, classify_citations, generate, load_catalogue,
    metadata_block, normalise_citation, scan_prose,
)
from cuneiform.domain3_generative.verbalise import verbalise
from cuneiform.orchestrator.fusion import (
    Detection, fused_predictions, load_sequence_model, support_ids,
)
from cuneiform.orchestrator.live_adapter import (
    build_slots, grid_components, load_detections, load_sign_names,
)

# The frozen configuration
LIVE_CONF = 0.15
LIVE_CELL = 2.0
LIVE_MIN_SIZE = 3
LIVE_EDGE_EXTEND = False
LIVE_LAMBDA = 1.0          # the best of a plateau from 0.25 to 1.0

# Posterior floor for showing a restoration. Precise only to about one bucket.
LM_FLOOR = 0.08

MAX_LEN = 128
MIN_BLOCK = 4              # same as dataset.build_blocks
RULE = "=" * 78


def segment(dets, cell_factor=LIVE_CELL, min_size=LIVE_MIN_SIZE):
    """Physical surfaces in reading order (top to bottom, then left to right)."""
    if not dets:
        return []
    cx = np.array([d["cx"] for d in dets], float)
    cy = np.array([d["cy"] for d in dets], float)
    w = np.array([d["x2"] - d["x1"] for d in dets], float)
    cell = max(float(np.median(w)) * cell_factor, 1.0)
    labels = grid_components(cx, cy, cell)
    groups = [np.flatnonzero(labels == l) for l in np.unique(labels)]
    groups = [g for g in groups if len(g) >= min_size]
    groups.sort(key=lambda ix: (float(cy[ix].min()), float(cx[ix].min())))
    return groups


def to_detections(slots, tok) -> list[Detection]:
    """Slot list to Detection list, applying the 82-class emit filter."""
    out = []
    for _, s in slots:
        if s.kind == "inserted" or s.label is None:
            out.append(Detection("missed"))
        elif int(s.label) in tok.vision_emit_classes:
            out.append(Detection("detected", train_label=int(s.label),
                                 confidence=float(s.confidence)))
        else:
            out.append(Detection("unknown"))
    return out


def build_live_blocks(slots, max_len: int = MAX_LEN) -> list[list[int]]:
    """
    Pack lines into blocks of at most max_len - 2 slots, returned as indices
    into the flat slot list. A block never spans two segments.
    """
    budget = max_len - 2

    lines: list[tuple[tuple, list[int]]] = []
    cur_key = None
    for i, (seg, s) in enumerate(slots):
        key = (seg, s.line)
        if key != cur_key:
            lines.append((key, []))
            cur_key = key
        lines[-1][1].append(i)

    blocks: list[list[int]] = []
    cur: list[int] = []
    cur_seg = None

    for (seg, _line), idxs in lines:
        if seg != cur_seg:
            if len(cur) >= MIN_BLOCK:
                blocks.append(cur)
            cur, cur_seg = [], seg

        if len(idxs) > budget:
            if len(cur) >= MIN_BLOCK:
                blocks.append(cur)
            cur = []
            for j in range(0, len(idxs), budget):
                piece = idxs[j:j + budget]
                if len(piece) >= MIN_BLOCK:
                    blocks.append(piece)
            continue

        if len(cur) + len(idxs) > budget:
            if len(cur) >= MIN_BLOCK:
                blocks.append(cur)
            cur = list(idxs)
        else:
            cur.extend(idxs)

    if len(cur) >= MIN_BLOCK:
        blocks.append(cur)
    return blocks


def run_fusion(model, tok, slots, detections, blocks, lam=LIVE_LAMBDA):
    """
    Fuse every block. Returns (preds, lm_conf), both keyed by slot index:
    the predicted train_label, and the LM's posterior at its argmax
    renormalised over the 282 signs.

    Each block has <mask> wherever vision had nothing, and fused_predictions
    re-masks each scored position on its own. lm_conf is kept for every
    position, but only acted on at restored ones.
    """
    sup = support_ids(tok)
    preds: dict[int, int] = {}
    lm_conf: dict[int, float] = {}

    for bi, block in enumerate(blocks, 1):
        body, dets_map = [], {}
        for pos, si in enumerate(block):
            d = detections[si]
            if d.kind == "detected":
                tid = tok.sign_to_id.get(int(d.train_label), tok.mask_id)
            else:
                tid = tok.mask_id
            body.append(tid)
            dets_map[pos + 1] = d          # +1 for <s>

        ids = torch.tensor([tok.bos_id] + body + [tok.eos_id], dtype=torch.long)
        t0 = time.time()
        positions, labels, p_lm = fused_predictions(model, ids, dets_map, lam, tok)

        for r, (p, lbl) in enumerate(zip(positions, labels)):
            si = block[p - 1]
            preds[si] = int(lbl)
            row = p_lm[r, sup]
            total = float(row.sum())
            lm_conf[si] = float(row.max()) / total if total > 0 else 0.0

        print(f"    block {bi:>2}/{len(blocks)}  {len(block):>3} slots  "
              f"{time.time() - t0:>5.1f}s")
    return preds, lm_conf


def build_domain3_input(tablet, slots, detections, preds, lm_conf, floor):
    """
    Slot list to TabletInput, applying the posterior floor.
    Returns (TabletInput, d3, restored_flags, n_demoted).

    I build one Verbalisation per (segment, line) so the model sees the line
    structure. A position outside any block or below the floor becomes [x]
    (counted separately), a vision read keeps its label, and an LM-supplied
    sign is marked restored and shown as [NAME].
    """
    d3: list[int | None] = []
    restored_flags: list[bool] = []
    n_demoted = 0

    for si in range(len(slots)):
        if si not in preds:
            d3.append(None)
            restored_flags.append(False)
        elif detections[si].is_informative:
            d3.append(preds[si])
            restored_flags.append(False)
        elif lm_conf.get(si, 0.0) < floor:
            # Below the floor the LM is no better than guessing the commonest sign.
            d3.append(None)
            restored_flags.append(False)
            n_demoted += 1
        else:
            d3.append(preds[si])
            restored_flags.append(True)

    # Group into tablet lines, keeping order.
    lines = []
    cur_key, cur_lab, cur_res = None, [], []
    for si, (seg, s) in enumerate(slots):
        key = (seg, s.line)
        if key != cur_key:
            if cur_lab:
                lines.append(verbalise(cur_lab, restored=cur_res))
            cur_key, cur_lab, cur_res = key, [], []
        cur_lab.append(d3[si])
        cur_res.append(restored_flags[si])
    if cur_lab:
        lines.append(verbalise(cur_lab, restored=cur_res))

    return TabletInput(tablet=tablet, lines=lines), d3, restored_flags, n_demoted


def call_domain3(t, catalogue):
    """
    The Domain 3 call with one retry on a truncated response. This is for
    serving only; an evaluation must not retry.
    """
    try:
        return generate(t, catalogue), 1
    except ValueError as exc:
        if "TRUNCATED AT CAP" not in str(exc):
            raise
        print(f"    first attempt truncated at the cap; retrying once.")
        print(f"    ({exc})")
        return generate(t, catalogue), 2


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tablet", required=True)
    ap.add_argument("--conf", type=float, default=LIVE_CONF)
    ap.add_argument("--cell", type=float, default=LIVE_CELL)
    ap.add_argument("--lam", type=float, default=LIVE_LAMBDA)
    ap.add_argument("--edge", action="store_true", default=LIVE_EDGE_EXTEND)
    ap.add_argument("--lm-floor", type=float, default=LM_FLOOR,
                    help="FIX 3 posterior floor. 0 disables demotion and "
                         "reproduces the pre-FIX-3 behaviour for comparison.")
    ap.add_argument("--generate", action="store_true",
                    help="call Domain 3. Off by default so the structural "
                         "runs stay fast. Ollama tray app must be running.")
    args = ap.parse_args()

    print(RULE)
    print(f"LIVE PIPELINE -- {args.tablet}")
    print(RULE)
    print(f"  conf {args.conf}  cell {args.cell}  lambda {args.lam}  "
          f"edge {'ON' if args.edge else 'off'}  lm-floor {args.lm_floor}")

    tok = SignTokenizer()
    names = load_sign_names()

    dets = load_detections(args.tablet, args.conf)
    groups = segment(dets, args.cell)
    kept = sum(len(g) for g in groups)
    slots = build_slots(dets, groups, extend_edges=args.edge)
    detections = to_detections(slots, tok)
    blocks = build_live_blocks(slots)

    in_blocks = sum(len(b) for b in blocks)
    n_det = sum(1 for d in detections if d.kind == "detected")
    n_unk = sum(1 for d in detections if d.kind == "unknown")
    n_mis = sum(1 for d in detections if d.kind == "missed")

    print(f"\n  detections        : {len(dets)}  "
          f"(kept {kept}, debris {len(dets) - kept})")
    print(f"  segments          : {len(groups)}")
    print(f"  slots             : {len(slots)}   "
          f"detected {n_det} | unknown {n_unk} | missed {n_mis}")
    print(f"  blocks            : {len(blocks)}  "
          f"covering {in_blocks}/{len(slots)} slots "
          f"(dropped {len(slots) - in_blocks} in under-{MIN_BLOCK} remnants)")

    if not blocks:
        print("\n  No block reached the minimum length. Nothing to fuse.")
        return

    print(f"\n  loading Domain 2 checkpoint...")
    model = load_sequence_model()
    print(f"  fusing {in_blocks} positions (leave-one-out, CPU):")
    preds, lm_conf = run_fusion(model, tok, slots, detections, blocks, args.lam)

    changed = sum(1 for si, lbl in preds.items()
                  if detections[si].kind == "detected"
                  and lbl != detections[si].train_label)
    print(f"\n  fusion CHANGED {changed} of {n_det} vision-named positions.")
    print("  NOT an accuracy figure -- no position has a known truth in the")
    print("  live path. Whether a change is a fix is for the live evaluation.")

    # What Domain 3 actually receives
    t, d3, restored_flags, n_demoted = build_domain3_input(
        args.tablet, slots, detections, preds, lm_conf, args.lm_floor
    )

    n_restorable = sum(1 for si in range(len(slots))
                       if si in preds and not detections[si].is_informative)
    n_named = sum(1 for x in d3 if x is not None)
    n_kept = n_restorable - n_demoted

    print(f"\n  restorations      : {n_restorable} candidates -> "
          f"{n_kept} kept as [NAME] | {n_demoted} demoted to [x] "
          f"at floor {args.lm_floor}")
    print("  DEMOTION IS NOT AN ACCURACY FIGURE EITHER. The floor was")
    print("  calibrated on the test split (arm D); how many of these demotions")
    print("  were correct is unknowable here, for the same reason as above.")

    if n_restorable:
        confs = sorted(lm_conf[si] for si in range(len(slots))
                       if si in preds and not detections[si].is_informative)
        mid = confs[len(confs) // 2]
        print(f"  restoration posterior: min {confs[0]:.3f} | "
              f"median {mid:.3f} | max {confs[-1]:.3f}")

    print(f"\n  Domain 3 receives  : {n_named}/{len(d3)} named positions "
          f"across {len(t.lines)} lines")

    catalogue = load_catalogue()
    prompt, had_meta = build_prompt(t, catalogue)
    plines = prompt.splitlines()
    print(f"  catalogue metadata : {had_meta}")

    # Print the metadata so the historical context can be checked against it.
    meta_text, _ = metadata_block(args.tablet, catalogue)
    print("\n  CATALOGUE METADATA AS SUPPLIED (audit historical_context "
          "against this):")
    print("    " + meta_text.replace("\n", "\n    "))
    print("\n  THE PROMPT HEADER Domain 3 will see:")
    try:
        i = next(k for k, l in enumerate(plines)
                 if l.startswith("TRANSLITERATION:"))
        for l in plines[i:]:
            if not l.strip():
                break
            print("    " + l)
    except StopIteration:
        print("    (header not found -- build_prompt's format changed)")

    # The sequence dump
    print("\n" + "-" * 78)
    print("  THE FUSED SEQUENCE  (* = fusion overrode vision)")
    print("-" * 78)
    cur, line_out = (None, None), []
    for si, ((seg, s), d) in enumerate(zip(slots, detections)):
        if (seg, s.line) != cur:
            if line_out:
                print("    " + " ".join(line_out))
            line_out, cur = [], (seg, s.line)
            print(f"\n  segment {seg}  line {s.line}")
        if si not in preds:
            line_out.append("\u00b7")
        elif d.is_informative:
            lbl = preds[si]
            mark = "*" if lbl != d.train_label else ""
            line_out.append(names.get(lbl, f"#{lbl}") + mark)
        elif not restored_flags[si]:
            line_out.append("[x]")
        else:
            line_out.append(f"[{names.get(preds[si], '?')}]")
    if line_out:
        print("    " + " ".join(line_out))

    print("\n  Legend: PLAIN = vision read, kept. NAME* = vision read, LM")
    print("  overrode. [NAME] = vision failed, LM inferred and cleared the")
    print("  floor. [x] = LM inferred below the floor, or the position fell")
    print("  outside a block. \u00b7 = outside a block (shown separately above).")

    # The Domain 3 call
    if not args.generate:
        print("\n  Domain 3 not called. Re-run with --generate (Ollama tray")
        print("  app must be running) to complete the chain.")
        print("\n" + RULE)
        return

    print("\n" + "-" * 78)
    print("  DOMAIN 3 -- local LLaMA-3.1 8B via Ollama")
    print("-" * 78)
    t0 = time.time()
    try:
        out, attempts = call_domain3(t, catalogue)
    except Exception as exc:
        print(f"  CALL FAILED: {type(exc).__name__}: {exc}")
        print("  Check the Ollama tray app is running.")
        print("\n" + RULE)
        return

    print(f"  returned in {time.time() - t0:.1f}s "
          f"({attempts} attempt{'s' if attempts > 1 else ''})")
    print(f"  schema-conformant  : "
          f"{'PASS' if all(k in out for k in ('tentative_reading', 'confidence', 'signs_used', 'uncertainty_notes', 'historical_context', 'metadata_used')) else 'FAIL'}")
    print(f"  confidence         : {out.get('confidence')}")
    print(f"  metadata_used      : claimed {out.get('metadata_used')} | "
          f"actual {out.get('_had_metadata')}")
    if out.get("_hit_cap"):
        print("  NOTE: parsed, but the model was still writing at the cap.")

    print(f"\n  tentative_reading  :\n    {out.get('tentative_reading', '')[:600]}")
    print(f"\n  uncertainty_notes  :\n    {out.get('uncertainty_notes', '')[:400]}")
    print(f"\n  historical_context :\n    {out.get('historical_context', '')[:400]}")

    supplied = t.all_names | t.displayed_names
    raw_cited = [s.strip() for s in out.get("signs_used", []) if s.strip()]
    cited = sorted(set(raw_cited))
    exact, normalised, invented = classify_citations(cited, supplied)
    print(f"\n  FAITHFULNESS (live input, so the reference set includes")
    print(f"  restored forms on both surfaces -- FIX 2):")
    print(f"    distinct supplied : {len(supplied)}")
    print(f"    citations         : {len(raw_cited)} ({len(cited)} distinct)")
    print(f"    cited exactly     : {len(exact)}")
    # Print the matched pairs, not just the count.
    print(f"    after normalising : {len(normalised)}")
    for c, matched in normalised[:8]:
        print(f"        {c}  ->  {matched}")
    print(f"    NOT in input      : {len(invented)}  {invented[:12]}")
    if not cited:
        print("    --> VACUOUS: nothing cited, so nothing could be checked.")
        print("        A gate failure, not a model success.")
    elif invented:
        print(f"    --> FAIL: {len(invented)} of {len(cited)} distinct "
              f"citations are not in the input.")
    elif normalised:
        print(f"    --> PASS, with {len(normalised)} citation(s) matching only")
        print(f"        after compound/restoration notation was stripped.")
        print(f"        Report that count; it measures how often the model")
        print(f"        renames what it was shown.")
    else:
        print("    --> PASS: every cited sign was present in the input.")

    # Prose scan, using the shared generate.scan_prose.
    print("\n  PROSE SCAN (the text a researcher actually reads):")
    scan = scan_prose(out, supplied)
    candidates, unmatched = scan["candidates"], scan["unmatched"]
    print(f"    sign-shaped tokens in prose : {len(candidates)}")
    print(f"    NOT in the input            : {len(unmatched)}  {unmatched[:12]}")

    print("\n  !! WHAT NEITHER GATE ABOVE CAN SEE. Both check SIGN NAMES. A")
    print("  PROPER NOUN in the prose -- a king, a city, a date, a person --")
    print("  is invisible to both, so this run can report PASS on a response")
    print("  that violates prompt rule 1. That is a scope limit of the")
    print("  faithfulness measure, not a bug: read the prose by eye and check")
    print("  every named entity against the metadata block printed above.")

    print("\n  AND AT 22.2% TOP-1 most restorations are wrong, so a fluent")
    print("  reading here is not evidence the tablet was read correctly.")
    print("  This run demonstrates the CHAIN, not accuracy.")
    print("\n" + RULE)


if __name__ == "__main__":
    main()