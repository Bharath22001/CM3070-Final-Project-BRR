"""
Simulates Domain 1's measured behaviour as Detection objects, so I could
evaluate fusion before the live detector was wired in. Every result produced
through it is simulated and has to be labelled that way. The live evaluation
is in evaluate_live.py; this stays as the calibrated counterfactual behind
artifacts/fusion_results.json.

Calibrated from the trained checkpoint: recall 0.243, precision 0.457, the 82
emit classes, and AP50 by training frequency. Recall counts boxes placed but
unnamed too, so it governs detected + unknown.

Assumed rather than measured: confidences come from Beta(5,2) when correct and
Beta(2,3) when wrong, wrong labels are drawn weighted by frequency, and there
are no spurious detections. AP50 is only used as a relative weight per class,
rescaled so the totals match the measured recall and precision.

Every draw is seeded from (seed, block_id, position, sign), so a rerun
reproduces a block exactly and different blocks stay independent.

Run the self-test from the project root:
    python -m cuneiform.orchestrator.simulate_vision
"""

from __future__ import annotations

import random

from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.domain2_sequence.build_vocab import count_vision_labels
from cuneiform.orchestrator.fusion import Detection

# Measured on the trained checkpoint
RECALL = 0.243
PRECISION = 0.457

# AP50 by number of training examples for the class.
AP50_STRATA = [
    (0, 0, 0.000),
    (1, 4, 0.140),
    (5, 19, 0.205),
    (20, 49, 0.305),
    (50, 199, 0.476),
    (200, 10**9, 0.568),
]

SEED = 1729


def stratum_ap50(n_train_boxes: int) -> float:
    for lo, hi, ap in AP50_STRATA:
        if lo <= n_train_boxes <= hi:
            return ap
    return 0.0


class VisionSimulator:
    """
    Turns a sequence of true sign tokens into per-position Detection objects.
    Each position is correct, wrong (the case a threshold design can't
    represent), unknown (a box outside the 82 emit classes) or missed. Unknown
    and missed both become <mask> but are counted separately.
    """

    def __init__(self, tok: SignTokenizer, seed: int = SEED, verbose: bool = True):
        self.tok = tok
        self.seed = seed

        if verbose:
            print("Reading Domain 1 training label counts")
            print("-" * 68)
        self.train_boxes = count_vision_labels("train")

        self.emit = set(tok.vision_emit_classes)
        self.ap50 = {
            label: stratum_ap50(self.train_boxes.get(label, 0))
            for label in self.emit
        }

        # Frequency weights for drawing a wrong label.
        self.wrong_pool = sorted(self.emit)
        self.wrong_weights = [
            float(self.train_boxes.get(l, 0)) + 1.0 for l in self.wrong_pool
        ]

        self._scale = 0.0

    def _calibrate(self, labels: list[int]) -> None:
        """
        Pick k so that k * AP50, averaged over the positions being simulated,
        equals RECALL * PRECISION (found and correctly named). Positions outside
        the emit set count in the denominator.
        """
        target = RECALL * PRECISION
        aps = [self.ap50.get(l, 0.0) for l in labels if l in self.emit]
        if not aps or not labels:
            self._scale = 0.0
            return
        mean_ap = sum(aps) / len(labels)
        self._scale = 0.0 if mean_ap <= 0 else target / mean_ap

    def _correct_prob(self, label: int) -> float:
        return min(RECALL, self._scale * self.ap50.get(label, 0.0))

    def simulate(self, ids: list[int], block_id) -> dict[int, Detection]:
        """
        Simulate one block. ids includes <s> and </s>; block_id must be unique
        per block. Returns {position: Detection} for every real, recoverable
        sign (specials, <brk> and <unk> are skipped).
        """
        tok = self.tok

        targets = [
            i for i, t in enumerate(ids)
            if tok.is_predictable(t) and t != tok.unk_id
        ]
        labels = [tok.id_to_sign[ids[i]] for i in targets]
        self._calibrate(labels)

        out: dict[int, Detection] = {}

        for pos, label in zip(targets, labels):
            # block_id in the seed keeps blocks independent.
            rng = random.Random(f"{self.seed}:{block_id}:{pos}:{label}")

            if label not in self.emit:
                # Outside the detector's vocabulary: an unnamed box or a miss, <mask> either way.
                kind = "unknown" if rng.random() < RECALL else "missed"
                out[pos] = Detection(kind)
                continue

            p_correct = self._correct_prob(label)
            p_wrong = max(0.0, RECALL - p_correct)

            roll = rng.random()
            if roll < p_correct:
                conf = rng.betavariate(5, 2)
                out[pos] = Detection("detected", train_label=label,
                                     confidence=conf, scores={label: conf})
            elif roll < p_correct + p_wrong:
                wrong = rng.choices(self.wrong_pool,
                                    weights=self.wrong_weights, k=1)[0]
                if wrong == label:
                    wrong = rng.choice([l for l in self.wrong_pool if l != label])
                conf = rng.betavariate(2, 3)
                out[pos] = Detection("detected", train_label=wrong,
                                     confidence=conf, scores={wrong: conf})
            else:
                out[pos] = Detection("missed")

        return out


def audit(sim: VisionSimulator, tok: SignTokenizer, split: str = "test",
          n_blocks: int = 200) -> dict:
    """
    Check the simulator against its own specification, as a separate function.
    Recall and precision are both checked over localised boxes (detected + unknown).
    """
    from cuneiform.domain2_sequence.dataset import build_blocks, load_corpus

    records = load_corpus(split)
    blocks = build_blocks(records, tok)[:n_blocks]

    n_pos = n_named = n_correct = n_unknown = n_missed = n_outside = 0
    by_stratum: dict[str, list[int]] = {}

    for block_id, body in enumerate(blocks):
        ids = [tok.bos_id] + body + [tok.eos_id]
        dets = sim.simulate(ids, block_id)
        for pos, det in dets.items():
            truth = tok.id_to_sign[ids[pos]]
            n_pos += 1
            if truth not in sim.emit:
                n_outside += 1

            if det.kind == "unknown":
                n_unknown += 1
            elif det.kind == "missed":
                n_missed += 1
            else:
                n_named += 1
                hit = det.train_label == truth
                n_correct += hit
                if truth in sim.emit:
                    key = f"AP50={sim.ap50[truth]:.3f}"
                    b = by_stratum.setdefault(key, [0, 0])
                    b[0] += hit
                    b[1] += 1

    n_boxes = n_named + n_unknown
    pct = lambda a, b: 100 * a / max(1, b)

    print("\n" + "=" * 68)
    print(f"SIMULATOR AUDIT  ({split} split, {len(blocks)} blocks)")
    print("=" * 68)
    print(f"  positions simulated         : {n_pos:,}")
    print(f"  outside the 82 emit classes : {n_outside:,} "
          f"({pct(n_outside, n_pos):.1f}%)   [build_vocab predicts ~18.5%]")
    print()
    print("  Against the measured detector figures:")
    print(f"    localised boxes           : {n_boxes:,} "
          f"({pct(n_boxes, n_pos):.1f}%)   [RECALL target "
          f"{100 * RECALL:.1f}%]")
    print(f"    of those, correctly named : {n_correct:,} "
          f"({pct(n_correct, n_boxes):.1f}%)   [PRECISION target "
          f"{100 * PRECISION:.1f}%]")
    print()
    print("  Outcome breakdown:")
    print(f"    correct detection         : {n_correct:,} "
          f"({pct(n_correct, n_pos):.1f}%)")
    print(f"    WRONG but named           : {n_named - n_correct:,} "
          f"({pct(n_named - n_correct, n_pos):.1f}%)"
          f"   <- invisible to a threshold")
    print(f"    unknown (box, no name)    : {n_unknown:,} "
          f"({pct(n_unknown, n_pos):.1f}%)")
    print(f"    missed entirely           : {n_missed:,} "
          f"({pct(n_missed, n_pos):.1f}%)")
    print()
    print(f"  reaching Domain 2 as <mask> : "
          f"{pct(n_unknown + n_missed, n_pos):.1f}%   "
          f"(vocabulary ceiling alone would give ~18.5%)")

    print("\n  accuracy by AP50 stratum (should rise with AP50):")
    for key in sorted(by_stratum):
        hit, tot = by_stratum[key]
        print(f"    {key} : {pct(hit, tot):>5.1f}%  ({tot:,} positions)")
    print("=" * 68)

    return {"n_pos": n_pos,
            "recall": n_boxes / max(1, n_pos),
            "precision": n_correct / max(1, n_boxes)}


def main() -> None:
    tok = SignTokenizer()
    print(f"\n  {tok}\n")
    sim = VisionSimulator(tok)
    print(f"\n  emit classes: {len(sim.emit)} | AP50 range "
          f"{min(sim.ap50.values()):.3f}-{max(sim.ap50.values()):.3f} "
          f"(low strata unreachable: the emit cut is the same >= 20 boxes)")
    audit(sim, tok, "test")


if __name__ == "__main__":
    main()