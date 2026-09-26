"""
Tests for the remaining modules that only have a few pure functions each:
evaluate_fusion (vision_context, synthetic_damage, context_density,
load_sign_names), evaluate_live (live_tablets, _slot_sources), simulate_vision
(stratum_ap50, VisionSimulator), baselines (build_ngram_tables),
evaluate_domain3 (norm_sign, gate_can_see, test_tablets, prose_of) and
identify_checkpoint (describe).

The simulator is only run in memory and never regenerates its artifact.
prepare_cuneiform_data.py, inspect_box_pitch.py and inspect_segment_choice.py
are deliberately untested: the first is abandoned code and the other two are
kept as records of approaches that failed.

Run from the project root:
    python -m pytest tests/test_orchestration_misc.py -q
"""

from __future__ import annotations

import random

import pytest
import torch

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.domain3_generative.evaluate_domain3 import (
    gate_can_see, norm_sign, prose_of,
)
from cuneiform.orchestrator.evaluate_fusion import (
    context_density, load_sign_names, synthetic_damage, vision_context,
)
from cuneiform.orchestrator.fusion import Detection
from cuneiform.orchestrator.simulate_vision import AP50_STRATA, stratum_ap50


@pytest.fixture(scope="module")
def tok() -> SignTokenizer:
    return SignTokenizer()


# vision_context: arm 2 vs arm 3

def test_a_detection_at_or_above_the_threshold_becomes_context(tok):
    label = sorted(tok.vision_emit_classes)[0]
    ids = torch.tensor([tok.bos_id, tok.mask_id, tok.eos_id])
    dets = {1: Detection("detected", train_label=label, confidence=0.6)}
    out = vision_context(ids, dets, tok, threshold=0.5)
    assert out[1].item() == tok.sign_to_id[label]


def test_a_detection_below_the_threshold_becomes_mask(tok):
    label = sorted(tok.vision_emit_classes)[0]
    ids = torch.tensor([tok.bos_id, tok.mask_id, tok.eos_id])
    dets = {1: Detection("detected", train_label=label, confidence=0.4)}
    assert vision_context(ids, dets, tok, threshold=0.5)[1].item() == tok.mask_id


def test_the_threshold_boundary_is_inclusive(tok):
    label = sorted(tok.vision_emit_classes)[0]
    ids = torch.tensor([tok.bos_id, tok.mask_id, tok.eos_id])
    dets = {1: Detection("detected", train_label=label, confidence=0.5)}
    assert vision_context(ids, dets, tok, 0.5)[1].item() == tok.sign_to_id[label]


def test_a_zero_threshold_writes_in_every_named_detection_including_wrong_ones(tok):
    """At threshold 0 every named detection becomes context, wrong ones included (arm 3)."""
    a, b = sorted(tok.vision_emit_classes)[:2]
    ids = torch.tensor([tok.bos_id, tok.mask_id, tok.mask_id, tok.eos_id])
    dets = {1: Detection("detected", train_label=a, confidence=0.01),
            2: Detection("detected", train_label=b, confidence=0.99)}
    out = vision_context(ids, dets, tok, threshold=0.0)
    assert out[1].item() == tok.sign_to_id[a]
    assert out[2].item() == tok.sign_to_id[b]


def test_an_uninformative_position_is_masked_at_any_threshold(tok):
    ids = torch.tensor([tok.bos_id, tok.mask_id, tok.mask_id, tok.eos_id])
    dets = {1: Detection("unknown"), 2: Detection("missed")}
    for thr in (0.0, 0.5, 1.0):
        out = vision_context(ids, dets, tok, thr)
        assert out[1].item() == out[2].item() == tok.mask_id


def test_a_label_outside_the_vocabulary_falls_back_to_mask(tok):
    bogus = max(tok.sign_to_id) + 10_000
    ids = torch.tensor([tok.bos_id, tok.mask_id, tok.eos_id])
    dets = {1: Detection("detected", train_label=bogus, confidence=0.99)}
    assert vision_context(ids, dets, tok, 0.0)[1].item() == tok.mask_id


def test_vision_context_does_not_mutate_the_input_tensor(tok):
    """vision_context doesn't change its input, since several arms share the same block."""
    label = sorted(tok.vision_emit_classes)[0]
    ids = torch.tensor([tok.bos_id, tok.mask_id, tok.eos_id])
    before = ids.clone()
    vision_context(ids, {1: Detection("detected", train_label=label,
                                      confidence=0.9)}, tok, 0.5)
    assert torch.equal(ids, before)


def test_positions_with_no_detection_are_left_alone(tok):
    label = sorted(tok.vision_emit_classes)[0]
    tid = tok.sign_to_id[label]
    ids = torch.tensor([tok.bos_id, tid, tok.mask_id, tok.eos_id])
    out = vision_context(ids, {2: Detection("missed")}, tok, 0.5)
    assert out[1].item() == tid
    assert out[0].item() == tok.bos_id
    assert out[3].item() == tok.eos_id


# synthetic_damage / context_density

def test_synthetic_damage_hides_the_requested_fraction(tok):
    ids = torch.full((20,), 7, dtype=torch.long)
    positions = list(range(20))
    out = synthetic_damage(ids, positions, 0.5, tok, random.Random(0))
    assert int((out == tok.mask_id).sum()) == 10


def test_a_damage_rate_of_zero_changes_nothing(tok):
    ids = torch.full((10,), 7, dtype=torch.long)
    out = synthetic_damage(ids, list(range(10)), 0.0, tok, random.Random(0))
    assert torch.equal(out, ids)


def test_synthetic_damage_only_touches_the_positions_it_is_given(tok):
    ids = torch.full((10,), 7, dtype=torch.long)
    out = synthetic_damage(ids, [0, 1, 2, 3], 1.0, tok, random.Random(0))
    assert (out[:4] == tok.mask_id).all()
    assert (out[4:] == 7).all()


def test_synthetic_damage_is_reproducible_from_its_rng(tok):
    ids = torch.full((30,), 7, dtype=torch.long)
    pos = list(range(30))
    a = synthetic_damage(ids, pos, 0.4, tok, random.Random(99))
    b = synthetic_damage(ids, pos, 0.4, tok, random.Random(99))
    assert torch.equal(a, b)


def test_synthetic_damage_does_not_mutate_its_input(tok):
    ids = torch.full((10,), 7, dtype=torch.long)
    before = ids.clone()
    synthetic_damage(ids, list(range(10)), 1.0, tok, random.Random(0))
    assert torch.equal(ids, before)


def test_context_density_is_the_fraction_carrying_a_real_token(tok):
    ids = torch.tensor([7, tok.mask_id, 8, tok.mask_id])
    assert context_density(ids, [0, 1, 2, 3], tok) == pytest.approx(0.5)
    assert context_density(ids, [0, 2], tok) == pytest.approx(1.0)
    assert context_density(ids, [1, 3], tok) == pytest.approx(0.0)


def test_context_density_of_no_positions_is_zero_rather_than_a_division_error(tok):
    assert context_density(torch.tensor([7]), [], tok) == 0.0


def test_the_two_arms_differ_in_context_density(tok):
    """The threshold and fusion contexts differ in how much is masked."""
    labels = sorted(tok.vision_emit_classes)[:4]
    ids = torch.full((6,), tok.mask_id, dtype=torch.long)
    dets = {i + 1: Detection("detected", train_label=l, confidence=c)
            for i, (l, c) in enumerate(zip(labels, (0.1, 0.2, 0.8, 0.9)))}
    positions = list(dets)

    thr = vision_context(ids, dets, tok, 0.5)
    full = vision_context(ids, dets, tok, 0.0)
    assert context_density(thr, positions, tok) == pytest.approx(0.5)
    assert context_density(full, positions, tok) == pytest.approx(1.0)


# load_sign_names

def test_sign_names_are_keyed_by_train_label_and_never_empty():
    names = load_sign_names()
    assert names
    assert all(isinstance(k, int) for k in names)
    assert all(isinstance(v, str) and v for v in names.values())


def test_the_unnamed_signs_fall_back_to_mzl_so_cer_stays_defined():
    """Unnamed signs fall back to MZL#### so CER never sees an empty name."""
    names = load_sign_names()
    assert not any(v.strip() == "" for v in names.values())
    assert any(v.startswith("MZL") for v in names.values())


def test_every_domain_2_vocabulary_sign_has_a_name(tok):
    names = load_sign_names()
    missing = [s for s in tok.sign_to_id if s not in names]
    assert not missing, f"{len(missing)} vocabulary signs have no name"


# stratum_ap50

def test_ap50_rises_monotonically_with_training_examples():
    """AP50 rises with the number of training examples, as measured on the real detector."""
    aps = [ap for _, _, ap in AP50_STRATA]
    assert aps == sorted(aps)


def test_the_strata_tile_the_range_without_gaps_or_overlaps():
    for (_, hi_a, _), (lo_b, _, _) in zip(AP50_STRATA, AP50_STRATA[1:]):
        assert lo_b == hi_a + 1


@pytest.mark.parametrize("n", [1, 4, 5, 19, 20, 49, 50, 199, 200, 10_000])
def test_every_box_count_lands_in_a_stratum(n):
    assert stratum_ap50(n) > 0.0


def test_a_class_with_no_training_boxes_scores_zero():
    """A class with no training boxes scores 0: it can't be found at all."""
    assert stratum_ap50(0) == 0.0


def test_the_bounds_are_inclusive_at_both_ends():
    lo, hi, ap = AP50_STRATA[1]
    assert stratum_ap50(lo) == ap
    assert stratum_ap50(hi) == ap


# VisionSimulator

def _block(tok, labels):
    """One block as simulate expects it: token ids, including <s> and </s>."""
    return [tok.bos_id] + [tok.sign_to_id[l] for l in labels] + [tok.eos_id]


@pytest.mark.slow
def test_the_simulator_is_reproducible_from_its_seed(tok):
    """
    The simulator reproduces from its seed. Each draw is seeded per (seed,
    block_id, position, label), so the result doesn't depend on block order.
    """
    from cuneiform.orchestrator.simulate_vision import VisionSimulator

    ids = _block(tok, sorted(tok.vision_emit_classes) * 3)
    a = VisionSimulator(tok, seed=1729, verbose=False).simulate(ids, "blk")
    b = VisionSimulator(tok, seed=1729, verbose=False).simulate(ids, "blk")
    assert {p: (d.kind, d.train_label, d.confidence) for p, d in a.items()} == \
        {p: (d.kind, d.train_label, d.confidence) for p, d in b.items()}


@pytest.mark.slow
def test_a_different_block_id_gives_a_different_draw(tok):
    """A different block id gives a different draw, so blocks don't repeat each other."""
    from cuneiform.orchestrator.simulate_vision import VisionSimulator

    ids = _block(tok, sorted(tok.vision_emit_classes) * 3)
    sim = VisionSimulator(tok, seed=1729, verbose=False)
    a = sim.simulate(ids, "block_one")
    b = sim.simulate(ids, "block_two")
    assert [a[p].kind for p in sorted(a)] != [b[p].kind for p in sorted(b)]


@pytest.mark.slow
def test_a_different_seed_gives_a_different_draw(tok):
    from cuneiform.orchestrator.simulate_vision import VisionSimulator

    ids = _block(tok, sorted(tok.vision_emit_classes) * 3)
    a = VisionSimulator(tok, seed=1, verbose=False).simulate(ids, "blk")
    b = VisionSimulator(tok, seed=2, verbose=False).simulate(ids, "blk")
    assert [a[p].kind for p in sorted(a)] != [b[p].kind for p in sorted(b)]


@pytest.mark.slow
def test_the_simulator_emits_only_the_documented_outcomes(tok):
    from cuneiform.orchestrator.simulate_vision import VisionSimulator

    labels = sorted(tok.vision_emit_classes) * 3
    ids = _block(tok, labels)
    dets = VisionSimulator(tok, seed=7, verbose=False).simulate(ids, "blk")

    assert len(dets) == len(labels)
    assert {d.kind for d in dets.values()} <= {"detected", "unknown", "missed"}
    for d in dets.values():
        if d.kind == "detected":
            assert d.train_label in tok.vision_emit_classes
            assert 0.0 <= d.confidence <= 1.0
            assert d.scores == {d.train_label: d.confidence}
        else:
            assert d.train_label is None


@pytest.mark.slow
def test_the_sentinels_are_never_given_a_detection(tok):
    """<s>, </s>, <brk> and <unk> never get a Detection."""
    from cuneiform.orchestrator.simulate_vision import VisionSimulator

    labels = sorted(tok.vision_emit_classes)[:6]
    ids = ([tok.bos_id] + [tok.sign_to_id[l] for l in labels]
           + [tok.brk_id, tok.unk_id, tok.eos_id])
    dets = VisionSimulator(tok, seed=11, verbose=False).simulate(ids, "blk")
    assert sorted(dets) == list(range(1, len(labels) + 1))


@pytest.mark.slow
def test_a_sign_outside_the_emit_classes_can_never_be_detected(tok):
    """
    A sign outside the emit classes is never detected, only "unknown" or
    "missed", so the simulator is no better than the real detector.
    """
    from cuneiform.orchestrator.simulate_vision import VisionSimulator

    outside = [s for s in sorted(tok.sign_to_id)
               if s not in tok.vision_emit_classes][:40]
    ids = _block(tok, outside)
    dets = VisionSimulator(tok, seed=3, verbose=False).simulate(ids, "blk")
    kinds = {d.kind for d in dets.values()}
    assert "detected" not in kinds
    assert kinds <= {"unknown", "missed"}


@pytest.mark.slow
def test_a_confident_but_wrong_detection_is_produced_at_all(tok):
    """The simulator does produce confident but wrong detections."""
    from cuneiform.orchestrator.simulate_vision import VisionSimulator

    labels = sorted(tok.vision_emit_classes) * 6
    ids = _block(tok, labels)
    dets = VisionSimulator(tok, seed=5, verbose=False).simulate(ids, "blk")
    wrong = [p for i, p in enumerate(sorted(dets), start=0)
             if dets[p].kind == "detected" and dets[p].train_label != labels[i]]
    assert wrong, "the simulator produced no wrong detections at all"


# build_ngram_tables

@pytest.mark.slow
def test_the_ngram_baselines_are_built_on_the_train_split_only(tok):
    """The n-gram baselines never see a sign that only appears in the test split."""
    from cuneiform.domain2_sequence.baselines import build_ngram_tables
    from cuneiform.domain2_sequence.dataset import load_corpus

    unigram, bigram = build_ngram_tables(tok)
    assert unigram

    train_ids = set()
    for rec in load_corpus("train"):
        train_ids.update(tok.encode_line(rec["tokens"]))
    assert set(unigram) <= train_ids


@pytest.mark.slow
def test_broken_positions_are_excluded_from_the_ngram_statistics(tok):
    """<brk> is left out of the n-gram statistics."""
    from cuneiform.domain2_sequence.baselines import build_ngram_tables

    unigram, bigram = build_ngram_tables(tok)
    assert tok.brk_id not in unigram
    assert tok.brk_id not in bigram
    assert all(tok.brk_id not in nxt for nxt in bigram.values())


@pytest.mark.slow
def test_the_bigram_table_is_a_subset_of_the_unigram_vocabulary(tok):
    from cuneiform.domain2_sequence.baselines import build_ngram_tables

    unigram, bigram = build_ngram_tables(tok)
    assert set(bigram) <= set(unigram)
    for nxt in bigram.values():
        assert set(nxt) <= set(unigram)


# norm_sign / gate_can_see

def test_a_trailing_gloss_is_stripped():
    assert norm_sign("LUGAL (lugal)") == "LUGAL"


def test_the_numeral_notation_survives_because_no_space_precedes_its_bracket():
    """3(DIS) keeps its bracket, since no space comes before it."""
    assert norm_sign("3(DIS)") == "3(DIS)"
    assert norm_sign("3(DIS) (3(dis))") == "3(DIS)"


def test_pipe_wrappers_and_trailing_punctuation_are_dropped():
    assert norm_sign("|ME.U.U.U|") == "ME.U.U.U"
    assert norm_sign("LUGAL.") == "LUGAL"
    assert norm_sign("  LUGAL  ") == "LUGAL"


def test_nested_brackets_are_matched_by_depth_not_by_the_first_one_seen():
    assert norm_sign("A (b(c))") == "A"


def test_a_name_with_no_gloss_is_unchanged():
    assert norm_sign("LUGAL") == "LUGAL"


def test_gate_can_see_is_a_self_audit_of_the_prose_regex():
    """gate_can_see tells whether the prose regex would find a token at all."""
    assert gate_can_see("LUGAL")
    assert gate_can_see("|ME.U.U.U|")
    assert not gate_can_see("lugal")


def test_prose_of_joins_only_the_two_prose_fields():
    out = {"tentative_reading": "one", "uncertainty_notes": "two",
           "historical_context": "three"}
    got = prose_of(out)
    assert "one" in got and "two" in got
    assert "three" not in got


def test_prose_of_tolerates_a_missing_field():
    assert prose_of({}) == " "


# evaluate_live: the population (read-only)

@pytest.mark.slow
def test_live_tablets_is_the_thirty_tablet_population():
    """live_tablets() returns exactly the 30 evaluation tablets."""
    from cuneiform.orchestrator.evaluate_live import live_tablets

    tablets = live_tablets()
    assert len(tablets) == 30
    assert len(set(tablets)) == 30
    assert tablets == sorted(tablets)


@pytest.mark.slow
def test_no_upload_has_leaked_into_the_live_detection_population():
    """Every file in the live detections folder is one of the expected 30, so no upload has leaked in."""
    from cuneiform.orchestrator.evaluate_live import live_tablets

    on_disk = {p.stem for p in config.D1_LIVE_DETECTIONS.glob("*.json")
               if not p.stem.startswith("_")}
    assert on_disk == set(live_tablets())


def test_the_uploads_directory_is_not_the_live_detections_directory():
    assert config.UPLOADS_DIR.resolve() != config.D1_LIVE_DETECTIONS.resolve()
    assert config.D1_LIVE_DETECTIONS.resolve() not in config.UPLOADS_DIR.resolve().parents


# identify_checkpoint

def test_identify_checkpoint_reports_a_missing_file_rather_than_raising(capsys):
    """Pointed at a missing file, identify_checkpoint reports it instead of crashing."""
    from cuneiform.domain1_vision.identify_checkpoint import describe

    describe("D:/definitely/not/a/checkpoint.pt")
    out = capsys.readouterr().out
    assert out.strip()
