"""
Tests for live_adapter, which turns saved detections into fusion.Detection
objects. The geometry functions are tested on synthetic detections whose
grouping can be worked out by hand.

The adapter's own defaults (conf 0.25, cell 3.0, min size 5) deliberately
differ from the production pipeline's (0.15, 2.0, 3), and a test asserts that
they differ. Nothing here writes into the live detections folder.

Run from the project root:
    python -m pytest tests/test_live_adapter.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.orchestrator import live_adapter as LA
from cuneiform.orchestrator import live_pipeline as LP
from cuneiform.orchestrator.insert_gaps import DEFAULT_PITCH_RATIO, Slot
from cuneiform.orchestrator.live_adapter import (
    build_slots, grid_components, load_detections, load_sign_names, segment,
    to_detections,
)


@pytest.fixture(scope="module")
def tok() -> SignTokenizer:
    return SignTokenizer()


def _det(cx, cy, size=100.0, conf=0.9, label=1):
    """A detection record shaped exactly as run_yolo_inference.py saves it."""
    return {"cx": float(cx), "cy": float(cy),
            "x1": float(cx - size / 2), "x2": float(cx + size / 2),
            "y1": float(cy - size / 2), "y2": float(cy + size / 2),
            "conf": float(conf), "train_label": int(label)}


# The deliberate constant difference

def test_the_adapter_constants_deliberately_differ_from_the_frozen_ones():
    """
    The adapter keeps its own values for standalone runs, and live_pipeline
    imports none of them. If this fails, one of the two files has changed.
    """
    assert (LA.DEFAULT_CONF, LA.SEGMENT_CELL_FACTOR, LA.SEGMENT_MIN_SIZE) == \
        (0.25, 3.0, 5)
    assert (LP.LIVE_CONF, LP.LIVE_CELL, LP.LIVE_MIN_SIZE) == (0.15, 2.0, 3)
    assert LA.DEFAULT_CONF != LP.LIVE_CONF
    assert LA.SEGMENT_CELL_FACTOR != LP.LIVE_CELL
    assert LA.SEGMENT_MIN_SIZE != LP.LIVE_MIN_SIZE


def test_live_pipeline_does_not_import_the_adapters_constants():
    """live_pipeline imports functions from the adapter but none of its constants."""
    import inspect

    src = inspect.getsource(LP)
    imported = src.split("from cuneiform.orchestrator.live_adapter import")[1]
    imported = imported.split(")")[0]
    assert "SEGMENT_CELL_FACTOR" not in imported
    assert "SEGMENT_MIN_SIZE" not in imported
    assert "DEFAULT_CONF" not in imported


# grid_components

def test_no_points_gives_no_labels():
    out = grid_components(np.array([]), np.array([]), 10.0)
    assert len(out) == 0


def test_two_far_apart_groups_become_two_components():
    cx = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    cy = np.zeros(6)
    labels = grid_components(cx, cy, 5.0)
    assert len(set(labels.tolist())) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_a_large_enough_cell_merges_everything():
    cx = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    labels = grid_components(cx, np.zeros(6), 200.0)
    assert len(set(labels.tolist())) == 1


def test_connectivity_is_eight_way_so_diagonal_neighbours_join():
    """Diagonal neighbours join, since the grid is 8-connected."""
    labels = grid_components(np.array([0.0, 10.0]), np.array([0.0, 10.0]), 10.0)
    assert labels[0] == labels[1]


def test_every_point_receives_a_label():
    rng = np.random.default_rng(0)
    cx, cy = rng.uniform(0, 1000, 50), rng.uniform(0, 1000, 50)
    labels = grid_components(cx, cy, 30.0)
    assert len(labels) == 50
    assert (labels >= 0).all()


def test_component_labels_are_invariant_to_translation():
    cx = np.array([0.0, 5.0, 300.0, 305.0])
    cy = np.zeros(4)
    a = grid_components(cx, cy, 20.0)
    b = grid_components(cx + 7.0, cy + 7.0, 20.0)
    # The grouping must match even if the labels are renumbered.
    assert [a[0] == a[i] for i in range(4)] == [b[0] == b[i] for i in range(4)]


# segment

def test_no_detections_gives_no_segments():
    assert segment([]) == []


def test_two_surfaces_separated_in_x_are_split():
    """
    Two surfaces that overlap in y but not in x are split, as on P334932, which
    holds two separate objects side by side.
    """
    left = [_det(x, 500.0) for x in np.arange(6) * 120.0]
    right = [_det(5000.0 + x, 500.0) for x in np.arange(6) * 120.0]
    groups = segment(left + right, cell_factor=3.0, min_size=5)
    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == [6, 6]


def test_clusters_below_min_size_are_dropped_not_merged():
    """A cluster below min_size is dropped as debris, not merged into another."""
    main = [_det(x, 500.0) for x in np.arange(8) * 120.0]
    debris = [_det(9000.0 + x, 500.0) for x in (0.0, 5.0, 10.0)]
    groups = segment(main + debris, cell_factor=3.0, min_size=5)
    assert len(groups) == 1
    assert len(groups[0]) == 8
    assert sum(len(g) for g in groups) == 8      # the 3 are gone, not absorbed


def test_segments_are_ordered_top_to_bottom_then_left_to_right():
    """Segments go top to bottom, then left to right. This is a convention; geometry can't tell obverse from reverse."""
    lower = [_det(x, 5000.0) for x in np.arange(6) * 120.0]
    upper_right = [_det(9000.0 + x, 100.0) for x in np.arange(6) * 120.0]
    upper_left = [_det(x, 100.0) for x in np.arange(6) * 120.0]

    groups = segment(upper_right + lower + upper_left,
                     cell_factor=3.0, min_size=5)
    dets = upper_right + lower + upper_left
    cy_min = [min(dets[i]["cy"] for i in g) for g in groups]
    cx_min = [min(dets[i]["cx"] for i in g) for g in groups]
    assert cy_min == sorted(cy_min)
    # The two top segments tie on y, so the left one comes first.
    assert cx_min[0] < cx_min[1]


def test_the_indices_returned_point_back_into_the_input_list():
    dets = [_det(x, 500.0) for x in np.arange(8) * 120.0]
    groups = segment(dets, cell_factor=3.0, min_size=5)
    flat = sorted(int(i) for g in groups for i in g)
    assert flat == list(range(8))


def test_the_cell_is_scaled_by_the_median_box_width():
    """The same layout at 10x segments identically."""
    small = [_det(x, 500.0, size=100.0) for x in np.arange(6) * 120.0] + \
        [_det(5000.0 + x, 500.0, size=100.0) for x in np.arange(6) * 120.0]
    big = [_det(x * 10, 5000.0, size=1000.0) for x in np.arange(6) * 120.0] + \
        [_det(50000.0 + x * 10, 5000.0, size=1000.0) for x in np.arange(6) * 120.0]
    assert len(segment(small)) == len(segment(big)) == 2


def test_live_pipelines_own_segment_agrees_with_the_adapters_on_the_same_arguments():
    """live_pipeline.segment and the adapter's segment agree when given the same parameters."""
    dets = [_det(x, 500.0) for x in np.arange(10) * 120.0] + \
        [_det(6000.0 + x, 3000.0) for x in np.arange(10) * 120.0]
    for cell, min_size in ((2.0, 3), (3.0, 5), (1.0, 2)):
        a = segment(dets, cell, min_size)
        b = LP.segment(dets, cell, min_size)
        assert [sorted(g.tolist()) for g in a] == [sorted(g.tolist()) for g in b]


# build_slots

def test_build_slots_carries_the_segment_index_alongside_each_slot():
    """
    Each slot carries its segment index, since every segment numbers its lines
    from 0 and they would otherwise merge.
    """
    dets = [_det(x, 500.0) for x in np.arange(8) * 120.0] + \
        [_det(x, 9000.0) for x in np.arange(8) * 120.0]
    groups = segment(dets, cell_factor=2.0, min_size=3)
    assert len(groups) == 2

    slots = build_slots(dets, groups)
    seg_ids = {seg for seg, _ in slots}
    assert seg_ids == {0, 1}
    # Both segments do produce a line 0, which is why the index is needed.
    assert all(any(s.line == 0 for sg, s in slots if sg == g) for g in (0, 1))


def test_build_slots_returns_pairs_of_index_and_slot():
    dets = [_det(x, 500.0) for x in np.arange(8) * 120.0]
    slots = build_slots(dets, segment(dets, 2.0, 3))
    assert slots
    for seg_i, s in slots:
        assert isinstance(seg_i, int)
        assert isinstance(s, Slot)


def test_build_slots_on_no_groups_gives_no_slots():
    assert build_slots([], []) == []


def test_build_slots_passes_the_edge_flag_through():
    """
    The edge flag is passed through. The rows get a few pixels of jitter, like
    real detections, so the grouper keeps them as separate lines.
    """
    rng = np.random.default_rng(21)
    rows = [(0.0, 12), (4 * 120.0, 6), (0.0, 10), (2 * 120.0, 8)]
    dets = []
    for r, (off_x, ncol) in enumerate(rows):
        for i in range(ncol):
            dets.append(_det(off_x + i * 120.0 + rng.normal(0, 3),
                             300.0 + r * 160.0 + rng.normal(0, 3)))
    groups = segment(dets, 2.0, 3)
    off = build_slots(dets, groups, extend_edges=False)
    on = build_slots(dets, groups, extend_edges=True)
    assert len(on) > len(off)
    assert sum(1 for _, s in on if s.kind == "detected") ==         sum(1 for _, s in off if s.kind == "detected")


def test_build_slots_uses_the_pitch_ratio_it_is_given():
    dets = [_det(0.0, 500.0), _det(120.0, 500.0), _det(1200.0, 500.0),
            _det(1320.0, 500.0), _det(1440.0, 500.0)]
    groups = segment(dets, 20.0, 3)
    wide = build_slots(dets, groups, pitch_ratio=DEFAULT_PITCH_RATIO * 4)
    narrow = build_slots(dets, groups, pitch_ratio=DEFAULT_PITCH_RATIO)
    n_wide = sum(1 for _, s in wide if s.kind == "inserted")
    n_narrow = sum(1 for _, s in narrow if s.kind == "inserted")
    assert n_narrow > n_wide


# to_detections: the three kinds

def test_an_inserted_slot_becomes_a_missed_detection(tok):
    out = to_detections([(0, Slot("inserted", 0, 1.0))], tok)
    assert [d.kind for d in out] == ["missed"]


def test_a_box_inside_the_emit_classes_becomes_a_detected_detection(tok):
    label = sorted(tok.vision_emit_classes)[0]
    out = to_detections([(0, Slot("detected", 0, 1.0, label, 0.77))], tok)
    assert out[0].kind == "detected"
    assert out[0].train_label == label
    assert out[0].confidence == pytest.approx(0.77)


def test_a_box_outside_the_emit_classes_becomes_unknown_not_missed(tok):
    """
    A box outside the emit classes is "unknown", not "missed": the detector did
    find something there, even though both reach Domain 2 as <mask>.
    """
    outside = next(s for s in sorted(tok.sign_to_id)
                   if s not in tok.vision_emit_classes)
    out = to_detections([(0, Slot("detected", 0, 1.0, outside, 0.9))], tok)
    assert out[0].kind == "unknown"
    assert out[0].train_label is None


def test_a_detected_slot_with_no_label_becomes_missed(tok):
    out = to_detections([(0, Slot("detected", 0, 1.0, None, 0.9))], tok)
    assert out[0].kind == "missed"


def test_one_detection_is_produced_per_slot_in_order(tok):
    emit = sorted(tok.vision_emit_classes)[0]
    outside = next(s for s in sorted(tok.sign_to_id)
                   if s not in tok.vision_emit_classes)
    slots = [(0, Slot("detected", 0, 0.0, emit, 0.5)),
             (0, Slot("inserted", 0, 1.0)),
             (1, Slot("detected", 0, 2.0, outside, 0.5))]
    out = to_detections(slots, tok)
    assert [d.kind for d in out] == ["detected", "missed", "unknown"]


def test_only_a_detected_kind_is_informative_to_fusion(tok):
    emit = sorted(tok.vision_emit_classes)[0]
    outside = next(s for s in sorted(tok.sign_to_id)
                   if s not in tok.vision_emit_classes)
    slots = [(0, Slot("detected", 0, 0.0, emit, 0.5)),
             (0, Slot("inserted", 0, 1.0)),
             (0, Slot("detected", 0, 2.0, outside, 0.5))]
    assert [d.is_informative for d in to_detections(slots, tok)] == \
        [True, False, False]


def test_live_pipelines_own_to_detections_agrees_with_the_adapters(tok):
    """The same agreement check for the emit filter."""
    emit = sorted(tok.vision_emit_classes)[:3]
    outside = [s for s in sorted(tok.sign_to_id)
               if s not in tok.vision_emit_classes][:3]
    slots = [(0, Slot("detected", 0, float(i), lab, 0.5))
             for i, lab in enumerate(emit + outside)]
    slots.append((1, Slot("inserted", 0, 99.0)))

    a = to_detections(slots, tok)
    b = LP.to_detections(slots, tok)
    assert [(d.kind, d.train_label, d.confidence) for d in a] == \
        [(d.kind, d.train_label, d.confidence) for d in b]


# Loading, read-only against the real artifacts

def test_load_detections_rejects_a_tablet_that_does_not_exist():
    with pytest.raises(FileNotFoundError):
        load_detections("P000000_not_a_tablet", 0.15)


@pytest.mark.slow
def test_load_detections_applies_the_threshold_and_never_writes(tmp_path):
    """Applies the threshold and leaves the detections folder exactly as it was."""
    before = sorted(p.name for p in config.D1_LIVE_DETECTIONS.iterdir())
    tablets = [p.stem for p in config.D1_LIVE_DETECTIONS.glob("P*.json")]
    assert tablets, "the live detection population is missing"
    tablet = sorted(tablets)[0]

    low = load_detections(tablet, 0.0)
    frozen = load_detections(tablet, LP.LIVE_CONF)
    high = load_detections(tablet, 0.9)

    assert len(low) >= len(frozen) >= len(high)
    assert all(d["conf"] >= LP.LIVE_CONF for d in frozen)
    assert all(d["conf"] >= 0.9 for d in high)
    assert len(frozen) < len(low), "the frozen threshold discards nothing here"

    assert sorted(p.name for p in config.D1_LIVE_DETECTIONS.iterdir()) == before


@pytest.mark.slow
def test_every_persisted_detection_carries_the_fields_the_adapter_reads():
    """Every saved detection has the fields segment and build_slots read by name."""
    needed = {"cx", "cy", "x1", "x2", "y1", "y2", "conf", "train_label"}
    n = 0
    for path in sorted(config.D1_LIVE_DETECTIONS.glob("P*.json")):
        with path.open("r", encoding="utf-8") as fh:
            rec = json.load(fh)
        assert "detections" in rec, path.name
        for d in rec["detections"]:
            assert needed <= set(d), f"{path.name}: {needed - set(d)}"
        n += 1
    assert n == 30, f"expected the 30-tablet live population, found {n}"


def test_load_sign_names_returns_integer_keys_and_never_raises():
    """load_sign_names returns integer keys and returns {} instead of raising."""
    names = load_sign_names()
    assert isinstance(names, dict)
    assert all(isinstance(k, int) for k in names)
    assert all(isinstance(v, str) and v for v in names.values())


def test_load_sign_names_falls_back_rather_than_omitting_an_unnamed_sign():
    """Unnamed signs still appear, as MZL{n}, instead of vanishing from the dump."""
    names = load_sign_names()
    if not names:
        pytest.skip("sign_lookup.json is not on disk")
    assert not any(v.startswith("None") for v in names.values())
