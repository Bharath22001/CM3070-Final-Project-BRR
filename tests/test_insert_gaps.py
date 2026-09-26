"""
Tests for insert_masks, which orders detections into lines and inserts a <mask>
wherever a sign should be but wasn't detected.

The geometry is synthetic so every expected count can be worked out by hand:
boxes of width 100 at the pitch ratio 0.92 give a pitch of exactly 92 px, and
a gap inserts round(gap / pitch) - 1 masks (capped at MAX_INSERT) once it is
wider than MIN_GAP_PITCHES pitches.

Run from the project root:
    python -m pytest tests/test_insert_gaps.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cuneiform.orchestrator.insert_gaps import (
    DEFAULT_PITCH_RATIO, MAX_INSERT, MIN_GAP_PITCHES, Slot,
    calibrate_pitch_ratio, insert_masks,
)

BOX = 100.0
PITCH = BOX * DEFAULT_PITCH_RATIO          # 92.0 px


def _row(xs, box: float = BOX, y: float = 500.0, first_label: int = 10):
    """One horizontal row of boxes at the given x centres."""
    xs = np.asarray(xs, float)
    n = len(xs)
    return dict(
        cx=xs, cy=np.full(n, y), w=np.full(n, box), h=np.full(n, box),
        labels=np.arange(first_label, first_label + n),
        confs=np.linspace(0.2, 0.9, n),
    )


def _run(d, **kw):
    return insert_masks(d["cx"], d["cy"], d["w"], d["h"],
                        d["labels"], d["confs"], DEFAULT_PITCH_RATIO, **kw)


def _kinds(slots):
    return [s.kind for s in slots]


# The constants the arithmetic depends on

def test_the_documented_constants_are_the_ones_in_force():
    """Every expected count here comes from these constants, so they are pinned."""
    assert DEFAULT_PITCH_RATIO == 0.92
    assert MIN_GAP_PITCHES == 1.5
    assert MAX_INSERT == 8


# Degenerate inputs

def test_no_detections_gives_no_slots():
    assert insert_masks([], [], [], [], [], [], DEFAULT_PITCH_RATIO) == []


@pytest.mark.parametrize("n", [1, 2])
def test_fewer_than_three_boxes_bypasses_the_line_grouper(n):
    """With one or two boxes the grouper is skipped: boxes come back sorted by x, with no insertion."""
    d = _row([1000.0, 0.0][:n])
    slots = _run(d)
    assert len(slots) == n
    assert all(s.kind == "detected" and s.line == 0 for s in slots)
    assert [s.x for s in slots] == sorted(s.x for s in slots)


def test_two_far_apart_boxes_are_not_gap_filled():
    slots = _run(_row([0.0, 5000.0]))
    assert _kinds(slots) == ["detected", "detected"]


# The insertion rule

def test_evenly_spaced_boxes_produce_no_insertions():
    """Ordinary spacing is not mistaken for a missing sign."""
    slots = _run(_row(np.arange(6) * PITCH))
    assert _kinds(slots) == ["detected"] * 6


def test_a_gap_just_under_the_threshold_inserts_nothing():
    gap = 1.4 * PITCH
    slots = _run(_row([0.0, PITCH, PITCH + gap, PITCH + gap + PITCH]))
    assert "inserted" not in _kinds(slots)


def test_a_gap_of_n_pitches_inserts_n_minus_one_masks():
    """A gap of 6 pitches means 5 missing signs."""
    for spans in (2, 3, 4, 6):
        xs = [0.0, PITCH, PITCH + spans * PITCH, PITCH + (spans + 1) * PITCH]
        slots = _run(_row(xs))
        assert _kinds(slots).count("inserted") == spans - 1, f"spans={spans}"


def test_insertion_is_capped_at_max_insert():
    """A 40-pitch gap still inserts only MAX_INSERT masks."""
    xs = [0.0, PITCH, PITCH + 40 * PITCH, PITCH + 41 * PITCH]
    assert _kinds(_run(_row(xs))).count("inserted") == MAX_INSERT


def test_inserted_slots_carry_no_label_and_no_confidence():
    """An inserted slot has no label and no confidence."""
    xs = [0.0, PITCH, PITCH + 5 * PITCH, PITCH + 6 * PITCH]
    for s in _run(_row(xs)):
        if s.kind == "inserted":
            assert s.label is None
            assert s.confidence == 0.0


def test_inserted_slots_sit_between_their_two_anchors_in_x():
    xs = [0.0, PITCH, PITCH + 6 * PITCH, PITCH + 7 * PITCH]
    slots = _run(_row(xs))
    assert [s.x for s in slots] == sorted(s.x for s in slots)
    for s in slots:
        if s.kind == "inserted":
            assert PITCH < s.x < PITCH + 6 * PITCH


def test_pitch_is_a_ratio_of_box_width_so_the_result_is_scale_invariant():
    """The same layout scaled by 10x gives the same sequence."""
    xs = np.array([0.0, 1.0, 6.0, 7.0])
    small = _run(_row(xs * PITCH, box=BOX))
    big = _run(_row(xs * PITCH * 10, box=BOX * 10, y=5000.0))
    assert _kinds(small) == _kinds(big)


# Ordering

def test_boxes_are_emitted_left_to_right_within_a_line():
    """Boxes come out left to right within a line."""
    shuffled = np.array([3, 0, 4, 1, 2], float) * PITCH
    slots = _run(_row(shuffled))
    assert [s.x for s in slots] == sorted(s.x for s in slots)


def test_lines_are_emitted_top_to_bottom():
    rng = np.random.default_rng(11)
    n_rows, n_cols = 4, 10
    cx = np.tile(np.arange(n_cols) * PITCH, n_rows) + rng.normal(0, 3, n_rows * n_cols)
    cy = np.repeat([900.0, 300.0, 600.0, 1200.0], n_cols) + rng.normal(0, 3, n_rows * n_cols)
    n = cx.size
    slots = insert_masks(cx, cy, np.full(n, BOX), np.full(n, BOX),
                         np.arange(n), np.full(n, 0.5), DEFAULT_PITCH_RATIO)
    # Each line's mean y must increase with the line index.
    by_line: dict[int, list[float]] = {}
    for s in slots:
        if s.kind == "detected":
            by_line.setdefault(s.line, []).append(cy[int(s.label)])
    means = [np.mean(by_line[l]) for l in sorted(by_line)]
    assert means == sorted(means)


def test_line_numbers_are_contiguous_from_zero():
    d = _row(np.arange(8) * PITCH)
    slots = _run(d)
    assert sorted({s.line for s in slots}) == [0]


# The pandas index trap

def test_labels_are_read_positionally_even_from_a_reindexed_series():
    """
    Labels are read by position even from a Series indexed from 1000, so a
    label-based lookup would raise instead of silently giving the wrong sign.
    """
    xs = np.arange(5) * PITCH
    idx = pd.RangeIndex(start=1000, stop=1005)
    labels = pd.Series([11, 22, 33, 44, 55], index=idx)
    confs = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)

    slots = insert_masks(xs, np.full(5, 500.0), np.full(5, BOX),
                         np.full(5, BOX), labels, confs, DEFAULT_PITCH_RATIO)
    assert [int(s.label) for s in slots] == [11, 22, 33, 44, 55]
    assert [round(float(s.confidence), 1) for s in slots] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_labels_survive_a_reversed_input_order():
    """Each label moves with its own box when the input is reversed."""
    xs = (np.arange(5) * PITCH)[::-1]
    labels = np.array([55, 44, 33, 22, 11])
    slots = insert_masks(xs, np.full(5, 500.0), np.full(5, BOX), np.full(5, BOX),
                         labels, np.full(5, 0.5), DEFAULT_PITCH_RATIO)
    assert [int(s.label) for s in slots] == [11, 22, 33, 44, 55]


# Edge extrapolation (off by default)

def _ragged_segment(seed: int = 3):
    """Rows whose ends don't line up, with a spare row built in."""
    rng = np.random.default_rng(seed)
    offsets = [0.0, 4 * PITCH, 0.0, 2 * PITCH]
    widths = [12, 6, 10, 8]
    cx, cy = [], []
    for r, (off, ncol) in enumerate(zip(offsets, widths)):
        cx.append(np.arange(ncol) * PITCH + off)
        cy.append(np.full(ncol, 300.0 + r * 300.0))
    cx = np.concatenate(cx) + rng.normal(0, 3, sum(widths))
    cy = np.concatenate(cy) + rng.normal(0, 3, sum(widths))
    return cx, cy


def test_edge_extrapolation_is_off_by_default():
    """Edge extrapolation is off by default, and turning it on only ever adds slots."""
    cx, cy = _ragged_segment()
    n = cx.size
    args = (cx, cy, np.full(n, BOX), np.full(n, BOX), np.arange(n),
            np.full(n, 0.5), DEFAULT_PITCH_RATIO)
    off = insert_masks(*args)
    on = insert_masks(*args, extend_edges=True)
    assert _kinds(off).count("inserted") < _kinds(on).count("inserted")
    assert len(on) > len(off)


def test_edge_extrapolation_never_adds_a_detected_slot():
    """It only adds masks, never a reading."""
    cx, cy = _ragged_segment(seed=4)
    n = cx.size
    on = insert_masks(cx, cy, np.full(n, BOX), np.full(n, BOX), np.arange(n),
                      np.full(n, 0.5), DEFAULT_PITCH_RATIO, extend_edges=True)
    assert _kinds(on).count("detected") == n


# Every detection survives

def test_no_detection_is_ever_dropped():
    """Insertion is purely additive; no detection is ever lost."""
    rng = np.random.default_rng(5)
    n_rows, n_cols = 3, 12
    cx = np.tile(np.arange(n_cols) * PITCH * 2.5, n_rows) + rng.normal(0, 4, n_rows * n_cols)
    cy = np.repeat([200.0, 500.0, 800.0], n_cols) + rng.normal(0, 4, n_rows * n_cols)
    n = cx.size
    slots = insert_masks(cx, cy, np.full(n, BOX), np.full(n, BOX), np.arange(n),
                         np.full(n, 0.5), DEFAULT_PITCH_RATIO)
    got = sorted(int(s.label) for s in slots if s.kind == "detected")
    assert got == list(range(n))


def test_the_slot_dataclass_defaults_are_the_honest_ones():
    s = Slot("inserted", 0, 12.5)
    assert s.label is None and s.confidence == 0.0


# calibrate_pitch_ratio on the real annotation

@pytest.mark.slow
def test_the_shipped_pitch_ratio_reproduces_from_the_function_its_comment_names():
    """
    calibrate_pitch_ratio() on the real annotation gives about 0.929, within
    1% of the shipped 0.92. The constant stays at 0.92 because every published
    slot count used it.
    """
    from cuneiform.domain1_vision.group_lines import load_boxes

    measured = calibrate_pitch_ratio(load_boxes(), verbose=False)
    assert measured == pytest.approx(0.9290, abs=0.001), (
        f"the measurement has moved: {measured:.4f}. With the grouper repaired "
        f"it should be 0.9290; before the repair it was 0.8932. Re-read the "
        f"docstring above before touching DEFAULT_PITCH_RATIO."
    )
    assert abs(DEFAULT_PITCH_RATIO - measured) / measured < 0.02, (
        f"the shipped {DEFAULT_PITCH_RATIO} no longer reproduces from "
        f"calibrate_pitch_ratio()'s {measured:.4f}, which is what its comment "
        f"claims it was measured by"
    )
