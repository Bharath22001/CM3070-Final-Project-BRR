"""
Tests for group_lines, which turns bounding-box geometry into reading lines.
Everything downstream (gap insertion, the slot sequence, every live CER and
WER) depends on it.

The last tests cover a bug these tests found: the grouper handled a density
peak in the first histogram bin but not the last, so a segment's final row
merged into the row above. It is fixed, and those tests now require the fix.

Run from the project root:
    python -m pytest tests/test_group_lines.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cuneiform.domain1_vision.group_lines import (
    BAND_ROWS, HIST_BINS_PER_H, MIN_PEAK_SEP, SLOPES, _peakiness,
    fit_local_slopes, group_segment, load_boxes,
)


def _rows(n_rows: int, n_cols: int, pitch_x: float = 110.0,
          pitch_y: float = 160.0, box: float = 100.0, slope: float = 0.0,
          jitter: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A synthetic segment: `n_rows` straight rows of `n_cols` boxes."""
    rng = np.random.default_rng(seed)
    cx = np.tile(np.arange(n_cols) * pitch_x, n_rows)
    cy = np.repeat(np.arange(n_rows) * pitch_y, n_cols) + slope * cx
    if jitter:
        cx = cx + rng.normal(0, jitter, cx.size)
        cy = cy + rng.normal(0, jitter, cy.size)
    return pd.DataFrame({"cx": cx, "cy": cy,
                         "w": np.full(cx.size, box),
                         "h": np.full(cx.size, box)})


# _peakiness: the score the slope search maximises

def test_peakiness_is_maximal_when_points_concentrate():
    """Four points in one bin score 16, spread over four bins they score 4."""
    assert _peakiness(np.array([0.0, 0.0, 0.0, 0.0]), 1.0) == 16.0
    assert _peakiness(np.array([0.0, 1.0, 2.0, 3.0]), 1.0) == 4.0


def test_peakiness_is_monotone_as_points_are_pulled_together():
    spread = _peakiness(np.array([0.0, 3.0, 6.0, 9.0]), 1.0)
    tight = _peakiness(np.array([0.0, 0.5, 1.0, 1.5]), 1.0)
    assert tight > spread


def test_peakiness_is_invariant_to_translation():
    a = np.array([0.0, 1.0, 1.0, 5.0])
    assert _peakiness(a, 1.0) == _peakiness(a + 1000.0, 1.0)


# fit_local_slopes

def test_a_flat_segment_is_fitted_at_slope_zero():
    g = _rows(6, 12)
    bands = fit_local_slopes(g["cx"].to_numpy(float), g["cy"].to_numpy(float), 100.0)
    assert all(abs(s) < 0.01 for _, _, s in bands)


def test_a_known_slant_is_recovered():
    """Rows tilted by exactly 0.10 recover a slope of 0.10."""
    slope = 0.10
    assert np.isclose(SLOPES, slope).any(), "the search grid must contain 0.10"
    g = _rows(8, 12, pitch_y=160.0, box=90.0, slope=slope)
    bands = fit_local_slopes(g["cx"].to_numpy(float), g["cy"].to_numpy(float), 90.0)
    assert all(abs(s - slope) < 1e-6 for _, _, s in bands)


def test_the_bands_tile_the_segment_without_gaps_or_overlaps():
    g = _rows(20, 10, box=40.0)
    cy = g["cy"].to_numpy(float)
    bands = fit_local_slopes(g["cx"].to_numpy(float), cy, 40.0)
    assert bands[0][0] == pytest.approx(cy.min())
    assert bands[-1][1] >= cy.max()
    for (_, hi_a, _), (lo_b, _, _) in zip(bands, bands[1:]):
        assert hi_a == pytest.approx(lo_b)


def test_a_tall_segment_is_split_into_several_bands():
    """A tall segment gets more than one slope estimate, so it can follow curvature."""
    box = 40.0
    tall = _rows(BAND_ROWS * 4, 10, pitch_y=box * 2.5, box=box)
    bands = fit_local_slopes(tall["cx"].to_numpy(float),
                             tall["cy"].to_numpy(float), box)
    assert len(bands) > 1


def test_a_sparse_band_falls_back_rather_than_fitting_noise():
    """With every band too sparse to fit, every slope falls back to 0.0, not NaN."""
    g = pd.DataFrame({"cx": [0.0, 50.0, 100.0], "cy": [0.0, 1.0, 2.0],
                      "w": [10.0] * 3, "h": [10.0] * 3})
    bands = fit_local_slopes(g["cx"].to_numpy(float), g["cy"].to_numpy(float), 10.0)
    assert all(np.isfinite(s) for _, _, s in bands)


def test_slopes_are_searched_only_within_the_declared_grid():
    g = _rows(6, 12, slope=0.9)          # far outside the search range
    bands = fit_local_slopes(g["cx"].to_numpy(float), g["cy"].to_numpy(float), 100.0)
    for _, _, s in bands:
        assert SLOPES.min() <= s <= SLOPES.max()


# group_segment

def test_one_line_id_is_returned_per_box_and_none_is_negative():
    """
    One line id per box and none negative. Ids can skip a number (it happens on
    1 of 132 real segments), which is harmless because insert_masks renumbers them.
    """
    g = _rows(5, 10, jitter=3.0, seed=1)
    line_of = group_segment(g)
    assert len(line_of) == len(g)
    assert line_of.min() >= 0
    assert line_of.dtype.kind == "i"


def test_line_ids_increase_down_the_projected_axis():
    """Line ids increase down the projected axis, matching the reading order."""
    g = _rows(5, 10, jitter=2.0, seed=2)
    line_of = group_segment(g)
    cy = g["cy"].to_numpy(float)
    means = [cy[line_of == l].mean() for l in sorted(np.unique(line_of))]
    assert means == sorted(means)


def test_well_separated_rows_are_recovered_as_separate_lines():
    g = _rows(4, 10, pitch_y=160.0, box=100.0, jitter=3.0, seed=3)
    assert len(np.unique(group_segment(g))) == 4


def test_a_slanted_segment_groups_as_well_as_a_flat_one():
    """Rows tilted by 0.10 still come back as rows."""
    flat = group_segment(_rows(4, 10, jitter=3.0, seed=4))
    slanted = group_segment(_rows(4, 10, slope=0.10, jitter=3.0, seed=4))
    assert len(np.unique(slanted)) == len(np.unique(flat)) == 4


def test_a_single_dense_row_is_not_split_into_several(tmp_path):
    """One genuine row stays one line (peaks within 0.75 box heights merge)."""
    one = _rows(1, 20, jitter=2.0, seed=5)
    assert len(np.unique(group_segment(one))) == 1


def test_boxes_within_a_line_are_not_reordered_by_the_grouper():
    """group_segment only assigns line ids; sorting by x happens later."""
    g = _rows(3, 8, jitter=2.0, seed=6)
    before = g["cx"].to_numpy(float).copy()
    group_segment(g)
    assert np.array_equal(g["cx"].to_numpy(float), before)


def test_a_degenerate_segment_with_no_peak_returns_one_line():
    g = pd.DataFrame({"cx": [0.0, 1.0, 2.0], "cy": [0.0, 0.0, 0.0],
                      "w": [10.0] * 3, "h": [10.0] * 3})
    assert set(group_segment(g).tolist()) == {0}


# Against the real annotation

@pytest.mark.slow
def test_the_grouper_runs_on_every_real_annotated_segment():
    """
    Runs group_segment over every annotated segment on disk and checks the
    invariants hold: valid ids, monotone in mean y, never more lines than boxes.
    """
    bb = load_boxes()
    n = 0
    for _, g in bb.groupby(["split", "tablet_CDLI", "segm_idx"]):
        if len(g) < 3:
            continue
        n += 1
        line_of = group_segment(g)
        assert line_of.min() >= 0
        assert len(np.unique(line_of)) <= len(g)
        cy = g["cy"].to_numpy(float)
        means = [cy[line_of == l].mean() for l in sorted(np.unique(line_of))]
        assert means == sorted(means)
    assert n > 100, f"only {n} segments reached; the annotation may be missing"


# The final-row bug, now a requirement

def test_the_final_row_is_not_merged_into_the_one_above():
    """
    Three cleanly separated rows must come back as three lines. Before the fix
    the last row merged into the one above because the final histogram bin could
    never be a peak.

    On the 132 real segments, against the independent line anchors, the median
    line-count ratio went from 0.902 to 0.983, but per segment it is 13 better,
    7 worse and 16 unchanged.
    """
    g = _rows(3, 6, pitch_y=160.0, box=100.0)
    line_of = group_segment(g)

    assert len(np.unique(line_of)) == 3, (
        "the final row merged into the one above; the trailing-edge peak case "
        "in group_segment has been removed or broken"
    )
    # Each built row sits whole in one line, and the three lines are distinct.
    blocks = [set(line_of[i:i + 6].tolist()) for i in range(0, 18, 6)]
    assert all(len(b) == 1 for b in blocks), f"a row was split across lines: {blocks}"
    assert len({next(iter(b)) for b in blocks}) == 3, f"two rows share a line: {blocks}"


def test_both_the_leading_and_the_trailing_edge_case_are_in_the_source():
    """Both the leading and trailing edge cases are in the source, checked directly."""
    import inspect

    src = inspect.getsource(group_segment)
    assert "peaks.insert(0, 0)" in src, "the leading-edge case has moved"
    assert "peaks.append(" in src, (
        "the trailing-edge peak case has gone; see the test above for what it "
        "is worth on the independent line anchors"
    )
    assert "range(1, len(smooth) - 1)" in src


def test_insert_masks_renumbers_the_lines_it_is_given_contiguously():
    """insert_masks renumbers lines from 0 with no gaps, which is why skipped ids don't matter."""
    from cuneiform.orchestrator.insert_gaps import DEFAULT_PITCH_RATIO, insert_masks

    g = _rows(4, 10, jitter=3.0, seed=7)
    n = len(g)
    slots = insert_masks(g["cx"], g["cy"], g["w"], g["h"],
                         np.arange(n), np.full(n, 0.9), DEFAULT_PITCH_RATIO)
    lines = sorted({s.line for s in slots})
    assert lines == list(range(len(lines)))
