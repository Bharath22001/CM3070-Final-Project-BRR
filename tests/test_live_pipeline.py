"""
Tests for live_pipeline's frozen configuration and build_live_blocks.

The six frozen values are easy to move by accident, and every live figure
depends on them. build_live_blocks decides what context the language model
sees, so I check that a block never spans two segments and that short runs are
dropped rather than padded.

Run from the project root:
    python -m pytest tests/test_live_pipeline.py -q
"""

from __future__ import annotations

import pytest

from cuneiform.orchestrator import live_pipeline as LP
from cuneiform.orchestrator.insert_gaps import Slot
from cuneiform.orchestrator.live_pipeline import MAX_LEN, MIN_BLOCK, build_live_blocks


def _slots(spec: list[tuple[int, int, int]]) -> list[tuple[int, Slot]]:
    """spec is [(segment, line, how_many_slots)], in emission order."""
    out: list[tuple[int, Slot]] = []
    for seg, line, count in spec:
        for i in range(count):
            out.append((seg, Slot("detected", line, float(i), 1, 0.5)))
    return out


# The frozen configuration

def test_the_frozen_configuration_is_the_one_claude_md_section_5_records():
    """
    conf 0.15, cell 2.0, edge extrapolation off, min cluster 3, lambda 1.0,
    LM floor 0.08. If this fails, don't update the test: changing any of these
    changes every live result.
    """
    assert LP.LIVE_CONF == 0.15
    assert LP.LIVE_CELL == 2.0
    assert LP.LIVE_MIN_SIZE == 3
    assert LP.LIVE_EDGE_EXTEND is False
    assert LP.LIVE_LAMBDA == 1.0
    assert LP.LM_FLOOR == 0.08


def test_the_lm_floor_is_quoted_to_the_precision_it_has():
    """0.08 is only accurate to about one bucket, so it must not be quoted with more precision."""
    assert f"{LP.LM_FLOOR:g}" == "0.08"
    assert 0.06 <= LP.LM_FLOOR <= 0.10


def test_the_block_geometry_matches_the_domain_2_training_regime():
    """MAX_LEN matches Domain 2's training length and MIN_BLOCK matches dataset.build_blocks."""
    assert MAX_LEN == 128
    assert MIN_BLOCK == 4


def test_segment_and_to_detections_default_to_the_frozen_values():
    """service.py and evaluate_live.py call these with no arguments, so the defaults must be the frozen values."""
    import inspect

    sig = inspect.signature(LP.segment)
    assert sig.parameters["cell_factor"].default == LP.LIVE_CELL
    assert sig.parameters["min_size"].default == LP.LIVE_MIN_SIZE
    assert inspect.signature(LP.run_fusion).parameters["lam"].default == \
        LP.LIVE_LAMBDA


# build_live_blocks

def test_no_slots_gives_no_blocks():
    assert build_live_blocks([]) == []


def test_a_run_shorter_than_min_block_is_dropped():
    """A run shorter than MIN_BLOCK is dropped, not padded."""
    assert build_live_blocks(_slots([(0, 0, MIN_BLOCK - 1)])) == []


def test_a_run_of_exactly_min_block_is_kept():
    blocks = build_live_blocks(_slots([(0, 0, MIN_BLOCK)]))
    assert [len(b) for b in blocks] == [MIN_BLOCK]


def test_lines_within_one_segment_are_packed_together():
    blocks = build_live_blocks(_slots([(0, 0, 10), (0, 1, 10)]))
    assert [len(b) for b in blocks] == [20]


def test_a_block_never_spans_two_segments():
    """
    A block never spans two segments (two physical surfaces). Both runs would
    fit in one block by length, so only the segment rule can split them.
    """
    blocks = build_live_blocks(_slots([(0, 0, 10), (1, 0, 10)]))
    assert [len(b) for b in blocks] == [10, 10]


def test_a_short_run_at_a_segment_boundary_is_dropped_not_carried_over():
    """Segment 0's three slots must not be carried into segment 1's block."""
    blocks = build_live_blocks(_slots([(0, 0, 3), (1, 0, 10)]))
    assert [len(b) for b in blocks] == [10]
    assert blocks[0] == list(range(3, 13))


def test_a_line_longer_than_the_budget_is_split_into_budget_sized_pieces():
    budget = MAX_LEN - 2
    blocks = build_live_blocks(_slots([(0, 0, 300)]))
    assert [len(b) for b in blocks] == [budget, budget, 300 - 2 * budget]


def test_a_tail_piece_below_min_block_is_dropped_when_a_line_is_split():
    budget = MAX_LEN - 2
    blocks = build_live_blocks(_slots([(0, 0, budget + 2)]))
    assert [len(b) for b in blocks] == [budget]


def test_packing_starts_a_new_block_rather_than_overflowing_the_budget():
    budget = MAX_LEN - 2
    blocks = build_live_blocks(_slots([(0, i, 50) for i in range(6)]))
    assert all(len(b) <= budget for b in blocks)
    assert [len(b) for b in blocks] == [100, 100, 100]


def test_a_line_is_never_split_across_two_blocks_unless_it_exceeds_the_budget():
    """Packing moves whole lines, so a line that doesn't fit starts the next block intact."""
    blocks = build_live_blocks(_slots([(0, 0, 100), (0, 1, 40), (0, 2, 40)]))
    assert [len(b) for b in blocks] == [100, 80]


def test_the_indices_returned_are_positions_in_the_flat_slot_list():
    slots = _slots([(0, 0, 6), (1, 0, 6)])
    blocks = build_live_blocks(slots)
    flat = [i for b in blocks for i in b]
    assert flat == sorted(flat)
    assert all(0 <= i < len(slots) for i in flat)


def test_no_slot_appears_in_two_blocks():
    slots = _slots([(0, 0, 40), (0, 1, 40), (1, 0, 40), (1, 1, 200)])
    seen: list[int] = []
    for b in build_live_blocks(slots):
        seen.extend(b)
    assert len(seen) == len(set(seen))


def test_blocks_preserve_the_reading_order_of_the_slots():
    slots = _slots([(0, 0, 10), (0, 1, 10), (1, 0, 10)])
    for b in build_live_blocks(slots):
        assert b == sorted(b)
    flat = [i for b in build_live_blocks(slots) for i in b]
    assert flat == sorted(flat)


def test_the_budget_leaves_room_for_the_two_sentinel_tokens():
    """Every block is wrapped in <s> ... </s>, so the budget leaves room for both."""
    budget = MAX_LEN - 2
    longest = max(len(b) for b in build_live_blocks(_slots([(0, 0, 1000)])))
    assert longest + 2 <= MAX_LEN
    assert longest == budget
