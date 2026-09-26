"""
Tests for the pure arithmetic in Domain 1: box_iou, match_predictions,
compute_ap, ap_per_class, percentiles, tablet_of and bucket_for from
evaluate_domain1, and tile_origins, crop_tile and nms from run_yolo_inference.

Every Domain 1 figure (recall 0.6753, top-1 0.7161, the 0.9143 mAP50 cap) comes
from these functions, so this is where a wrong number would start. compute_ap
uses four sentinels like Ultralytics; the older two-sentinel form overstated
mAP50, and a test below fails if it comes back. Nothing here loads a
checkpoint, runs inference or re-tiles.

Run from the project root:
    python -m pytest tests/test_domain1_metrics.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from cuneiform import config
from cuneiform.domain1_vision.evaluate_domain1 import (
    SCALE_BUCKETS, ap_per_class, box_iou, bucket_for, compute_ap,
    match_predictions, percentiles, tablet_of,
)
from cuneiform.domain1_vision.run_yolo_inference import (
    STEP, TILE, crop_tile, nms, tile_origins,
)


# box_iou

def test_a_box_against_itself_is_iou_one():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)


def test_disjoint_boxes_are_iou_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert box_iou(a, b)[0, 0] == 0.0


def test_touching_boxes_are_iou_zero_not_negative():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[10.0, 0.0, 20.0, 10.0]])
    assert box_iou(a, b)[0, 0] == 0.0


def test_a_hand_computable_overlap():
    """Two 10x10 boxes offset by 5 in x: intersection 50, union 150, IoU 1/3."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    assert box_iou(a, b)[0, 0] == pytest.approx(1 / 3)


def test_a_box_wholly_inside_another():
    outer = np.array([[0.0, 0.0, 10.0, 10.0]])
    inner = np.array([[2.0, 2.0, 4.0, 4.0]])
    assert box_iou(outer, inner)[0, 0] == pytest.approx(4 / 100)


def test_the_matrix_has_one_row_per_a_and_one_column_per_b():
    a = np.zeros((3, 4))
    a[:, 2:] = 10.0
    b = np.zeros((5, 4))
    b[:, 2:] = 10.0
    assert box_iou(a, b).shape == (3, 5)


def test_an_empty_side_gives_an_empty_matrix_rather_than_raising():
    a = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert box_iou(a, np.zeros((0, 4))).shape == (1, 0)
    assert box_iou(np.zeros((0, 4)), a).shape == (0, 1)


def test_a_degenerate_zero_area_box_does_not_divide_by_zero():
    a = np.array([[5.0, 5.0, 5.0, 5.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert box_iou(a, b)[0, 0] == 0.0


def test_iou_is_symmetric():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 50, (6, 2))
    a = np.hstack([a, a + rng.uniform(1, 30, (6, 2))])
    b = rng.uniform(0, 50, (7, 2))
    b = np.hstack([b, b + rng.uniform(1, 30, (7, 2))])
    assert np.allclose(box_iou(a, b), box_iou(b, a).T)


# match_predictions

def test_no_predictions_or_no_ground_truth_matches_nothing():
    correct, gt_to_pred = match_predictions(
        np.array([]), np.array([1]), np.zeros((1, 0)), 0.5)
    assert correct.size == 0
    assert gt_to_pred.tolist() == [-1]


def test_a_perfect_overlap_of_the_same_class_matches():
    iou = np.array([[1.0]])
    correct, gt_to_pred = match_predictions(
        np.array([7]), np.array([7]), iou, 0.5)
    assert correct.tolist() == [True]
    assert gt_to_pred.tolist() == [0]


def test_class_aware_matching_rejects_the_wrong_class():
    """Class-aware matching rejects a box with the wrong class."""
    iou = np.array([[1.0]])
    aware, _ = match_predictions(np.array([9]), np.array([7]), iou, 0.5,
                                 class_aware=True)
    agnostic, _ = match_predictions(np.array([9]), np.array([7]), iou, 0.5,
                                    class_aware=False)
    assert aware.tolist() == [False]
    assert agnostic.tolist() == [True]


def test_an_overlap_below_the_threshold_does_not_match():
    iou = np.array([[0.4]])
    correct, _ = match_predictions(np.array([1]), np.array([1]), iou, 0.5)
    assert correct.tolist() == [False]


def test_when_two_predictions_compete_the_lowest_index_wins_not_the_best_iou():
    """
    When two predictions of the same class compete for one ground-truth box, the
    lowest prediction index wins, not the best IoU, because np.unique re-sorts
    the rows before the ground-truth dedup. That's Ultralytics' own behaviour,
    so I keep it.
    """
    iou = np.array([[0.6, 0.9]])          # one gt row, two predictions
    correct, gt_to_pred = match_predictions(
        np.array([1, 1]), np.array([1]), iou, 0.5)
    assert correct.tolist() == [True, False]
    assert gt_to_pred.tolist() == [0]


def test_the_other_axis_is_iou_greedy_as_documented():
    """
    One prediction contending for two ground truths goes to the better IoU,
    because the prediction dedup runs first, while the IoU order still holds.
    """
    iou = np.array([[0.6], [0.9]])        # two gt rows, one prediction
    correct, gt_to_pred = match_predictions(
        np.array([1]), np.array([1, 1]), iou, 0.5)
    assert correct.tolist() == [True]
    assert gt_to_pred.tolist() == [-1, 0]


def test_each_ground_truth_is_matched_at_most_once():
    iou = np.array([[0.9, 0.8, 0.7]])
    correct, gt_to_pred = match_predictions(
        np.array([1, 1, 1]), np.array([1]), iou, 0.5)
    assert correct.sum() == 1
    assert (gt_to_pred >= 0).sum() == 1


def test_each_prediction_is_matched_at_most_once():
    iou = np.array([[0.9], [0.8], [0.7]])
    correct, gt_to_pred = match_predictions(
        np.array([1]), np.array([1, 1, 1]), iou, 0.5)
    assert correct.sum() == 1
    assert (gt_to_pred >= 0).sum() == 1


def test_two_independent_pairs_both_match():
    iou = np.array([[0.9, 0.0], [0.0, 0.8]])
    correct, gt_to_pred = match_predictions(
        np.array([1, 2]), np.array([1, 2]), iou, 0.5)
    assert correct.tolist() == [True, True]
    assert gt_to_pred.tolist() == [0, 1]


def test_the_input_iou_matrix_is_not_mutated():
    """The caller's IoU matrix isn't modified, since it is reused for the class-agnostic pass."""
    iou = np.array([[0.9, 0.2], [0.1, 0.8]])
    before = iou.copy()
    match_predictions(np.array([1, 2]), np.array([3, 4]), iou, 0.5)
    assert np.array_equal(iou, before)


# compute_ap: the four sentinels

def test_a_perfect_detector_scores_ap_essentially_one():
    """
    A perfect detector scores 0.995, not 1.0: the final vertical drop lands on
    the last of the 101 points. Same for every detector, as in COCO.
    """
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert compute_ap(recall, precision) == pytest.approx(0.995, abs=1e-6)


def test_ap_is_between_zero_and_one():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = rng.integers(1, 20)
        recall = np.sort(rng.uniform(0, 1, n))
        precision = rng.uniform(0, 1, n)
        assert 0.0 <= compute_ap(recall, precision) <= 1.0


def test_no_area_is_credited_for_recall_the_detector_never_reached():
    """
    Perfect precision up to recall 0.6 must score AP of about 0.6. The old
    two-sentinel form would ramp precision down across the unreached recall and
    give about 0.8, so this fails if it comes back.
    """
    recall = np.array([0.2, 0.4, 0.6])
    precision = np.array([1.0, 1.0, 1.0])
    got = compute_ap(recall, precision)
    assert got == pytest.approx(0.6, abs=0.02)
    assert got < 0.7, "precision is ramping into recall that was never reached"


def test_precision_is_monotonised_into_an_envelope():
    """A dip in precision followed by a recovery is filled in by the envelope."""
    recall = np.array([0.25, 0.5, 0.75, 1.0])
    dipped = compute_ap(recall, np.array([1.0, 0.4, 1.0, 1.0]))
    flat = compute_ap(recall, np.array([1.0, 1.0, 1.0, 1.0]))
    assert dipped == pytest.approx(flat, abs=1e-9)


def test_a_detector_that_finds_nothing_scores_zero():
    assert compute_ap(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0, abs=1e-6)


def test_ap_is_computed_on_101_points():
    """AP uses the 101-point COCO grid."""
    import inspect

    assert "np.linspace(0, 1, 101)" in inspect.getsource(compute_ap)


# ap_per_class

def test_a_class_with_ground_truth_but_no_prediction_scores_zero_and_is_kept():
    """
    A class with ground truth but no prediction scores 0 and stays in the
    average, as in Ultralytics. This is what caps mAP50 at 0.9143.
    """
    tp = np.array([True])
    conf = np.array([0.9])
    pred_cls = np.array([1])
    target_cls = np.array([1, 2])          # class 2 is never predicted
    out = ap_per_class(tp, conf, pred_cls, target_cls)
    assert set(out) == {1, 2}
    assert out[2] == 0.0
    assert out[1] > 0.0


def test_a_class_predicted_but_absent_from_the_ground_truth_is_skipped():
    tp = np.array([False])
    conf = np.array([0.9])
    out = ap_per_class(tp, conf, np.array([5]), np.array([1]))
    assert set(out) == {1}


def test_all_true_positives_on_one_class_scores_near_one():
    tp = np.array([True, True, True])
    conf = np.array([0.9, 0.8, 0.7])
    out = ap_per_class(tp, conf, np.array([1, 1, 1]), np.array([1, 1, 1]))
    assert out[1] == pytest.approx(1.0, abs=0.01)


def test_predictions_are_sorted_by_confidence_before_accumulating():
    """Predictions are sorted by confidence first, so input order doesn't matter."""
    tp = np.array([False, True, True])
    conf = np.array([0.1, 0.9, 0.8])
    pred_cls = target_cls = np.array([1, 1, 1])
    a = ap_per_class(tp, conf, pred_cls, target_cls)

    order = np.array([1, 2, 0])
    b = ap_per_class(tp[order], conf[order], pred_cls, target_cls)
    assert a[1] == pytest.approx(b[1])


# percentiles / tablet_of / bucket_for

def test_percentiles_of_an_empty_series_are_nan_rather_than_an_exception():
    out = percentiles([])
    assert set(out) == {"p1", "p5", "p25", "p50", "p75", "p95", "p99"}
    assert all(np.isnan(v) for v in out.values())


def test_percentiles_are_nondecreasing_and_within_the_data_range():
    values = list(range(100))
    out = percentiles(values)
    keys = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
    got = [out[k] for k in keys]
    assert got == sorted(got)
    assert min(got) >= 0 and max(got) <= 99


def test_percentiles_of_a_single_value():
    out = percentiles([42.0])
    assert all(v == 42.0 for v in out.values())


def test_tablet_of_strips_the_tile_suffix():
    assert tablet_of("P334926_tile_0_384") == "P334926"


def test_tablet_of_handles_the_eight_tablets_that_are_not_p_numbers():
    """tablet_of handles museum ids as well as P-numbers."""
    assert tablet_of("VAT13604Vs_tile_512_0") == "VAT13604Vs"
    assert tablet_of("K.11530_tile_0_0") == "K.11530"
    assert tablet_of("BM_123456_tile_384_768") == "BM_123456"


def test_tablet_of_refuses_a_name_that_breaks_the_convention():
    """A name without the _tile_ marker raises instead of returning a made-up tablet id."""
    with pytest.raises(ValueError, match="_tile_"):
        tablet_of("P334926")


@pytest.mark.parametrize("height,label", [
    (10.0, "tiny <45px"),
    (44.9, "tiny <45px"),
    (45.0, "small 45-60"),
    (59.9, "small 45-60"),
    (60.0, "mid 60-90"),
    (89.9, "mid 60-90"),
    (90.0, "large 90-130"),
    (129.9, "large 90-130"),
    (130.0, "huge >130"),
    (100000.0, "huge >130"),
])
def test_bucket_boundaries_are_half_open_from_below(height, label):
    assert bucket_for(height) == label


def test_the_buckets_tile_the_range_without_gaps():
    for (_, hi_a, _), (lo_b, _, _) in zip(SCALE_BUCKETS, SCALE_BUCKETS[1:]):
        assert hi_a == lo_b


# tile_origins: the flush edge tile

def test_the_tile_geometry_is_the_one_the_checkpoint_was_trained_with():
    assert (TILE, STEP) == (config.D1_TILE_SIZE, config.D1_TILE_STEP) == (512, 384)


def test_an_image_smaller_than_one_tile_gives_a_single_origin():
    assert tile_origins(300) == [0]
    assert tile_origins(TILE) == [0]


def test_a_strided_grid_that_lands_flush_gets_no_extra_tile():
    dim = TILE + STEP                      # 896; origins 0 and 384 (= dim - tile)
    assert tile_origins(dim) == [0, STEP]


def test_a_strided_grid_that_stops_short_gets_a_flush_final_tile():
    """
    When the stride stops short, a flush final tile is added, so the last pixels
    of the image still get looked at. The training tiles are left as they were.
    """
    origins = tile_origins(1000)
    assert origins[-1] == 1000 - TILE
    assert origins[-1] + TILE == 1000


def test_every_pixel_of_the_axis_is_covered():
    for dim in (513, 700, 896, 897, 1000, 1537, 4096):
        origins = tile_origins(dim)
        covered = np.zeros(dim, dtype=bool)
        for o in origins:
            covered[o:o + TILE] = True
        assert covered.all(), f"dim={dim} leaves a gap"


def test_origins_are_strictly_increasing_and_never_negative():
    for dim in (100, 512, 513, 999, 5000):
        origins = tile_origins(dim)
        assert origins == sorted(set(origins))
        assert origins[0] == 0
        assert all(o >= 0 for o in origins)


def test_the_flush_tile_is_added_at_most_once():
    dim = 1000
    origins = tile_origins(dim)
    assert origins.count(dim - TILE) == 1


# crop_tile

def test_an_interior_crop_needs_no_padding():
    arr = np.arange(800 * 800 * 3, dtype=np.uint8).reshape(800, 800, 3)
    sub, vw, vh = crop_tile(arr, 100, 100)
    assert sub.shape == (TILE, TILE, 3)
    assert (vw, vh) == (TILE, TILE)
    assert np.array_equal(sub, arr[100:100 + TILE, 100:100 + TILE])


def test_a_crop_running_off_the_edge_is_padded_to_a_full_tile():
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    sub, vw, vh = crop_tile(arr, 0, 0)
    assert sub.shape == (TILE, TILE, 3)
    assert (vw, vh) == (300, 300)


def test_the_padding_replicates_the_edge_rather_than_filling_black():
    """Padding repeats the edge pixels, because a black border would attract false detections."""
    arr = np.full((300, 300, 3), 200, dtype=np.uint8)
    sub, vw, vh = crop_tile(arr, 0, 0)
    assert (sub[vh:, :vw] == 200).all()
    assert (sub[:vh, vw:] == 200).all()
    assert (sub == 200).all()


def test_the_valid_extents_report_the_real_pixels_only():
    arr = np.zeros((600, 700, 3), dtype=np.uint8)
    _, vw, vh = crop_tile(arr, 400, 400)
    assert (vw, vh) == (700 - 400, 600 - 400)


# nms

def test_nms_on_no_boxes_returns_nothing():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def test_the_highest_scoring_box_is_always_kept():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    scores = np.array([0.4, 0.9])
    assert nms(boxes, scores, 0.5)[0] == 1


def test_a_heavily_overlapping_box_is_suppressed():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]])
    assert nms(boxes, np.array([0.9, 0.8]), 0.5) == [0]


def test_a_barely_overlapping_box_survives():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [9.0, 0.0, 19.0, 10.0]])
    assert sorted(nms(boxes, np.array([0.9, 0.8]), 0.5)) == [0, 1]


def test_suppression_is_class_agnostic():
    """
    NMS ignores class: two overlapping boxes with different classes are one sign
    with two guesses. nms isn't even given the classes, which the signature check
    pins.
    """
    import inspect

    params = list(inspect.signature(nms).parameters)
    assert params == ["boxes", "scores", "iou_thr"]

    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.2, 0.2, 10.2, 10.2]])
    assert nms(boxes, np.array([0.9, 0.85]), 0.5) == [0]


def test_results_are_ordered_by_descending_score():
    boxes = np.array([[0.0, 0.0, 5.0, 5.0], [100.0, 100.0, 105.0, 105.0],
                      [200.0, 200.0, 205.0, 205.0]])
    scores = np.array([0.3, 0.9, 0.6])
    assert nms(boxes, scores, 0.5) == [1, 2, 0]


def test_disjoint_boxes_are_all_kept():
    boxes = np.array([[i * 100.0, 0.0, i * 100.0 + 10.0, 10.0] for i in range(6)])
    assert sorted(nms(boxes, np.linspace(0.1, 0.9, 6), 0.5)) == list(range(6))


def test_a_lower_threshold_keeps_fewer_boxes():
    rng = np.random.default_rng(17)
    xy = rng.uniform(0, 60, (40, 2))
    boxes = np.hstack([xy, xy + 20.0])
    scores = rng.uniform(0, 1, 40)
    assert len(nms(boxes, scores, 0.2)) <= len(nms(boxes, scores, 0.8))


def test_a_zero_area_box_does_not_divide_by_zero():
    boxes = np.array([[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 10.0, 10.0]])
    assert set(nms(boxes, np.array([0.9, 0.8]), 0.5)) <= {0, 1}
