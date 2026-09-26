"""
Tests for significance and significance_live: the permutation tests, the
bootstrap interval and the selection correction behind every p-value.

The original selection correction flips the signs of uncentred rows, so its
null maximum is set by the worst lambda's size, not by chance. It is valid but
far too conservative. Besides checking the statistics on inputs whose answer
is known, these tests reproduce that problem and show the studentized version
doesn't have it. The saved artifacts are only read, in the tests marked slow.

Run from the project root:
    python -m pytest tests/test_significance.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cuneiform.orchestrator.significance import (
    bootstrap_ci, paired_permutation, series,
)
from cuneiform.orchestrator.significance_live import (
    DROP_UNINFORMATIVE_ARMS, _diff_matrix, lambda_arms, length_regime, load,
    mde, selection_corrected_shipped, selection_corrected_studentized,
)

LIVE_JSON = "artifacts/live_fusion_results.json"
SIM_JSON = "artifacts/fusion_results.json"


def _rng(seed=0):
    return np.random.default_rng(seed)


# paired_permutation

def test_a_large_consistent_difference_is_significant():
    d = np.full(30, 0.05)
    assert paired_permutation(d, _rng(), n_perm=2000) < 0.001


def test_a_null_difference_is_not_significant():
    d = _rng(1).normal(0, 1, 40)
    assert paired_permutation(d, _rng(2), n_perm=2000) > 0.05


def test_the_p_value_can_never_be_zero():
    """The +1 on both sides means p can never be 0."""
    d = np.full(50, 1000.0)
    p = paired_permutation(d, _rng(), n_perm=1000)
    assert p > 0.0
    assert p == pytest.approx(1 / 1001)


def test_the_p_value_is_bounded_by_one():
    for seed in range(5):
        d = _rng(seed).normal(0, 1, 20)
        assert 0.0 < paired_permutation(d, _rng(seed + 100), n_perm=500) <= 1.0


def test_the_test_is_two_sided():
    """It tests |mean|, so an effect of the same size in either direction gets the same p."""
    d = _rng(3).normal(0.4, 1, 30)
    a = paired_permutation(d, _rng(9), n_perm=4000)
    b = paired_permutation(-d, _rng(9), n_perm=4000)
    assert a == b


def test_an_all_zero_difference_gives_p_of_one():
    assert paired_permutation(np.zeros(20), _rng(), n_perm=500) == 1.0


# bootstrap_ci

def test_the_interval_brackets_the_observed_mean():
    d = _rng(4).normal(0.3, 0.1, 40)
    lo, hi = bootstrap_ci(d, _rng(5), n_boot=4000)
    assert lo < d.mean() < hi


def test_the_interval_is_ordered_and_finite():
    d = _rng(6).normal(0, 1, 25)
    lo, hi = bootstrap_ci(d, _rng(7), n_boot=2000)
    assert lo < hi
    assert np.isfinite([lo, hi]).all()


def test_a_constant_series_gives_a_degenerate_interval():
    lo, hi = bootstrap_ci(np.full(20, 0.07), _rng(), n_boot=500)
    assert lo == pytest.approx(0.07)
    assert hi == pytest.approx(0.07)


def test_more_data_gives_a_narrower_interval():
    small = _rng(8).normal(0.2, 1, 15)
    big = _rng(8).normal(0.2, 1, 400)
    w_small = np.diff(bootstrap_ci(small, _rng(1), n_boot=3000))[0]
    w_big = np.diff(bootstrap_ci(big, _rng(1), n_boot=3000))[0]
    assert w_big < w_small


def test_an_interval_excluding_zero_agrees_with_a_significant_permutation_test():
    """On a clear-cut case the interval and the permutation test agree."""
    d = _rng(10).normal(0.5, 0.3, 40)
    lo, hi = bootstrap_ci(d, _rng(11), n_boot=4000)
    assert lo > 0
    assert paired_permutation(d, _rng(12), n_perm=4000) < 0.05


# mde: what n = 30 can and can't detect

def test_the_minimum_detectable_effect_shrinks_with_n():
    assert mde(120)["d_detectable_p05"] < mde(30)["d_detectable_p05"]


def test_the_thirty_tablet_mde_is_the_figure_the_report_must_quote():
    """
    At n = 30, d below about 0.36 is undetectable at p < 0.05 and about 0.51 is
    needed for 80% power, so a null result here isn't evidence of no effect.
    """
    got = mde(30)
    assert got["n"] == 30
    assert got["d_detectable_p05"] == pytest.approx(1.96 / 30 ** 0.5, rel=1e-9)
    assert got["d_detectable_p05"] == pytest.approx(0.3578, abs=0.0005)
    assert got["d_80pct_power"] == pytest.approx(0.5115, abs=0.0005)


def test_eighty_percent_power_needs_a_larger_effect_than_bare_significance():
    got = mde(30)
    assert got["d_80pct_power"] > got["d_detectable_p05"]


# length_regime

def test_the_drop_uninformative_arms_are_labelled_as_such():
    for arm in DROP_UNINFORMATIVE_ARMS:
        assert length_regime(arm) == "drop-uninformative"


def test_every_other_arm_is_length_matched():
    assert length_regime("3_fusion_lam1.0") == "length-matched"
    assert length_regime("2_threshold_mask") == "length-matched"


def test_the_vision_only_and_padded_arms_are_in_different_regimes():
    """
    1_vision_only is shorter than the reference, 1b_vision_only_padded is not, so
    the headline is quoted against the padded arm.
    """
    assert length_regime("1_vision_only") == "drop-uninformative"
    assert length_regime("1b_vision_only_padded") == "length-matched"


# load / series / lambda_arms

def test_load_returns_none_for_a_path_that_does_not_exist(tmp_path):
    assert load(tmp_path / "nope.json") is None


def test_series_reads_one_metric_out_of_one_arm():
    data = {"per_block": {"a": [{"wer": 0.1}, {"wer": 0.3}]}}
    assert series(data, "a", "wer").tolist() == [0.1, 0.3]


def test_lambda_arms_are_returned_sorted_numerically():
    """Lambda arms sort as numbers, not strings ("10.0" would come before "2.0")."""
    data = {"per_block": {"3_fusion_lam0.25": [], "3_fusion_lam10.0": [],
                          "3_fusion_lam2.0": [], "2_threshold_mask": []}}
    assert lambda_arms(data) == [0.25, 2.0, 10.0]


def test_the_diff_matrix_is_positive_when_fusion_is_better():
    """baseline - fusion is positive when fusion has the lower error."""
    data = {"per_block": {
        "2_threshold_mask": [{"wer": 0.5}, {"wer": 0.6}],
        "3_fusion_lam1.0": [{"wer": 0.4}, {"wer": 0.5}],
    }}
    D = _diff_matrix(data, [1.0], "wer")
    assert D.shape == (1, 2)
    assert (D > 0).all()


# The original correction's weakness, reproduced

_LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def _synthetic(n_lambda: int, n_units: int = 30, seed: int = 0,
               effect: float = 0.02) -> dict:
    """
    A lambda sweep with the same real effect on every arm, plus (when n_lambda
    is 6) one extra arm with no effect and thirty times the variance. The winner
    is identical either way.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.05, n_units)
    per_block = {"2_threshold_mask": [{"wer": float(x)} for x in base]}
    for i, lam in enumerate(_LAMBDAS[:n_lambda]):
        if i == n_lambda - 1 and n_lambda == 6:
            diff = rng.normal(0.0, 0.30, n_units)      # noisy, no effect
        else:
            diff = rng.normal(effect, 0.01, n_units)
        per_block[f"3_fusion_lam{lam}"] = [
            {"wer": float(x)} for x in base - diff]
    return {"per_block": per_block}


def test_the_shipped_correction_loses_a_real_effect_to_one_noisy_arm():
    """
    Adding one noisy, no-effect arm leaves the winner's statistic at 0.0228 but
    takes the original correction's p from about 0.0002 to about 0.33.
    """
    quiet = _synthetic(n_lambda=5)
    noisy = _synthetic(n_lambda=6)

    a = selection_corrected_shipped(quiet, lambda_arms(quiet), "wer", n_perm=4000)
    b = selection_corrected_shipped(noisy, lambda_arms(noisy), "wer", n_perm=4000)

    assert a["obs"] == pytest.approx(b["obs"], abs=1e-12), (
        "the winner must be identical in both sweeps, or this is not a "
        "controlled comparison"
    )
    assert a["p"] < 0.01
    assert b["p"] > 0.10, (
        f"the noisy arm no longer buries the effect (p {a['p']:.4f} -> "
        f"{b['p']:.4f}); the weakness may have been fixed, which would change "
        f"every corrected p-value I report"
    )
    assert b["null_p95"] > a["null_p95"] * 5


def test_the_studentized_correction_is_unmoved_by_the_same_noisy_arm():
    """The studentized correction isn't moved by the same noisy arm."""
    quiet = _synthetic(n_lambda=5)
    noisy = _synthetic(n_lambda=6)

    a = selection_corrected_studentized(quiet, lambda_arms(quiet), "wer",
                                        n_perm=4000)
    b = selection_corrected_studentized(noisy, lambda_arms(noisy), "wer",
                                        n_perm=4000)

    assert a["obs_t"] == pytest.approx(b["obs_t"], abs=1e-9)
    assert a["p"] < 0.01 and b["p"] < 0.01


def test_both_corrections_report_their_null_mean_and_p95():
    """Both corrections return their null mean and p95 beside the p-value."""
    data = _synthetic(n_lambda=6)
    lams = lambda_arms(data)
    for fn in (selection_corrected_shipped, selection_corrected_studentized):
        got = fn(data, lams, "wer", n_perm=500)
        assert {"p", "null_mean", "null_p95", "best_lambda", "stat"} <= set(got)
        assert np.isfinite([got["null_mean"], got["null_p95"], got["p"]]).all()


def test_the_two_studentization_variants_are_both_available_and_named():
    """
    Both studentized variants (SE held fixed, SE refit per permutation) are
    available and labelled, since they give very different p-values.
    """
    data = _synthetic(n_lambda=6)
    lams = lambda_arms(data)
    fixed = selection_corrected_studentized(data, lams, "wer",
                                            refit_se=False, n_perm=500)
    refit = selection_corrected_studentized(data, lams, "wer",
                                            refit_se=True, n_perm=500)
    assert fixed["stat"] != refit["stat"]
    assert fixed["refit_se"] is False and refit["refit_se"] is True
    assert "SE" in fixed["stat"] and "SE" in refit["stat"]


def test_a_selection_corrected_p_is_never_smaller_than_it_can_justify():
    data = _synthetic(n_lambda=6)
    lams = lambda_arms(data)
    for fn in (selection_corrected_shipped, selection_corrected_studentized):
        p = fn(data, lams, "wer", n_perm=1000)["p"]
        assert 1 / 1001 <= p <= 1.0


# Against the saved artifacts (read-only)

@pytest.mark.slow
def test_g5_is_still_visible_in_the_published_simulated_artifact():
    """
    On the simulated results at lambda 1.0 (observed mean improvement 0.01696
    WER), the original correction's null p95 (0.0179) sits above the observed
    value, while the studentized test is clearly significant. The null mean
    should be about +0.0068.
    """
    from pathlib import Path

    path = Path(SIM_JSON)
    if not path.exists():
        pytest.skip("the simulated artifact is not on disk")
    data = load(path)
    lams = lambda_arms(data)

    shipped = selection_corrected_shipped(data, lams, "wer", n_perm=20000)
    stud = selection_corrected_studentized(data, lams, "wer", n_perm=20000)

    assert shipped["null_mean"] == pytest.approx(0.0068, abs=0.0015), (
        f"the shipped null mean is now {shipped['null_mean']:.4f}; I recorded "
        f"+0.0068 and the write-up rests on it"
    )
    assert shipped["obs"] < shipped["null_p95"], (
        "the shipped statistic no longer buries the observed effect under its "
        "own null p95, which is the visible form of the weakness"
    )
    assert stud["obs_t"] > stud["null_p95"]
    assert stud["p"] < shipped["p"] / 10, (
        "the studentized correction is no longer an order of magnitude more "
        "powerful than the shipped one, which is the whole finding"
    )
    assert shipped["best_lambda"] == stud["best_lambda"] == 1.0, (
        "the two corrections disagree about which lambda wins, which would be "
        "a far larger problem than the power difference"
    )


@pytest.mark.slow
def test_the_live_artifact_is_the_real_detector_scored_per_tablet():
    """The live artifact says simulated_vision is false and the unit is the tablet."""
    from pathlib import Path

    path = Path(LIVE_JSON)
    if not path.exists():
        pytest.skip("the live artifact is not on disk")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["simulated_vision"] is False
    assert data["unit"] == "tablet"
    assert data["n_units"] == 30
    assert len(data["tablet_order"]) == 30


@pytest.mark.slow
def test_reading_the_live_artifact_does_not_modify_it():
    """Reading the live results file doesn't change its md5."""
    import hashlib
    from pathlib import Path

    path = Path(LIVE_JSON)
    if not path.exists():
        pytest.skip("the live artifact is not on disk")
    before = hashlib.md5(path.read_bytes()).hexdigest()
    load(path)
    assert hashlib.md5(path.read_bytes()).hexdigest() == before
    assert before.upper() == "A8E22A8C0D70E6440B8009220C3BACFD", (
        f"the pinned live artifact has changed: md5 is now {before.upper()}. "
        f"Stop and find out why before changing anything else."
    )
