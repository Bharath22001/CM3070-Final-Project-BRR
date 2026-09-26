"""
Tests for the edit-distance code behind every CER and WER figure:
evaluate_fusion.levenshtein / score and evaluate_live.align / score_unit.

The full evaluation checks these over the real corpus but is run by hand, since
it rewrites the results file. Here I check the same four identities on small
hand-checkable inputs and a random sweep:

    dist == levenshtein(pred, truth)
    S + D + I == dist
    I - D == len(pred) - len(truth)
    dist(prefer="sub") == dist(prefer="indel")

I also check that the two tie-breaks really give different S/D/I splits on
ambiguous inputs, which is why both are published.

Run from the project root:
    python -m pytest tests/test_alignment.py -q
"""

from __future__ import annotations

import random

import pytest

from cuneiform.orchestrator.evaluate_fusion import levenshtein, score
from cuneiform.orchestrator.evaluate_live import align, score_unit

NAMES = {1: "AN", 2: "KI", 3: "LUGAL", 4: "E2"}


# levenshtein

def test_identical_sequences_are_at_distance_zero():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([], []) == 0


def test_distance_to_the_empty_sequence_is_the_length():
    assert levenshtein([1, 2, 3], []) == 3
    assert levenshtein([], [1, 2, 3]) == 3


@pytest.mark.parametrize("a,b,want", [
    ([1, 2, 3], [1, 9, 3], 1),          # one substitution
    ([1, 2, 3], [1, 3], 1),             # one deletion
    ([1, 3], [1, 2, 3], 1),             # one insertion
    ([1, 2, 3], [3, 2, 1], 2),          # two substitutions, no reversal cost
    ([1, 1, 1], [1], 2),
])
def test_known_distances(a, b, want):
    assert levenshtein(a, b) == want


def test_the_distance_is_symmetric():
    """The distance doesn't depend on argument order (the code swaps them internally)."""
    rng = random.Random(7)
    for _ in range(200):
        a = [rng.randrange(4) for _ in range(rng.randrange(0, 9))]
        b = [rng.randrange(4) for _ in range(rng.randrange(0, 9))]
        assert levenshtein(a, b) == levenshtein(b, a)


def test_the_triangle_inequality_holds():
    rng = random.Random(8)
    for _ in range(100):
        a = [rng.randrange(3) for _ in range(rng.randrange(0, 7))]
        b = [rng.randrange(3) for _ in range(rng.randrange(0, 7))]
        c = [rng.randrange(3) for _ in range(rng.randrange(0, 7))]
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)


def test_the_distance_is_bounded_by_the_longer_length():
    rng = random.Random(9)
    for _ in range(100):
        a = [rng.randrange(3) for _ in range(rng.randrange(0, 8))]
        b = [rng.randrange(3) for _ in range(rng.randrange(0, 8))]
        assert levenshtein(a, b) <= max(len(a), len(b))


def test_it_works_on_characters_as_well_as_ints():
    """score calls it on a list of characters for CER."""
    assert levenshtein(list("kitten"), list("sitting")) == 3


# score: WER and CER

def test_a_perfect_reading_scores_zero_on_both_metrics():
    assert score([1, 2, 3], [1, 2, 3], NAMES) == {"wer": 0.0, "cer": 0.0}


def test_wer_is_the_edit_distance_over_the_reference_length():
    got = score([1, 9, 3], [1, 2, 3], NAMES)
    assert got["wer"] == pytest.approx(1 / 3)


def test_wer_can_exceed_one_when_the_prediction_is_longer():
    """WER above 1.0 is expected when the prediction is longer, and is normal on live tablets."""
    got = score([1] * 10, [1, 2], NAMES)
    assert got["wer"] > 1.0


def test_an_empty_reference_does_not_divide_by_zero():
    got = score([1, 2], [], NAMES)
    assert got["wer"] == 2.0
    assert got["cer"] >= 0.0


def test_cer_is_computed_over_the_rendered_sign_names():
    """CER runs over the joined sign names, and an unnamed sign still counts through its fallback."""
    got = score([1], [3], NAMES)              # "AN" against "LUGAL"
    assert got["cer"] == pytest.approx(levenshtein(list("AN"), list("LUGAL")) / 5)


def test_an_unnamed_sign_still_contributes_to_cer():
    unnamed = score([99], [99], {})
    assert unnamed["cer"] == 0.0
    differing = score([99], [98], {})
    assert differing["cer"] > 0.0


# align: the four identities

def _identities(pred, truth):
    a = align(pred, truth, "sub")
    b = align(pred, truth, "indel")
    assert a["dist"] == levenshtein(pred, truth)
    assert a["dist"] == b["dist"]
    for r in (a, b):
        assert r["S"] + r["D"] + r["I"] == r["dist"]
        assert r["I"] - r["D"] == len(pred) - len(truth)
    return a, b


def test_the_four_identities_hold_on_hand_checkable_cases():
    for pred, truth in [
        ([], []),
        ([1], []),
        ([], [1]),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 9, 3]),
        ([1, 2, 3, 4], [1, 4]),
        ([1, 4], [1, 2, 3, 4]),
        ([3, 2, 1], [1, 2, 3]),
        ([1, 1, 1, 1], [1]),
    ]:
        _identities(pred, truth)


def test_the_four_identities_hold_under_a_randomised_sweep():
    """The four identities on 400 random pairs."""
    rng = random.Random(1729)
    for _ in range(400):
        pred = [rng.randrange(5) for _ in range(rng.randrange(0, 12))]
        truth = [rng.randrange(5) for _ in range(rng.randrange(0, 12))]
        _identities(pred, truth)


def test_rows_are_the_reference_so_a_missing_sign_is_a_deletion():
    """Rows are the reference: a missing reference sign is a deletion, an extra prediction an insertion."""
    r = align([1, 3], [1, 2, 3], "sub")
    assert (r["D"], r["I"], r["S"]) == (1, 0, 0)

    r = align([1, 2, 3], [1, 3], "sub")
    assert (r["I"], r["D"], r["S"]) == (1, 0, 0)


def test_an_exact_match_diagonal_wins_an_equal_cost_tie_in_both_variants():
    """An exact match wins an equal-cost tie under both tie-breaks."""
    for prefer in ("sub", "indel"):
        r = align([1, 2, 3], [1, 2, 3], prefer)
        assert (r["S"], r["D"], r["I"], r["dist"]) == (0, 0, 0, 0)


def test_the_two_tie_breaks_really_do_produce_different_splits():
    """The two tie-breaks do give different splits on some inputs."""
    rng = random.Random(4242)
    differing = 0
    for _ in range(300):
        pred = [rng.randrange(3) for _ in range(rng.randrange(1, 10))]
        truth = [rng.randrange(3) for _ in range(rng.randrange(1, 10))]
        a = align(pred, truth, "sub")
        b = align(pred, truth, "indel")
        if (a["S"], a["D"], a["I"]) != (b["S"], b["D"], b["I"]):
            differing += 1
    assert differing > 0


def test_the_sub_tie_break_is_not_always_the_minimum_insertion_count():
    """
    A counterexample where prefer="sub" reports one insertion and prefer="indel"
    none, so neither tie-break is a guaranteed extreme. On the saved results
    this happens in 1 of 570 tablet-arm rows, and dist is the same either way.
    If this fails, the aligner's backtrace has changed.
    """
    pred = [0, 2, 2, 0, 0, 0, 1, 0]
    truth = [1, 1, 1, 1, 2, 0, 2, 0, 2]
    a = align(pred, truth, "sub")
    b = align(pred, truth, "indel")

    assert a["dist"] == b["dist"] == 7          # the published quantity agrees
    assert (a["S"], a["D"], a["I"]) == (4, 2, 1)
    assert (b["S"], b["D"], b["I"]) == (6, 1, 0)
    assert a["I"] > b["I"], "the inversion this test documents has gone"


def test_a_concrete_case_where_the_two_tie_breaks_disagree():
    """
    Reference [1, 0], prediction [0, 1]: two substitutions or a deletion plus an
    insertion, both cost 2. sub takes the substitutions, indel the indel pair.
    """
    a = align([0, 1], [1, 0], "sub")
    b = align([0, 1], [1, 0], "indel")
    assert a["dist"] == b["dist"] == 2
    assert (a["S"], a["D"], a["I"]) == (2, 0, 0)
    assert (b["S"], b["D"], b["I"]) == (0, 1, 1)


def test_align_does_not_mutate_its_inputs():
    pred, truth = [1, 2, 3], [1, 9, 3]
    p, t = list(pred), list(truth)
    align(pred, truth, "sub")
    assert pred == p and truth == t


# score_unit

def test_score_unit_reuses_the_simulated_metric_definition_verbatim():
    """The live metric is literally the same function as the simulated one."""
    pred, truth = [1, 2, 3, 4], [1, 9, 3]
    base = score(pred, truth, NAMES)
    unit = score_unit(pred, truth, NAMES)
    assert unit["wer"] == base["wer"]
    assert unit["cer"] == base["cer"]


def test_score_unit_reports_both_tie_breaks():
    unit = score_unit([0, 1], [1, 0], NAMES)
    assert unit["sdi_sub"] == {"S": 2, "D": 0, "I": 0}
    assert unit["sdi_indel"] == {"S": 0, "D": 1, "I": 1}


def test_score_unit_reports_the_length_ratio_that_sets_the_wer_floor():
    """score_unit reports slots/real, which sets each tablet's WER floor."""
    unit = score_unit([1] * 9, [1, 2, 3], NAMES)
    assert unit["len_pred"] == 9
    assert unit["len_truth"] == 3
    assert unit["ratio"] == pytest.approx(3.0)
    assert unit["wer"] >= abs(unit["ratio"] - 1)


def test_the_cer_numerator_and_denominator_reconstruct_the_cer():
    """The CER numerator and denominator rebuild the per-tablet CER exactly."""
    pred, truth = [1, 2, 4], [1, 3, 3, 2]
    unit = score_unit(pred, truth, NAMES)
    assert unit["cer_len"] == len(" ".join(NAMES[s] for s in truth))
    assert unit["cer_dist"] / unit["cer_len"] == pytest.approx(unit["cer"], abs=1e-9)


def test_score_unit_on_an_empty_prediction_is_a_total_deletion():
    unit = score_unit([], [1, 2, 3], NAMES)
    assert unit["sdi_sub"] == {"S": 0, "D": 3, "I": 0}
    assert unit["wer"] == pytest.approx(1.0)
    assert unit["ratio"] == 0.0


def test_score_unit_identities_hold_under_a_randomised_sweep():
    rng = random.Random(31337)
    for _ in range(200):
        pred = [rng.randrange(1, 5) for _ in range(rng.randrange(0, 10))]
        truth = [rng.randrange(1, 5) for _ in range(rng.randrange(1, 10))]
        unit = score_unit(pred, truth, NAMES)
        assert unit["dist"] == levenshtein(pred, truth)
        for key in ("sdi_sub", "sdi_indel"):
            s = unit[key]
            assert s["S"] + s["D"] + s["I"] == unit["dist"]
            assert s["I"] - s["D"] == len(pred) - len(truth)
        assert unit["wer"] == pytest.approx(unit["dist"] / len(truth))


# What the two tie-breaks really give

def _all_minimum_cost_sdi(pred, truth):
    """
    Every (S, D, I) reachable by some minimum-cost alignment. It builds the same
    DP table as align, then walks back through every optimal predecessor.
    Rows are the reference and columns the prediction, as in align.
    """
    from functools import lru_cache

    n, m = len(truth), len(pred)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    d[0] = list(range(m + 1))
    for i in range(1, n + 1):
        d[i][0] = i
        for j in range(1, m + 1):
            c = 0 if truth[i - 1] == pred[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + c, d[i - 1][j] + 1, d[i][j - 1] + 1)

    @lru_cache(maxsize=None)
    def walk(i, j):
        if i == 0 and j == 0:
            return frozenset({(0, 0, 0)})
        out = set()
        if i and j:
            c = 0 if truth[i - 1] == pred[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + c:
                out |= {(S + c, D, I) for S, D, I in walk(i - 1, j - 1)}
        if i and d[i][j] == d[i - 1][j] + 1:
            out |= {(S, D + 1, I) for S, D, I in walk(i - 1, j)}
        if j and d[i][j] == d[i][j - 1] + 1:
            out |= {(S, D, I + 1) for S, D, I in walk(i, j - 1)}
        return frozenset(out)

    return walk(n, m), d[n][m]


def test_the_brute_force_oracle_agrees_with_align_on_the_distance():
    """The brute-force walk gives the same distance as align, so it solves the same problem."""
    rng = random.Random(90210)
    for _ in range(100):
        pred = [rng.randrange(3) for _ in range(rng.randrange(0, 7))]
        truth = [rng.randrange(3) for _ in range(rng.randrange(0, 7))]
        triples, dist = _all_minimum_cost_sdi(pred, truth)
        assert dist == levenshtein(pred, truth)
        assert triples, "no alignment enumerated"
        for S, D, I in triples:
            assert S + D + I == dist
            assert I - D == len(pred) - len(truth)
        for prefer in ("sub", "indel"):
            a = align(pred, truth, prefer)
            assert (a["S"], a["D"], a["I"]) in triples, (
                f"align({prefer}) produced a split no minimum-cost alignment "
                f"can reach: {a} not in {sorted(triples)}"
            )


def test_both_tie_breaks_lie_inside_the_true_insertion_interval():
    """
    Both tie-breaks give insertion counts inside the true range of all
    minimum-cost alignments, so the published pair understates the ambiguity
    rather than overstating it. At least one pair shows a strict gap.
    """
    rng = random.Random(1848)
    strict = 0
    for _ in range(200):
        pred = [rng.randrange(3) for _ in range(rng.randrange(0, 8))]
        truth = [rng.randrange(3) for _ in range(rng.randrange(0, 8))]
        triples, _ = _all_minimum_cost_sdi(pred, truth)
        true_lo = min(I for _, _, I in triples)
        true_hi = max(I for _, _, I in triples)
        i_sub = align(pred, truth, "sub")["I"]
        i_ind = align(pred, truth, "indel")["I"]
        assert true_lo <= min(i_sub, i_ind), (pred, truth, true_lo, i_sub, i_ind)
        assert max(i_sub, i_ind) <= true_hi, (pred, truth, true_hi, i_sub, i_ind)
        if true_lo < min(i_sub, i_ind) or max(i_sub, i_ind) < true_hi:
            strict += 1
    assert strict > 0, (
        "the two tie-breaks hit the true extremes on every pair sampled, which "
        "would make the correction to the docstring unnecessary -- check the "
        "oracle before believing it"
    )


def test_the_module_no_longer_claims_the_tie_breaks_are_the_extremes():
    """
    Checks evaluate_live's source: neither docstring may claim a tie-break is
    an extreme, and the I_min / I_max keys must survive because they are keys in
    the saved results file.
    """
    import inspect

    from cuneiform.orchestrator import evaluate_live

    src = inspect.getsource(evaluate_live)
    low = src.lower()

    # Each occurrence must be negated on its own line or the line before; just
    # checking the words are absent would fail correct prose that denies the claim.
    NEGATORS = ("neither", "not the", "no longer", "never", "nor ")
    lines = low.split("\n")
    for phrase in ("minimum insertion count", "maximum insertion count"):
        for i, line in enumerate(lines):
            if phrase not in line:
                continue
            context = (lines[i - 1] if i else "") + " " + line
            assert any(n in context for n in NEGATORS), (
                f"{phrase!r} is asserted rather than denied on line {i + 1} of "
                f"the module. Neither tie-break is optimised over all "
                f"minimum-cost alignments, so neither is an extreme. "
                f"Context: {context.strip()!r}"
            )

    # Two phrasings that can't be read as a denial.
    assert "gives the minimum" not in low
    assert "gives the maximum" not in low

    assert "the two tie-breaks" in low, (
        "the corrected wording about the two tie-breaks has gone"
    )
    assert "1 of 570" in src, (
        "the measured incidence no longer travels with the claim"
    )

    assert '"I_min": sum(' in src, "the published artifact key has been renamed"
    assert '"I_max": sum(' in src, "the published artifact key has been renamed"
