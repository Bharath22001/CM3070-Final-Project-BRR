"""
Tests for fusion, the log-linear combination at the centre of my project.

The two key invariants: at lambda = 0 fusion reproduces the vision argmax
exactly, and at a large lambda it reproduces the LM argmax, so the sweep really
runs between the two single-modality systems. Most tests build the posteriors
by hand so the expected answer can be worked out on paper; one slow test
repeats A and B on the real checkpoint and a real block.

Skip the slow one with:
    python -m pytest tests/test_fusion.py -q -m "not slow"

Run from the project root:
    python -m pytest tests/test_fusion.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.orchestrator.fusion import (
    LOG_FLOOR, Detection, argmax_labels, fuse, support_ids, vision_posterior,
)


@pytest.fixture(scope="module")
def tok() -> SignTokenizer:
    return SignTokenizer()


def _uniform_over_support(tok: SignTokenizer, rows: int) -> torch.Tensor:
    """A flat distribution on the sign support and nothing elsewhere."""
    sup = support_ids(tok)
    p = torch.zeros(rows, tok.vocab_size, dtype=torch.float64)
    p[:, sup] = 1.0 / len(sup)
    return p


def _spiked(tok: SignTokenizer, labels: list[int], mass: float = 0.9
            ) -> torch.Tensor:
    """One row per label, `mass` on that label's token and the rest spread flat."""
    sup = support_ids(tok)
    p = torch.zeros(len(labels), tok.vocab_size, dtype=torch.float64)
    p[:, sup] = (1.0 - mass) / len(sup)
    for r, lab in enumerate(labels):
        p[r, tok.sign_to_id[lab]] += mass
    return p / p.sum(dim=-1, keepdim=True)


# support_ids

def test_support_is_exactly_the_282_sign_tokens(tok):
    sup = support_ids(tok)
    assert sup.dtype == torch.long
    assert len(sup) == 282
    assert set(sup.tolist()) == set(tok.id_to_sign)


def test_support_excludes_every_special_including_unk_and_brk(tok):
    """None of the six special tokens can ever be an argmax."""
    assert not (set(support_ids(tok).tolist()) & set(tok.specials.values()))


def test_support_is_sorted(tok):
    sup = support_ids(tok).tolist()
    assert sup == sorted(sup)


# Detection

def test_only_a_detected_box_is_informative():
    assert Detection("detected", train_label=1, confidence=0.5).is_informative
    assert not Detection("unknown").is_informative
    assert not Detection("missed").is_informative


# vision_posterior

def test_every_row_is_a_normalised_distribution(tok):
    label = sorted(tok.vision_emit_classes)[0]
    dets = [Detection("detected", train_label=label, confidence=0.9),
            Detection("unknown"), Detection("missed")]
    p = vision_posterior(dets, tok)
    assert p.shape == (3, tok.vocab_size)
    assert torch.allclose(p.sum(dim=-1), torch.ones(3, dtype=torch.float64))
    assert (p >= 0).all()


def test_mass_sits_only_on_the_sign_support(tok):
    """A special token never carries vision mass."""
    dets = [Detection("detected", train_label=sorted(tok.vision_emit_classes)[0],
                      confidence=0.9)]
    p = vision_posterior(dets, tok)
    for special_id in tok.specials.values():
        assert p[0, special_id].item() == 0.0


def test_a_confident_detection_puts_its_argmax_on_its_own_class(tok):
    label = sorted(tok.vision_emit_classes)[3]
    p = vision_posterior(
        [Detection("detected", train_label=label, confidence=0.9)], tok)
    assert tok.id_to_sign[int(p[0].argmax())] == label


def test_an_uninformative_position_is_exactly_uniform(tok):
    """An uninformative position gets an exactly flat row, so the LM decides alone."""
    sup = support_ids(tok)
    for kind in ("unknown", "missed"):
        row = vision_posterior([Detection(kind)], tok)[0]
        vals = row[sup]
        assert torch.allclose(vals, vals[0].expand_as(vals))


def test_the_residual_leaves_the_detected_class_short_of_certainty(tok):
    """The residual 1 - c keeps log P_vision finite, so the LM can still move the argmax."""
    label = sorted(tok.vision_emit_classes)[0]
    row = vision_posterior(
        [Detection("detected", train_label=label, confidence=1.0)], tok)[0]
    top = row[tok.sign_to_id[label]].item()
    assert top < 1.0
    assert row[support_ids(tok)].min().item() > 0.0


def test_a_full_top_k_is_used_rather_than_only_the_argmax(tok):
    """A full score vector lets fusion use vision's second choice."""
    a, b = sorted(tok.vision_emit_classes)[:2]
    row = vision_posterior(
        [Detection("detected", train_label=a, confidence=0.3,
                   scores={a: 0.3, b: 0.6})], tok)[0]
    assert tok.id_to_sign[int(row.argmax())] == b


def test_an_empty_scores_dict_falls_back_to_the_single_label(tok):
    label = sorted(tok.vision_emit_classes)[5]
    row = vision_posterior(
        [Detection("detected", train_label=label, confidence=0.7)], tok)[0]
    assert tok.id_to_sign[int(row.argmax())] == label


def test_a_label_outside_the_vocabulary_degrades_to_uniform_not_to_a_crash(tok):
    """A label outside the vocabulary gives a uniform row, not a crash."""
    bogus = max(tok.sign_to_id) + 10_000
    row = vision_posterior(
        [Detection("detected", train_label=bogus, confidence=0.9)], tok)[0]
    sup = support_ids(tok)
    assert torch.allclose(row[sup], row[sup][0].expand(len(sup)))


def test_no_detections_gives_an_empty_matrix_of_the_right_width(tok):
    p = vision_posterior([], tok)
    assert p.shape == (0, tok.vocab_size)


# fuse: the two invariants

def test_lambda_zero_reproduces_the_vision_argmax_exactly(tok):
    """Invariant A. The LM confidently disagrees everywhere, so any LM leak at lambda 0 would show."""
    emit = sorted(tok.vision_emit_classes)
    vision_labels = emit[:20]
    lm_labels = emit[20:40]
    assert not set(vision_labels) & set(lm_labels)

    p_v = _spiked(tok, vision_labels, mass=0.6)
    p_l = _spiked(tok, lm_labels, mass=0.99)

    assert argmax_labels(fuse(p_v, p_l, 0.0, tok), tok) == vision_labels


def test_a_large_lambda_reproduces_the_lm_argmax(tok):
    """Invariant B, with vision pulling hard the other way."""
    emit = sorted(tok.vision_emit_classes)
    vision_labels, lm_labels = emit[:20], emit[20:40]

    p_v = _spiked(tok, vision_labels, mass=0.99)
    p_l = _spiked(tok, lm_labels, mass=0.6)

    assert argmax_labels(fuse(p_v, p_l, 1e6, tok), tok) == lm_labels


def test_the_two_ends_of_the_sweep_are_the_two_single_modality_systems(tok):
    """A and B together: the sweep runs between vision only and LM only."""
    emit = sorted(tok.vision_emit_classes)
    v_lab, l_lab = emit[:12], emit[12:24]
    p_v, p_l = _spiked(tok, v_lab, 0.7), _spiked(tok, l_lab, 0.7)

    vision_only = argmax_labels(
        fuse(p_v, _uniform_over_support(tok, 12), 1.0, tok), tok)
    lm_only = argmax_labels(
        fuse(_uniform_over_support(tok, 12), p_l, 1.0, tok), tok)

    assert argmax_labels(fuse(p_v, p_l, 0.0, tok), tok) == vision_only == v_lab
    assert argmax_labels(fuse(p_v, p_l, 1e6, tok), tok) == lm_only == l_lab


def test_a_uniform_vision_row_hands_the_decision_entirely_to_the_lm(tok):
    """At an uninformative position and lambda 1.0, the fused argmax is the LM's argmax."""
    emit = sorted(tok.vision_emit_classes)[:8]
    p_l = _spiked(tok, emit, 0.5)
    assert argmax_labels(
        fuse(_uniform_over_support(tok, 8), p_l, 1.0, tok), tok) == emit


def test_a_confident_but_wrong_detection_can_be_outvoted_as_lambda_rises(tok):
    """
    A confident but wrong detection gets outvoted as lambda rises. I check for
    a monotone crossing rather than a specific lambda.
    """
    right, wrong = sorted(tok.vision_emit_classes)[:2]
    p_v = _spiked(tok, [wrong], mass=0.55)
    p_l = _spiked(tok, [right], mass=0.95)

    got = [argmax_labels(fuse(p_v, p_l, lam, tok), tok)[0]
           for lam in (0.0, 0.25, 1.0, 4.0, 16.0)]
    assert got[0] == wrong
    assert got[-1] == right
    # Once the LM wins it must stay won; the score is linear in lambda.
    first_win = got.index(right)
    assert all(g == right for g in got[first_win:])


def test_non_support_columns_are_minus_infinity(tok):
    """So an argmax can never pick a special token."""
    scores = fuse(_uniform_over_support(tok, 2), _uniform_over_support(tok, 2),
                  1.0, tok)
    for special_id in tok.specials.values():
        assert scores[0, special_id].item() == float("-inf")
    assert torch.isfinite(scores[:, support_ids(tok)]).all()


def test_fuse_is_the_documented_arithmetic(tok):
    """Checked against log P_vision + lambda * log P_LM computed independently."""
    emit = sorted(tok.vision_emit_classes)[:4]
    p_v, p_l = _spiked(tok, emit, 0.8), _spiked(tok, emit[::-1], 0.3)
    lam = 0.7
    got = fuse(p_v, p_l, lam, tok)
    sup = support_ids(tok)
    want = torch.log(p_v[:, sup].clamp_min(LOG_FLOOR)) + \
        lam * torch.log(p_l[:, sup].clamp_min(LOG_FLOOR))
    assert torch.allclose(got[:, sup], want)


def test_a_zero_probability_is_floored_rather_than_producing_nan(tok):
    """A zero probability is floored, so no NaN row silently picks position 0."""
    sup = support_ids(tok)
    p_v = torch.zeros(1, tok.vocab_size, dtype=torch.float64)
    p_v[0, sup[0]] = 1.0
    p_l = _uniform_over_support(tok, 1)
    scores = fuse(p_v, p_l, 1.0, tok)
    assert torch.isfinite(scores[0, sup]).all()
    assert math.isclose(scores[0, sup[1]].item(),
                        math.log(LOG_FLOOR) + math.log(1.0 / len(sup)),
                        rel_tol=1e-9)


def test_a_shape_mismatch_raises_rather_than_broadcasting(tok):
    with pytest.raises(ValueError, match="shape mismatch"):
        fuse(_uniform_over_support(tok, 3), _uniform_over_support(tok, 4),
             1.0, tok)


# argmax_labels

def test_argmax_labels_returns_train_labels_not_token_ids(tok):
    """argmax_labels returns train_labels, not token ids."""
    emit = sorted(tok.vision_emit_classes)[:5]
    scores = fuse(_spiked(tok, emit, 0.9), _uniform_over_support(tok, 5),
                  0.0, tok)
    got = argmax_labels(scores, tok)
    assert got == emit
    assert all(g in tok.sign_to_id for g in got)


def test_argmax_labels_never_returns_a_special(tok):
    scores = fuse(_uniform_over_support(tok, 6), _uniform_over_support(tok, 6),
                  1.0, tok)
    assert all(lbl in tok.sign_to_id for lbl in argmax_labels(scores, tok))


# The real checkpoint (slow)

@pytest.mark.slow
def test_invariants_hold_on_the_real_checkpoint_and_a_real_block(tok):
    """
    Checks A and B on the real 175 MB checkpoint and a real test-split block,
    through fused_predictions, which is what the live pipeline calls.
    """
    from cuneiform.domain2_sequence.dataset import build_blocks, load_corpus
    from cuneiform.orchestrator.fusion import fused_predictions, load_sequence_model

    model = load_sequence_model()
    blocks = build_blocks(load_corpus("test"), tok)

    chosen = None
    for b in blocks:
        emit = [i for i, t in enumerate(b)
                if tok.is_predictable(t) and t != tok.unk_id
                and tok.id_to_sign[t] in tok.vision_emit_classes]
        if len(emit) >= 5:
            chosen = b
            break
    assert chosen is not None, "no test block carries 5 emittable signs"

    ids = torch.tensor([tok.bos_id] + chosen + [tok.eos_id], dtype=torch.long)
    dets: dict[int, Detection] = {}
    for i, t in enumerate(ids.tolist()):
        if not tok.is_predictable(t) or t == tok.unk_id:
            continue
        label = tok.id_to_sign[t]
        dets[i] = (Detection("detected", train_label=label, confidence=0.90)
                   if label in tok.vision_emit_classes else Detection("missed"))

    masked = ids.clone()
    for pos, d in dets.items():
        if not d.is_informative:
            masked[pos] = tok.mask_id

    positions, pred0, p_lm = fused_predictions(model, masked, dets, 0.0, tok)
    got = dict(zip(positions, pred0))

    # A
    for pos in positions:
        if dets[pos].is_informative:
            assert got[pos] == dets[pos].train_label

    # B
    lm_only = argmax_labels(
        fuse(torch.ones_like(p_lm) / p_lm.shape[1], p_lm, 1.0, tok), tok)
    _, pred_big, _ = fused_predictions(model, masked, dets, 1e6, tok,
                                       lm_cache=p_lm)
    assert pred_big == lm_only
