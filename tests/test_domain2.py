"""
Tests for the Domain 2 package: the corpus loader, span measurement, block
packing, the masking dataset and the model builder.

Nothing here runs build_vocab.py, trains, or writes into artifacts/domain2/.
build_model(pretrained=False) is tested only as the ablation arm; it is never
the deployed model. The key dataset checks: labels are -100 except where a
token was hidden, <brk> is never a target, and masked spans never merge.

Run from the project root:
    python -m pytest tests/test_domain2.py -q
"""

from __future__ import annotations

import pytest
import torch

from cuneiform import config
from cuneiform.domain2_sequence.dataset import (
    MASK_RATE, MAX_LEN, MAX_SPAN, CuneiformMLMDataset, build_blocks,
    load_corpus, measure_span_distribution,
)
from cuneiform.domain2_sequence.model import count_parameters
from cuneiform.domain2_sequence.tokenizer import SignTokenizer


@pytest.fixture(scope="module")
def tok() -> SignTokenizer:
    return SignTokenizer()


def _rec(tablet, view, segm, line, tokens):
    return {"tablet": tablet, "view": view, "segm_idx": segm,
            "line_idx": line, "tokens": list(tokens), "split": "test"}


# The training constants

def test_the_training_regime_constants_are_the_deployed_ones():
    """MAX_LEN matches live_pipeline.MAX_LEN; MAX_SPAN is the longest real damage run."""
    from cuneiform.orchestrator.live_pipeline import MAX_LEN as LIVE_MAX_LEN

    assert MAX_LEN == 128 == LIVE_MAX_LEN
    assert MASK_RATE == 0.25
    assert MAX_SPAN == 27


# measure_span_distribution

def test_a_corpus_with_no_damage_falls_back_to_single_token_spans():
    dist = measure_span_distribution([_rec("T", "obv", 0, 0, [7, 8, 9])])
    assert dist == {1: 1.0}


def test_runs_of_broken_signs_are_counted_by_length():
    B = config.BROKEN_LABEL
    recs = [_rec("T", "obv", 0, 0, [1, B, 2, B, B, 3])]
    dist = measure_span_distribution(recs)
    assert dist == {1: 0.5, 2: 0.5}


def test_a_run_reaching_the_end_of_a_line_is_still_counted():
    """A damage run that reaches the end of a line is still counted."""
    B = config.BROKEN_LABEL
    dist = measure_span_distribution([_rec("T", "obv", 0, 0, [1, B, B, B])])
    assert dist == {3: 1.0}


def test_runs_longer_than_max_span_are_discarded():
    B = config.BROKEN_LABEL
    recs = [_rec("T", "obv", 0, 0, [1] + [B] * (MAX_SPAN + 5) + [1]),
            _rec("T", "obv", 0, 1, [1, B, 1])]
    dist = measure_span_distribution(recs)
    assert dist == {1: 1.0}
    assert all(k <= MAX_SPAN for k in dist)


def test_the_distribution_is_normalised_and_sorted():
    B = config.BROKEN_LABEL
    recs = [_rec("T", "obv", 0, i, [1, B, 1, B, B, 1]) for i in range(3)]
    dist = measure_span_distribution(recs)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert list(dist) == sorted(dist)
    assert all(v > 0 for v in dist.values())


def test_runs_do_not_bridge_two_lines():
    """A run at the end of one line and one at the start of the next are two runs."""
    B = config.BROKEN_LABEL
    recs = [_rec("T", "obv", 0, 0, [1, B]), _rec("T", "obv", 0, 1, [B, 1])]
    assert measure_span_distribution(recs) == {1: 1.0}


# build_blocks

def test_lines_from_one_segment_are_packed_together(tok):
    recs = [_rec("T", "obv", 0, i, [7] * 10) for i in range(3)]
    blocks = build_blocks(recs, tok)
    assert [len(b) for b in blocks] == [30]


def test_a_block_never_spans_two_tablets(tok):
    """A block never spans two tablets."""
    recs = [_rec("A", "obv", 0, 0, [7] * 10), _rec("B", "obv", 0, 0, [8] * 10)]
    blocks = build_blocks(recs, tok)
    assert len(blocks) == 2
    assert all(len(set(b)) == 1 for b in blocks)


def test_a_block_never_spans_two_views_or_two_segments(tok):
    recs = [_rec("A", "obv", 0, 0, [7] * 10), _rec("A", "rev", 0, 0, [8] * 10),
            _rec("A", "obv", 1, 0, [9] * 10)]
    assert len(build_blocks(recs, tok)) == 3


def test_lines_are_packed_in_line_index_order(tok):
    """Records supplied out of order still pack in line order."""
    recs = [_rec("T", "obv", 0, 2, [9, 9]), _rec("T", "obv", 0, 0, [7, 7]),
            _rec("T", "obv", 0, 1, [8, 8])]
    blocks = build_blocks(recs, tok)
    ids = [tok.encode_sign(s) for s in (7, 7, 8, 8, 9, 9)]
    assert blocks == [ids]


def test_a_run_shorter_than_four_tokens_is_dropped(tok):
    assert build_blocks([_rec("T", "obv", 0, 0, [7, 8, 9])], tok) == []


def test_an_empty_line_is_skipped_without_breaking_the_block(tok):
    recs = [_rec("T", "obv", 0, 0, [7] * 5), _rec("T", "obv", 0, 1, []),
            _rec("T", "obv", 0, 2, [8] * 5)]
    assert [len(b) for b in build_blocks(recs, tok)] == [10]


def test_the_budget_leaves_room_for_the_two_sentinels(tok):
    budget = MAX_LEN - 2
    recs = [_rec("T", "obv", 0, 0, [7] * 500)]
    blocks = build_blocks(recs, tok)
    assert max(len(b) for b in blocks) == budget
    assert max(len(b) for b in blocks) + 2 <= MAX_LEN


def test_an_overlong_line_is_split_and_short_tails_are_dropped(tok):
    budget = MAX_LEN - 2
    recs = [_rec("T", "obv", 0, 0, [7] * (budget + 2))]
    assert [len(b) for b in build_blocks(recs, tok)] == [budget]


def test_a_line_that_does_not_fit_starts_the_next_block_intact(tok):
    budget = MAX_LEN - 2
    recs = [_rec("T", "obv", 0, 0, [7] * (budget - 10)),
            _rec("T", "obv", 0, 1, [8] * 30)]
    assert [len(b) for b in build_blocks(recs, tok)] == [budget - 10, 30]


def test_blocks_contain_token_ids_not_train_labels(tok):
    """Blocks hold token ids, not train_labels."""
    sign = sorted(tok.sign_to_id)[0]
    recs = [_rec("T", "obv", 0, 0, [sign] * 8)]
    block = build_blocks(recs, tok)[0]
    assert set(block) == {tok.sign_to_id[sign]}
    assert all(0 <= t < tok.vocab_size for t in block)


def test_broken_signs_become_brk_tokens_inside_a_block(tok):
    recs = [_rec("T", "obv", 0, 0, [config.BROKEN_LABEL] * 8)]
    assert set(build_blocks(recs, tok)[0]) == {tok.brk_id}


# load_corpus

@pytest.mark.slow
def test_the_three_splits_are_disjoint_and_non_empty():
    """Train, val and test are disjoint and non-empty (read-only against the real corpus)."""
    splits = {s: load_corpus(s) for s in ("train", "val", "test")}
    for name, recs in splits.items():
        assert recs, f"split {name} is empty"

    tablets = {name: {r["tablet"] for r in recs} for name, recs in splits.items()}
    assert not (tablets["train"] & tablets["test"])
    assert not (tablets["train"] & tablets["val"])
    assert not (tablets["val"] & tablets["test"])


@pytest.mark.slow
def test_every_corpus_record_carries_the_fields_the_block_packer_reads():
    needed = {"tablet", "view", "segm_idx", "line_idx", "tokens", "split"}
    for split in ("train", "val", "test"):
        for rec in load_corpus(split):
            assert needed <= set(rec)
            assert isinstance(rec["tokens"], list)


def test_an_unknown_split_gives_no_records():
    assert load_corpus("not_a_split") == []


# CuneiformMLMDataset

@pytest.fixture(scope="module")
def dataset(tok) -> CuneiformMLMDataset:
    return CuneiformMLMDataset("test", tok, deterministic=True)


@pytest.mark.slow
def test_every_example_is_a_triple_of_equal_length_tensors(dataset):
    ids, attn, labels = dataset[0]
    assert ids.shape == attn.shape == labels.shape
    assert ids.dtype == labels.dtype == torch.long


@pytest.mark.slow
def test_the_loss_is_computed_only_on_positions_that_were_actually_hidden(dataset):
    """Labels are -100 everywhere except the positions that were actually masked."""
    for i in range(min(60, len(dataset))):
        ids, _, labels = dataset[i]
        scored = (labels != -100)
        assert (ids[scored] == dataset.tok.mask_id).all()
        assert (labels[~scored] == -100).all()


@pytest.mark.slow
def test_known_limitation_a_small_fraction_of_blocks_carry_no_training_signal():
    """
    A known limitation: about 1.7% of blocks come back with every label at
    -100, because they are mostly <brk> and no span fits. Measured:

        train   64 of 3,847 blocks   1.66%
        val      9 of   430 blocks   2.09%
        test     1 of    74 blocks   1.35%

    Nothing breaks and no published figure changes. Fixing it would mean
    retraining Domain 2, so it stays.
    """
    tok = SignTokenizer()
    ds = CuneiformMLMDataset("test", tok, deterministic=True)
    empty = sum(1 for i in range(len(ds)) if not (ds[i][2] != -100).any())
    assert empty / len(ds) < 0.05, (
        f"{empty} of {len(ds)} test blocks carry no training signal, which is "
        f"far above the 1.35% recorded above"
    )
    assert empty > 0, (
        "no empty block found on the test split; the measurement above said "
        "there is exactly one, so either the masking or the corpus has moved"
    )


@pytest.mark.slow
def test_the_label_at_a_masked_position_is_the_token_that_was_there(dataset):
    for i in range(min(20, len(dataset))):
        ids, _, labels = dataset[i]
        body = [dataset.tok.bos_id] + list(dataset.blocks[i]) + [dataset.tok.eos_id]
        for pos in (labels != -100).nonzero().flatten().tolist():
            assert labels[pos].item() == body[pos]


@pytest.mark.slow
def test_a_broken_position_is_never_a_prediction_target(dataset):
    """<brk> is never a prediction target, since nobody knows what the sign was."""
    for i in range(min(40, len(dataset))):
        ids, _, labels = dataset[i]
        for pos in (labels != -100).nonzero().flatten().tolist():
            assert labels[pos].item() not in dataset.tok.non_target_ids


@pytest.mark.slow
def test_the_sentinels_are_never_masked(dataset):
    for i in range(min(40, len(dataset))):
        ids, _, labels = dataset[i]
        assert labels[0].item() == -100
        assert ids[0].item() == dataset.tok.bos_id


@pytest.mark.slow
def test_the_attention_mask_covers_the_real_tokens(dataset):
    for i in range(min(20, len(dataset))):
        ids, attn, _ = dataset[i]
        assert attn[ids != dataset.tok.pad_id].all()


@pytest.mark.slow
def test_deterministic_mode_gives_the_same_masking_on_every_access(dataset):
    """Two reads of the same index give identical masking in deterministic mode."""
    a = dataset[3]
    b = dataset[3]
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.slow
def test_non_deterministic_mode_resamples(tok):
    ds = CuneiformMLMDataset("test", tok, deterministic=False)
    seen = {tuple((ds[0][2] != -100).nonzero().flatten().tolist())
            for _ in range(12)}
    assert len(seen) > 1, "the training dataset is not resampling its masks"


@pytest.mark.slow
def test_the_realised_mask_rate_is_near_the_target(dataset):
    """The realised mask rate is near the target (a wide band, since spans are placed whole)."""
    hidden = total = 0
    for i in range(len(dataset)):
        ids, _, labels = dataset[i]
        hidden += int((labels != -100).sum())
        total += int(sum(1 for t in ids.tolist()
                         if dataset.tok.is_predictable(t) or t == dataset.tok.mask_id))
    assert 0.10 <= hidden / max(1, total) <= 0.45


@pytest.mark.slow
def test_masked_runs_stay_isolated_from_one_another(dataset):
    """Masked runs stay separate, so every run length appears in the measured distribution."""
    allowed = set(dataset.span_lengths)
    for i in range(min(60, len(dataset))):
        _, _, labels = dataset[i]
        marks = (labels != -100).tolist()
        run = 0
        for m in marks + [False]:
            if m:
                run += 1
            elif run:
                assert run in allowed, (
                    f"example {i} realised a run of {run}, which is not a "
                    f"length the measured span distribution contains"
                )
                run = 0


# count_parameters

def test_count_parameters_splits_the_embedding_from_the_body():
    class _Fake:
        def __init__(self):
            self.emb = torch.nn.Embedding(10, 4)      # 40
            self.lin = torch.nn.Linear(4, 4, bias=False)   # 16

        def parameters(self):
            return list(self.emb.parameters()) + list(self.lin.parameters())

        def get_input_embeddings(self):
            return self.emb

    got = count_parameters(_Fake())
    assert got == {"total": 56, "embedding": 40, "transformer": 16}


# build_model: both arms

@pytest.mark.slow
def test_the_pretrained_arm_keeps_the_body_and_replaces_the_vocabulary(tok):
    """The deployed arm keeps distilroberta-base's body and replaces the vocabulary."""
    from cuneiform.domain2_sequence.model import BASE_CHECKPOINT, build_model

    model = build_model(tok, pretrained=True)
    assert model.config.vocab_size == tok.vocab_size
    assert model.get_input_embeddings().weight.shape[0] == tok.vocab_size
    assert "distilroberta" in BASE_CHECKPOINT


@pytest.mark.slow
def test_our_special_token_ids_reach_both_the_config_and_the_embedding(tok):
    """
    My special-token ids reach both the config and the embedding module;
    otherwise RoBERTa would compute position ids from the wrong pad id.
    """
    from cuneiform.domain2_sequence.model import build_model

    model = build_model(tok, pretrained=True)
    assert model.config.pad_token_id == tok.pad_id
    assert model.config.bos_token_id == tok.bos_id
    assert model.config.eos_token_id == tok.eos_id
    assert model.config.mask_token_id == tok.mask_id
    assert model.get_input_embeddings().padding_idx == tok.pad_id
    assert model.roberta.embeddings.padding_idx == tok.pad_id


@pytest.mark.slow
def test_the_pad_embedding_row_is_zeroed(tok):
    from cuneiform.domain2_sequence.model import build_model

    model = build_model(tok, pretrained=True)
    assert torch.allclose(model.get_input_embeddings().weight[tok.pad_id],
                          torch.zeros(model.config.hidden_size))


@pytest.mark.slow
def test_the_random_init_arm_is_an_ablation_and_differs_from_the_pretrained_one(tok):
    """
    The random-init arm is only for the ablation and is never deployed. It
    shares the architecture but has different body weights.
    """
    from cuneiform.domain2_sequence.model import build_model

    pre = build_model(tok, pretrained=True)
    rnd = build_model(tok, pretrained=False)

    assert pre.config.num_hidden_layers == rnd.config.num_hidden_layers
    assert pre.config.hidden_size == rnd.config.hidden_size
    assert count_parameters(pre)["total"] == count_parameters(rnd)["total"]

    a = pre.roberta.encoder.layer[0].attention.self.query.weight
    b = rnd.roberta.encoder.layer[0].attention.self.query.weight
    assert not torch.allclose(a, b), "the ablation arm kept the pretrained body"
