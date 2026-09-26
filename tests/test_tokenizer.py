"""
Tests for SignTokenizer. The sign_to_id / id_to_sign round trip is the most
important invariant here, since a mismatch would silently return wrong labels.
These tests only read vocab.json and never rebuild it, and the exact sizes
(288 tokens, 282 signs, 82 emit classes) are pinned because the vocabulary is frozen.

Run from the project root:
    python -m pytest tests/test_tokenizer.py -q
"""

from __future__ import annotations

import json

import pytest

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import VOCAB_JSON, SignTokenizer


@pytest.fixture(scope="module")
def tok() -> SignTokenizer:
    return SignTokenizer()


# The frozen shape

def test_vocabulary_is_the_frozen_288_token_layout(tok):
    """The Domain 2 vocabulary is frozen at 288 tokens."""
    assert tok.vocab_size == 288
    assert len(tok.sign_to_id) == 282
    assert len(tok.id_to_sign) == 282
    assert len(tok.vision_emit_classes) == 82


def test_specials_occupy_ids_zero_to_five_in_the_documented_order(tok):
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3
    assert tok.mask_id == 4
    assert tok.brk_id == 5


def test_sign_tokens_start_after_the_specials(tok):
    """Ids 6..287 are signs; no sign may collide with a special."""
    assert min(tok.id_to_sign) == 6
    assert max(tok.id_to_sign) == 287
    assert not (set(tok.id_to_sign) & set(tok.specials.values()))


def test_emit_classes_are_a_subset_of_the_sign_vocabulary(tok):
    """Every emit class must have a token, or vision_posterior would silently drop its vote."""
    assert tok.vision_emit_classes <= set(tok.sign_to_id)


# The round trip

def test_sign_to_id_and_id_to_sign_are_exact_inverses(tok):
    for sign, tid in tok.sign_to_id.items():
        assert tok.id_to_sign[tid] == sign
    for tid, sign in tok.id_to_sign.items():
        assert tok.sign_to_id[sign] == tid


def test_encode_line_round_trips_every_sign_in_the_vocabulary(tok):
    signs = sorted(tok.sign_to_id)
    ids = tok.encode_line(signs)
    assert [tok.id_to_sign[i] for i in ids] == signs


# encode_sign: the three cases

def test_broken_label_encodes_to_brk_not_to_unk(tok):
    """<brk> is damage with no ground truth; <unk> is a real but rare sign. They must stay separate."""
    assert tok.encode_sign(config.BROKEN_LABEL) == tok.brk_id
    assert tok.brk_id != tok.unk_id


def test_a_sign_outside_the_vocabulary_encodes_to_unk(tok):
    unseen = max(tok.sign_to_id) + 10_000
    assert unseen not in tok.sign_to_id
    assert tok.encode_sign(unseen) == tok.unk_id


def test_a_sign_inside_the_vocabulary_encodes_to_its_own_token(tok):
    sign = sorted(tok.sign_to_id)[0]
    assert tok.encode_sign(sign) == tok.sign_to_id[sign]


# is_predictable: what may become a masking target

@pytest.mark.parametrize("attr", ["pad_id", "bos_id", "eos_id", "mask_id", "brk_id"])
def test_specials_that_must_never_be_prediction_targets(tok, attr):
    assert not tok.is_predictable(getattr(tok, attr))


def test_unk_is_predictable_but_every_other_special_is_not(tok):
    """
    <unk> can be a masking target (it's a real sign); fusion excludes it
    separately at prediction time. The two rules must stay different.
    """
    assert tok.is_predictable(tok.unk_id)
    assert tok.non_target_ids == {
        tok.pad_id, tok.bos_id, tok.eos_id, tok.mask_id, tok.brk_id,
    }


def test_every_sign_token_is_predictable(tok):
    assert all(tok.is_predictable(i) for i in tok.id_to_sign)


# decode

def test_decode_names_the_specials_and_the_signs(tok):
    assert tok.decode_id(tok.mask_id) == "<mask>"
    assert tok.decode_id(tok.brk_id) == "<brk>"
    first_sign_id = min(tok.id_to_sign)
    assert tok.decode_id(first_sign_id) == f"sign_{tok.id_to_sign[first_sign_id]}"


def test_decode_id_flags_an_out_of_range_token_rather_than_crashing(tok):
    assert tok.decode_id(tok.vocab_size + 5) == f"<invalid:{tok.vocab_size + 5}>"


def test_decode_maps_a_whole_sequence(tok):
    ids = [tok.bos_id, min(tok.id_to_sign), tok.mask_id, tok.eos_id]
    assert tok.decode(ids) == [
        "<s>", f"sign_{tok.id_to_sign[min(tok.id_to_sign)]}", "<mask>", "</s>",
    ]


# Construction

def test_a_missing_vocabulary_file_raises_rather_than_producing_an_empty_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        SignTokenizer(vocab_path=tmp_path / "nope.json")


def test_the_constructor_reads_the_file_it_is_given(tmp_path, tok):
    """A shrunk copy of vocab.json must change what the object reports, so the tests above really read the file."""
    with open(VOCAB_JSON, "r", encoding="utf-8") as fh:
        v = json.load(fh)
    v["vocab_size"] = 999
    mutant = tmp_path / "vocab.json"
    with open(mutant, "w", encoding="utf-8") as fh:
        json.dump(v, fh)

    assert SignTokenizer(vocab_path=mutant).vocab_size == 999
    assert tok.vocab_size == 288          # the real one is untouched


def test_repr_reports_the_three_sizes(tok):
    r = repr(tok)
    assert "288" in r and "282" in r and "82" in r
