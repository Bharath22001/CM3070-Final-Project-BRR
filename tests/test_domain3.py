"""
Tests for the pure parts of Domain 3: sign verbalisation, prompt assembly and
the two faithfulness checks. Nothing here calls Ollama.

The model's output isn't reproducible, so these checks are what stand between
the dashboard and a confident invented reading. There are two because one
wasn't enough: the prose scan caught an invented sign (DUN₄) that the citation
check passed. bare_name is shared by both halves of the reference set so they
can't drift apart.

Run from the project root:
    python -m pytest tests/test_domain3.py -q
"""

from __future__ import annotations

import pytest

from cuneiform import config
from cuneiform.domain3_generative.generate import (
    PROSE_SCAN_FIELDS, PROSE_STOP_WORDS, TabletInput, build_prompt,
    classify_citations, display_tokens, gloss, metadata_block,
    normalise_citation, scan_prose,
)
from cuneiform.domain3_generative.verbalise import (
    KIND_BROKEN, KIND_SIGN, KIND_UNKNOWN, KIND_UNNAMED, SignLookup, bare_name,
    get_lookup, resolve, verbalise,
)


@pytest.fixture(scope="module")
def lookup() -> SignLookup:
    return get_lookup()


@pytest.fixture(scope="module")
def named_label(lookup) -> int:
    """A train_label the lookup gives a real OGSL sign name."""
    for key, rec in lookup.lookup.items():
        if rec.get("sign") and rec.get("status") != "damage_placeholder":
            return int(key)
    pytest.skip("no named sign in the lookup")


# bare_name

def test_bare_name_removes_restoration_brackets():
    assert bare_name("[LUGAL]") == "LUGAL"
    assert bare_name("LUGAL") == "LUGAL"


def test_bare_name_keeps_compound_pipes():
    """|SHU2.DUN4| is one sign, so the pipes stay; only restoration brackets go."""
    assert bare_name("|SHU2.DUN4|") == "|SHU2.DUN4|"
    assert bare_name("[|SHU2.DUN4|]") == "|SHU2.DUN4|"


def test_bare_name_removes_brackets_wherever_they_sit():
    """Brackets are removed wherever they sit, not only at the ends."""
    assert bare_name("A[B]C") == "ABC"


def test_generate_and_verbalise_share_one_bare_name():
    """generate and verbalise use the very same bare_name function, checked by identity."""
    from cuneiform.domain3_generative import generate, verbalise as verb_mod

    assert generate.bare_name is verb_mod.bare_name


# resolve: the four slot kinds

def test_none_resolves_to_an_unknown_position(lookup):
    s = resolve(None, index=3, lookup=lookup)
    assert s.kind == KIND_UNKNOWN
    assert s.train_label is None
    assert s.restored is False
    assert not s.is_named


def test_the_broken_label_resolves_to_a_broken_position(lookup):
    s = resolve(config.BROKEN_LABEL, lookup=lookup)
    assert s.kind == KIND_BROKEN
    assert not s.is_named


def test_a_named_sign_resolves_to_its_ogsl_name(lookup, named_label):
    s = resolve(named_label, lookup=lookup)
    assert s.kind == KIND_SIGN
    assert s.is_named
    assert s.text == lookup.record(named_label)["sign"]


def test_a_restored_sign_is_wrapped_in_restoration_brackets(lookup, named_label):
    """A restored sign is wrapped in [ ], so it never looks like a vision read."""
    plain = resolve(named_label, lookup=lookup, restored=False)
    restored = resolve(named_label, lookup=lookup, restored=True)
    assert restored.text == f"[{plain.text}]"
    assert restored.restored is True
    assert bare_name(restored.text) == plain.text


def test_a_broken_position_is_never_marked_restored(lookup):
    """A broken position is never marked restored, since Domain 2 can't predict <brk>."""
    assert resolve(config.BROKEN_LABEL, lookup=lookup, restored=True).restored is False


def test_an_unknown_position_is_never_marked_restored(lookup):
    assert resolve(None, lookup=lookup, restored=True).restored is False


def test_an_unresolvable_label_raises_in_strict_mode(lookup):
    """An unresolvable label raises in strict mode instead of producing a wrong prompt."""
    bogus = max(int(k) for k in lookup.lookup) + 10_000
    with pytest.raises(ValueError, match="vocabulary contract"):
        resolve(bogus, lookup=lookup, strict=True)


def test_an_unresolvable_label_degrades_to_unknown_when_not_strict(lookup):
    bogus = max(int(k) for k in lookup.lookup) + 10_000
    s = resolve(bogus, lookup=lookup, strict=False)
    assert s.kind == KIND_UNKNOWN
    assert s.status == "absent_from_lookup"
    assert s.restored is False


def test_the_slot_index_is_carried_through(lookup, named_label):
    assert resolve(named_label, index=17, lookup=lookup).index == 17


@pytest.mark.slow
def test_every_label_in_the_domain_2_vocabulary_resolves_strictly():
    """Every label in the Domain 2 vocabulary resolves in strict mode."""
    from cuneiform.domain2_sequence.tokenizer import SignTokenizer

    tok = SignTokenizer()
    lk = get_lookup()
    for label in sorted(tok.sign_to_id):
        slot = resolve(label, lookup=lk, strict=True)
        assert slot.kind in (KIND_SIGN, KIND_UNNAMED, KIND_BROKEN)
        assert slot.text


@pytest.mark.slow
def test_the_seven_unnameable_signs_still_carry_an_identity():
    """The seven unnameable signs fall back to their MZL number instead of going blank."""
    from cuneiform.domain2_sequence.tokenizer import SignTokenizer

    tok = SignTokenizer()
    lk = get_lookup()
    unnamed = [resolve(l, lookup=lk) for l in sorted(tok.sign_to_id)]
    unnamed = [s for s in unnamed if s.kind == KIND_UNNAMED]
    assert unnamed, "no unnameable signs found; the lookup may have changed"
    for s in unnamed:
        assert s.text.startswith("MZL") or s.text.startswith("LABEL")
        assert s.text


# verbalise

def test_verbalise_produces_one_slot_per_label(lookup, named_label):
    v = verbalise([named_label, None, config.BROKEN_LABEL], lookup=lookup)
    assert len(v.slots) == 3
    assert [s.kind for s in v.slots] == [KIND_SIGN, KIND_UNKNOWN, KIND_BROKEN]


def test_verbalise_indexes_the_slots_in_order(lookup, named_label):
    v = verbalise([named_label] * 5, lookup=lookup)
    assert [s.index for s in v.slots] == [0, 1, 2, 3, 4]


def test_verbalise_text_is_the_slot_texts_joined(lookup, named_label):
    v = verbalise([named_label, None], lookup=lookup)
    assert v.text == " ".join(s.text for s in v.slots)


def test_the_restored_mask_must_match_the_label_count(lookup, named_label):
    with pytest.raises(ValueError, match="restored mask"):
        verbalise([named_label, named_label], lookup=lookup, restored=[True])


def test_the_restored_mask_is_applied_positionally(lookup, named_label):
    v = verbalise([named_label, named_label], lookup=lookup,
                  restored=[False, True])
    assert v.slots[0].restored is False
    assert v.slots[1].restored is True
    assert v.slots[1].text.startswith("[")


def test_n_restored_counts_only_restored_positions(lookup, named_label):
    v = verbalise([named_label] * 4, lookup=lookup,
                  restored=[True, False, True, False])
    assert v.n_restored == 2


def test_an_empty_sequence_verbalises_to_nothing(lookup):
    v = verbalise([], lookup=lookup)
    assert v.slots == []
    assert v.text == ""


# display_tokens / gloss: what the model sees

def test_a_reading_is_shown_only_when_it_was_verified_canonical(lookup):
    """A reading is shown only when it is known to be the sign's own, since OGSL lists readings alphabetically."""
    with_reading = [rec for rec in lookup.lookup.values()
                    if rec.get("sign") and rec.get("readings")]
    canonical = [r for r in with_reading if r.get("canonical_reading_first")]
    uncanonical = [r for r in with_reading if not r.get("canonical_reading_first")]
    if not canonical or not uncanonical:
        pytest.skip("the lookup does not contain both cases")

    def label_of(rec):
        return next(int(k) for k, v in lookup.lookup.items() if v is rec)

    v_ok = verbalise([label_of(canonical[0])], lookup=lookup)
    v_no = verbalise([label_of(uncanonical[0])], lookup=lookup)

    assert display_tokens(v_ok)[0].endswith(f"({canonical[0]['readings'][0]})")
    assert "(" not in display_tokens(v_no)[0]


def test_display_tokens_has_one_entry_per_slot(lookup, named_label):
    v = verbalise([named_label, None, config.BROKEN_LABEL], lookup=lookup)
    assert len(display_tokens(v)) == len(v.slots)


def test_gloss_is_the_display_tokens_joined(lookup, named_label):
    v = verbalise([named_label, None], lookup=lookup)
    assert gloss(v) == " ".join(display_tokens(v))


def test_the_faithfulness_reference_set_holds_both_surfaces(lookup, named_label):
    """displayed_names holds both "[LUGAL] (lugal)" and "LUGAL (lugal)", so either citation counts as faithful."""
    v = verbalise([named_label], lookup=lookup, restored=[True])
    t = TabletInput(tablet="TEST", lines=[v])
    shown = display_tokens(v)[0]
    assert shown in t.displayed_names
    assert bare_name(shown) in t.displayed_names


def test_displayed_names_excludes_broken_and_unknown_positions(lookup):
    v = verbalise([None, config.BROKEN_LABEL], lookup=lookup)
    assert TabletInput(tablet="TEST", lines=[v]).displayed_names == set()


# normalise_citation

def test_normalise_strips_the_two_notations_that_carry_no_identity():
    assert normalise_citation("[LUGAL]") == "LUGAL"
    assert normalise_citation("|ME.U.U.U|") == "ME.U.U.U"
    assert normalise_citation("[|ME.U.U.U|]") == "ME.U.U.U"
    assert normalise_citation("  LUGAL  ") == "LUGAL"


def test_normalisation_tracks_notation_and_does_not_excuse_error():
    """
    Normalising doesn't excuse real errors: "U.U" cited against an input of
    "|U.U.U|" is still an invention.
    """
    supplied = {"|U.U.U|"}
    _, normalised, invented = classify_citations(["U.U"], supplied)
    assert normalised == []
    assert invented == ["U.U"]

    _, normalised, invented = classify_citations(["U.U.U"], supplied)
    assert normalised == [("U.U.U", "|U.U.U|")]
    assert invented == []


# classify_citations: three tiers

def test_an_exact_citation_is_exact():
    exact, normalised, invented = classify_citations(["LUGAL"], {"LUGAL"})
    assert (exact, normalised, invented) == (["LUGAL"], [], [])


def test_a_renamed_citation_lands_in_the_middle_tier_with_what_it_matched():
    """A renamed citation lands in the middle tier, paired with the form it matched."""
    exact, normalised, invented = classify_citations(["ME.U.U.U"], {"|ME.U.U.U|"})
    assert exact == []
    assert normalised == [("ME.U.U.U", "|ME.U.U.U|")]
    assert invented == []


def test_a_sign_never_shown_is_invented():
    exact, normalised, invented = classify_citations(["DUN4"], {"LUGAL"})
    assert invented == ["DUN4"]


def test_the_three_tiers_partition_the_citations():
    cited = ["LUGAL", "ME.U.U.U", "DUN4", "LUGAL"]
    supplied = {"LUGAL", "|ME.U.U.U|"}
    exact, normalised, invented = classify_citations(cited, supplied)
    assert len(exact) + len(normalised) + len(invented) == len(cited)


def test_no_citations_gives_three_empty_lists():
    assert classify_citations([], {"LUGAL"}) == ([], [], [])


# scan_prose

def test_an_invented_sign_in_the_prose_is_reported():
    """An invented sign in the prose is reported even when the citation list is clean."""
    out = {"tentative_reading": "a record mentioning DUN and LUGAL",
           "uncertainty_notes": ""}
    got = scan_prose(out, {"LUGAL"})
    assert "DUN" in got["unmatched"]
    assert "LUGAL" not in got["unmatched"]


def test_a_supplied_sign_in_the_prose_is_not_reported():
    out = {"tentative_reading": "LUGAL appears twice", "uncertainty_notes": ""}
    assert scan_prose(out, {"LUGAL"})["unmatched"] == []


def test_an_unpiped_compound_in_the_prose_is_matched_rather_than_split():
    """
    An unpiped compound like "ME.U.U.U" in the prose is matched whole, not split
    into fragments that look like invented signs. Both audits now share one
    sign pattern.
    """
    out = {"tentative_reading": "the sign ME.U.U.U is present",
           "uncertainty_notes": ""}
    assert scan_prose(out, {"|ME.U.U.U|"})["unmatched"] == [], (
        "an unpiped compound is still being split into fragments and reported "
        "as invented signs"
    )

    # The citation check has always handled this correctly.
    _, normalised, invented = classify_citations(["ME.U.U.U"], {"|ME.U.U.U|"})
    assert normalised and not invented


def test_a_compound_that_disappeared_entirely_is_now_seen():
    """|U.U.U| (all one-character fragments) is now seen at all."""
    out = {"tentative_reading": "the sign U.U.U is present", "uncertainty_notes": ""}
    got = scan_prose(out, {"|U.U.U|"})
    assert "U.U.U" in got["candidates"], "the compound is still invisible to the scan"
    assert got["unmatched"] == []


def test_a_dotted_english_initialism_is_not_read_as_a_sign():
    """Dotted initialisms like U.S and B.C are not read as signs, and a sentence's full stop is ignored."""
    out = {"tentative_reading": "a tablet from the U.S. collection, dated 700 B.C.",
           "uncertainty_notes": "compare the U.K. holdings"}
    assert scan_prose(out, set())["unmatched"] == []

    # Also list the stop words by name, because the parametrised test below would
    # silently lose a case if a word were deleted from the set.
    for word in ("US", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI",
                 "XII", "U.S", "U.K", "U.S.A", "A.D", "B.C", "B.C.E", "C.E",
                 "N.B", "NAME"):
        assert word in PROSE_STOP_WORDS, (
            f"{word!r} has been dropped from the shared stop-word set; the "
            f"widened pattern matches it and the summary pane will report it "
            f"as an invented sign"
        )


def test_a_piped_compound_in_the_prose_is_matched_correctly():
    """A piped compound in the prose is matched correctly."""
    out = {"tentative_reading": "the sign |ME.U.U.U| is present",
           "uncertainty_notes": ""}
    assert scan_prose(out, {"|ME.U.U.U|"})["unmatched"] == []


def test_our_own_restoration_marker_is_not_read_as_a_citation():
    """NAME, my own restoration marker, is never reported as a sign citation."""
    out = {"tentative_reading": "several positions marked [NAME]",
           "uncertainty_notes": ""}
    assert scan_prose(out, set())["unmatched"] == []


@pytest.mark.parametrize("word", sorted(PROSE_STOP_WORDS))
def test_every_stop_word_is_suppressed(word):
    out = {"tentative_reading": f"mentioning {word} here", "uncertainty_notes": ""}
    assert word not in scan_prose(out, set())["unmatched"]


def test_a_compound_sign_in_the_prose_is_matched_as_one_token():
    out = {"tentative_reading": "the compound |SHU2.DUN4| occurs",
           "uncertainty_notes": ""}
    assert scan_prose(out, {"|SHU2.DUN4|"})["unmatched"] == []


def test_subscript_digits_are_part_of_a_sign_name():
    """Subscript digits (as in LUGAL₂) are part of a sign name."""
    out = {"tentative_reading": "the sign LUGAL₂ occurs",
           "uncertainty_notes": ""}
    assert scan_prose(out, {"LUGAL₂"})["unmatched"] == []


def test_lowercase_prose_is_not_mistaken_for_sign_names():
    out = {"tentative_reading": "this appears to be a receipt for barley",
           "uncertainty_notes": "the reading is uncertain"}
    assert scan_prose(out, set())["unmatched"] == []


def test_the_scan_names_the_fields_it_read():
    """The result lists the fields it scanned and every candidate, not just a verdict."""
    got = scan_prose({"tentative_reading": "", "uncertainty_notes": ""}, set())
    assert got["fields"] == list(PROSE_SCAN_FIELDS)
    assert set(got) == {"fields", "candidates", "unmatched"}


def test_the_scan_ignores_fields_outside_its_declared_list():
    """Fields outside the declared list (like historical_context) are not scanned."""
    out = {"tentative_reading": "", "uncertainty_notes": "",
           "historical_context": "mentions DUN4 at length"}
    assert scan_prose(out, set())["unmatched"] == []


def test_a_missing_field_does_not_raise():
    assert scan_prose({}, set())["unmatched"] == []


def test_candidates_and_unmatched_are_sorted():
    out = {"tentative_reading": "ZZZ AAA MMM", "uncertainty_notes": ""}
    got = scan_prose(out, set())
    assert got["candidates"] == sorted(got["candidates"])
    assert got["unmatched"] == sorted(got["unmatched"])


# metadata_block / build_prompt

def test_a_tablet_with_no_catalogue_record_gets_an_explicit_refusal_to_speculate():
    """A tablet with no catalogue record gets an explicit instruction not to speculate."""
    text, had = metadata_block("P000000", {})
    assert had is False
    assert "none available" in text
    assert "Do not speculate" in text


def test_only_populated_catalogue_fields_are_emitted():
    cat = {"P1": {"designation": "SAA 01 001", "period": None,
                  "provenience": "", "genre": []}}
    text, had = metadata_block("P1", cat)
    assert had is True
    assert "SAA 01 001" in text
    assert "None" not in text
    assert text.count("\n") == 1


def test_the_prompt_reports_restored_positions_as_less_certain(lookup, named_label):
    """Restored signs are reported to the model as less certain, not counted as legible."""
    v = verbalise([named_label] * 4, lookup=lookup,
                  restored=[True, True, False, False])
    prompt, _ = build_prompt(TabletInput(tablet="TEST", lines=[v]), {})
    assert "2 (50%) were read directly from the tablet." in prompt
    assert "2 were restored by the sequence model [NAME] and are less certain." \
        in prompt


def test_the_ground_truth_wording_is_byte_identical_when_nothing_is_restored(
        lookup, named_label):
    """With nothing restored, the prompt wording is exactly the original, so earlier results stay comparable."""
    v = verbalise([named_label] * 4, lookup=lookup)
    prompt, _ = build_prompt(TabletInput(tablet="TEST", lines=[v]), {})
    assert "4 (100%) carry a legible sign name." in prompt
    assert "restored by the sequence model" not in prompt


def test_the_prompt_numbers_the_lines_from_one(lookup, named_label):
    lines = [verbalise([named_label], lookup=lookup) for _ in range(3)]
    prompt, _ = build_prompt(TabletInput(tablet="TEST", lines=lines), {})
    assert "    1: " in prompt
    assert "    3: " in prompt
    assert "3 lines, 3 sign positions" in prompt


def test_the_prompt_counts_broken_and_unidentified_separately(lookup, named_label):
    v = verbalise([named_label, config.BROKEN_LABEL, None, None], lookup=lookup)
    prompt, _ = build_prompt(TabletInput(tablet="TEST", lines=[v]), {})
    assert "1 are broken [...] and 2 unidentified [x]." in prompt


# TabletInput rates

def test_restored_rate_is_the_fraction_named_by_domain_2(lookup, named_label):
    v = verbalise([named_label] * 4, lookup=lookup,
                  restored=[True, False, False, False])
    t = TabletInput(tablet="TEST", lines=[v])
    assert t.n_restored == 1
    assert t.restored_rate == pytest.approx(0.25)


def test_damage_rate_excludes_restorations_and_is_the_wrong_number_live(
        lookup, named_label):
    """
    damage_rate leaves out restorations, so on a live tablet it can read 0.0
    while many positions are guesses; restored_rate has to be reported with it.
    """
    v = verbalise([named_label] * 4, lookup=lookup, restored=[True] * 4)
    t = TabletInput(tablet="TEST", lines=[v])
    assert t.damage_rate == 0.0
    assert t.restored_rate == 1.0


def test_the_rates_are_zero_rather_than_undefined_on_an_empty_tablet():
    t = TabletInput(tablet="TEST", lines=[])
    assert t.damage_rate == 0.0
    assert t.restored_rate == 0.0
