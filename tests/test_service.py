"""
Checks that the service layer gives exactly the same reading as running
`python -m cuneiform.orchestrator.live_pipeline` in a separate process.

I compare against the real CLI every run instead of a saved fixture, so if the
pipeline changes and the service doesn't follow, this fails. The comparison is
the sequence dump, the same strings the browser shows, decoded as UTF-8.

Run from the project root:
    python -m pytest tests/test_service.py -q
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cuneiform.api import service

ROOT = Path(__file__).resolve().parent.parent
TABLET = "P335940"

# The header the CLI prints above each row of the dump.
HEAD = re.compile(r"^  segment (\d+)  line (\d+)$")


def _run_cli(tablet: str) -> str:
    """The CLI's stdout, decoded as UTF-8. Raises if the CLI itself failed."""
    proc = subprocess.run(
        [sys.executable, "-m", "cuneiform.orchestrator.live_pipeline",
         "--tablet", tablet],
        cwd=str(ROOT), capture_output=True, timeout=600,
    )
    assert proc.returncode == 0, (
        "the CLI itself failed, so there is nothing to compare against:\n"
        + proc.stderr.decode("utf-8", "replace")[-2000:]
    )
    return proc.stdout.decode("utf-8", "replace")


def _parse_cli_dump(text: str) -> list[dict]:
    """
    Pull the sequence dump out of the CLI's stdout. Each row's tokens follow
    its header.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.strip().startswith("THE FUSED SEQUENCE"))
    except StopIteration:
        pytest.fail("the CLI printed no 'THE FUSED SEQUENCE' block")

    rows: list[dict] = []
    pending: tuple[int, int] | None = None
    for l in lines[start:]:
        m = HEAD.match(l)
        if m:
            pending = (int(m.group(1)), int(m.group(2)))
            continue
        if l.startswith("  Legend:"):
            break
        if pending is not None and l.startswith("    ") and l.strip():
            seg, ln = pending
            rows.append({"segment": seg, "line": ln, "text": l.strip()})
            pending = None
    return rows


@pytest.fixture(scope="module")
def cli_dump() -> list[dict]:
    rows = _parse_cli_dump(_run_cli(TABLET))
    # An empty parse would let every assertion pass vacuously.
    assert rows, "parsed 0 rows out of the CLI dump -- the parser is broken"
    return rows


@pytest.fixture(scope="module")
def result() -> dict:
    return service.analyse_tablet(TABLET)


# Acceptance checks

def test_sequence_matches_the_cli_sign_for_sign(result, cli_dump):
    """Compared row by row so a failure names the row."""
    svc = result["sequence_dump"]

    assert len(svc) == len(cli_dump), (
        f"row count differs: service {len(svc)}, CLI {len(cli_dump)}"
    )

    mismatches = []
    for a, b in zip(svc, cli_dump):
        if (a["segment"], a["line"]) != (b["segment"], b["line"]):
            mismatches.append(f"header {a['segment']}/{a['line']} "
                              f"vs {b['segment']}/{b['line']}")
        elif a["text"] != b["text"]:
            mismatches.append(
                f"segment {a['segment']} line {a['line']}:\n"
                f"    service: {a['text']}\n"
                f"    CLI    : {b['text']}"
            )

    assert not mismatches, (
        f"{len(mismatches)} of {len(svc)} rows differ:\n"
        + "\n".join(mismatches[:10])
    )


def test_result_is_json_serialisable(result):
    """The result must be JSON-serialisable (it once held Verbalisation objects)."""
    blob = json.dumps(result, ensure_ascii=False)
    assert json.loads(blob) == result


def test_returns_under_ten_seconds():
    """
    A warm analysis must take under ten seconds, since the no-job-queue design
    depends on it. The server warms the model at startup.
    """
    service.warm()
    t0 = time.perf_counter()
    service.analyse_tablet(TABLET)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"analyse_tablet took {elapsed:.2f}s"


def test_defaults_are_the_frozen_configuration(result):
    """A non-frozen operating point would mislead if the UI didn't say so."""
    cfg = result["config"]
    assert cfg["is_frozen"] is True
    assert cfg["conf"] == 0.15
    assert cfg["cell"] == 2.0
    assert cfg["lambda"] == 1.0
    assert cfg["lm_floor"] == 0.08
    assert cfg["min_cluster"] == 3
    assert cfg["edge_extrapolation"] is False


def test_every_detected_slot_found_its_box(result):
    """Every detected slot must find its box, or the image pane would draw it in the wrong place."""
    assert result["counts"]["box_match_misses"] == 0


def test_provenance_states_are_closed_and_consistent(result):
    """Each slot's rendered token must agree with its declared state."""
    seen = set()
    for row in result["lines"]:
        for s in row["slots"]:
            seen.add(s["state"])
            tok, st = s["token"], s["state"]
            if st == "vision":
                assert not tok.startswith("[") and not tok.endswith("*")
            elif st == "override":
                assert tok.endswith("*")
            elif st == "restored":
                assert tok.startswith("[") and tok != "[x]"
                assert s["label"] is not None
            elif st == "unnamed":
                assert tok in ("[x]", "·")
                assert s["label"] is None
                assert s["unnamed_reason"] in ("below_floor", "outside_block")
            else:
                pytest.fail(f"unknown provenance state {st!r}")
    assert seen <= {"vision", "override", "restored", "unnamed"}


def test_no_pane_payload_says_translation(result):
    """The caveats are in the payload and never call the output a translation."""
    assert result["caveats"], "caveats must travel with the result"
    assert any("does not translate" in c for c in result["caveats"])
    assert any("22.2%" in c for c in result["caveats"])


def test_population_is_the_thirty_test_tablets():
    """
    The UI's tablets must be exactly the evaluation's 30. I check both the count
    and that list_tablets delegates to evaluate_live.live_tablets.
    """
    from cuneiform.orchestrator.evaluate_live import live_tablets

    tablets = service.list_tablets()
    assert len(tablets) == 30, f"expected 30 tablets, found {len(tablets)}"
    assert TABLET in tablets
    assert "_summary" not in tablets
    assert tablets == live_tablets()


def test_every_tablet_has_an_image_on_disk():
    """Every one of the 30 tablets has a photograph on disk."""
    missing = [t for t in service.list_tablets()
               if not service.image_path(t).exists()]
    assert not missing, f"no source image for: {missing}"


def test_unknown_tablet_raises_rather_than_returning_empty():
    with pytest.raises(FileNotFoundError):
        service.analyse_tablet("P000000")


# Domain 3 plumbing (no language model is called)

def test_the_prose_scan_has_exactly_one_implementation():
    """The CLI and the API must call the same prose scan function, not a copy."""
    from cuneiform.domain3_generative import generate
    from cuneiform.orchestrator import live_pipeline

    assert live_pipeline.scan_prose is generate.scan_prose


def test_the_prose_scan_still_reports_a_sign_that_was_never_supplied():
    """
    Stop words (NAME, JSON) and an unpiped compound that was supplied aren't
    reported; an uppercase sign that was never supplied is.
    """
    from cuneiform.domain3_generative.generate import scan_prose

    out = {"tentative_reading": "the [NAME] marker, JSON, |ME.U.U.U| and GESZ",
           "uncertainty_notes": ""}
    scan = scan_prose(out, {"|ME.U.U.U|"})

    assert "GESZ" in scan["unmatched"]
    assert "NAME" not in scan["unmatched"]
    assert "JSON" not in scan["unmatched"]
    assert "|ME.U.U.U|" not in scan["unmatched"]


def test_the_ollama_client_is_unbounded_until_we_bound_it():
    """
    The installed ollama client still has no timeout, and bound_ollama_deadline
    applies one to the client generate() actually uses.
    """
    import importlib
    import ollama

    importlib.reload(ollama)
    assert ollama._client._client.timeout.connect is None, (
        "ollama now ships a default deadline; Bite 1 needs re-measuring")

    applied = service.bound_ollama_deadline()
    assert applied["applied"] is True
    assert ollama._client._client.timeout.connect == \
        service.OLLAMA_CONNECT_TIMEOUT
    assert ollama._client._client.timeout.read == service.OLLAMA_READ_TIMEOUT


def test_the_analysis_cache_hands_back_the_live_tablet_input():
    """
    The summarise path must get the cached TabletInput object itself, since the
    faithfulness reference set comes from its slots and a rebuild would be smaller.
    """
    service.clear_analysis_cache()
    result, t, was_cached = service.analyse_tablet_frozen_with_input(TABLET)

    assert was_cached is False
    assert t is not None
    assert t.lines and not isinstance(t.lines[0], str), (
        "lines must be Verbalisation objects, not strings")

    supplied = t.all_names | t.displayed_names
    assert supplied, "empty faithfulness reference set"

    # The reference set is larger than anything the JSON could rebuild.
    assert len(supplied) > len(t.all_names) or not any(
        s.restored for v in t.lines for s in v.slots)

    again, t2, cached2 = service.analyse_tablet_frozen_with_input(TABLET)
    assert cached2 is True
    assert t2 is t, "the cached pair must hand back the same object"


def test_nothing_to_fuse_is_an_error_type_not_a_silent_empty():
    """A tablet with nothing to fuse raises its own error type instead of returning an empty result."""
    assert issubclass(service.NothingToFuse, Exception)
    assert issubclass(service.OllamaUnreachable, service.Domain3Unavailable)
    assert service.OllamaUnreachable.http_status == 503
    assert service.OllamaTimedOut.http_status == 504
    assert service.Domain3Malformed.http_status == 502
