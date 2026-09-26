"""
Endpoint tests for my FastAPI app, run through TestClient against the real
artifacts with no mocking. The health tests check the contract, so they pass
whether or not Ollama is running. Several tests make sure no route can quietly
return nothing, since an empty answer would look like the system declining to
name a sign.

Run from the project root:
    python -m pytest tests/test_api.py -q
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

from cuneiform import config
from cuneiform.api import service
from cuneiform.api.main import IMAGE_MAX_LONG_SIDE, app

TABLET = "P335940"


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan, which warms the model.
    with TestClient(app) as c:
        yield c


# /api/health

def test_health_reports_the_domains_separately(client):
    """Model loaded and Ollama reachable must be two separate facts."""
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()

    assert body["domain2"]["loaded"] is True, (
        "the lifespan should have warmed the sequence model")
    assert isinstance(body["domain3"]["reachable"], bool)
    # Two separate keys under two separate objects.
    assert "loaded" not in body["domain3"]
    assert "reachable" not in body["domain2"]


def test_health_status_is_ok_or_degraded_never_an_error(client):
    """Domain 3 being down is "degraded", never an error, since the reading still works."""
    body = client.get("/api/health").json()
    assert body["status"] in ("ok", "degraded")
    if body["status"] == "degraded":
        assert body["notes"], "a degraded status must explain itself"
    if not body["domain3"]["reachable"]:
        assert body["status"] == "degraded"
        assert any("Ollama" in n for n in body["notes"])
        # Unreachable means the question was never asked, not that the answer was no.
        assert body["domain3"]["model_available"] is None


def test_health_names_the_domain2_checkpoint_not_the_detector(client):
    """
    The Domain 2 lamp must name the sequence model's checkpoint, not the
    detector's. Checked against the path load_sequence_model actually opens.
    """
    from cuneiform import config
    from cuneiform.orchestrator.fusion import CKPT_DIR

    shown = client.get("/api/health").json()["domain2"]["checkpoint"]
    assert shown == CKPT_DIR.name, (
        f"the Domain 2 lamp shows {shown!r}, but Domain 2 loads from "
        f"{CKPT_DIR}")
    assert shown != config.D1_WEIGHTS.name, (
        "the Domain 2 lamp is showing the Domain 1 detector's checkpoint")


def test_health_ollama_probe_is_bounded(client):
    """The Ollama probe finishes in bounded time whatever the outcome."""
    body = client.get("/api/health").json()
    took = body["domain3"]["checked_in_seconds"]
    assert took is not None
    assert took < 10.0, f"the Ollama probe took {took}s"


def test_health_reports_the_frozen_configuration(client):
    """The operating point behind the numbers is visible."""
    cfg = client.get("/api/health").json()["config"]
    assert cfg["conf"] == 0.15
    assert cfg["cell"] == 2.0
    assert cfg["lambda"] == 1.0
    assert cfg["lm_floor"] == 0.08
    assert cfg["min_cluster"] == 3
    assert cfg["edge_extrapolation"] is False


# /api/tablets

def test_tablet_list_returns_the_thirty(client):
    r = client.get("/api/tablets")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 30
    assert len(body["tablets"]) == 30
    assert {t["tablet"] for t in body["tablets"]} >= {TABLET}
    assert body["population_note"]


def test_tablet_list_never_leaks_ground_truth(client):
    """n_gt_boxes (a ground-truth count) must not appear anywhere in the raw response text."""
    raw = client.get("/api/tablets").text
    for forbidden in ("n_gt_boxes", "gt_box", "ground_truth", "overlay"):
        assert forbidden not in raw, f"{forbidden!r} leaked into /api/tablets"


def test_tablet_list_counts_detections_at_the_frozen_threshold(client):
    """The count shown is at the frozen threshold, not the raw count down to the 0.01 floor."""
    body = client.get("/api/tablets").json()
    for t in body["tablets"]:
        assert t["conf"] == 0.15
        assert 0 < t["detections_above_conf"] <= t["detections_in_file"]
    row = next(t for t in body["tablets"] if t["tablet"] == TABLET)
    assert row["detections_in_file"] == 255
    assert row["detections_above_conf"] == 141


def test_single_tablet_matches_its_row_in_the_list(client):
    """The list and the detail view come from one code path, so they agree."""
    body = client.get("/api/tablets").json()
    row = next(t for t in body["tablets"] if t["tablet"] == TABLET)
    assert client.get(f"/api/tablets/{TABLET}").json() == row


def test_unknown_tablet_is_a_404(client):
    assert client.get("/api/tablets/P000000").status_code == 404
    assert client.get("/api/tablets/P000000/image").status_code == 404


# /api/tablets/{tablet}/image

def test_image_is_a_real_decodable_jpeg(client):
    """The image must actually decode, not just return 200 with a JPEG content type."""
    from PIL import Image

    r = client.get(f"/api/tablets/{TABLET}/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 10_000

    with Image.open(io.BytesIO(r.content)) as im:
        im.load()
        assert im.format == "JPEG"
        assert max(im.size) <= IMAGE_MAX_LONG_SIDE


def test_image_carries_the_source_dimensions_boxes_are_drawn_in(client):
    """
    Boxes are in full-image pixels but the served image is downscaled, so the
    source dimensions must travel with it and match the tablet metadata.
    """
    from PIL import Image

    meta = client.get(f"/api/tablets/{TABLET}").json()
    r = client.get(f"/api/tablets/{TABLET}/image")

    assert int(r.headers["X-Source-Width"]) == meta["image"]["width"]
    assert int(r.headers["X-Source-Height"]) == meta["image"]["height"]

    with Image.open(io.BytesIO(r.content)) as im:
        served_w, served_h = im.size
    assert (served_w, served_h) != (meta["image"]["width"],
                                    meta["image"]["height"]), (
        "this tablet is expected to be downscaled, so the test that the two "
        "spaces differ is meaningful here")
    # The aspect ratio must survive, or scaling boxes by one factor would be off on one axis.
    src_ar = meta["image"]["width"] / meta["image"]["height"]
    assert abs(served_w / served_h - src_ar) < 0.01


def test_every_tablet_serves_an_image(client):
    """Every one of the 30 tablets serves an image."""
    failures = []
    for t in client.get("/api/tablets").json()["tablets"]:
        r = client.get(t["image"]["url"])
        if r.status_code != 200 or len(r.content) < 5_000:
            failures.append((t["tablet"], r.status_code, len(r.content)))
    assert not failures, f"image route failed for: {failures}"


def test_no_route_serves_the_ground_truth_overlay(client):
    """No route serves the overlay JPEGs, which have ground-truth boxes drawn in green."""
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert not any("overlay" in p for p in paths)
    assert client.get(f"/api/tablets/{TABLET}/overlay").status_code == 404


# POST /api/tablets/{tablet}/analyse

def test_analyse_validates_and_matches_the_service_exactly(client):
    """
    The route must return what the service computed: same sequence, counts and
    fusion stats (the response adds `cached` and some types are coerced).
    """
    from cuneiform.api import service

    r = client.post(f"/api/tablets/{TABLET}/analyse")
    assert r.status_code == 200
    body = r.json()

    ref = service.analyse_tablet(TABLET)
    assert body["sequence_dump"] == ref["sequence_dump"]
    assert body["counts"] == ref["counts"]
    assert body["fusion"] == ref["fusion"]
    assert body["tablet"] == TABLET


def test_analysis_is_deterministic_so_caching_cannot_lie(client):
    """
    Analysing the same tablet twice without the cache must give the same result
    (timings aside). If this ever fails, the cache has to go.
    """
    from cuneiform.api import service

    a = service.analyse_tablet(TABLET)
    b = service.analyse_tablet(TABLET)
    a.pop("timings"), b.pop("timings")
    assert a == b


def test_analyse_is_cached_and_says_so(client):
    """The second call says cached: true and keeps the original timings."""
    from cuneiform.api import service

    service.clear_analysis_cache()

    first = client.post(f"/api/tablets/{TABLET}/analyse").json()
    assert first["cached"] is False

    second = client.post(f"/api/tablets/{TABLET}/analyse").json()
    assert second["cached"] is True
    assert second["timings"] == first["timings"], (
        "a cached response must report the original computation's timings")
    assert second["sequence_dump"] == first["sequence_dump"]


def test_analyse_returns_under_ten_seconds_for_a_typical_tablet(client):
    """
    A typical tablet (P335940) analyses in under 10 s from cold. This is one
    tablet near the median, not the worst case; see the next test.
    """
    from cuneiform.api import service

    service.clear_analysis_cache()
    t0 = time.perf_counter()
    r = client.post(f"/api/tablets/{TABLET}/analyse")
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 10.0, f"analyse took {elapsed:.2f}s"


def test_the_slow_tablets_are_still_the_slow_ones(client):
    """
    The largest tablet (P334932, 632 slots) is not under 10 s. I only check its
    cost is bounded and that the second request is fast.
    """
    from cuneiform.api import service

    service.clear_analysis_cache()

    t0 = time.perf_counter()
    first = client.post("/api/tablets/P334932/analyse")
    cold = time.perf_counter() - t0
    assert first.status_code == 200
    assert cold < 60.0, f"cold analyse of the largest tablet took {cold:.1f}s"

    t0 = time.perf_counter()
    second = client.post("/api/tablets/P334932/analyse")
    warm = time.perf_counter() - t0
    assert second.json()["cached"] is True
    assert warm < 2.0, f"cached analyse took {warm:.2f}s"


def test_analyse_ships_its_caveats(client):
    """The payload carries the specific caveats that must travel with the numbers, not a placeholder."""
    body = client.post(f"/api/tablets/{TABLET}/analyse").json()
    joined = " ".join(body["caveats"])
    assert "does not translate" in joined
    assert "22.2%" in joined
    assert "declining to guess" in joined
    assert "CHANGED" in joined


def test_analyse_reports_the_operating_point(client):
    body = client.post(f"/api/tablets/{TABLET}/analyse").json()
    assert body["config"]["is_frozen"] is True
    assert body["config"]["conf"] == 0.15
    assert body["config"]["frozen"]["lm_floor"] == 0.08


def test_analyse_never_leaks_ground_truth(client):
    """The same raw-text ground-truth check as the tablet list."""
    raw = client.post(f"/api/tablets/{TABLET}/analyse").text
    for forbidden in ("n_gt_boxes", "gt_box", "ground_truth", "overlay"):
        assert forbidden not in raw


def test_analyse_boxes_are_in_full_image_space(client):
    """Detection coordinates are in the source image's pixels, not the downscaled copy's."""
    body = client.post(f"/api/tablets/{TABLET}/analyse").json()
    w, h = body["image"]["size"]                   # [W, H], width first
    assert body["detections"], "expected detections for this tablet"
    for d in body["detections"]:
        assert 0 <= d["x1"] < d["x2"] <= w + 1
        assert 0 <= d["y1"] < d["y2"] <= h + 1
    # At least one box must be beyond the downscaled size, or this couldn't tell the spaces apart.
    assert max(d["x2"] for d in body["detections"]) > IMAGE_MAX_LONG_SIDE or \
        max(d["y2"] for d in body["detections"]) > IMAGE_MAX_LONG_SIDE


def test_analyse_slot_states_are_closed(client):
    """Every slot has one of the four states and a matching token."""
    body = client.post(f"/api/tablets/{TABLET}/analyse").json()
    states = set()
    for line in body["lines"]:
        for s in line["slots"]:
            states.add(s["state"])
            if s["state"] == "unnamed":
                assert s["label"] is None
                assert s["unnamed_reason"] in ("below_floor", "outside_block")
            else:
                assert s["label"] is not None
    assert states <= {"vision", "override", "restored", "unnamed"}
    assert "vision" in states and "restored" in states


def test_analyse_unknown_tablet_is_a_404(client):
    assert client.post("/api/tablets/P000000/analyse").status_code == 404


def test_analyse_is_post_not_get(client):
    """The expensive route is a POST so nothing prefetches it."""
    assert client.get(f"/api/tablets/{TABLET}/analyse").status_code == 405


# The app itself

def test_openapi_schema_is_generated(client):
    """The React client is written against this schema."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for expected in ("/api/health", "/api/tablets",
                     "/api/tablets/{tablet}/image"):
        assert expected in paths


def test_cors_allows_the_vite_dev_server(client):
    """The Vite dev server on port 5173 can call the API."""
    r = client.get("/api/tablets",
                   headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == \
        "http://localhost:5173"


def test_no_part_one_route_runs_the_pipeline(client):
    """The cheap routes stay cheap and never run the pipeline."""


    t0 = time.perf_counter()
    client.get("/api/tablets")
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"/api/tablets took {elapsed:.2f}s"


# POST /api/tablets/{tablet}/summarise
#
# Some of these only run when Ollama is up and some only when it is down. The
# 503 test needs Ollama really stopped, not mocked, so it checks what Ollama
# itself raises. The contract tests run either way.


def _ollama_is_up() -> bool:
    from cuneiform.api.main import _check_ollama
    return _check_ollama().reachable


requires_ollama = pytest.mark.skipif(
    not _ollama_is_up(),
    reason="Ollama tray app is not running; the Domain 3 success path cannot "
           "be exercised. Start it and re-run.")

requires_ollama_down = pytest.mark.skipif(
    _ollama_is_up(),
    reason="Ollama tray app IS running, so the unreachable path cannot be "
           "exercised. Stop it and re-run -- Part 3's acceptance criterion "
           "requires this to have been run for real.")


def test_summarise_route_is_declared_with_its_failure_modes(client):
    """The route is a POST and declares its failure statuses in the OpenAPI document."""
    schema = client.get("/openapi.json").json()
    path = schema["paths"]["/api/tablets/{tablet}/summarise"]
    assert "post" in path and "get" not in path
    for status in ("200", "404", "409", "502", "503", "504"):
        assert status in path["post"]["responses"], (
            f"the summarise route does not declare {status}")


def test_no_response_field_is_called_a_translation(client):
    """No field in the published schema is named as a translation (the prose may deny it)."""
    schema = client.get("/openapi.json").json()
    offenders = []
    for name, model in schema["components"]["schemas"].items():
        for field in (model.get("properties") or {}):
            if "translat" in field.lower():
                offenders.append(f"{name}.{field}")
    assert not offenders, f"fields named as a translation: {offenders}"


def test_summarise_unknown_tablet_is_404_and_costs_nothing(client):
    """A bad id fails before any language model is called."""
    t0 = time.perf_counter()
    r = client.post("/api/tablets/P000000/summarise")
    assert r.status_code == 404
    assert time.perf_counter() - t0 < 2.0


@requires_ollama
def test_summarise_returns_a_reading_with_its_audit(client):
    """A summary comes back with its faithfulness audit populated, not just prose."""
    r = client.post(f"/api/tablets/{TABLET}/summarise?regenerate=true")
    assert r.status_code == 200, r.text
    b = r.json()

    assert b["tablet"] == TABLET
    assert b["schema_complete"] is True
    assert b["summary"]["tentative_reading"].strip()
    assert b["summary"]["confidence"] in ("none", "low", "medium", "high")
    assert b["attempts"] in (1, 2)

    # The audit and its evidence
    f = b["faithfulness"]
    assert f["verdict"] in ("pass", "pass_after_normalising", "fail",
                            "vacuous")
    assert f["supplied_distinct"] > 0, (
        "the faithfulness reference set is empty, so the check could not have "
        "run -- this is the symptom of a TabletInput rebuilt from JSON")
    assert f["citations_distinct"] == len(set(
        f["exact"] + [n[0] for n in f["normalised"]] + f["invented"]))
    assert isinstance(b["prose_scan"]["unmatched"], list)

    # The deadline really landed on the client Domain 3 uses.
    assert b["deadline"]["applied"] is True, (
        "ollama builds its client with timeout=None; if the deadline could "
        "not be applied, a hung model can hold a worker forever")


@requires_ollama
def test_summarise_ships_its_caveats(client):
    """The summary ships with its caveats."""
    b = client.post(f"/api/tablets/{TABLET}/summarise").json()
    joined = " ".join(b["caveats"])
    assert "NOT A TRANSLATION" in joined.upper()
    assert "22.2%" in joined
    assert "SIGN NAMES ONLY" in joined.upper()
    assert "temperature" in joined.lower(), (
        "the payload must say the summary is not reproducible")


@requires_ollama
def test_summarise_cache_says_what_it_is_and_regenerate_bypasses_it(client):
    """
    Domain 3 isn't deterministic, so a repeat request returns the same summary
    with cached: true, and regenerate gets a fresh one.
    """
    from cuneiform.api import service

    service.clear_analysis_cache()

    first = client.post(f"/api/tablets/{TABLET}/summarise").json()
    assert first["cached"] is False

    second = client.post(f"/api/tablets/{TABLET}/summarise").json()
    assert second["cached"] is True
    assert second["summary"] == first["summary"], (
        "a cached summary must be the one already shown, not a new call")

    fresh = client.post(f"/api/tablets/{TABLET}/summarise?regenerate=true")
    assert fresh.json()["cached"] is False


@requires_ollama
def test_summarise_never_leaks_ground_truth(client):
    """No ground-truth count in the raw summary response."""
    r = client.post(f"/api/tablets/{TABLET}/summarise")
    assert "n_gt_boxes" not in r.text
    assert "ground_truth" not in r.text


@requires_ollama_down
def test_summarise_is_a_clean_503_when_ollama_is_down(client):
    """
    With Ollama really stopped, summarise returns a clean 503 with a hint, in
    bounded time.
    """
    t0 = time.perf_counter()
    r = client.post(f"/api/tablets/{TABLET}/summarise?regenerate=true")
    elapsed = time.perf_counter() - t0

    assert r.status_code == 503, r.text
    detail = r.json()["detail"]
    assert detail["error"] in ("OllamaUnreachable",)
    assert "Ollama" in detail["hint"]
    assert detail["tablet"] == TABLET
    assert elapsed < 30.0, (
        f"the unreachable path took {elapsed:.1f}s; the connect deadline is "
        f"3.0 s and a failure this slow means it was not applied")


@requires_ollama_down
def test_health_degrades_rather_than_failing_when_ollama_is_down(client):
    """The page stays usable when Ollama is down."""
    b = client.get("/api/health").json()
    assert b["status"] == "degraded"
    assert b["domain3"]["reachable"] is False
    assert b["domain2"]["loaded"] is True
    assert any("Ollama" in n for n in b["notes"])


# The boxes the image pane draws

def test_served_photograph_is_in_the_same_frame_as_the_boxes(client):
    """
    The detector works on the EXIF-transposed image, so the served image must be
    in the same frame. Its aspect must match the recorded image_size.
    """
    from PIL import Image

    meta = client.get(f"/api/tablets/{TABLET}").json()
    r = client.get(f"/api/tablets/{TABLET}/image")
    with Image.open(io.BytesIO(r.content)) as im:
        served_w, served_h = im.size

    src_w, src_h = meta["image"]["width"], meta["image"]["height"]
    # A transpose swaps the aspect ratio; a downscale doesn't.
    assert abs(served_w / served_h - src_w / src_h) < 0.01, (
        f"served {served_w}x{served_h} is not the same shape as the frame the "
        f"boxes are in, {src_w}x{src_h} -- an orientation mismatch would put "
        f"every box in the wrong place"
    )
    assert (served_w > served_h) == (src_w > src_h)


def test_every_slot_points_at_a_real_box_and_no_box_is_claimed_twice(client):
    """
    detection_i must index the right box: in range, the same rectangle, never
    claimed twice, and absent for inserted positions.
    """
    b = client.post(f"/api/tablets/{TABLET}/analyse").json()
    dets = b["detections"]
    slots = [s for line in b["lines"] for s in line["slots"]]

    claimed = []
    for s in slots:
        di = s["detection_i"]
        if di is None:
            assert s["box"] is None, (
                f"slot {s['i']} has no detection index but does have a box")
            continue
        assert 0 <= di < len(dets), f"slot {s['i']} indexes detection {di}"
        claimed.append(di)
        # And it points at the same rectangle, not just a valid one.
        for k in ("x1", "y1", "x2", "y2"):
            assert s["box"][k] == dets[di][k], (
                f"slot {s['i']} box {k} disagrees with detection {di}")

    assert len(claimed) == len(set(claimed)), "a box is claimed by two slots"
    debris = len(dets) - len(claimed)
    assert debris == b["counts"]["detections_debris"], (
        f"{debris} unclaimed boxes but counts says "
        f"{b['counts']['detections_debris']} debris")


def test_no_route_serves_the_ground_truth_overlay_even_now_that_boxes_exist(client):
    """Still no overlay route now that the payload carries boxes."""
    body = client.post(f"/api/tablets/{TABLET}/analyse").text
    assert "overlay" not in body.lower()
    assert "n_gt_boxes" not in body
    assert client.get(f"/api/tablets/{TABLET}/overlay").status_code == 404


# Uploaded photographs
#
# The first test matters most: an upload must never land in the evaluation's
# detections folder, or the 30 evaluation tablets would silently become 31.


def _test_photograph(scale: float = 1.0) -> bytes:
    """
    A real tablet photograph, optionally rescaled. A blank image would give no
    detections and every check below would pass without testing anything.
    """
    from PIL import Image

    src = Image.open(config.IMAGES_DIR / "P335940.jpg").convert("RGB")
    if scale != 1.0:
        src = src.resize((int(src.width * scale), int(src.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    src.save(buf, "JPEG", quality=88)
    return buf.getvalue()


@pytest.fixture
def uploaded(client):
    """One uploaded photograph, deleted afterwards so tests don't depend on leftovers."""
    body = _test_photograph(2.0)          # 2x, so the rescale rule kicks in
    r = client.post("/api/uploads?filename=field_photo.jpg", content=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    yield payload
    for p in config.UPLOADS_DIR.glob(payload["tablet"] + ".*"):
        p.unlink(missing_ok=True)


def test_upload_never_joins_the_evaluated_population(client, uploaded):
    """An upload never joins the 30 evaluation tablets."""
    assert len(service.list_tablets()) == 30
    assert client.get("/api/tablets").json()["count"] == 30

    ids = [t["tablet"] for t in client.get("/api/tablets").json()["tablets"]]
    assert uploaded["tablet"] not in ids

    # And the file is in the uploads folder.
    assert (config.UPLOADS_DIR / f"{uploaded['tablet']}.json").exists()
    assert not (config.D1_LIVE_DETECTIONS / f"{uploaded['tablet']}.json").exists()

    # Ask the evaluation's own function, since the results depend on it.
    from cuneiform.orchestrator.evaluate_live import live_tablets
    assert len(live_tablets()) == 30


def test_upload_runs_the_detector_and_reports_both_passes(client, uploaded):
    up = uploaded["upload"]
    assert uploaded["detections_above_conf"] > 0
    assert up["original_size"] == [2356, 5052]

    # At 2x the signs are outside the measured band, so a rescale must happen
    # and be reported.
    assert up["scale"]["applied"] is True
    assert 0.25 <= up["scale"]["factor"] <= 1.0
    assert up["scale"]["reason"]
    assert len(up["passes"]) == 2

    before, after = up["passes"]
    assert before["median_sign_px"] > after["median_sign_px"], (
        "the rescale did not make the signs smaller, so it did not do the "
        "thing its own reason says it did")
    # And it landed inside the population's range.
    assert 42.7 <= after["median_sign_px"] <= 112.7


def test_a_photograph_already_at_the_right_scale_is_left_alone(client):
    """A photograph already at the right scale is not rescaled."""
    body = _test_photograph(1.0)
    r = client.post("/api/uploads?filename=native.jpg", content=body)
    assert r.status_code == 200
    payload = r.json()
    try:
        assert payload["upload"]["scale"]["applied"] is False
        assert payload["upload"]["scale"]["factor"] == 1.0
        assert len(payload["upload"]["passes"]) == 1
        assert "inside" in payload["upload"]["scale"]["reason"]
    finally:
        for p in config.UPLOADS_DIR.glob(payload["tablet"] + ".*"):
            p.unlink(missing_ok=True)


def test_uploaded_tablet_analyses_like_any_other(client, uploaded):
    """An upload works with analyse and image like any other tablet id."""
    tid = uploaded["tablet"]
    r = client.post(f"/api/tablets/{tid}/analyse")
    assert r.status_code == 200
    body = r.json()
    assert body["tablet"] == tid
    assert body["counts"]["slots"] > 0
    assert body["config"]["is_frozen"] is True

    # The image route resolves it too and serves a real JPEG.
    img = client.get(f"/api/tablets/{tid}/image")
    assert img.status_code == 200
    assert img.content[:2] == b"\xff\xd8"


def test_an_uploaded_analysis_carries_MORE_caveats_than_a_corpus_one(client, uploaded):
    """An upload has no ground truth or evaluation, so it carries more caveats, not fewer."""
    mine = client.post(f"/api/tablets/{uploaded['tablet']}/analyse").json()
    theirs = client.post("/api/tablets/P335940/analyse").json()
    assert len(mine["caveats"]) > len(theirs["caveats"])

    text = " ".join(mine["caveats"]).lower()
    assert "not part of any evaluation" in text
    assert "no ground truth exists" in text
    assert "81 annotated museum plates" in text


def test_upload_rejects_what_it_cannot_read(client):
    """
    A file that isn't an image gets a 422 with a message, never a 500. Once
    ultralytics is loaded a bad file raises ModuleNotFoundError, so the handler
    catches everything.
    """
    r = client.post("/api/uploads?filename=notes.txt", content=b"not an image")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "UploadRejected"
    # HEIC is named because it's the iPhone default.
    assert "heic" in detail["detail"].lower()

    assert client.post("/api/uploads", content=b"").status_code == 422


def test_upload_does_not_leak_ground_truth(client, uploaded):
    """No ground-truth count in an upload payload."""
    tid = uploaded["tablet"]
    for r in (client.get("/api/uploads"),
              client.post(f"/api/tablets/{tid}/analyse")):
        assert "n_gt_boxes" not in r.text


def test_uploads_are_listed_separately_with_their_own_note(client, uploaded):
    body = client.get("/api/uploads").json()
    assert body["count"] >= 1
    assert uploaded["tablet"] in [u["tablet"] for u in body["uploads"]]
    note = body["note"].lower()
    assert "no evaluation" in note and "no ground truth" in note


def test_the_tablet_note_no_longer_claims_there_is_no_upload(client):
    """The tablet list note no longer says there is no upload."""
    note = client.get("/api/tablets").json()["population_note"]
    assert "no upload" not in note.lower()
    assert "30" in note
