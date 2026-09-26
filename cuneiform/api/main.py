"""
My FastAPI app for the dashboard.

Routes: /api/health, /api/tablets (the 30 evaluation tablets), one tablet, its
image, POST analyse (Domains 1 and 2), POST summarise (Domain 3), and
/api/uploads for new photographs. Uploads are listed separately from the 30
because nothing has evaluated them.

Run from the project root:
    uvicorn cuneiform.api.main:app --reload
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from cuneiform import config
from cuneiform.api import service
from cuneiform.api.models import (
    AnalysisResponse,
    Domain2Health,
    Domain3Error,
    Domain3Health,
    FrozenConfig,
    HealthResponse,
    ImageInfo,
    SummaryResponse,
    TabletListResponse,
    TabletSummary,
    UploadInfo,
    UploadListResponse,
    UploadResponse,
)
from cuneiform.domain3_generative.generate import MODEL as D3_MODEL

# Long side of the display copy of a photograph, in pixels.
IMAGE_MAX_LONG_SIDE = 1600
IMAGE_JPEG_QUALITY = 85

# Kept short so the health check can never hang.
OLLAMA_PROBE_TIMEOUT = 3.0

# The Vite dev server, under both spellings of localhost.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Filled in at startup and read by /api/health.
_WARM: dict[str, float] | None = None
_WARM_ERROR: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _WARM, _WARM_ERROR
    try:
        _WARM = service.warm()
    except Exception as exc:                        # noqa: BLE001
        # I catch everything so the server still starts and health can report it.
        _WARM_ERROR = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="Cuneiform Translator API",
    description=(
        "A read-and-analyse interface to a three-model cuneiform pipeline: "
        "YOLOv8s detection, a masked sequence model over a logographic sign "
        "vocabulary, and LLaMA-3.1 8B for a tentative reading. "
        "THE SYSTEM DOES NOT TRANSLATE -- Domain 3 produces a genre-level "
        "gist, which is why its field is named tentative_reading."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Health

def _frozen_config() -> FrozenConfig:
    """The frozen configuration, as the service layer reports it."""
    cfg = service.frozen_config()
    return FrozenConfig(
        conf=cfg["conf"], cell=cfg["cell"], **{"lambda": cfg["lambda"]},
        edge_extrapolation=cfg["edge_extrapolation"],
        lm_floor=cfg["lm_floor"], min_cluster=cfg["min_cluster"],
    )


def _check_ollama(timeout: float = OLLAMA_PROBE_TIMEOUT) -> Domain3Health:
    """
    Check whether Ollama is running and holds the Domain 3 model.

    Never raises and never waits longer than `timeout`, and reports "not
    running" and "model not pulled" separately since they need different fixes.
    """
    t0 = time.perf_counter()
    try:
        import ollama

        # My own client with a deadline, since the default one has no timeout.
        client = ollama.Client(timeout=timeout)
        listing = client.list()

        held: list[str] = []
        for m in getattr(listing, "models", None) or listing.get("models", []):
            name = getattr(m, "model", None) or (
                m.get("model") if isinstance(m, dict) else None)
            if name:
                held.append(str(name))

        return Domain3Health(
            reachable=True,
            model=D3_MODEL,
            model_available=D3_MODEL in held,
            models_held=sorted(held),
            detail=None if D3_MODEL in held else (
                f"Ollama is running but does not hold {D3_MODEL}. "
                f"Run: ollama pull {D3_MODEL}"),
            checked_in_seconds=round(time.perf_counter() - t0, 3),
        )

    except ConnectionError as exc:
        # The usual case when the tray app is closed.
        return Domain3Health(
            reachable=False, model=D3_MODEL, model_available=None,
            detail=f"ConnectionError: {exc}. Start the Ollama tray app.",
            checked_in_seconds=round(time.perf_counter() - t0, 3),
        )
    except Exception as exc:                        # noqa: BLE001
        return Domain3Health(
            reachable=False, model=D3_MODEL, model_available=None,
            detail=f"{type(exc).__name__}: {exc}",
            checked_in_seconds=round(time.perf_counter() - t0, 3),
        )


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Per-domain status. Domain 3 being down only makes the system "degraded",
    because the reading pane does not need the language model.
    """
    d2_loaded = service.is_warm()
    d3 = _check_ollama()

    notes: list[str] = []
    if not d2_loaded:
        notes.append(
            "The Domain 2 sequence model is not loaded, so no reading can be "
            "produced. " + (_WARM_ERROR or "Warming has not been attempted."))
    if not d3.reachable:
        notes.append(
            "Ollama is not reachable, so the tentative reading and historical "
            "summary are unavailable. Detection and the fused sign sequence "
            "are unaffected. Start the Ollama tray app.")
    elif not d3.model_available:
        notes.append(d3.detail or "The Domain 3 model is not available.")

    return HealthResponse(
        status="ok" if (d2_loaded and d3.reachable and d3.model_available)
               else "degraded",
        tablets=len(service.list_tablets()),
        domain2=Domain2Health(
            loaded=d2_loaded,
            # The sequence model's checkpoint, not the detector's.
            checkpoint=service.sequence_checkpoint_name(),
            warm_seconds=_WARM,
        ),
        domain3=d3,
        config=_frozen_config(),
        notes=notes,
    )


# The tablet list

def _summarise(tablet: str) -> TabletSummary:
    rec = service.tablet_metadata(tablet)
    return TabletSummary(
        tablet=tablet,
        image=ImageInfo(
            width=rec["width"], height=rec["height"],
            url=f"/api/tablets/{tablet}/image",
            served_max_long_side=IMAGE_MAX_LONG_SIDE,
        ),
        detections_in_file=rec["detections_in_file"],
        detections_above_conf=rec["detections_above_conf"],
        conf=rec["conf"],
        detector_seconds=rec["detector_seconds"],
    )


@app.get("/api/tablets", response_model=TabletListResponse)
def tablets() -> TabletListResponse:
    """The 30 tablets the live evaluation was measured on."""
    ids = service.list_tablets()
    return TabletListResponse(
        tablets=[_summarise(t) for t in ids],
        count=len(ids),
        population_note=(
            "These are the 30 tablets with persisted detections -- exactly the "
            "population of the live fusion evaluation, enumerated by the same "
            "function. Every accuracy figure in this interface was measured on "
            "these and traces to that evaluation. Photographs you upload are "
            "listed separately and no evaluation covers them."
        ),
    )


@app.get("/api/tablets/{tablet}", response_model=TabletSummary)
def tablet(tablet: str) -> TabletSummary:
    if not service.tablet_exists(tablet):
        raise HTTPException(404, f"No tablet {tablet}.")
    return _summarise(tablet)


# Uploaded photographs
#
# The image comes in as a raw request body instead of multipart form data,
# because multipart needs an extra package and this has to work offline. The
# filename is only a label; I identify the file by decoding it.

def _upload_summary(rec: dict) -> UploadResponse:
    """An upload record as the API's own shape."""
    tablet_id = rec["tablet"]
    meta = service.tablet_metadata(tablet_id)
    return UploadResponse(
        tablet=tablet_id,
        image=ImageInfo(
            width=meta["width"], height=meta["height"],
            url=f"/api/tablets/{tablet_id}/image",
            served_max_long_side=IMAGE_MAX_LONG_SIDE,
        ),
        detections_in_file=meta["detections_in_file"],
        detections_above_conf=meta["detections_above_conf"],
        conf=meta["conf"],
        detector_seconds=meta["detector_seconds"],
        upload=UploadInfo(**rec["upload"]),
        caveats=service.upload_caveats(rec),
    )


@app.post("/api/uploads", response_model=UploadResponse,
          responses={413: {"description": "Too large to decode."},
                     422: {"description": "Not a readable photograph."}})
async def upload(request: Request, filename: str = "") -> UploadResponse:
    """
    Take a photograph and run Domain 1 on it.

    This is my slowest route outside Domain 3: it detects once and, if the signs
    are the wrong size for the detector, resizes and detects again. The reading
    itself comes from a separate call to analyse, like any other tablet.
    """
    data = await request.body()
    try:
        rec = service.ingest_upload(data, filename)
    except service.UploadRejected as exc:
        # 422: the request was fine, but I could not read the image in it.
        raise HTTPException(422, {"error": "UploadRejected",
                                  "detail": str(exc),
                                  "hint": "Try a JPEG or PNG photograph of a "
                                          "single tablet."}) from exc
    return _upload_summary(rec)


@app.get("/api/uploads", response_model=UploadListResponse)
def uploads() -> UploadListResponse:
    """Uploaded photographs, newest first. Kept apart from /api/tablets."""
    recs = [service.upload_record(t) for t in service.list_uploads()]
    return UploadListResponse(
        uploads=[_upload_summary(r) for r in recs],
        count=len(recs),
        note=("Photographs uploaded to this server. No evaluation in this "
              "project covers them and no ground truth exists for them, so "
              "no accuracy figure shown anywhere in this interface describes "
              "these."),
    )


# The photograph

@lru_cache(maxsize=32)
def _display_jpeg(tablet: str, max_side: int, quality: int) -> bytes:
    """
    Return the photograph downscaled to `max_side`, cached in memory.

    I apply exif_transpose here because the detector does too, so the boxes
    always land in the same frame as the image. LANCZOS keeps the fine wedges
    sharp.
    """
    from PIL import Image, ImageOps

    path = service.image_path(tablet)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im.convert("RGB"))
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# The analysis (Domains 1 and 2)

@app.post("/api/tablets/{tablet}/analyse", response_model=AnalysisResponse)
def analyse(tablet: str) -> AnalysisResponse:
    """
    Run Domains 1 and 2 for one tablet and return the whole result.

    It is a POST so browsers never prefetch it, and it runs synchronously
    (the slowest tablet takes about 14 s). Results are cached per tablet, so a
    repeat request comes straight from memory. Domain 3 lives on its own route.
    """
    if not service.tablet_exists(tablet):
        raise HTTPException(404, f"No tablet {tablet}.")

    result, was_cached = service.analyse_tablet_frozen(tablet)
    # `timings` describe the original run; `cached` tells the reader it was reused.
    return AnalysisResponse(**result, cached=was_cached)


# Domain 3

@app.post(
    "/api/tablets/{tablet}/summarise",
    response_model=SummaryResponse,
    responses={
        404: {"description": "No such tablet. The population is the "
                             "30 tablets of the live evaluation."},
        503: {"model": Domain3Error,
              "description": "Ollama is not reachable. Start the tray app."},
        504: {"model": Domain3Error,
              "description": "Ollama did not answer within the read deadline."},
        502: {"model": Domain3Error,
              "description": "Ollama answered, but not with valid JSON."},
        409: {"model": Domain3Error,
              "description": "Nothing was fused, so there is no sequence to "
                             "read. A real outcome, not a fault."},
    },
)
def summarise(tablet: str, regenerate: bool = False) -> SummaryResponse:
    """
    Run Domain 3 for one tablet and return the tentative reading with its audit.

    This is not a translation, only a genre-level gist. It has its own route so
    nobody waits for the language model just to see the sign sequence.
    `regenerate=true` forces a fresh call; otherwise I return the summary I
    already made, which is not deterministic.
    """
    if not service.tablet_exists(tablet):
        raise HTTPException(404, f"No tablet {tablet}.")

    try:
        payload, was_cached = service.summarise_tablet(
            tablet, regenerate=regenerate)

    except service.NothingToFuse as exc:
        raise HTTPException(
            409,
            detail={
                "error": "NothingToFuse",
                "detail": str(exc),
                "hint": "This tablet produced no block long enough to fuse. "
                        "There is no sign sequence for Domain 3 to read.",
                "tablet": tablet,
            },
        ) from exc

    except service.Domain3Unavailable as exc:
        # The status code and hint come from the exception class in service.py.
        raise HTTPException(
            exc.http_status,
            detail={
                "error": type(exc).__name__,
                "detail": str(exc),
                "hint": exc.hint,
                "tablet": tablet,
            },
        ) from exc

    return SummaryResponse(**payload, cached=was_cached)


@app.get("/api/tablets/{tablet}/image")
def tablet_image(tablet: str) -> Response:
    """
    The source photograph for the image pane (never the overlay files, which
    have ground-truth boxes drawn on them). It is downscaled, so the client
    scales boxes by rendered width over the original width.
    """
    if not service.tablet_exists(tablet):
        raise HTTPException(404, f"No tablet {tablet}.")
    path = service.image_path(tablet)
    if not path.exists():
        raise HTTPException(404, f"No source image on disk for {tablet}.")

    data = _display_jpeg(tablet, IMAGE_MAX_LONG_SIDE, IMAGE_JPEG_QUALITY)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-Source-Width": str(service.tablet_metadata(tablet)["width"]),
            "X-Source-Height": str(service.tablet_metadata(tablet)["height"]),
        },
    )
