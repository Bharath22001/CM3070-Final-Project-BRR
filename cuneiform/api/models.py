"""
My pydantic response models for the API.

They are the contract the React client is written against (FastAPI publishes
them at /docs), and they make a renamed field fail loudly instead of reaching
the browser as a blank pane. I never return the ground-truth box counts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FrozenConfig(BaseModel):
    """The operating point, sent with every health check and analysis."""
    conf: float = Field(description="Domain 1 confidence threshold. 0.15.")
    cell: float = Field(description="Segmentation cell factor. 2.0.")
    lambda_: float = Field(alias="lambda", description="Fusion weight. 1.0.")
    edge_extrapolation: bool
    lm_floor: float = Field(
        description="Domain 2 posterior floor, 0.08. Precision is ONE BUCKET "
                    "WIDE, about 0.06-0.10; do not display more than two "
                    "significant figures.")
    min_cluster: int

    model_config = {"populate_by_name": True}


class Domain2Health(BaseModel):
    """Is the sequence model in memory, and what did loading it cost?"""
    loaded: bool
    checkpoint: str | None = Field(
        default=None,
        description="The checkpoint Domain 2 loads, read from fusion.CKPT_DIR. "
                    "This field reported the DETECTOR's weights file until "
                    "2026-09-19, so the health strip printed the wrong model "
                    "under the Domain 2 lamp.")
    warm_seconds: dict[str, float] | None = Field(
        default=None,
        description="Per-asset load time from the last warm() call. Zeros on "
                    "a second call are the cache-hit evidence.")


class Domain3Health(BaseModel):
    """
    Whether Ollama is reachable and holds the model I ask for. I keep these
    as two facts because they need different fixes.
    """
    reachable: bool
    model: str = Field(description="The model Domain 3 asks for.")
    model_available: bool | None = Field(
        default=None,
        description="Whether that model is among the ones Ollama holds. None "
                    "when Ollama could not be reached, because the question "
                    "was never asked -- not because the answer was no.")
    models_held: list[str] | None = None
    detail: str | None = None
    checked_in_seconds: float | None = None


class HealthResponse(BaseModel):
    """
    `status` is "ok" only when everything works. "degraded" means Domain 3 is
    down but the reading still works.
    """
    status: str = Field(description='"ok" | "degraded"')
    tablets: int
    domain2: Domain2Health
    domain3: Domain3Health
    config: FrozenConfig
    notes: list[str] = Field(
        default_factory=list,
        description="Plain-language explanation of any non-ok status, "
                    "renderable directly.")


class ImageInfo(BaseModel):
    """
    Photograph size. `width` and `height` are the full-image pixels the boxes
    are in, not the size of the downscaled copy served at `url`.
    """
    width: int
    height: int
    url: str
    served_max_long_side: int = Field(
        description="The derivative at `url` is downscaled to this on its long "
                    "side. Box coordinates are NOT in this space.")


class TabletSummary(BaseModel):
    """
    One row of the tablet list. `detections_above_conf` is what the pipeline
    actually uses at the frozen threshold.
    """
    tablet: str
    image: ImageInfo
    detections_in_file: int
    detections_above_conf: int
    conf: float
    detector_seconds: float | None = Field(
        default=None,
        description="Domain 1 inference time recorded when the detections "
                    "were persisted. Historical, not measured now.")


class TabletListResponse(BaseModel):
    tablets: list[TabletSummary]
    count: int
    population_note: str


# Uploaded photographs. These get their own models and route so they never
# mix with the 30 evaluation tablets.

class UploadPass(BaseModel):
    """One detector pass over an uploaded photograph."""
    width: int
    height: int
    n_detections: int
    median_sign_px: float | None = Field(
        description="Median detected sign size at the frozen threshold, in "
                    "this pass's own pixels. null when too few signs were "
                    "found to measure one.")
    seconds: float


class UploadScale(BaseModel):
    """
    Whether I resized the photograph before reading it, and why. The accepted
    range (42.7 to 112.7 px) and target (58.5 px) come from the 30 tablets'
    median sign sizes.
    """
    applied: bool
    factor: float
    measured_px: float | None
    reason: str


class UploadInfo(BaseModel):
    """Everything about an upload that the 30 corpus tablets do not have."""
    filename: str
    uploaded_at: str
    original_size: list[int] = Field(description="[W, H] as uploaded.")
    original_format: str
    exif_transposed: bool = Field(
        description="Whether EXIF orientation changed the frame. TRUE for most "
                    "phone photographs and FALSE for all 30 corpus images, "
                    "which is why the served image is the transposed one.")
    scale: UploadScale
    passes: list[UploadPass] = Field(
        description="One entry, or two when a rescale was applied. Both are "
                    "reported so the rescale can be judged by its effect.")
    decode_seconds: float


class UploadResponse(BaseModel):
    """The result of ingesting a photograph. Its id works with analyse, summarise and image."""
    tablet: str
    image: ImageInfo
    detections_in_file: int
    detections_above_conf: int
    conf: float
    detector_seconds: float | None = None
    upload: UploadInfo
    caveats: list[str] = Field(
        min_length=1,
        description="What is true of this photograph and not of the 30: no "
                    "evaluation covers it, no ground truth exists for it, and "
                    "it may be outside the conditions the detector was trained "
                    "on. Shipped here as well as on the analysis so the upload "
                    "pane can show them the moment the image arrives.")


class UploadListResponse(BaseModel):
    uploads: list[UploadResponse]
    count: int
    note: str


# The analysis payload

class AnalysisConfig(FrozenConfig):
    """The operating point an analysis ran at, and whether it is the frozen one."""
    is_frozen: bool
    frozen: FrozenConfig = Field(
        description="What the frozen values ARE, so a non-frozen run can be "
                    "compared against them without the client hardcoding them.")


class Box(BaseModel):
    """A detection box in full-image pixels (not the downscaled image's)."""
    x1: float
    y1: float
    x2: float
    y2: float


class Slot(BaseModel):
    """
    One position in the reconstructed reading.

    `state` is worked out on the server:
      "vision"    vision read it and fusion kept it         -> NAME
      "override"  vision read it and the LM overrode it     -> NAME*
      "restored"  vision missed it and the LM filled it in  -> [NAME]
      "unnamed"   below the floor, or outside any block     -> [x] or a dot

    Restorations are usually wrong (Domain 2 top-1 is 22.2%), so the client
    must never style them as confidently as vision reads.
    """
    i: int = Field(description="Index into the flat slot list.")
    kind: str = Field(description='"detected" (a box) | "inserted" (a gap)')
    x: float
    state: str = Field(description='"vision"|"override"|"restored"|"unnamed"')
    unnamed_reason: str | None = Field(
        default=None, description='"below_floor" | "outside_block" | null')
    token: str = Field(
        description="The rendered token, IDENTICAL to the CLI's dump. Rendered "
                    "server-side so the browser cannot drift from what the "
                    "tests checked.")
    label: int | None
    name: str | None
    vision_label: int | None
    vision_name: str | None
    vision_conf: float | None
    lm_posterior: float | None = Field(
        default=None,
        description="The language model's posterior at its argmax over the "
                    "282-sign support. Acted on only at restored positions, "
                    "where it is compared against the 0.08 floor.")
    detection_i: int | None = Field(
        default=None,
        description="Index into this response's `detections` list, or null "
                    "for an inserted position. It is what lets the image "
                    "pane and the reading pane point at the same thing "
                    "without the browser matching float coordinates. "
                    "Measured over all 30 tablets: 3,678 of 3,714 boxes "
                    "above the threshold carry one, the other 36 being "
                    "debris clusters the pipeline discarded.")
    box: Box | None


class Line(BaseModel):
    """One (segment, line) row. `segment` is a physical surface of the tablet."""
    segment: int
    line: int
    slots: list[Slot]


class SequenceRow(BaseModel):
    """One row of the reading as a single string."""
    segment: int
    line: int
    text: str


class Detection(BaseModel):
    """One Domain 1 box above the confidence threshold, for the image pane."""
    i: int
    x1: float
    y1: float
    x2: float
    y2: float
    train_label: int
    name: str
    conf: float


class Counts(BaseModel):
    """
    How many detections and slots survive each stage, for the provenance pane.
    Debris and dropped remnants are reported, not hidden.
    """
    detections_in_file: int
    detections_above_conf: int
    detections_kept: int
    detections_debris: int
    segments: int
    slots: int
    slots_detected: int
    slots_unknown: int
    slots_missed: int
    blocks: int
    slots_in_blocks: int
    slots_dropped_in_remnants: int
    min_block: int
    box_match_misses: int


class RestorationPosterior(BaseModel):
    min: float
    median: float
    max: float


class Fusion(BaseModel):
    """
    Counts of what fusion changed. None of these is an accuracy figure, since no
    live position has a known answer.
    """
    vision_named_positions: int
    fusion_changed: int
    restoration_candidates: int
    restorations_kept: int
    restorations_demoted: int
    named_positions: int
    total_positions: int
    lm_floor: float
    restoration_posterior: RestorationPosterior | None = None


class VerbalisedLine(BaseModel):
    """One prompt line as Domain 3 will see it, plus its per-kind counts."""
    text: str
    counts: dict[str, int]


class Domain3Input(BaseModel):
    """
    Exactly what the generative stage receives, shown so the prose can be
    checked against the metadata it was given.
    """
    lines: list[VerbalisedLine]
    catalogue_metadata: str
    had_metadata: bool


class AnalysisImage(BaseModel):
    size: list[int] = Field(description="[W, H] -- width FIRST.")
    url: str


class AnalysisResponse(BaseModel):
    """
    The full Domains 1 and 2 result for one tablet.

    `caveats` must be non-empty, so no client can show a reading without also
    holding the facts that qualify it.
    """
    tablet: str
    config: AnalysisConfig
    image: AnalysisImage
    counts: Counts
    fusion: Fusion | None = Field(
        description="null when no block reached the minimum length -- a real "
                    "outcome, not an error.")
    lines: list[Line]
    sequence_dump: list[SequenceRow]
    detections: list[Detection]
    domain3_input: Domain3Input | None
    caveats: list[str] = Field(min_length=1)
    timings: dict[str, float]
    cached: bool = Field(
        default=False,
        description="True when served from the per-tablet cache. `timings` "
                    "then describes the ORIGINAL computation, not this "
                    "request -- so a cached response never reports a speed it "
                    "did not achieve.")


# Domain 3

class Summary(BaseModel):
    """
    The six fields Domain 3 returns. There is deliberately no `translation`
    field: the output is a genre-level gist.
    """
    tentative_reading: str
    confidence: str = Field(
        description="one of none / low / medium / high. The MODEL's own "
                    "estimate, unvalidated against any truth: no position in "
                    "the live path has a known answer.")
    signs_used: list[str] = Field(
        description="the model's sign citations, one entry PER OCCURRENCE, so "
                    "the list is longer than the distinct set. Audited by "
                    "`faithfulness` below.")
    uncertainty_notes: str
    historical_context: str
    metadata_used: bool = Field(
        description="the model's CLAIM that it used the catalogue metadata. "
                    "Compare with metadata.had_metadata, which is what was "
                    "actually supplied.")


class SummaryMetadata(BaseModel):
    """What Domain 3 was given, so the prose can be audited against it."""
    had_metadata: bool
    had_metadata_at_call: bool = Field(
        description="what generate() recorded supplying. Must equal "
                    "had_metadata; a disagreement means the tablet summarised "
                    "is not the tablet analysed.")
    claimed_metadata_used: bool
    catalogue_metadata: str = Field(
        description="the metadata block verbatim. Shown because the "
                    "faithfulness checks cover sign names only -- a king or a "
                    "city invented in the prose is invisible to them, and the "
                    "only way to catch one is to have the source on screen.")


class Faithfulness(BaseModel):
    """
    Whether the cited signs were really in the input. `normalised` keeps the
    [citation, matched_form] pairs so each match can be inspected, and
    "vacuous" means nothing was cited, which is not a pass.
    """
    supplied_distinct: int
    citations: int
    citations_distinct: int
    exact: list[str]
    normalised: list[list[str]]
    invented: list[str]
    verdict: str = Field(
        description="pass | pass_after_normalising | fail | vacuous")
    coverage_of_input: float | None


class ProseScan(BaseModel):
    """
    Sign-shaped tokens found in the prose itself, which catches invented signs
    that never appear in `signs_used`.
    """
    fields: list[str]
    candidates: list[str]
    unmatched: list[str]


class Positions(BaseModel):
    """How much of the tablet Domain 3 was actually shown."""
    named: int
    total: int
    lines: int


class SummaryResponse(BaseModel):
    """
    One Domain 3 result with its audit attached.

    Domain 3 is not deterministic, so `cached: true` here only means "the
    summary I already showed you for this tablet". Use `regenerate=true` for a
    fresh call.
    """
    tablet: str
    model: str
    attempts: int = Field(
        description="1, or 2 when the first response was truncated at the "
                    "token cap and the serving path retried. An evaluation "
                    "must not retry; a serving path should.")
    seconds: float
    hit_cap: bool = Field(
        description="the response parsed but the model was still writing at "
                    "the 4096-token cap, so a field was cut short.")
    schema_complete: bool
    summary: Summary
    metadata: SummaryMetadata
    faithfulness: Faithfulness
    prose_scan: ProseScan
    positions: Positions
    deadline: dict = Field(
        description="whether an explicit deadline was applied to the ollama "
                    "client. ollama builds its client with timeout=None, so "
                    "applied=false means this call could have blocked a "
                    "worker indefinitely.")
    caveats: list[str] = Field(min_length=1)
    cached: bool = False


class Domain3Error(BaseModel):
    """A Domain 3 failure, written for a person, with what to do about it."""
    error: str
    detail: str
    hint: str
    tablet: str
