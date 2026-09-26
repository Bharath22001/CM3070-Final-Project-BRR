/**
 * The API's response shapes, copied from cuneiform/api/models.py. I only
 * declare the fields the panes actually read. The server validates every
 * response against its own schema, so a mistake here shows up as `undefined`
 * rather than as a wrong number.
 */

export interface FrozenConfig {
  conf: number;
  cell: number;
  lambda: number;
  edge_extrapolation: boolean;
  lm_floor: number;
  min_cluster: number;
}

export interface Domain2Health {
  loaded: boolean;
  checkpoint: string;
  warm_seconds: Record<string, number> | null;
}

export interface Domain3Health {
  reachable: boolean;
  model: string;
  /** null when Ollama couldn't be reached at all: "unknown", not "no". */
  model_available: boolean | null;
  models_held: string[] | null;
  detail: string | null;
  checked_in_seconds: number;
}

export interface Health {
  /** "ok" or "degraded", never "error"; Domain 3 being down only affects one pane. */
  status: string;
  tablets: number;
  domain2: Domain2Health;
  domain3: Domain3Health;
  config: FrozenConfig;
  notes: string[];
}

export interface ImageInfo {
  /** The source photograph's size, which is the space the boxes are in,
   *  not the size of the image the route serves. */
  width: number;
  height: number;
  url: string;
  served_max_long_side: number;
}

export interface TabletSummary {
  tablet: string;
  image: ImageInfo;
  detections_in_file: number;
  detections_above_conf: number;
  conf: number;
  /** Can be null, matching the server's `float | None`. */
  detector_seconds: number | null;
}

export interface TabletList {
  tablets: TabletSummary[];
  count: number;
  population_note: string;
}

// The analysis payload (POST /api/tablets/{id}/analyse)

export interface AnalysisConfig extends FrozenConfig {
  /** Whether this analysis ran at the frozen operating point. */
  is_frozen: boolean;
  frozen: FrozenConfig;
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** The four provenance states, worked out on the server. */
export type SlotState = "vision" | "override" | "restored" | "unnamed";

export type UnnamedReason = "below_floor" | "outside_block";

export interface Slot {
  i: number;
  /** "detected" (a box the detector found) or "inserted" (a gap that should
   *  hold a sign). */
  kind: string;
  x: number;
  state: SlotState;
  unnamed_reason: UnnamedReason | null;
  /**
   * Render this string as-is. It's built on the server by the same code the
   * CLI uses, so never rebuild a token from `label` and `name`.
   */
  token: string;
  label: number | null;
  name: string | null;
  vision_label: number | null;
  vision_name: string | null;
  vision_conf: number | null;
  lm_posterior: number | null;
  /** Index into `Analysis.detections`, or null for an inserted position, so
   *  the two panes can point at the same box. */
  detection_i: number | null;
  box: Box | null;
}

export interface Line {
  /** A physical surface of the tablet, ordered top to bottom on the plate. */
  segment: number;
  line: number;
  slots: Slot[];
}

export interface SequenceRow {
  segment: number;
  line: number;
  /** The row as one string, shown verbatim in the plain-text view. */
  text: string;
}

export interface Counts {
  detections_in_file: number;
  detections_above_conf: number;
  detections_kept: number;
  detections_debris: number;
  segments: number;
  slots: number;
  slots_detected: number;
  slots_unknown: number;
  slots_missed: number;
  blocks: number;
  slots_in_blocks: number;
  slots_dropped_in_remnants: number;
  min_block: number;
  box_match_misses: number;
}

export interface RestorationPosterior {
  min: number;
  median: number;
  max: number;
}

export interface Fusion {
  vision_named_positions: number;
  /** How many predictions fusion changed, not how many it fixed. */
  fusion_changed: number;
  restoration_candidates: number;
  restorations_kept: number;
  restorations_demoted: number;
  named_positions: number;
  total_positions: number;
  lm_floor: number;
  restoration_posterior: RestorationPosterior | null;
}

/**
 * One Domain 1 box above the frozen confidence threshold.
 *
 * Coordinates are in full-image source pixels (`Analysis.image.size`), not the
 * downscaled image. The image pane draws into an SVG whose viewBox is that pixel
 * space, so no manual rescaling is needed. `name` can be `label NNN` for the
 * few labels missing from the sign lookup, or `<BROKEN>` for class 0 (damage).
 */
export interface Detection {
  i: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  train_label: number;
  name: string;
  conf: number;
}

export interface Analysis {
  tablet: string;
  config: AnalysisConfig;
  image: { size: number[]; url: string };
  counts: Counts;
  /** null when no block reached the minimum length, which is a real outcome. */
  fusion: Fusion | null;
  lines: Line[];
  sequence_dump: SequenceRow[];
  /** Every box above the threshold, including debris the pipeline dropped
   *  (the boxes no slot points at). */
  detections: Detection[];
  /** Never empty: the caveats travel with the result. */
  caveats: string[];
  timings: Record<string, number>;
  /** When true, `timings` describe the original computation, not this request. */
  cached: boolean;
}

// The Domain 3 payload (POST /api/tablets/{id}/summarise)
//
// There is deliberately no `translation` field; `tentative_reading` keeps its
// real name all the way through.

export interface Domain3Summary {
  tentative_reading: string;
  /**
   * none / low / medium / high. This is the model's own estimate, checked
   * against nothing, so the pane never styles it as a system confidence.
   */
  confidence: string;
  /** One entry per occurrence, so it's longer than the distinct set. */
  signs_used: string[];
  uncertainty_notes: string;
  historical_context: string;
  /** The model's claim that it used the metadata; compare had_metadata. */
  metadata_used: boolean;
}

export interface SummaryMetadata {
  had_metadata: boolean;
  /** Must equal had_metadata; a mismatch means the summary is for the wrong tablet. */
  had_metadata_at_call: boolean;
  claimed_metadata_used: boolean;
  /** The metadata block verbatim, so invented names or places can be spotted. */
  catalogue_metadata: string;
}

export type FaithfulnessVerdict =
  | "pass"
  | "pass_after_normalising"
  | "fail"
  /** The model cited nothing, so the check never ran. */
  | "vacuous";

export interface Faithfulness {
  supplied_distinct: number;
  citations: number;
  citations_distinct: number;
  exact: string[];
  /** [citation, matched_form] pairs, so each match can be checked. */
  normalised: string[][];
  invented: string[];
  verdict: FaithfulnessVerdict | string;
  coverage_of_input: number | null;
}

export interface ProseScan {
  fields: string[];
  candidates: string[];
  /** Sign-shaped tokens in the prose that weren't in the input. Names of people
   *  or places aren't caught, so an empty list isn't a clean bill. */
  unmatched: string[];
}

export interface Positions {
  named: number;
  total: number;
  lines: number;
}

export interface SummaryResult {
  tablet: string;
  model: string;
  /** 2 when the first response was cut off at the token cap and retried. */
  attempts: number;
  seconds: number;
  hit_cap: boolean;
  schema_complete: boolean;
  summary: Domain3Summary;
  metadata: SummaryMetadata;
  faithfulness: Faithfulness;
  prose_scan: ProseScan;
  positions: Positions;
  deadline: Record<string, unknown>;
  caveats: string[];
  /**
   * Weaker than `Analysis.cached`: generation isn't reproducible, so this only
   * means "the summary this server already showed you for this tablet".
   */
  cached: boolean;
}

// Uploaded photographs
//
// Kept as a separate type and list from the 30 evaluation tablets, since an
// upload has never been evaluated and has no ground truth.

export interface UploadPass {
  width: number;
  height: number;
  n_detections: number;
  /** Median detected sign size at the frozen threshold, in this pass's pixels.
   *  null when too few signs were found. */
  median_sign_px: number | null;
  seconds: number;
}

export interface UploadScale {
  applied: boolean;
  factor: number;
  measured_px: number | null;
  /** Why it was or wasn't rescaled, shown verbatim to the user. */
  reason: string;
}

export interface UploadInfo {
  filename: string;
  uploaded_at: string;
  /** [W, H] as uploaded, before EXIF transposition or any rescale. */
  original_size: number[];
  original_format: string;
  exif_transposed: boolean;
  scale: UploadScale;
  /** One entry, or two when a rescale was applied. */
  passes: UploadPass[];
  decode_seconds: number;
}

export interface Upload {
  tablet: string;
  image: ImageInfo;
  detections_in_file: number;
  detections_above_conf: number;
  conf: number;
  detector_seconds: number | null;
  upload: UploadInfo;
  caveats: string[];
}

export interface UploadList {
  uploads: Upload[];
  count: number;
  note: string;
}
