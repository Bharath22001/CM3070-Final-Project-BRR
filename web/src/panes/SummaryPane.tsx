/**
 * Domain 3: the tentative reading, with the audit that makes it safe to show.
 * The output is never called a translation.
 *
 * The model's `confidence` is shown as its own claim, not as a system
 * confidence, since nothing checks it. A "vacuous" faithfulness verdict (no
 * signs cited) is shown as a failure, not a pass. Results aren't reproducible,
 * so `cached` only means "already shown to you". While waiting I show elapsed
 * seconds and measured figures instead of a fake progress bar.
 *
 * The call is a button rather than automatic, because a call usually takes
 * about 5 s, can take a minute, and the first one after idle takes over 80 s
 * while the model loads. The button says so before it's pressed.
 */

import { useEffect, useRef, useState } from "react";
import { ApiError, summariseTablet } from "../api";
import type { Faithfulness, SummaryResult } from "../types";

interface Props {
  tablet: string | null;
  /** True once the analysis has arrived; Domain 3 reads the fused sequence. */
  analysisReady: boolean;
  /**
   * Called on a 503 (Ollama stopped), the one moment the health strip is
   * certainly out of date.
   */
  onDomain3Unreachable: () => void;
}

/**
 * What the pane says while it waits, by elapsed seconds. The figures are
 * measured (five tablets, one call each), and the thresholds mark where a
 * different explanation becomes more likely.
 */
function waitingMessage(seconds: number): string {
  if (seconds < 6) {
    return (
      "Generating. Five tablets were measured at a median of 5.1 seconds, so " +
      "this is the ordinary case."
    );
  }
  if (seconds < 20) {
    return (
      "Past the measured median of 5.1 seconds. The slowest of the five " +
      "measured tablets took 53 seconds, so this is still within what has " +
      "been seen."
    );
  }
  return (
    "Still generating. The first call after Ollama has been idle loads the " +
    "model into video memory before any generation starts: measured once at " +
    "86 seconds against 3.8 seconds for the same tablet immediately " +
    "afterwards, so almost all of that wait is loading, not reading."
  );
}

/**
 * How a faithfulness verdict is shown. "vacuous" means the model cited nothing,
 * so it's a failure. "pass" isn't called "faithful", since the checks only
 * cover sign names; the catalogue metadata is shown so invented names or places
 * can be spotted.
 */
function verdictView(f: Faithfulness): {
  tone: "good" | "warn" | "bad";
  head: string;
  body: string;
} {
  switch (f.verdict) {
    case "pass":
      return {
        tone: "good",
        head: "Every cited sign was in the input",
        body:
          `All ${f.citations_distinct} distinct signs the model cited appear ` +
          `in the ${f.supplied_distinct} it was shown, matched exactly. This ` +
          `says the citations are real; it says nothing about the prose.`,
      };
    case "pass_after_normalising":
      return {
        tone: "warn",
        head: "Cited signs matched only after normalising",
        body:
          `${f.exact.length} citations matched exactly and ` +
          `${f.normalised.length} matched only after normalising the written ` +
          `form. The pairs are listed below so the match can be judged rather ` +
          `than taken on trust.`,
      };
    case "vacuous":
      return {
        tone: "bad",
        head: "THE CHECK DID NOT RUN. The model cited no signs",
        body:
          "There were no citations to audit, so this is an absent result, " +
          "not a clean one. Nothing below constrains the prose above it.",
      };
    case "fail":
      return {
        tone: "bad",
        head: `${f.invented.length} cited signs were never in the input`,
        body:
          "The model cited signs it was not shown. They are listed below. " +
          "This is the failure the check exists to catch.",
      };
    default:
      return {
        tone: "bad",
        head: `Unrecognised verdict: ${f.verdict}`,
        body:
          "The server reported a verdict this pane does not know how to " +
          "read, so it is shown as a failure rather than guessed at.",
      };
  }
}

/**
 * Is this an uploaded photograph rather than one of the 30? Used so the
 * corpus-only metadata message isn't shown under an upload.
 */
function isUploadId(tablet: string): boolean {
  return tablet.startsWith("upload_");
}

export function SummaryPane({ tablet, analysisReady, onDomain3Unreachable }: Props) {
  const isUpload = tablet !== null && isUploadId(tablet);
  const [result, setResult] = useState<SummaryResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [showInput, setShowInput] = useState(false);
  const started = useRef(0);

  // A new tablet clears the previous summary so it never shows under the wrong heading.
  useEffect(() => {
    setResult(null);
    setError(null);
    setRunning(false);
    setShowInput(false);
  }, [tablet]);

  // The elapsed counter, cleared on unmount and when the call finishes.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setElapsed((Date.now() - started.current) / 1000);
    }, 500);
    return () => clearInterval(id);
  }, [running]);

  async function run(regenerate: boolean) {
    if (!tablet) return;
    started.current = Date.now();
    setElapsed(0);
    setRunning(true);
    setError(null);
    try {
      setResult(await summariseTablet(tablet, regenerate));
    } catch (err) {
      const e = err as ApiError;
      setError(e);
      // A 503 means Ollama is down, so re-check the health strip now.
      if (e.status === 503) onDomain3Unreachable();
    } finally {
      setRunning(false);
    }
  }

  if (!tablet) {
    return (
      <section className="pane pane--summary" aria-label="Tentative reading">
        <header className="pane__head"><h2>Tentative reading</h2></header>
        <p className="empty">
          Select a tablet first. The generative stage reads the fused sign
          sequence, so there is nothing for it to read until the tablet has
          been analysed.
        </p>
      </section>
    );
  }

  return (
    <section className="pane pane--summary" aria-label="Tentative reading">
      <header className="pane__head">
        <h2>Tentative reading</h2>
        <span className="pane__count">{tablet}</span>
        {result && !running && (
          <button
            type="button"
            className="ghost"
            onClick={() => void run(true)}
          >
            Generate again
          </button>
        )}
      </header>

      {/* What it is, up front, before the button. */}
      <p className="note note--warn">
        The last stage is a local LLaMA-3.1 8B reading the sign sequence above.
        It offers a genre-level gist and a guess at context. <strong>It does not
        translate the tablet</strong>, and it is working from a sequence in
        which most restored signs are wrong.
      </p>

      {!result && !running && !error && (
        <>
          <p className="note">
            Not run automatically, because it is slow and it is not
            reproducible. Measured on five tablets, one call each: a median of
            5.1 seconds and a slowest of 53 seconds. <strong>The first call
            after the model has been idle is much slower</strong>, because the model
            is being loaded into video memory. Measured twice: 86 and 67
            seconds, against 3.8 seconds for the same tablet on the next call.
          </p>
          <button
            type="button"
            className="primary"
            disabled={!analysisReady}
            onClick={() => void run(false)}
          >
            Offer a tentative reading
          </button>
          {!analysisReady && (
            <p className="note dim">
              Waiting for the detector and the sequence model to finish. The
              generative stage reads their output, not the photograph.
            </p>
          )}
        </>
      )}

      {/* Waiting */}
      {running && (
        <div className="waiting" role="status" aria-live="polite">
          <p className="waiting__clock">{elapsed.toFixed(0)}s elapsed</p>
          <p className="waiting__text">{waitingMessage(elapsed)}</p>
          <p className="note dim">
            There is no progress figure here because nothing in this system
            knows how far along a generation is. The seconds are real; a bar
            would not be.
          </p>
        </div>
      )}

      {/* Error */}
      {error && !running && (
        <div className="strip strip--bad">
          <strong>
            {error.status === 503
              ? "The language model is not reachable."
              : "Domain 3 did not return a usable summary."}
          </strong>{" "}
          {error.message} {error.hint}
          <p className="note">
            Only this pane is affected. The reading and the photograph come
            from Domains 1 and 2 and need no language model.
          </p>
          <button type="button" className="ghost" onClick={() => void run(false)}>
            Try again
          </button>
        </div>
      )}

      {/* Result */}
      {result && !running && (
        <>
          {/* Caveats first, above the prose */}
          <div className="caveats">
            <h3 className="caveats__head">Before you read this</h3>
            <ul>
              {result.caveats.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </div>

          <h3 className="sub">Tentative reading</h3>
          <p className="prose prose--reading">
            {result.summary.tentative_reading}
          </p>

          {/* The model's own estimate, muted like a restored sign so it doesn't look measured. */}
          <p className="claim">
            <span className="claim__label">the model's own estimate of its confidence:</span>{" "}
            <span className="claim__value">{result.summary.confidence}</span>
            <span className="claim__note">
              {" "}Validated against nothing. No position in this pipeline has
              a known correct answer, so there is no truth for this word to be
              right or wrong about.
            </span>
          </p>

          <h3 className="sub">What the model says it is unsure of</h3>
          <p className="prose">{result.summary.uncertainty_notes}</p>

          <h3 className="sub">Historical context, as offered</h3>
          <p className="prose">{result.summary.historical_context}</p>

          {/* The audit panel */}
          <h3 className="sub">Was it faithful to what it was shown?</h3>
          {(() => {
            const f = result.faithfulness;
            const v = verdictView(f);
            return (
              <div className={`audit audit--${v.tone}`}>
                <p className="audit__head">{v.head}</p>
                <p className="audit__body">{v.body}</p>
                <p className="audit__counts">
                  {f.citations} citations ({f.citations_distinct} distinct)
                  against {f.supplied_distinct} distinct signs supplied
                  {f.coverage_of_input === null
                    ? ""
                    : ` · ${(f.coverage_of_input * 100).toFixed(1)}% of the input cited`}
                  {" · verdict "}
                  <code>{f.verdict}</code>
                </p>
                {f.invented.length > 0 && (
                  <p className="audit__list">
                    <strong>signs it was never shown:</strong>{" "}
                    {f.invented.join(", ")}
                  </p>
                )}
                {f.normalised.length > 0 && (
                  <p className="audit__list">
                    <strong>matched after normalising:</strong>{" "}
                    {f.normalised.map((pair) => `${pair[0]} → ${pair[1]}`).join(", ")}
                  </p>
                )}
              </div>
            );
          })()}

          <h3 className="sub">Sign-shaped words inside the prose itself</h3>
          <p className="note">
            The citation audit above reads the model's own list. This reads the
            prose a person actually reads, because an invented sign written into
            a sentence never appears in that list. It matches sign-SHAPED tokens
            only: a king, a city or a date is invisible to it, so an empty
            result is not a clean bill of health.
          </p>
          {result.prose_scan.unmatched.length > 0 ? (
            <div className="audit audit--bad">
              <p className="audit__head">
                {result.prose_scan.unmatched.length} sign-shaped words in the
                prose were not in the input
              </p>
              <p className="audit__list">
                {result.prose_scan.unmatched.join(", ")}
              </p>
              <p className="audit__body">
                Found in: {result.prose_scan.fields.join(", ") || "none"}. This is
                the check that caught an invented sign the citation audit
                passed.
              </p>
            </div>
          ) : (
            <div className="audit audit--warn">
              <p className="audit__head">
                No sign-shaped word in the prose was missing from the input
              </p>
              <p className="audit__body">
                {result.prose_scan.candidates.length} sign-shaped tokens were
                examined. Proper nouns were not, and cannot be by this check.
              </p>
            </div>
          )}

          {/* What the model was given */}
          <h3 className="sub">What the model was given</h3>
          <p className="note">
            On screen because the two checks above cover sign names only. A king
            or a city in the prose can be judged only against this block, by
            eye.
          </p>
          {result.metadata.had_metadata !== result.metadata.had_metadata_at_call && (
            <div className="strip strip--bad">
              <strong>The tablet summarised is not the tablet analysed.</strong>{" "}
              The analysis recorded metadata
              {result.metadata.had_metadata ? " present" : " absent"} while the
              call recorded it
              {result.metadata.had_metadata_at_call ? " present" : " absent"}.
            </div>
          )}
          {/* A <pre> for the metadata block (its line breaks matter) and a
              paragraph for the no-metadata message, so the sentence wraps. */}
          {result.metadata.had_metadata ? (
            <pre className="dump">{result.metadata.catalogue_metadata}</pre>
          ) : (
            <p className="note">
              {isUpload
                ? "No catalogue metadata exists for a photograph you uploaded, " +
                  "and none can: the metadata comes from the Oracc catalogue, " +
                  "which is keyed on a museum P-number this image does not have. " +
                  "The model was given the sign sequence and nothing else."
                : "No catalogue metadata exists for this tablet. Seven of the " +
                  "30 have none."}
            </p>
          )}
          <p className="note">
            The model <strong>claims</strong> it{" "}
            {result.metadata.claimed_metadata_used ? "used" : "did not use"} that
            block; what was actually supplied is{" "}
            {result.metadata.had_metadata ? "a block" : "nothing"}. The claim is
            the unreliable half of that sentence.
          </p>

          <p className="note">
            <button
              type="button"
              className="ghost ghost--inline"
              aria-pressed={showInput}
              onClick={() => setShowInput((s) => !s)}
            >
              {showInput ? "Hide the cited signs" : "Show the cited signs"}
            </button>{" "}
            {result.summary.signs_used.length} citations, one entry per
            occurrence.
          </p>
          {showInput && (
            <pre className="dump dump--wrap">
              {result.summary.signs_used.join(" · ")}
            </pre>
          )}

          {/* Footer */}
          <p className="note dim">
            {result.model} · {result.positions.named} of{" "}
            {result.positions.total} positions were named well enough to show
            it, over {result.positions.lines} lines ·{" "}
            {result.attempts === 2
              ? "two attempts, the first truncated at the token cap"
              : "one attempt"}
            {result.hit_cap ? " · still writing at the cap, so a field is cut short" : ""}
            {result.schema_complete ? "" : " · the schema came back incomplete"} ·{" "}
            {result.seconds.toFixed(2)} s
          </p>
          <p className="note dim">
            {result.cached
              ? `Shown from this server's cache. That claims only that this is the
                 summary it already showed you for this tablet, not that the
                 same text would come back again.`
              : "Generated for this request."}{" "}
            Generation runs at temperature 0.2 with no fixed seed, so pressing
            “Generate again” can legitimately return different prose from the
            same sign sequence. Unlike the analysis, this result is not
            reproducible.
          </p>
        </>
      )}
    </section>
  );
}
