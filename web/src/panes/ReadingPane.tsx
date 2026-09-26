/**
 * The fused sign sequence, the main pane of the dashboard.
 *
 * It renders `slot.token` from the payload as-is and never rebuilds tokens, so
 * the browser shows exactly what the CLI prints. Each state is marked by more
 * than colour, so it still works for colour-blind readers and in print:
 *
 *   vision     full contrast, upright, no rule          the detector read it
 *   override   accent colour, solid underline, `*`      the LM disagreed and won
 *   restored   muted, italic, dashed underline, `[ ]`   nothing usable from vision
 *   unnamed    dimmest, monospace, dotted outline       declining to name it
 *
 * Restored signs look deliberately less confident, since most of them are
 * wrong. An unnamed position is the system declining to guess: [x] means the
 * LM's answer fell below the floor, a middle dot means it was never asked. The
 * caveats sit above the sequence, fully expanded, so they are read first.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { Analysis, Detection, Line, Slot, SlotState } from "../types";

interface Props {
  tablet: string | null;
  analysis: Analysis | null;
  loading: boolean;
  error: string | null;
  /** The selected position, held by App because the image pane sets it too. */
  picked: number | null;
  onPick: (slotIndex: number | null) => void;
}

/**
 * The legend text. It describes the display only; every claim about the
 * pipeline itself comes from the payload's caveats.
 */
const LEGEND: { state: SlotState; mark: string; title: string; body: string }[] = [
  {
    state: "vision",
    mark: "NAME",
    title: "read by the detector",
    body: "The detector found a sign and the sequence model agreed with it.",
  },
  {
    state: "override",
    mark: "NAME*",
    title: "overridden by the sequence model",
    body:
      "The detector read one sign here and the sequence model read a different " +
      "one. The starred name is the sequence model's.",
  },
  {
    state: "restored",
    mark: "[NAME]",
    title: "restored, and most of these are wrong",
    body:
      "The detector gave nothing usable here: either no box, or a box whose " +
      "class we may not emit. The sequence model proposed a sign and it " +
      "cleared the floor. Read these as guesses, not as readings.",
  },
  {
    state: "unnamed",
    mark: "[x]  ·",
    title: "declining to guess: a result, not a gap",
    body:
      "We believe a sign sits here and we are not entitled to name it. [x] means " +
      "we asked the sequence model and its best answer fell below the floor; the " +
      "middle dot means the position fell outside any block, so we never asked.",
  },
];

/** The class for a slot's state. [x] and the dot look alike, so the reason is in the class too. */
function tokenClass(slot: Slot): string {
  if (slot.state === "unnamed") {
    return slot.unnamed_reason === "outside_block"
      ? "tok tok--unnamed tok--outside"
      : "tok tok--unnamed tok--floor";
  }
  return `tok tok--${slot.state}`;
}

/**
 * What the detector said at a position, for the inspector. A null vision_name
 * can mean no box at all, or a box whose class is outside the 82 emit classes;
 * the two get different wording so this agrees with the image pane.
 */
function detectorSaid(slot: Slot, dets: Detection[]): string {
  if (slot.vision_name !== null) {
    return slot.vision_conf === null
      ? slot.vision_name
      : `${slot.vision_name} at ${slot.vision_conf.toFixed(3)}`;
  }
  const d = slot.detection_i === null ? undefined : dets[slot.detection_i];
  if (d === undefined) return "nothing. No box here";
  return (
    `${d.name} at ${d.conf.toFixed(3)}, a class outside the 82 this ` +
    `pipeline may emit, so its vote was not used`
  );
}

/** The hover detail. Extra only; nothing essential is hidden here. */
function tokenTitle(slot: Slot): string {
  const parts = [`slot ${slot.i} · ${slot.kind}`];
  if (slot.vision_name !== null) {
    parts.push(
      `detector: ${slot.vision_name}` +
        (slot.vision_conf === null ? "" : ` (${slot.vision_conf.toFixed(3)})`),
    );
  } else {
    // A null vision_name may still mean a box with an unusable class, so I don't claim there's no box.
    parts.push("detector: nothing we could use");
  }
  if (slot.lm_posterior !== null) {
    parts.push(`sequence model: ${slot.lm_posterior.toFixed(3)}`);
  }
  return parts.join("\n");
}

/** Consecutive lines that share a segment (a physical surface). */
function bySegment(lines: Line[]): { segment: number; lines: Line[] }[] {
  const groups: { segment: number; lines: Line[] }[] = [];
  for (const line of lines) {
    const last = groups[groups.length - 1];
    if (last && last.segment === line.segment) last.lines.push(line);
    else groups.push({ segment: line.segment, lines: [line] });
  }
  return groups;
}

export function ReadingPane({
  tablet, analysis, loading, error, picked, onPick,
}: Props) {
  const [plain, setPlain] = useState(false);
  const pickedRef = useRef<HTMLButtonElement>(null);

  const pickedSlot = useMemo(() => {
    if (picked === null) return null;
    for (const line of analysis?.lines ?? []) {
      for (const slot of line.slots) if (slot.i === picked) return slot;
    }
    return null;
  }, [picked, analysis]);

  // A position picked by clicking a box may be far down the list, so scroll to it.
  useEffect(() => {
    // Optional call: jsdom has no scrollIntoView, and scrolling is only a nicety.
    pickedRef.current?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [picked]);

  // How many of each state are on screen, counted from what is rendered.
  const tally = useMemo(() => {
    const t: Record<string, number> = {
      vision: 0, override: 0, restored: 0, unnamed: 0,
      below_floor: 0, outside_block: 0,
    };
    for (const line of analysis?.lines ?? []) {
      for (const slot of line.slots) {
        t[slot.state] += 1;
        if (slot.unnamed_reason) t[slot.unnamed_reason] += 1;
      }
    }
    return t;
  }, [analysis]);

  if (!tablet) {
    return (
      <section className="pane pane--reading" aria-label="Reading">
        <header className="pane__head"><h2>Reading</h2></header>
        <p className="empty">Select a tablet to reconstruct its sign sequence.</p>
      </section>
    );
  }

  if (loading || (!analysis && !error)) {
    return (
      <section className="pane pane--reading" aria-label="Reading">
        <header className="pane__head">
          <h2>Reading</h2>
          <span className="pane__count">{tablet}</span>
        </header>
        <p className="empty">
          Running the detector and the sequence model over the full-resolution
          photograph. This takes a few seconds, and noticeably longer on the
          largest tablets.
        </p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="pane pane--reading" aria-label="Reading">
        <header className="pane__head">
          <h2>Reading</h2>
          <span className="pane__count">{tablet}</span>
        </header>
        <div className="strip strip--bad">
          <strong>Could not reconstruct a reading.</strong> {error}
        </div>
      </section>
    );
  }

  const a = analysis as Analysis;
  const groups = bySegment(a.lines);

  return (
    <section className="pane pane--reading" aria-label="Reading">
      <header className="pane__head">
        <h2>Reading</h2>
        <span className="pane__count">{a.tablet}</span>
        <button
          type="button"
          className="ghost"
          aria-pressed={plain}
          onClick={() => setPlain((p) => !p)}
        >
          {plain ? "Marked-up view" : "Plain text"}
        </button>
      </header>

      {/* Caveats first, above the sequence and fully expanded. */}
      <div className="caveats">
        <h3 className="caveats__head">Before you read this</h3>
        <ul>
          {a.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>

      {/* Legend */}
      <ul className="legend" aria-label="What the marks mean">
        {LEGEND.map((entry) => (
          <li key={entry.state} className="legend__row">
            <span className={`tok tok--${entry.state} legend__mark`}>
              {entry.mark}
            </span>
            <span className="legend__text">
              <strong>{entry.title}</strong>
              <span className="legend__count">
                {tally[entry.state]} here
                {entry.state === "unnamed" && tally.unnamed > 0
                  ? ` (${tally.below_floor} below the floor, ${tally.outside_block} outside a block)`
                  : ""}
              </span>
              <span className="legend__body">{entry.body}</span>
            </span>
          </li>
        ))}
      </ul>

      {/* Sequence */}
      {a.lines.length === 0 ? (
        <p className="empty">
          No block of positions reached the minimum length of {a.counts.min_block},
          so no sequence was reconstructed. That is a result about this tablet,
          not a failure of the software.
        </p>
      ) : plain ? (
        <>
          <p className="note">
            The sequence as one string per line, ready to copy. This is the
            same reading, sign for sign, that the command-line version of this
            pipeline prints for {a.tablet}.
          </p>
          <pre className="dump">
            {a.sequence_dump
              .map((r) => `${r.segment}.${r.line}\t${r.text}`)
              .join("\n")}
          </pre>
        </>
      ) : (
        <div className="reading">
          {groups.map((g) => (
            <div key={g.segment} className="segment">
              <h3 className="segment__head">
                Segment {g.segment}
                <span className="dim"> · {g.lines.length} lines</span>
              </h3>
              {g.lines.map((line) => (
                <div key={`${line.segment}.${line.line}`} className="row">
                  <span className="row__no">{line.line}</span>
                  <span className="row__slots">
                    {line.slots.map((slot) => (
                      <button
                        key={slot.i}
                        type="button"
                        ref={picked === slot.i ? pickedRef : undefined}
                        className={
                          tokenClass(slot) +
                          (picked === slot.i ? " tok--picked" : "")
                        }
                        title={tokenTitle(slot)}
                        onClick={() =>
                          onPick(picked === slot.i ? null : slot.i)
                        }
                      >
                        {slot.token}
                      </button>
                    ))}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* Slot inspector */}
      {pickedSlot && !plain && (
        <dl className="inspect" aria-label="Selected position">
          <div>
            <dt>position</dt>
            <dd>
              {pickedSlot.i} · {pickedSlot.kind}
              {pickedSlot.box ? " · has a box" : " · no box"}
            </dd>
          </div>
          <div>
            <dt>shown as</dt>
            <dd className={tokenClass(pickedSlot)}>{pickedSlot.token}</dd>
          </div>
          <div>
            <dt>detector said</dt>
            <dd>
              {detectorSaid(pickedSlot, a.detections)}
            </dd>
          </div>
          <div>
            <dt>sequence model</dt>
            <dd>
              {pickedSlot.lm_posterior === null
                ? "not asked: outside any block"
                : `${pickedSlot.name ?? "its best answer"} at ${pickedSlot.lm_posterior.toFixed(3)}`}
              {pickedSlot.state === "unnamed" &&
                pickedSlot.unnamed_reason === "below_floor" &&
                `, below the ${a.config.lm_floor} floor, so it is not named`}
            </dd>
          </div>
        </dl>
      )}

      {/* Footer */}
      <p className="note">
        {a.counts.slots.toLocaleString()} positions from{" "}
        {a.counts.detections_above_conf.toLocaleString()} boxes over{" "}
        {a.counts.segments} {a.counts.segments === 1 ? "segment" : "segments"},
        at{" "}
        {a.config.is_frozen ? "the frozen operating point" : "a non-frozen operating point"}
        : conf {a.config.conf} · cell {a.config.cell} · λ {a.config.lambda} ·
        edge extrapolation {a.config.edge_extrapolation ? "on" : "off"} · LM
        floor {a.config.lm_floor} · minimum cluster {a.config.min_cluster}.
      </p>
      <p className="note dim">
        {a.cached
          ? `Served from this server's cache. The timings below are the original
             computation's, not this request's: `
          : "Computed for this request: "}
        {Object.entries(a.timings)
          .map(([k, v]) => `${k} ${v.toFixed(2)} s`)
          .join(" · ")}
        .
      </p>
    </section>
  );
}
