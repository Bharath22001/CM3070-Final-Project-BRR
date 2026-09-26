/**
 * The photograph, with the detector's boxes drawn on top of it.
 *
 * I draw the boxes from the detection JSON rather than using the overlay
 * images, which also contain the human ground-truth boxes. The SVG sits over
 * the image with its viewBox in source pixels, so the browser maps the boxes
 * onto the rendered image itself, even while resizing, and nothing here
 * rescales coordinates by hand.
 *
 * Boxes are coloured by what the pipeline did with them (the reading pane's
 * four states plus debris), never by detector confidence, so the colour can't
 * be read as "how sure the system is". The confidence is shown as a number.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { imageUrl } from "../api";
import type { Analysis, Detection, Slot, TabletSummary } from "../types";

interface Props {
  tablet: TabletSummary | null;
  analysis: Analysis | null;
  /** The position selected in the reading pane, so both panes agree. */
  picked: number | null;
  onPick: (slotIndex: number | null) => void;
}

/** Detection index to the slot it became. Debris boxes aren't in it. */
function slotsByDetection(analysis: Analysis | null): Map<number, Slot> {
  const map = new Map<number, Slot>();
  for (const line of analysis?.lines ?? []) {
    for (const slot of line.slots) {
      if (slot.detection_i !== null) map.set(slot.detection_i, slot);
    }
  }
  return map;
}

/** The class a box is drawn in: its slot's provenance, or debris. */
function boxClass(slot: Slot | undefined): string {
  if (!slot) return "box box--debris";
  if (slot.state === "unnamed") {
    return slot.unnamed_reason === "outside_block"
      ? "box box--unnamed box--outside"
      : "box box--unnamed box--floor";
  }
  return `box box--${slot.state}`;
}

/** What the pipeline did with a box, in a clause a person can read. */
function outcome(slot: Slot | undefined): string {
  if (!slot) {
    return "discarded as debris. It was not in a large enough cluster to " +
      "join a line, so it contributed nothing to the reading";
  }
  switch (slot.state) {
    case "vision":
      return "kept. The sequence model agreed with the detector";
    case "override":
      return `overridden. The sequence model read ${slot.name} here instead`;
    case "restored":
      // A restored box means the detector's output here was uninformative.
      return `restored as ${slot.name} by the sequence model`;
    case "unnamed":
      return slot.unnamed_reason === "outside_block"
        ? "left unnamed. The position fell outside any block, so the " +
          "sequence model was never asked about it"
        : "left unnamed. The sequence model's best answer did not clear " +
          "the floor, so we decline to name it";
    default:
      return slot.state;
  }
}

export function ImagePane({ tablet, analysis, picked, onPick }: Props) {
  const [loaded, setLoaded] = useState(false);
  const [boxesOn, setBoxesOn] = useState(true);
  const [hovered, setHovered] = useState<number | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  // A new tablet means a new photograph, so reset loading; otherwise the old
  // image would sit on screen under the new heading.
  useEffect(() => {
    setLoaded(false);
    setHovered(null);
  }, [tablet?.tablet]);

  const bySlot = useMemo(() => slotsByDetection(analysis), [analysis]);
  const detections = analysis?.detections ?? [];

  // The box the reading pane is pointing at, if any.
  const pickedDetection = useMemo(() => {
    if (picked === null) return null;
    for (const line of analysis?.lines ?? []) {
      for (const slot of line.slots) {
        if (slot.i === picked) return slot.detection_i;
      }
    }
    return null;
  }, [picked, analysis]);

  if (!tablet) {
    return (
      <section className="pane pane--image" aria-label="Photograph">
        <header className="pane__head"><h2>Photograph</h2></header>
        <p className="empty">Select a tablet to see its photograph.</p>
      </section>
    );
  }

  const shown = hovered ?? pickedDetection;
  const shownDet: Detection | undefined =
    shown === null || shown === undefined ? undefined : detections[shown];
  const debris = detections.length - bySlot.size;

  return (
    <section className="pane pane--image" aria-label="Photograph">
      <header className="pane__head">
        <h2>Photograph</h2>
        <span className="pane__count">{tablet.tablet}</span>
        {detections.length > 0 && (
          <button
            type="button"
            className="ghost"
            aria-pressed={boxesOn}
            onClick={() => setBoxesOn((b) => !b)}
          >
            {boxesOn ? "Hide boxes" : "Show boxes"}
          </button>
        )}
      </header>

      {/* The caveat goes above the picture so it's read before the boxes. */}
      {analysis && boxesOn && detections.length > 0 && (
        <p className="note note--warn boxnote">
          Every box below is the <strong>detector's own</strong>. Nothing on
          this image is human annotation. The colours say what the pipeline
          then <em>did</em> with each box, not how confident it was. The
          detector's own confidence is a number, shown when you point at a box.
        </p>
      )}

      <div className="plate">
        {!loaded && <div className="plate__loading">Loading photograph…</div>}
        {/* Faded while loading rather than visibility: hidden, which would
            hide the image from screen readers. */}
        <div className="plate__stack">
          <img
            ref={imgRef}
            key={tablet.tablet}
            className={loaded ? "plate__img" : "plate__img plate__img--loading"}
            src={imageUrl(tablet)}
            alt={`Photograph of tablet ${tablet.tablet}`}
            onLoad={() => setLoaded(true)}
          />
          {analysis && boxesOn && loaded && detections.length > 0 && (
            <svg
              className="boxes"
              // Source pixels; the browser maps them onto the rendered image.
              viewBox={`0 0 ${analysis.image.size[0]} ${analysis.image.size[1]}`}
              preserveAspectRatio="xMidYMid meet"
              role="group"
              aria-label={`${detections.length} detector boxes`}
            >
              {detections.map((d) => {
                const slot = bySlot.get(d.i);
                const on = shown === d.i;
                return (
                  <rect
                    key={d.i}
                    className={boxClass(slot) + (on ? " box--on" : "")}
                    x={d.x1}
                    y={d.y1}
                    width={Math.max(d.x2 - d.x1, 1)}
                    height={Math.max(d.y2 - d.y1, 1)}
                    onMouseEnter={() => setHovered(d.i)}
                    onMouseLeave={() => setHovered(null)}
                    onClick={() => onPick(slot ? slot.i : null)}
                  >
                    {/* A native tooltip too, which keyboards and screen readers reach. */}
                    <title>
                      {`${d.name} at ${d.conf.toFixed(3)}. ${outcome(slot)}`}
                    </title>
                  </rect>
                );
              })}
            </svg>
          )}
        </div>
      </div>

      {/* A fixed detail strip under the image rather than a floating tooltip,
          so it can be read, screenshotted and tested. */}
      {analysis && detections.length > 0 && (
        <div className={shownDet ? "boxdetail boxdetail--on" : "boxdetail"}>
          {shownDet ? (
            <>
              <span className={boxClass(bySlot.get(shownDet.i)) + " boxdetail__swatch"} />
              <span>
                <strong>
                  {shownDet.name === "<BROKEN>"
                    ? "<BROKEN>, a damage placeholder, not a sign"
                    : shownDet.name}
                </strong>{" "}
                <span className="dim">
                  is what the detector proposed, at confidence{" "}
                  {shownDet.conf.toFixed(3)}.
                </span>{" "}
                {outcome(bySlot.get(shownDet.i))}.
              </span>
            </>
          ) : (
            <span className="dim">
              Point at a box to see the class the detector proposed, its
              confidence, and what the pipeline did with it.
            </span>
          )}
        </div>
      )}

      <p className="note">
        The source photograph, unmarked, served at a{" "}
        {tablet.image.served_max_long_side} px long side. Boxes are drawn from
        the detector's output in the source's own {tablet.image.width} ×{" "}
        {tablet.image.height} px coordinates.
        {analysis && detections.length > 0 && (
          <>
            {" "}
            {detections.length.toLocaleString()} boxes clear conf{" "}
            {analysis.config.conf}
            {debris > 0 && (
              <>
                , of which <strong>{debris}</strong>{" "}
                {debris === 1 ? "was" : "were"} discarded as debris and read as
                nothing
              </>
            )}
            .
          </>
        )}
      </p>
      <p className="note dim">
        Boxes smaller than a few pixels on screen are hard to point at. That is
        a property of the photograph's scale, not of the detection. The
        smallest box in this population renders about 3 px wide.
      </p>
    </section>
  );
}
