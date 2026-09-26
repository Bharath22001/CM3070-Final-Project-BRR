/**
 * Where every sign on screen came from, and at what settings.
 *
 * This pane stops the dashboard overclaiming. It shows the attrition from the
 * detector's boxes to named positions, what fusion changed, the operating
 * point and the timings, all read straight from the analysis payload with
 * nothing recomputed. Fusion's counts are changes, not fixes, since no live
 * position has a known answer, and that caveat sits above the numbers.
 */

import type { Analysis, Upload } from "../types";

interface Props {
  tablet: string | null;
  analysis: Analysis | null;
  /** Set when the selected tablet is an upload, so any resize is reported with its reason. */
  upload: Upload | null;
}

/** One row of the attrition ladder. `note` is why the drop happened. */
function Row({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td className="num">{value}</td>
      <td className="why">{note}</td>
    </tr>
  );
}

export function ProvenancePane({ tablet, analysis, upload }: Props) {
  if (!tablet || !analysis) {
    return (
      <section className="pane pane--prov" aria-label="Provenance">
        <header className="pane__head"><h2>Provenance</h2></header>
        <p className="empty">
          Select a tablet to see how many boxes became how many signs, and at
          what settings.
        </p>
      </section>
    );
  }

  const a = analysis;
  const c = a.counts;
  const f = a.fusion;
  const n = (v: number) => v.toLocaleString();

  return (
    <section className="pane pane--prov" aria-label="Provenance">
      <header className="pane__head">
        <h2>Provenance</h2>
        <span className="pane__count">{a.tablet}</span>
      </header>

      {/* The uploaded photograph comes first, since everything below was read
          from an image that may have been resized. */}
      {upload && (
        <>
          <h3 className="sub">Your photograph, before it was read</h3>
          <p className="note note--warn">
            No evaluation in this project covers this image and no ground truth
            exists for it, so no accuracy figure shown anywhere in this
            interface describes what you see here.
          </p>
          <table className="ladder">
            <tbody>
              <Row
                label="as uploaded"
                value={`${upload.upload.original_size[0]} × ${upload.upload.original_size[1]}`}
                note={
                  upload.upload.original_format +
                  (upload.upload.exif_transposed
                    ? ", rotated to its upright orientation before anything else"
                    : "")
                }
              />
              {upload.upload.passes.map((p, i) => (
                <Row
                  key={i}
                  label={`detector pass ${i + 1}`}
                  value={`${p.width} × ${p.height}`}
                  note={
                    `${p.n_detections.toLocaleString()} boxes, typical sign ` +
                    (p.median_sign_px === null
                      ? "not measurable"
                      : `${p.median_sign_px.toFixed(0)} px`) +
                    `, ${p.seconds.toFixed(1)} s`
                  }
                />
              ))}
              <Row
                label={upload.upload.scale.applied ? "resized by" : "not resized"}
                value={
                  upload.upload.scale.applied
                    ? `${upload.upload.scale.factor.toFixed(2)}`
                    : "1.00"
                }
                note={upload.upload.scale.reason}
              />
            </tbody>
          </table>
          <p className="note dim">
            The reference tablets' typical sign measures 43 to 113 px across,
            with a median of 59. Those two figures are the whole of the resize
            rule: they were measured on the 30 tablets and nothing about them
            was chosen.
          </p>
        </>
      )}

      {/* The attrition ladder */}
      <h3 className="sub">From photograph to sign sequence</h3>
      <p className="note">
        Each row is what survived the row above it. The drops are the argument:
        a reading of {n(c.slots)} positions is not a reading of a whole tablet.
      </p>
      <table className="ladder">
        <tbody>
          <Row
            label="boxes the detector emitted"
            value={n(c.detections_in_file)}
            note="every box YOLOv8s produced, at any confidence"
          />
          <Row
            label={`still above conf ${a.config.conf}`}
            value={n(c.detections_above_conf)}
            note="the frozen threshold, not a tuned value; see below"
          />
          <Row
            label="kept by the geometry"
            value={n(c.detections_kept)}
            note={`${n(c.detections_debris)} discarded as debris: boxes no line of the reconstructed layout could hold`}
          />
          {/* The note lists all three parts so they add up to the row's value. */}
          <Row
            label="positions in the sequence"
            value={n(c.slots)}
            note={`${n(c.slots_detected)} from a box, ${n(c.slots_missed)} inserted where the spacing says a sign is missing, ${n(c.slots_unknown)} holding a box we may not name (below)`}
          />
          <Row
            label={`positions inside a block of ${c.min_block} or more`}
            value={n(c.slots_in_blocks)}
            note={`${n(c.slots_dropped_in_remnants)} fell in fragments too short to ask the sequence model about`}
          />
          {/* Boxes whose class is outside the 82 emit classes. These aren't the
              [x] positions; those are the demoted restorations in the fusion table. */}
          <Row
            label="positions holding a box we may not name"
            value={n(c.slots_unknown)}
            note="the detector found something here and gave it a class outside the 82 this pipeline is allowed to emit"
          />
        </tbody>
      </table>
      <p className="note dim">
        {c.segments} {c.segments === 1 ? "segment" : "segments"} (physical
        {c.segments === 1 ? " surface" : " surfaces"}) · {c.blocks}{" "}
        {c.blocks === 1 ? "block" : "blocks"}
        {c.box_match_misses > 0
          ? ` · ${n(c.box_match_misses)} positions could not be matched back to their box, so they carry no detector detail`
          : ""}
      </p>

      {/* The fusion */}
      <h3 className="sub">What fusion did</h3>
      {f === null ? (
        <p className="note">
          No block reached the minimum length of {c.min_block}, so the sequence
          model was never asked and nothing was fused. That is a result about
          this tablet, not a failure of the software.
        </p>
      ) : (
        <>
          {/* The caveat, above the numbers. */}
          <div className="caveats caveats--tight">
            <h3 className="caveats__head">Read this before the numbers below</h3>
            <ul>
              <li>
                These count predictions the sequence model <strong>changed</strong>,
                not predictions it <strong>fixed</strong>. Nothing here is an
                accuracy figure: no position on a live tablet has a known
                correct answer, so the system cannot tell an improvement from a
                new error.
              </li>
              <li>
                Domain 2's top-1 accuracy is 22.2%, measured elsewhere on data
                that does have answers. Most restorations are therefore wrong,
                and a large “changed” count is not good news.
              </li>
            </ul>
          </div>
          <table className="ladder">
            <tbody>
              <Row
                label="positions the detector named"
                value={n(f.vision_named_positions)}
                note="a box whose class the detector was willing to call a sign"
              />
              <Row
                label="of those, changed by the sequence model"
                value={n(f.fusion_changed)}
                note="changed, not corrected; shown starred in the reading"
              />
              <Row
                label="empty positions the sequence model offered a sign for"
                value={n(f.restoration_candidates)}
                note="no box at all, or a box whose class we may not emit"
              />
              <Row
                label={`of those, above the ${f.lm_floor} floor and kept`}
                value={n(f.restorations_kept)}
                note="shown bracketed, and most of them are wrong"
              />
              <Row
                label="below the floor and refused"
                value={n(f.restorations_demoted)}
                note="the system declining to name a position it was offered a guess for. These are the [x] marks in the reading: a result, not a gap"
              />
              <Row
                label="named positions in the final reading"
                value={`${n(f.named_positions)} of ${n(f.total_positions)}`}
                note="what the generative stage is shown"
              />
            </tbody>
          </table>
          {f.restoration_posterior && (
            <p className="note dim">
              Sequence-model probability across the restorations it offered:
              lowest {f.restoration_posterior.min.toFixed(3)}, median{" "}
              {f.restoration_posterior.median.toFixed(3)}, highest{" "}
              {f.restoration_posterior.max.toFixed(3)}. These are the model's own
              probabilities, not measured accuracies.
            </p>
          )}
        </>
      )}

      {/* The operating point */}
      <h3 className="sub">The settings this was run at</h3>
      <p className="note">
        {a.config.is_frozen
          ? "This is the frozen configuration: the one every published number in this project was measured at."
          : "⚠ This run is NOT at the frozen configuration, so its numbers are not comparable with any published figure."}{" "}
        None of these is a tuned value, and none should be read as one:
        confidence 0.15 is the midpoint of a broad plateau, the cell factor was
        not selectable at all because the selection metric is non-monotonic on
        that axis, and the floor is precise to about one bucket, roughly 0.06 to
        0.10.
      </p>
      <table className="ladder ladder--config">
        <tbody>
          <Row label="detector confidence" value={String(a.config.conf)}
               note={a.config.conf === a.config.frozen.conf ? "frozen" : `frozen value is ${a.config.frozen.conf}`} />
          <Row label="cell factor" value={String(a.config.cell)}
               note={a.config.cell === a.config.frozen.cell ? "frozen" : `frozen value is ${a.config.frozen.cell}`} />
          <Row label="fusion λ" value={String(a.config.lambda)}
               note={a.config.lambda === a.config.frozen.lambda ? "frozen" : `frozen value is ${a.config.frozen.lambda}`} />
          <Row label="edge extrapolation" value={a.config.edge_extrapolation ? "on" : "off"}
               note="off by default: given a near-tie, take the weaker assumption" />
          <Row label="sequence-model floor" value={String(a.config.lm_floor)}
               note="below this, a proposed sign is refused rather than shown" />
          <Row label="minimum cluster" value={String(a.config.min_cluster)}
               note="shorter fragments are not asked about at all" />
        </tbody>
      </table>

      {/* Timings */}
      <h3 className="sub">Time spent</h3>
      <p className="note">
        {a.cached
          ? "This analysis came from the server's cache, so these are the ORIGINAL computation's timings, not this request's. The cache is honest because the analysis is deterministic: recomputing it gives a byte-identical result."
          : "Measured on this request."}
      </p>
      <table className="ladder">
        <tbody>
          {Object.entries(a.timings).map(([k, v]) => (
            <Row key={k} label={k.replace(/_/g, " ")} value={`${v.toFixed(3)} s`} note="" />
          ))}
        </tbody>
      </table>
    </section>
  );
}
