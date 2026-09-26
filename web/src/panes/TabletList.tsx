/**
 * The tablet browser: the 30 evaluated tablets, and any photographs the user
 * has uploaded, in two separate sections. Uploads have no ground truth and
 * were never evaluated, so they never appear under the evaluated heading. The
 * counts shown are the detector's own; human annotation counts are never shown.
 */

import { useEffect, useRef, useState } from "react";
import type { TabletSummary, Upload } from "../types";

interface Props {
  tablets: TabletSummary[];
  populationNote: string;
  uploads: Upload[];
  uploadNote: string;
  selected: string | null;
  onSelect: (tablet: string) => void;
  /** Runs the upload. Resolves when Domain 1 has finished with the new image. */
  onUpload: (file: File) => Promise<void>;
  uploading: boolean;
  uploadError: string | null;
}

/** One row, shared by both sections so they look the same. */
function Row({
  id, selected, onSelect, lines,
}: {
  id: string;
  selected: boolean;
  onSelect: () => void;
  lines: React.ReactNode[];
}) {
  return (
    <li>
      <button
        type="button"
        className={selected ? "tablet tablet--on" : "tablet"}
        aria-current={selected}
        onClick={onSelect}
      >
        <span className="tablet__id">{id}</span>
        {lines}
      </button>
    </li>
  );
}

export function TabletList({
  tablets, populationNote, uploads, uploadNote,
  selected, onSelect, onUpload, uploading, uploadError,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [elapsed, setElapsed] = useState(0);
  const started = useRef(0);

  // Count elapsed seconds while the detector runs; there's no real progress to show.
  useEffect(() => {
    if (!uploading) return;
    started.current = Date.now();
    setElapsed(0);
    const id = setInterval(
      () => setElapsed((Date.now() - started.current) / 1000), 500);
    return () => clearInterval(id);
  }, [uploading]);

  return (
    <div className="app__left">
    <section className="pane pane--list" aria-label="Test tablets">
      <header className="pane__head">
        <h2>Tablets</h2>
        <span className="pane__count">{tablets.length}</span>
      </header>

      <p className="note">{populationNote}</p>

      <ul className="tablets">
        {tablets.map((t) => (
          <Row
            key={t.tablet}
            id={t.tablet}
            selected={t.tablet === selected}
            onSelect={() => onSelect(t.tablet)}
            lines={[
              <span className="tablet__meta" key="a">
                {t.detections_above_conf.toLocaleString()} of{" "}
                {t.detections_in_file.toLocaleString()} boxes clear conf {t.conf}
              </span>,
              <span className="tablet__meta dim" key="b">
                {t.image.width} × {t.image.height} px
                {t.detector_seconds === null
                  ? ""
                  : ` · detector ${t.detector_seconds.toFixed(1)} s`}
              </span>,
            ]}
          />
        ))}
      </ul>

    </section>

    {/* Uploads get their own labelled section, so screen readers and tests
        see the two lists as separate regions. */}
    <section className="pane pane--list" aria-label="Your uploads">
      <header className="pane__head">
        <h2>Your uploads</h2>
        <span className="pane__count">{uploads.length}</span>
      </header>

      <p className="note note--warn">{uploadNote}</p>

      <input
        ref={fileRef}
        type="file"
        accept="image/jpeg,image/png,image/tiff"
        className="visually-hidden"
        aria-label="Photograph to read"
        onChange={(e) => {
          const file = e.target.files?.[0];
          // Clear the input so picking the same file again still triggers a change.
          e.target.value = "";
          if (file) void onUpload(file);
        }}
      />
      <button
        type="button"
        className="primary"
        disabled={uploading}
        onClick={() => fileRef.current?.click()}
      >
        {uploading ? "Reading the photograph…" : "Upload a photograph"}
      </button>

      {/* Name the readable formats up front: the file picker hides HEIC files,
          so the server's HEIC message would never be seen. */}
      {!uploading && (
        <p className="note dim">
          JPEG, PNG and TIFF can be read. HEIC cannot, and iPhones produce HEIC
          by default, so export or convert to JPEG first.
        </p>
      )}

      {uploading && (
        <div className="waiting" role="status" aria-live="polite">
          <p className="waiting__clock">{elapsed.toFixed(0)}s elapsed</p>
          <p className="waiting__text">
            Running the detector over the photograph in 512 pixel tiles. If the
            signs turn out to be a different size from the tablets the detector
            was trained on, it resizes the image once and reads it again, so
            this can take twice as long as it first appears.
          </p>
        </div>
      )}

      {uploadError && (
        <div className="strip strip--bad">
          <strong>That photograph could not be read.</strong> {uploadError}
        </div>
      )}

      {uploads.length === 0 && !uploading && (
        <p className="note dim">
          Nothing uploaded yet. A photograph you add is read by the same three
          models, at the same settings, and is kept separate from the 30 above.
        </p>
      )}

      <ul className="tablets">
        {uploads.map((u) => (
          <Row
            key={u.tablet}
            id={u.upload.filename || u.tablet}
            selected={u.tablet === selected}
            onSelect={() => onSelect(u.tablet)}
            lines={[
              <span className="tablet__meta" key="a">
                {u.detections_above_conf.toLocaleString()} of{" "}
                {u.detections_in_file.toLocaleString()} boxes clear conf {u.conf}
              </span>,
              <span className="tablet__meta dim" key="b">
                {u.image.width} × {u.image.height} px
                {u.upload.scale.applied
                  ? ` · resized ${u.upload.scale.factor.toFixed(2)} from ${u.upload.original_size[0]} × ${u.upload.original_size[1]}`
                  : " · not resized"}
              </span>,
              <span className="tablet__meta dim" key="c">
                not evaluated · no ground truth
              </span>,
            ]}
          />
        ))}
      </ul>
    </section>
    </div>
  );
}
