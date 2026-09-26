/**
 * The dashboard shell: the tablet browser, the reading, the photograph with the
 * detector's boxes, the tentative reading and the provenance.
 *
 * I don't poll /api/health. When Ollama is stopped each probe takes about 2 s
 * to time out, so instead the strip refreshes on demand and once more whenever
 * a summary comes back 503 (`onDomain3Unreachable`). The selected position
 * lives here because both the reading pane and the image pane set and read it.
 * The interface never calls the output a translation.
 */

import { useCallback, useEffect, useState } from "react";
import {
  analyseTablet, ApiError, getHealth, getTablets, getUploads, uploadTablet,
} from "./api";
import { HealthStrip } from "./panes/HealthStrip";
import { ImagePane } from "./panes/ImagePane";
import { ProvenancePane } from "./panes/ProvenancePane";
import { ReadingPane } from "./panes/ReadingPane";
import { SummaryPane } from "./panes/SummaryPane";
import { TabletList } from "./panes/TabletList";
import type { Analysis, Health, TabletSummary, Upload } from "./types";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  const [tablets, setTablets] = useState<TabletSummary[]>([]);
  const [populationNote, setPopulationNote] = useState("");
  const [listError, setListError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysing, setAnalysing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Uploads are kept in their own state, never merged into `tablets`, because
  // the 30 tablets are the evaluated ones and uploads aren't.
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [uploadNote, setUploadNote] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // The selected position, shared by the reading and image panes: click a
  // sign to light up its box, or a box to light up its sign.
  const [picked, setPicked] = useState<number | null>(null);

  const checkHealth = useCallback(async () => {
    setChecking(true);
    try {
      setHealth(await getHealth());
      setHealthError(null);
    } catch (err) {
      const e = err as ApiError;
      setHealthError(e.hint ? `${e.message} ${e.hint}` : e.message);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    let cancelled = false;
    getTablets()
      .then((list) => {
        if (cancelled) return;
        setTablets(list.tablets);
        setPopulationNote(list.population_note);
        setListError(null);
      })
      .catch((err: ApiError) => {
        if (cancelled) return;
        setListError(err.hint ? `${err.message} ${err.hint}` : err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshUploads = useCallback(async () => {
    try {
      const list = await getUploads();
      setUploads(list.uploads);
      setUploadNote(list.note);
    } catch {
      // A failed upload listing must not take the 30 tablets down with it.
    }
  }, []);

  useEffect(() => {
    void refreshUploads();
  }, [refreshUploads]);

  const handleUpload = useCallback(
    async (file: File) => {
      setUploading(true);
      setUploadError(null);
      try {
        const created = await uploadTablet(file);
        await refreshUploads();
        // Select the new upload straight away so its reading loads.
        setSelected(created.tablet);
      } catch (err) {
        const e = err as ApiError;
        setUploadError(e.hint ? `${e.message} ${e.hint}` : e.message);
      } finally {
        setUploading(false);
      }
    },
    [refreshUploads],
  );

  // Selecting a tablet runs Domains 1 and 2 for it. If the user moves on
  // before the response arrives, I drop the late result so one tablet's
  // reading never shows under another's heading.
  useEffect(() => {
    if (!selected) {
      setAnalysis(null);
      setAnalysisError(null);
      setPicked(null);
      return;
    }
    let cancelled = false;
    setAnalysis(null);
    setAnalysisError(null);
    // A slot index means nothing on a different tablet, so clear it.
    setPicked(null);
    setAnalysing(true);
    analyseTablet(selected)
      .then((result) => {
        if (cancelled) return;
        setAnalysis(result);
      })
      .catch((err: ApiError) => {
        if (cancelled) return;
        setAnalysisError(err.hint ? `${err.message} ${err.hint}` : err.message);
      })
      .finally(() => {
        if (!cancelled) setAnalysing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  // The selected row from either list. An Upload has all the TabletSummary
  // fields, so the image pane doesn't need to know which list it came from.
  const current: TabletSummary | null =
    tablets.find((t) => t.tablet === selected)
    ?? uploads.find((u) => u.tablet === selected)
    ?? null;

  return (
    <div className="app">
      <header className="app__head">
        <h1>Cuneiform pipeline: reading assistant</h1>
        <p className="tagline">
          Three pre-trained models in series: a YOLOv8s detector over tablet
          photographs, a masked sequence model over a logographic sign
          vocabulary, and a local LLaMA-3.1 8B that offers a tentative reading.
          <strong> It does not translate.</strong> The last stage produces a
          genre-level gist, and most restored signs are wrong.
        </p>
      </header>

      <HealthStrip
        health={health}
        error={healthError}
        refreshing={checking}
        onRefresh={() => void checkHealth()}
      />

      <main className="app__body">
        {listError ? (
          <div className="strip strip--bad">
            <strong>Could not load the tablet list.</strong> {listError}
          </div>
        ) : (
          <TabletList
            tablets={tablets}
            populationNote={populationNote}
            uploads={uploads}
            uploadNote={uploadNote}
            selected={selected}
            onSelect={setSelected}
            onUpload={handleUpload}
            uploading={uploading}
            uploadError={uploadError}
          />
        )}

        <div className="app__right">
          <ReadingPane
            tablet={selected}
            analysis={analysis}
            loading={analysing}
            error={analysisError}
            picked={picked}
            onPick={setPicked}
          />
          <ImagePane
            tablet={current}
            analysis={analysis}
            picked={picked}
            onPick={setPicked}
          />
          <SummaryPane
            tablet={selected}
            analysisReady={analysis !== null}
            onDomain3Unreachable={() => void checkHealth()}
          />
          <ProvenancePane
            tablet={selected}
            analysis={analysis}
            upload={uploads.find((u) => u.tablet === selected) ?? null}
          />
        </div>
      </main>

      <footer className="app__foot">
        Everything on this page is read from the orchestrator; nothing is a
        placeholder. The sign sequence is the same sequence{" "}
        <code>live_pipeline.py</code> prints at the command line, the boxes are
        drawn from the detector's own JSON rather than from any pre-rendered
        image, and every count in the provenance pane comes from the payload
        that produced the reading beside it.
      </footer>
    </div>
  );
}
