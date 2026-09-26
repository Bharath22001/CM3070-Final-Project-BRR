/**
 * The three models' status as three separate lamps, plus the operating point.
 * Domain 3 being down only makes the system "degraded", since the reading
 * doesn't need it. The strip is fetched once and refreshed on request, not
 * polled, because the health check takes about 2 s whenever Ollama is down.
 */

import type { Health } from "../types";

interface Props {
  health: Health | null;
  error: string | null;
  refreshing: boolean;
  onRefresh: () => void;
}

function Lamp({ on, label, detail }: { on: boolean; label: string; detail: string }) {
  return (
    <span className={on ? "lamp lamp--on" : "lamp lamp--off"} title={detail}>
      <span className="lamp__dot" aria-hidden="true" />
      {label}
    </span>
  );
}

export function HealthStrip({ health, error, refreshing, onRefresh }: Props) {
  if (error) {
    return (
      <div className="strip strip--bad">
        <strong>Orchestrator unreachable.</strong> {error}
      </div>
    );
  }

  if (!health) {
    return <div className="strip">Checking the pipeline…</div>;
  }

  const cfg = health.config;

  return (
    <div className="strip">
      <div className="strip__lamps">
        <Lamp
          on={true}
          label="Domain 1 · detector"
          detail="For the 30 reference tablets the YOLOv8s detections were computed ahead of time and are read from disk. For a photograph you upload the detector runs here, in this server."
        />
        <Lamp
          on={health.domain2.loaded}
          label="Domain 2 · sequence model"
          detail={`Checkpoint ${health.domain2.checkpoint}. Loaded at server start.`}
        />
        <Lamp
          on={health.domain3.reachable && health.domain3.model_available === true}
          label="Domain 3 · language model"
          detail={health.domain3.detail ?? `${health.domain3.model} reachable.`}
        />
      </div>

      <div className="strip__config">
        <span className="dim">frozen operating point:</span> conf {cfg.conf} · cell{" "}
        {cfg.cell} · λ {cfg.lambda} · floor {cfg.lm_floor} · min cluster{" "}
        {cfg.min_cluster} · edge{" "}
        {cfg.edge_extrapolation ? "on" : "off"}
        <span className="dim"> · not tuned values; see the caveats</span>
      </div>

      <button type="button" className="ghost" onClick={onRefresh} disabled={refreshing}>
        {refreshing ? "checking…" : "re-check"}
      </button>

      {health.notes.length > 0 && (
        <ul className="strip__notes">
          {health.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
