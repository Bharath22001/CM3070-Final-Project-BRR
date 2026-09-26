/**
 * Every call to my API goes through here, so there is one place that knows the
 * urls and one place that turns a failure into something a pane can show.
 * Errors keep their HTTP status and the server's hint, because a 503 (Ollama
 * down), a 409 (nothing to fuse) and a 404 (unknown tablet) mean different things.
 */

import type {
  Analysis, Health, SummaryResult, TabletList, TabletSummary, Upload,
  UploadList,
} from "./types";

// Empty by default, since Vite proxies /api to the FastAPI server. Set
// VITE_API_BASE to point a built bundle at a server somewhere else.
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;
  hint: string;
  /** The server's machine-readable name for the failure, when it sent one. */
  kind: string;

  constructor(status: number, message: string, hint = "", kind = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.hint = hint;
    this.kind = kind;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(BASE + path, init);
  } catch (cause) {
    // fetch only rejects on a network failure, which here nearly always means
    // the API isn't running.
    throw new ApiError(
      0,
      "Could not reach the orchestrator.",
      "Start the API from D:\\FYP with: uvicorn cuneiform.api.main:app",
      "NetworkError",
    );
  }

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let hint = "";
    let kind = "";
    try {
      const body = await response.json();
      // A structured failure puts an object under `detail`; a plain one puts a string.
      const detail = body?.detail;
      if (typeof detail === "string") {
        message = detail;
      } else if (detail && typeof detail === "object") {
        message = detail.detail ?? message;
        hint = detail.hint ?? "";
        kind = detail.error ?? "";
      }
    } catch {
      // A non-JSON error body isn't worth failing over.
    }
    throw new ApiError(response.status, message, hint, kind);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/api/health");
}

export function getTablets(): Promise<TabletList> {
  return request<TabletList>("/api/tablets");
}

/**
 * The url of a tablet's photograph, downscaled for display. Never the overlay
 * files (they have ground-truth boxes drawn on); boxes are drawn from the
 * detection JSON instead.
 */
export function imageUrl(tablet: TabletSummary | string): string {
  const id = typeof tablet === "string" ? tablet : tablet.tablet;
  return `${BASE}/api/tablets/${id}/image`;
}

/**
 * Run Domains 1 and 2 for one tablet. It's a POST because it does real work.
 * The server caches the result per tablet, so a repeat comes back with
 * `cached: true` and the original timings.
 */
export function analyseTablet(tablet: string): Promise<Analysis> {
  return request<Analysis>(`/api/tablets/${tablet}/analyse`, { method: "POST" });
}

/**
 * Ask Domain 3 for a tentative reading of one tablet. This is the slow route:
 * usually about 5 s, up to about a minute, plus the model load the first time.
 * `cached: true` only means "the summary already shown", since generation isn't
 * reproducible; `regenerate` forces a new one. Ollama being down gives a 503.
 */
export function summariseTablet(
  tablet: string,
  regenerate = false,
): Promise<SummaryResult> {
  const query = regenerate ? "?regenerate=true" : "";
  return request<SummaryResult>(`/api/tablets/${tablet}/summarise${query}`, {
    method: "POST",
  });
}

/**
 * Upload a photograph and run Domain 1 on it. The file is sent as the raw
 * request body, since multipart would need an extra package and this has to
 * work offline. It may detect twice if the signs need resizing.
 */
export function uploadTablet(file: File): Promise<Upload> {
  const name = encodeURIComponent(file.name ?? "");
  return request<Upload>(`/api/uploads?filename=${name}`, {
    method: "POST",
    body: file,
  });
}

/** Photographs already uploaded to this server, kept apart from getTablets. */
export function getUploads(): Promise<UploadList> {
  return request<UploadList>("/api/uploads");
}
