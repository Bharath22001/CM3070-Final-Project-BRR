/**
 * Component tests for the dashboard, with fetch stubbed. They check that the
 * components behave correctly given the API's shapes; tests/test_api.py checks
 * the API really returns those shapes. jsdom has no layout, so how the page
 * looks still has to be checked by eye. The honesty checks (no ground truth on
 * screen, never calling the output a translation) matter most.
 */

import {
  cleanup, fireEvent, render, screen, waitFor, within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import App from "./App";
import type {
  Analysis, Detection, Health, Slot, SummaryResult, TabletList, Upload,
  UploadList,
} from "./types";

const TABLET_IDS = Array.from({ length: 30 }, (_, i) => `P33${5000 + i}`);

const health: Health = {
  status: "ok",
  tablets: 30,
  domain2: { loaded: true, checkpoint: "best.pt", warm_seconds: { sequence_model: 1.4 } },
  domain3: {
    reachable: true,
    model: "llama3.1:8b",
    model_available: true,
    models_held: ["llama3.1:8b"],
    detail: null,
    checked_in_seconds: 0.03,
  },
  config: {
    conf: 0.15,
    cell: 2.0,
    lambda: 1.0,
    edge_extrapolation: false,
    lm_floor: 0.08,
    min_cluster: 3,
  },
  notes: [],
};

const tabletList: TabletList = {
  count: 30,
  population_note:
    "These are the 30 tablets with persisted detections -- exactly the population of the live fusion evaluation.",
  tablets: TABLET_IDS.map((id, i) => ({
    tablet: id,
    image: {
      width: 1507,
      height: 2464,
      url: `/api/tablets/${id}/image`,
      served_max_long_side: 1600,
    },
    detections_in_file: 400 + i,
    detections_above_conf: 230 + i,
    conf: 0.15,
    detector_seconds: 1.6,
  })),
};

// An analysis fixture, trimmed from a real payload to the smallest thing that
// covers all four provenance states and both unnamed reasons. `token` is the
// server's string and no test rebuilds one.

function slot(i: number, token: string, over: Partial<Slot> = {}): Slot {
  return {
    i,
    kind: "detected",
    x: 100 + i,
    state: "vision",
    unnamed_reason: null,
    token,
    label: 80,
    name: "I",
    vision_label: 80,
    vision_name: "I",
    vision_conf: 0.55,
    lm_posterior: 0.33,
    detection_i: i,
    box: { x1: 100 + i * 60, y1: 300, x2: 150 + i * 60, y2: 360 },
    ...over,
  };
}

// Five boxes for six positions: three positions are inserted and have no box,
// and box 4 belongs to no position (debris), as happens on real tablets.
const DETECTIONS: Detection[] = [
  { i: 0, x1: 100, y1: 300, x2: 150, y2: 360, train_label: 80, name: "I", conf: 0.554 },
  { i: 1, x1: 160, y1: 300, x2: 210, y2: 360, train_label: 65, name: "DA", conf: 0.346 },
  { i: 2, x1: 220, y1: 300, x2: 270, y2: 360, train_label: 12, name: "AN", conf: 0.802 },
  { i: 3, x1: 280, y1: 300, x2: 330, y2: 360, train_label: 0, name: "<BROKEN>", conf: 0.201 },
  { i: 4, x1: 900, y1: 1800, x2: 930, y2: 1830, train_label: 33, name: "KA", conf: 0.163 },
];

const CAVEAT_TRANSLATE =
  "This system does not translate. Domain 3 produces a genre-level gist, and " +
  "its field is named tentative_reading for that reason.";
const CAVEAT_TOP1 =
  "Domain 2's top-1 accuracy is 22.2%, so MOST restorations -- every " +
  "bracketed [NAME] -- are wrong.";
const CAVEAT_UNNAMED =
  "[x] is the system declining to guess: we hold a box, or a gap we believe " +
  "holds a sign, but nothing we are entitled to call a name. It is a feature, " +
  "not a gap.";

function analysisFor(tablet: string): Analysis {
  const lines = [
    {
      segment: 0,
      line: 0,
      slots: [
        slot(0, "I", { vision_conf: 0.554 }),
        // An override: `name` is the sequence model's reading and `vision_name`
        // the detector's, and they must disagree.
        slot(1, "DIS*", {
          state: "override",
          label: 14, name: "DIS",
          vision_label: 65, vision_name: "DA", vision_conf: 0.346,
          lm_posterior: 0.86,
        }),
      ],
    },
    {
      segment: 1,
      line: 0,
      slots: [
        slot(2, "AN", {
          label: 12, name: "AN",
          vision_label: 12, vision_name: "AN", vision_conf: 0.802,
        }),
        slot(3, "[AS]", {
          state: "restored",
          kind: "inserted",
          vision_label: null,
          vision_name: null,
          vision_conf: null,
          lm_posterior: 0.13,
          detection_i: null,
          box: null,
        }),
        slot(4, "[x]", {
          state: "unnamed",
          unnamed_reason: "below_floor",
          label: null,
          name: null,
          vision_label: null,
          vision_name: null,
          vision_conf: null,
          lm_posterior: 0.077,
          // The <BROKEN> box: a damage placeholder with no name.
          detection_i: 3,
          box: { x1: 280, y1: 300, x2: 330, y2: 360 },
        }),
        slot(5, "·", {
          state: "unnamed",
          unnamed_reason: "outside_block",
          kind: "inserted",
          label: null,
          name: null,
          vision_label: null,
          vision_name: null,
          vision_conf: null,
          lm_posterior: null,
          detection_i: null,
          box: null,
        }),
      ],
    },
  ];

  const frozen = {
    conf: 0.15, cell: 2.0, lambda: 1.0,
    edge_extrapolation: false, lm_floor: 0.08, min_cluster: 3,
  };

  return {
    tablet,
    config: { ...frozen, is_frozen: true, frozen },
    image: { size: [1178, 2526], url: `/api/tablets/${tablet}/image` },
    counts: {
      detections_in_file: 255, detections_above_conf: 141, detections_kept: 141,
      detections_debris: 0, segments: 2, slots: 6, slots_detected: 4,
      slots_unknown: 1, slots_missed: 1, blocks: 2, slots_in_blocks: 6,
      slots_dropped_in_remnants: 0, min_block: 4, box_match_misses: 0,
    },
    fusion: {
      vision_named_positions: 3, fusion_changed: 1, restoration_candidates: 2,
      restorations_kept: 1, restorations_demoted: 1, named_positions: 4,
      total_positions: 6, lm_floor: 0.08,
      restoration_posterior: { min: 0.06, median: 0.13, max: 0.89 },
    },
    lines,
    detections: DETECTIONS,
    // Built the way the server builds it: each row's tokens joined by a space.
    sequence_dump: lines.map((l) => ({
      segment: l.segment,
      line: l.line,
      text: l.slots.map((sl) => sl.token).join(" "),
    })),
    caveats: [CAVEAT_TRANSLATE, CAVEAT_TOP1, CAVEAT_UNNAMED],
    timings: { geometry: 0.006, model_load: 0.053, fusion: 2.518, total: 2.592 },
    cached: false,
  };
}

// A Domain 3 fixture, from a real payload. The default verdict is
// pass_after_normalising so the middle tier gets rendered; vacuous and fail get
// their own tests. The caveats are checked by DOM order, not just presence.

const D3_CAVEAT_NOT_TRANSLATION =
  "THIS IS NOT A TRANSLATION. Domain 3 produces a genre-level gist from a " +
  "sign sequence, which is why the field is called tentative_reading.";
const D3_CAVEAT_TOP1 =
  "Domain 2's top-1 accuracy is 22.2%, so most restored signs are wrong.";
const D3_CAVEAT_SIGN_NAMES_ONLY =
  "Both faithfulness checks below cover SIGN NAMES ONLY. A proper noun in " +
  "the prose is invisible to them, so a PASS is not evidence the prose is " +
  "faithful.";
const D3_CAVEAT_NOT_REPRODUCIBLE =
  "Generation runs at temperature 0.2 with no fixed seed, so a second call " +
  "can return a different summary.";

const TENTATIVE_READING =
  "A short administrative record listing quantities against personal names, " +
  "of a kind common in the Neo-Assyrian archives.";

function summaryFor(tablet: string, over: Partial<SummaryResult> = {}): SummaryResult {
  return {
    tablet,
    model: "llama3.1:8b",
    attempts: 1,
    seconds: 5.12,
    hit_cap: false,
    schema_complete: true,
    summary: {
      tentative_reading: TENTATIVE_READING,
      confidence: "low",
      signs_used: ["AN", "I", "DIS", "AN"],
      uncertainty_notes:
        "Many positions carry no name at all, so the sequence is fragmentary.",
      historical_context:
        "Neo-Assyrian administrative tablets of this shape are typically " +
        "seventh century BCE.",
      metadata_used: true,
    },
    metadata: {
      had_metadata: true,
      had_metadata_at_call: true,
      claimed_metadata_used: true,
      catalogue_metadata: "SAA 01 001. Letter. Nineveh.",
    },
    faithfulness: {
      supplied_distinct: 4,
      citations: 4,
      citations_distinct: 3,
      exact: ["AN", "I"],
      normalised: [["DIS", "DIŠ"]],
      invented: [],
      verdict: "pass_after_normalising",
      coverage_of_input: 0.75,
    },
    prose_scan: { fields: [], candidates: ["AN", "DIS"], unmatched: [] },
    positions: { named: 4, total: 6, lines: 2 },
    deadline: { applied: true, connect: 3.0, read: 300.0 },
    caveats: [
      D3_CAVEAT_NOT_TRANSLATION,
      D3_CAVEAT_TOP1,
      D3_CAVEAT_SIGN_NAMES_ONLY,
      D3_CAVEAT_NOT_REPRODUCIBLE,
    ],
    cached: false,
    ...over,
  };
}

// The uploaded population. Empty by default, like a fresh server, so the older
// tests see what they always saw.

const UPLOAD_NOTE =
  "Photographs uploaded to this server. No evaluation in this project covers " +
  "them and no ground truth exists for them, so no accuracy figure shown " +
  "anywhere in this interface describes these.";

const emptyUploads: UploadList = { uploads: [], count: 0, note: UPLOAD_NOTE };

function uploadFixture(over: Partial<Upload> = {}): Upload {
  return {
    tablet: "upload_abc123def0",
    image: {
      width: 1109, height: 2377,
      url: "/api/tablets/upload_abc123def0/image",
      served_max_long_side: 1600,
    },
    detections_in_file: 258,
    detections_above_conf: 148,
    conf: 0.15,
    detector_seconds: 1.1,
    upload: {
      filename: "field_photo.jpg",
      uploaded_at: "2026-09-20T04:10:00",
      original_size: [2356, 5052],
      original_format: "JPEG",
      exif_transposed: false,
      scale: {
        applied: true,
        factor: 0.471,
        measured_px: 124.3,
        reason:
          "the typical sign measured 124 px across, outside the 43 to 113 px " +
          "range spanned by the 30 reference tablets, so the photograph was " +
          "resized by 0.47 to bring it to 59 px and read again.",
      },
      passes: [
        { width: 2356, height: 5052, n_detections: 460, median_sign_px: 124.3, seconds: 4.8 },
        { width: 1109, height: 2377, n_detections: 258, median_sign_px: 61.8, seconds: 1.1 },
      ],
      decode_seconds: 0.3,
    },
    caveats: [
      "This photograph is not part of any evaluation in this project.",
      "No ground truth exists for this photograph.",
      "The detector was fine-tuned on 81 annotated museum plates.",
    ],
    ...over,
  };
}

function stubFetch(
  overrides: {
    health?: unknown;
    tablets?: unknown;
    analysis?: unknown;
    /** Part 7: the Domain 3 payload, or a failure to render. */
    summary?: unknown;
    summaryStatus?: number;
    /** The uploaded population. Defaults to none, like a fresh server. */
    uploads?: unknown;
    /** What POST /api/uploads returns. */
    created?: unknown;
  } = {},
) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    let body: unknown;
    if (url.includes("/api/health")) {
      body = overrides.health ?? health;
    } else if (url.includes("/api/uploads")) {
      // POST carries ?filename= and GET doesn't, so the stub can tell them apart.
      body = url.includes("filename=")
        ? (overrides.created ?? uploadFixture())
        : (overrides.uploads ?? emptyUploads);
    } else if (url.includes("/summarise")) {
      const id = url.split("/")[3];
      if (overrides.summaryStatus && overrides.summaryStatus >= 400) {
        return {
          ok: false,
          status: overrides.summaryStatus,
          statusText: "Service Unavailable",
          json: async () => ({
            detail: {
              error: "ollama_unreachable",
              detail: "Failed to connect to Ollama.",
              hint: "Start the Ollama tray app, then try again.",
              tablet: id,
            },
          }),
        } as Response;
      }
      body = overrides.summary ?? summaryFor(id);
    } else if (url.endsWith("/analyse")) {
      const id = url.split("/")[3];
      body = overrides.analysis ?? analysisFor(id);
    } else {
      body = overrides.tablets ?? tabletList;
    }
    return {
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => body,
    } as Response;
  });
}

beforeEach(() => {
  vi.stubGlobal("fetch", stubFetch());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Part 4: the tablet browser", () => {
  test("lists all thirty tablets, with the population note", async () => {
    render(<App />);

    const list = await screen.findByRole("region", { name: /test tablets/i });
    await waitFor(() => {
      expect(within(list).getAllByRole("button")).toHaveLength(30);
    });

      // The note says these are the evaluation's tablets, not a sample.
    expect(within(list).getByText(/30 tablets with persisted detections/i))
      .toBeTruthy();
  });

  test("each row states both box counts in one readable clause", async () => {
    // Both numbers must be in one clause: the kept boxes and the total.
    render(<App />);
    const list = await screen.findByRole("region", { name: /test tablets/i });
    const first = (await within(list).findAllByRole("button"))[0];

    expect(first.textContent).toMatch(/\d+ of \d+ boxes clear conf 0\.15/);
    expect(first.textContent).not.toMatch(/\(of /);
  });

  test("clicking a tablet shows that tablet's photograph", async () => {
    const user = userEvent.setup();
    render(<App />);

    const target = TABLET_IDS[7];
    const row = await screen.findByRole("button", { name: new RegExp(target) });

    // Before the click there is a prompt, not a picture (scoped to the photograph pane).
    expect(screen.queryByRole("img")).toBeNull();
    const plate = screen.getByRole("region", { name: /photograph/i });
    expect(within(plate).getByText(/select a tablet/i)).toBeTruthy();

    await user.click(row);

    const img = await screen.findByRole("img", {
      name: new RegExp(`photograph of tablet ${target}`, "i"),
    });
    expect(img.getAttribute("src")).toBe(`/api/tablets/${target}/image`);
  });

  test("selecting a second tablet replaces the first photograph", async () => {
    // A stale image under a new tablet's heading would be a wrong picture that looks right.
    const user = userEvent.setup();
    render(<App />);

    const first = TABLET_IDS[0];
    const second = TABLET_IDS[1];

    await user.click(await screen.findByRole("button", { name: new RegExp(first) }));
    expect(screen.getByRole("img").getAttribute("src")).toContain(first);

    await user.click(screen.getByRole("button", { name: new RegExp(second) }));
    await waitFor(() => {
      expect(screen.getByRole("img").getAttribute("src")).toContain(second);
    });
    expect(screen.queryByRole("img")?.getAttribute("src")).not.toContain(first);
  });
});

describe("Part 5: the reading pane", () => {
  /** Select a tablet and wait for its reading to arrive. */
  async function openReading(id = TABLET_IDS[3]) {
    const user = userEvent.setup();
    const view = render(<App />);
    await user.click(await screen.findByRole("button", { name: new RegExp(id) }));
    const pane = await screen.findByRole("region", { name: /^reading$/i });
    await waitFor(() => {
      expect(within(pane).getByText("AN")).toBeTruthy();
    });
    return { user, pane, container: view.container };
  }

  test("renders the server's tokens verbatim, and the rows reproduce the CLI dump", async () => {
    // Each rendered row must re-join to the server's sequence_dump string,
    // which is exactly what the CLI prints.
    const { pane } = await openReading();
    const expected = analysisFor(TABLET_IDS[3]).sequence_dump;

    const rows = pane.querySelectorAll(".row__slots");
    expect(rows).toHaveLength(expected.length);
    rows.forEach((row, i) => {
      const rendered = Array.from(row.querySelectorAll("button"))
        .map((b) => b.textContent)
        .join(" ");
      expect(rendered).toBe(expected[i].text);
    });
  });

  test("the four provenance states are visually distinct, and restored is not styled as vision", async () => {
    // jsdom can't check colour or contrast, but it can check each state gets its own class.
    const { pane } = await openReading();
    const cls = (text: string) =>
      within(pane).getAllByRole("button", { name: text })[0].className;

    const vision = cls("AN");
    const override = cls("DIS*");
    const restored = cls("[AS]");
    const belowFloor = cls("[x]");
    const outside = cls("·");

    const all = [vision, override, restored, belowFloor, outside];
    expect(new Set(all).size).toBe(all.length);

    // The class names themselves, since the stylesheet keys on them.
    expect(vision).toContain("tok--vision");
    expect(override).toContain("tok--override");
    expect(restored).toContain("tok--restored");
    expect(restored).not.toContain("tok--vision");
    expect(belowFloor).toContain("tok--floor");
    expect(outside).toContain("tok--outside");
  });

  test("the legend explains all four states, and says unnamed is a refusal rather than a gap", async () => {
    const { pane } = await openReading();
    const legend = within(pane).getByRole("list", { name: /what the marks mean/i });
    const text = legend.textContent ?? "";

    expect(text).toMatch(/read by the detector/i);
    expect(text).toMatch(/overridden by the sequence model/i);
    expect(text).toMatch(/most of these are wrong/i);
    // [x] is the system declining, not failing.
    expect(text).toMatch(/declining to guess/i);
    expect(text).toMatch(/not entitled to name it/i);
    // And the two unnamed reasons are told apart.
    expect(text).toMatch(/below the floor/i);
    expect(text).toMatch(/outside any block/i);
  });

  test("the caveats are on screen in full, and ABOVE the sequence", async () => {
    // Position, not just presence: the caveats must come before the sequence.
    const { pane } = await openReading();
    const caveats = pane.querySelector(".caveats");
    const sequence = pane.querySelector(".reading");
    expect(caveats).not.toBeNull();
    expect(sequence).not.toBeNull();

    for (const c of analysisFor(TABLET_IDS[3]).caveats) {
      expect(caveats?.textContent).toContain(c);
    }

    // DOCUMENT_POSITION_FOLLOWING: the sequence comes after the caveats.
    expect(
      caveats!.compareDocumentPosition(sequence!) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test("the plain-text view shows the dump the CLI comparison is made against", async () => {
    const { user, pane } = await openReading();
    await user.click(within(pane).getByRole("button", { name: /plain text/i }));

    const dump = pane.querySelector(".dump");
    expect(dump).not.toBeNull();
    for (const row of analysisFor(TABLET_IDS[3]).sequence_dump) {
      expect(dump?.textContent).toContain(row.text);
    }
  });

  test("clicking a position shows what each model actually said", async () => {
    const { user, pane } = await openReading();
    await user.click(within(pane).getAllByRole("button", { name: "[AS]" })[0]);

    const inspector = await within(pane).findByLabelText(/selected position/i);
    const text = inspector.textContent ?? "";

    // A restoration at an inserted position, where "no box" really is true.
    expect(text).toMatch(/nothing\. No box here/i);
    expect(text).toMatch(/0\.130/);
  });

  test("a box whose class we may not emit is not reported as no box", async () => {
    // A null vision_name can also mean a box whose class is outside the 82 emit
    // classes. The inspector must name that box, not say "no box", so it agrees
    // with the photograph pane. The fixture's [x] slot holds such a box.
    const { user, pane } = await openReading();
    await user.click(within(pane).getAllByRole("button", { name: "[x]" })[0]);

    const inspector = await within(pane).findByLabelText(/selected position/i);
    const text = inspector.textContent ?? "";

    expect(text).not.toMatch(/No box here/i);
    expect(text).toMatch(/<BROKEN>/);
    expect(text).toMatch(/0\.201/);
    // It gives the reason the vote was dropped, not just the class.
    expect(text).toMatch(/outside the 82/i);
  });

  test("selecting another tablet clears the previous reading before the new one arrives", async () => {
    // One tablet's signs must never stay under another tablet's id.
    const { user, pane } = await openReading(TABLET_IDS[3]);
    expect(within(pane).getByText("AN")).toBeTruthy();

    await user.click(
      screen.getByRole("button", { name: new RegExp(TABLET_IDS[9]) }),
    );
    const after = await screen.findByRole("region", { name: /^reading$/i });
    await waitFor(() => {
      expect(within(after).getByText(new RegExp(TABLET_IDS[9]))).toBeTruthy();
    });
  });
});

describe("Part 6: the detector's boxes on the photograph", () => {
  /** Select a tablet and wait for its boxes to be drawn. */
  async function openImage(id = TABLET_IDS[3]) {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: new RegExp(id) }));
    const pane = await screen.findByRole("region", { name: /photograph/i });
    // The overlay only draws after the image loads; jsdom never fires load itself.
    fireEvent.load(within(pane).getByRole("img"));
    await waitFor(() => {
      expect(pane.querySelectorAll("rect").length).toBe(DETECTIONS.length);
    });
    return { user, pane };
  }

  test("draws one box per detection, in SOURCE pixel coordinates", async () => {
    // The mapping is declared once as the SVG viewBox; coordinates are never pre-scaled in JS.
    const { pane } = await openImage();
    const svg = pane.querySelector("svg.boxes");
    expect(svg?.getAttribute("viewBox")).toBe("0 0 1178 2526");

    const rects = Array.from(pane.querySelectorAll("rect"));
    expect(rects).toHaveLength(DETECTIONS.length);
    rects.forEach((rect, i) => {
      const d = DETECTIONS[i];
      expect(rect.getAttribute("x")).toBe(String(d.x1));
      expect(rect.getAttribute("y")).toBe(String(d.y1));
      expect(rect.getAttribute("width")).toBe(String(d.x2 - d.x1));
      expect(rect.getAttribute("height")).toBe(String(d.y2 - d.y1));
    });
  });

  test("pointing at a box names the class and the confidence the DETECTOR gave it", async () => {
    const { pane } = await openImage();
    const rect = pane.querySelectorAll("rect")[1]; // DA at 0.346, overridden

    fireEvent.mouseEnter(rect);
    const detail = pane.querySelector(".boxdetail");
    const text = detail?.textContent ?? "";

    expect(text).toContain("DA");
    expect(text).toContain("0.346");
    // And what the pipeline then did, so the confidence isn't read as the final reading's.
    expect(text).toMatch(/overridden/i);
    expect(text).toMatch(/DIS/); // the name the sequence model put there
  });

  test("a discarded box is drawn, and says it contributed nothing", async () => {
    // Debris boxes are drawn, not hidden, so the detector doesn't look tidier than it is.
    const { pane } = await openImage();
    const rects = Array.from(pane.querySelectorAll("rect"));
    const debris = rects[4];
    expect(debris.getAttribute("class")).toContain("box--debris");

    fireEvent.mouseEnter(debris);
    expect(pane.querySelector(".boxdetail")?.textContent)
      .toMatch(/discarded as debris/i);
  });

  test("box colour is the fusion outcome, never the detector's confidence", async () => {
    // Colour shows what the pipeline did, using the reading pane's four classes, never confidence.
    const { pane } = await openImage();
    const classes = Array.from(pane.querySelectorAll("rect"))
      .map((r) => r.getAttribute("class") ?? "");

    expect(classes[0]).toContain("box--vision");
    expect(classes[1]).toContain("box--override");
    expect(classes[3]).toContain("box--unnamed");
    expect(classes[4]).toContain("box--debris");

    // And the caveat is above the picture.
    const note = pane.querySelector(".boxnote");
    const plate = pane.querySelector(".plate");
    expect(note?.textContent).toMatch(/colours say what the pipeline/i);
    expect(note?.textContent).toMatch(/nothing on this image is human annotation/i);
    expect(
      note!.compareDocumentPosition(plate!) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test("clicking a box selects the sign it became, and clicking a sign selects its box", async () => {
    // The two panes share one selection; drive it from each end and check the other.
    const { user, pane } = await openImage();
    const reading = await screen.findByRole("region", { name: /^reading$/i });

    // Box to sign
    await user.click(pane.querySelectorAll("rect")[1]);
    await waitFor(() => {
      const on = reading.querySelector(".tok--picked");
      expect(on?.textContent).toBe("DIS*");
    });

    // Sign to box. Slot 0 is "I", from detection 0.
    await user.click(within(reading).getAllByRole("button", { name: "I" })[0]);
    await waitFor(() => {
      const lit = Array.from(pane.querySelectorAll("rect"))
        .findIndex((r) => (r.getAttribute("class") ?? "").includes("box--on"));
      expect(lit).toBe(0);
    });
  });

  test("a restored position has no box to light up, and that does not break the link", async () => {
    // An inserted position has no box, so selecting it lights nothing and doesn't throw.
    const { user, pane } = await openImage();
    const reading = await screen.findByRole("region", { name: /^reading$/i });

    await user.click(within(reading).getAllByRole("button", { name: "[AS]" })[0]);
    await waitFor(() => {
      expect(reading.querySelector(".tok--picked")?.textContent).toBe("[AS]");
    });
    expect(pane.querySelector("rect.box--on")).toBeNull();
  });

  test("the boxes can be turned off, and the photograph survives it", async () => {
    const { user, pane } = await openImage();
    await user.click(within(pane).getByRole("button", { name: /hide boxes/i }));

    expect(pane.querySelectorAll("rect")).toHaveLength(0);
    // The photograph stays; only the drawing is hidden.
    expect(within(pane).getByRole("img")).toBeTruthy();
  });

  test("selecting another tablet drops the selection rather than carrying it over", async () => {
    // A selection doesn't carry over to a different tablet.
    const { user, pane } = await openImage(TABLET_IDS[3]);
    await user.click(pane.querySelectorAll("rect")[1]);
    await waitFor(() => {
      expect(pane.querySelector("rect.box--on")).not.toBeNull();
    });

    await user.click(screen.getByRole("button", { name: new RegExp(TABLET_IDS[9]) }));
    const after = await screen.findByRole("region", { name: /photograph/i });
    fireEvent.load(within(after).getByRole("img"));
    await waitFor(() => {
      expect(after.querySelectorAll("rect").length).toBe(DETECTIONS.length);
    });
    expect(after.querySelector("rect.box--on")).toBeNull();
  });

  test("the pane still refuses to fetch or link a ground-truth overlay", async () => {
    // Still no overlay image now that boxes are on screen.
    const { pane } = await openImage();

    const requested = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
      .mock.calls.map((c) => String(c[0]));
    expect(requested.some((u) => u.includes("overlay"))).toBe(false);

    const urls = Array.from(pane.querySelectorAll("img, image, a, source"))
      .map((el) => el.getAttribute("src") ?? el.getAttribute("href") ?? "");
    expect(urls.some((u) => u.includes("overlay"))).toBe(false);
    // Exactly one <img>; the boxes are vector shapes drawn from JSON.
    expect(pane.querySelectorAll("img")).toHaveLength(1);
  });
});


describe("honesty requirements", () => {
  test("no label anywhere calls the output a translation", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );

    const text = container.textContent ?? "";
    // Every occurrence of "translation" must be a denial (see unnegatedTranslationClaims).
    expect(unnegatedTranslationClaims(text)).toEqual([]);
    // And the denial itself is on the page.
    expect(text).toMatch(/does not translate/i);
  });

  test("never requests or displays the ground-truth overlay images", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );

    // Check what was fetched or linked, not the prose, which may mention the overlays.
    const requested = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
      .mock.calls.map((c) => String(c[0]));
    expect(requested.some((u) => u.includes("overlay"))).toBe(false);

    const urls = Array.from(container.querySelectorAll("img, a, source"))
      .map((el) => el.getAttribute("src") ?? el.getAttribute("href") ?? "");
    expect(urls.some((u) => u.includes("overlay"))).toBe(false);
    expect(urls.every((u) => u === "" || /^\/api\/tablets\/[^/]+\/image$/.test(u)))
      .toBe(true);

    expect(container.textContent ?? "").not.toMatch(/n_gt_boxes/);
  });

  test("the frozen operating point is on screen and flagged as untuned", async () => {
    render(<App />);
    const strip = await screen.findByText(/frozen operating point/i);
    const text = strip.parentElement?.textContent ?? "";
    expect(text).toContain("0.15"); // conf
    expect(text).toContain("0.08"); // lm floor
    expect(text).toMatch(/not tuned values/i);
  });

  test("the three domains are reported separately, and a dead Domain 3 does not black out the page", async () => {
    vi.stubGlobal(
      "fetch",
      stubFetch({
        health: {
          ...health,
          status: "degraded",
          domain3: {
            ...health.domain3,
            reachable: false,
            model_available: null,
            detail: "ConnectionError: Failed to connect to Ollama.",
          },
          notes: ["Ollama is not reachable, so the tentative reading is unavailable."],
        },
      }),
    );

    render(<App />);

    // The note is shown...
    expect(await screen.findByText(/Ollama is not reachable/i)).toBeTruthy();
    // ...and the tablet list still works.
    const list = await screen.findByRole("region", { name: /test tablets/i });
    await waitFor(() => {
      expect(within(list).getAllByRole("button")).toHaveLength(30);
    });
  });
});

// The summary pane and the provenance pane. The key tests: vacuous shows as a
// failure, confidence is labelled as the model's own, caveats come before the
// prose in DOM order, and a cached summary doesn't claim to be reproducible.

/** Select a tablet, wait for its analysis, and hand back the summary pane. */
async function openSummary() {
  const user = userEvent.setup();
  render(<App />);
  await user.click(
    await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
  );
  const pane = await screen.findByRole("region", { name: /tentative reading/i });
  await waitFor(() => {
    const go = within(pane).getByRole("button", {
      name: /offer a tentative reading/i,
    }) as HTMLButtonElement;
    expect(go.disabled).toBe(false);
  });
  return { user, pane };
}

async function generate() {
  const { user, pane } = await openSummary();
  await user.click(
    within(pane).getByRole("button", { name: /offer a tentative reading/i }),
  );
  await within(pane).findByText(TENTATIVE_READING);
  return { user, pane };
}

// The "translation" check. The word itself is allowed, since the server's own
// caveat says "THIS IS NOT A TRANSLATION"; what must never happen is the output
// being called one. So every occurrence must be negated, and the checker is
// tested below against a sentence that must fail it.

/** Occurrences of the word that aren't part of a denial. Empty means pass. */
function unnegatedTranslationClaims(text: string): string[] {
  const bad: string[] = [];
  const re = /translation/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const before = text.slice(Math.max(0, m.index - 24), m.index).toLowerCase();
    // "is not a translation", "not a translation", "never a translation".
    if (!/\b(not|never|no)\s+(a|an)?\s*$/.test(before)) {
      bad.push(text.slice(Math.max(0, m.index - 60), m.index + 40));
    }
  }
  return bad;
}

describe("Part 7: the tentative reading", () => {
  test("does not call Domain 3 until asked", async () => {
    // Selecting a tablet must not trigger the slow Domain 3 call.
    await openSummary();
    const requested = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
      .mock.calls.map((c) => String(c[0]));
    expect(requested.some((u) => u.includes("/summarise"))).toBe(false);
  });

  test("renders the tentative reading, and the caveats come BEFORE it", async () => {
    // Document order, not just presence: the caveats come before the prose.
    const { pane } = await generate();

    const caveat = within(pane).getByText(/NOT A TRANSLATION/i);
    const prose = within(pane).getByText(TENTATIVE_READING);
    expect(
      caveat.compareDocumentPosition(prose) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test("confidence is labelled as the model's own, not as a measurement", async () => {
    const { pane } = await generate();
    const claim = within(pane).getByText(/the model's own estimate of its confidence/i);
    const row = claim.parentElement?.textContent ?? "";
    expect(row).toMatch(/\blow\b/);
    expect(row).toMatch(/validated against nothing/i);
  });

  test("a 'vacuous' faithfulness verdict is rendered as a FAILURE, not a pass", async () => {
    // vacuous means the check never ran, so it must never look like a pass.
    vi.stubGlobal(
      "fetch",
      stubFetch({
        summary: summaryFor(TABLET_IDS[0], {
          faithfulness: {
            supplied_distinct: 4, citations: 0, citations_distinct: 0,
            exact: [], normalised: [], invented: [],
            verdict: "vacuous", coverage_of_input: null,
          },
        }),
      }),
    );
    const { pane } = await generate();

    const head = within(pane).getByText(/THE CHECK DID NOT RUN/i);
    expect(within(pane).getByText(/an absent result, not a clean one/i)).toBeTruthy();
    expect(head.closest(".audit")?.className).toMatch(/audit--bad/);
    expect(head.closest(".audit")?.className).not.toMatch(/audit--good/);
  });

  test("invented citations are named, not counted", async () => {
    vi.stubGlobal(
      "fetch",
      stubFetch({
        summary: summaryFor(TABLET_IDS[0], {
          faithfulness: {
            supplied_distinct: 4, citations: 5, citations_distinct: 4,
            exact: ["AN"], normalised: [], invented: ["DUN4", "GEME"],
            verdict: "fail", coverage_of_input: 0.25,
          },
        }),
      }),
    );
    const { pane } = await generate();
    expect(
      within(pane).getByText(/signs it was never shown/i).parentElement?.textContent,
    ).toMatch(/DUN4.*GEME/);
  });

  test("a sign-shaped word invented inside the prose is reported separately", async () => {
    // This reads the sentences themselves, not the model's own citation list.
    vi.stubGlobal(
      "fetch",
      stubFetch({
        summary: summaryFor(TABLET_IDS[0], {
          prose_scan: {
            fields: ["tentative_reading"],
            candidates: ["AN", "DUN4"],
            unmatched: ["DUN4"],
          },
        }),
      }),
    );
    const { pane } = await generate();
    const block = within(pane)
      .getByText(/sign-shaped words in the prose were not in the input/i)
      .closest(".audit");
    expect(block?.className).toMatch(/audit--bad/);
    expect(block?.textContent).toMatch(/DUN4/);
  });

  test("an empty prose scan is not presented as a clean bill of health", async () => {
    const { pane } = await generate();
    expect(within(pane).getByText(/not a clean bill of health/i)).toBeTruthy();
    expect(within(pane).getByText(/Proper nouns were not, and cannot be/i)).toBeTruthy();
  });

  test("a cached summary claims only what the cache can support", async () => {
    vi.stubGlobal(
      "fetch",
      stubFetch({ summary: summaryFor(TABLET_IDS[0], { cached: true }) }),
    );
    const { pane } = await generate();
    const text = pane.textContent ?? "";
    expect(text).toMatch(/summary it already showed you/i);
    expect(text).toMatch(/not\s+reproducible/i);
    expect(text).toMatch(/temperature 0\.2 with no fixed seed/i);
  });

  test("Ollama being down degrades this pane only, and re-checks health", async () => {
    // No polling, but a 503 triggers one health re-check.
    vi.stubGlobal("fetch", stubFetch({ summaryStatus: 503 }));
    const { user, pane } = await openSummary();

    const healthCalls = () =>
      (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
        .mock.calls.filter((c) => String(c[0]).includes("/api/health")).length;
    const before = healthCalls();

    await user.click(
      within(pane).getByRole("button", { name: /offer a tentative reading/i }),
    );

    expect(await within(pane).findByText(/language model is not reachable/i)).toBeTruthy();
    expect(within(pane).getByText(/Start the Ollama tray app/i)).toBeTruthy();
    expect(within(pane).getByText(/Only this pane is affected/i)).toBeTruthy();

    await waitFor(() => {
      expect(healthCalls()).toBeGreaterThan(before);
    });

    // The reading pane is untouched; it doesn't need the language model.
    const reading = screen.getByRole("region", { name: /^reading$/i });
    expect(within(reading).getAllByText("I").length).toBeGreaterThan(0);
  });

  test("no invented progress is shown while the model runs", async () => {
    const { user, pane } = await openSummary();
    await user.click(
      within(pane).getByRole("button", { name: /offer a tentative reading/i }),
    );
    // Whatever the timing, there must never be a progress bar.
    await within(pane).findByText(TENTATIVE_READING);
    expect(within(pane).queryByRole("progressbar")).toBeNull();
  });

  test("nothing on the page calls the output a translation, with prose on screen", async () => {
    // The same translation rule, now with Domain 3 prose on the page.
    const { pane } = await generate();
    expect(unnegatedTranslationClaims(document.body.textContent ?? "")).toEqual([]);
    expect(pane.textContent ?? "").toMatch(/does not\s+translate/i);
    // And the denial really is on screen.
    expect(pane.textContent ?? "").toMatch(/not a translation/i);
  });

  test("the translation gate itself rejects a page that DID claim one", async () => {
    // The checker must fail a real claim, like the draft report's sentence.
    expect(
      unnegatedTranslationClaims(
        "produces a fluent English translation together with a summary",
      ),
    ).toHaveLength(1);
    // And the two phrasings that must keep passing.
    expect(unnegatedTranslationClaims("THIS IS NOT A TRANSLATION.")).toEqual([]);
    expect(unnegatedTranslationClaims("It does not translate.")).toEqual([]);
  });
});

describe("Part 7: provenance", () => {
  async function openProvenance() {
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );
    return screen.findByRole("region", { name: /provenance/i });
  }

  test("shows the attrition from boxes to named positions", async () => {
    const pane = await openProvenance();
    // Read the number cell, not the whole row, or "6" next to "4 from a box" reads as "64".
    const value = (label: RegExp) =>
      within(pane).getByText(label).closest("tr")?.querySelector(".num")
        ?.textContent ?? "";

    expect(value(/boxes the detector emitted/i)).toBe("255");
    expect(value(/still above conf 0\.15/i)).toBe("141");
    expect(value(/positions in the sequence/i)).toBe("6");
    // This row counts boxes whose class is outside the 82 emit classes.
    expect(value(/positions holding a box we may not name/i)).toBe("1");
  });

  test("the 'changed, not fixed' caveat is ABOVE the fusion numbers", async () => {
    // The caveat comes before fusion_changed, the number most likely to be misread.
    const pane = await openProvenance();

    const caveat = within(pane).getByText(/not predictions it/i);
    const number = within(pane).getByText(/changed by the sequence model/i);
    expect(
      caveat.compareDocumentPosition(number) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(pane.textContent ?? "").toMatch(/Nothing here is an accuracy figure/i);
  });

  test("states the frozen operating point and that none of it is tuned", async () => {
    const pane = await openProvenance();
    const text = pane.textContent ?? "";

    expect(text).toMatch(/frozen configuration/i);
    expect(text).toMatch(/None of these is a tuned value/i);
    expect(
      within(pane).getByText(/sequence-model floor/i).closest("tr")?.textContent,
    ).toMatch(/0\.08/);
  });

  test("a tablet that fused nothing says so instead of showing empty tables", async () => {
    const user = userEvent.setup();
    const nothing = { ...analysisFor(TABLET_IDS[0]), fusion: null };
    vi.stubGlobal("fetch", stubFetch({ analysis: nothing }));
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );
    const pane = await screen.findByRole("region", { name: /provenance/i });
    expect(await within(pane).findByText(/a result about this tablet/i)).toBeTruthy();
  });
});

// Uploaded photographs. The first test matters most: an upload never appears
// among the evaluated tablets.

const UPLOAD_CAVEAT_NOT_EVALUATED =
  "This photograph is not part of any evaluation in this project. Every " +
  "accuracy figure quoted anywhere in this interface was measured on 30 other " +
  "tablets, and none of it describes this one.";
const UPLOAD_CAVEAT_NO_TRUTH =
  "No ground truth exists for this photograph, so nothing shown for it can be " +
  "checked against a known answer, not even in principle.";

/** An analysis for an uploaded tablet: the base caveats plus the upload ones. */
function uploadAnalysis(tablet: string): Analysis {
  const base = analysisFor(tablet);
  return {
    ...base,
    caveats: [...base.caveats, UPLOAD_CAVEAT_NOT_EVALUATED, UPLOAD_CAVEAT_NO_TRUTH],
  };
}

/** Render with one upload already on the server, and return both regions. */
async function withOneUpload(upload = uploadFixture()) {
  vi.stubGlobal(
    "fetch",
    stubFetch({
      uploads: { uploads: [upload], count: 1, note: UPLOAD_NOTE },
      analysis: uploadAnalysis(upload.tablet),
    }),
  );
  const user = userEvent.setup();
  render(<App />);
  const evaluated = await screen.findByRole("region", { name: /test tablets/i });
  const mine = await screen.findByRole("region", { name: /your uploads/i });
  return { user, evaluated, mine, upload };
}

describe("uploads: the two populations stay two populations", () => {
  test("an upload never appears in the evaluated list, which is still exactly 30", async () => {
    const { evaluated, mine, upload } = await withOneUpload();

    await waitFor(() => {
      expect(within(evaluated).getAllByRole("button")).toHaveLength(30);
    });
    // The uploaded id must be absent from the evaluated region, whatever its row count.
    expect(within(evaluated).queryByText(/field_photo/i)).toBeNull();
    expect(evaluated.textContent ?? "").not.toContain(upload.tablet);

    // And it is in the uploads region.
    expect(within(mine).getByText(/field_photo\.jpg/i)).toBeTruthy();
  });

  test("the uploaded population is labelled as unevaluated, in both places", async () => {
    const { mine } = await withOneUpload();

    // The note on the section...
    expect(within(mine).getByText(/no evaluation in this project covers them/i))
      .toBeTruthy();
    // ...and on the row itself, since a row can be read without the note.
    expect(within(mine).getByText(/not evaluated · no ground truth/i)).toBeTruthy();
  });

  test("the evaluated note no longer claims there is no upload", async () => {
    // The tablet note no longer says there is no upload.
    const { evaluated } = await withOneUpload();
    expect(evaluated.textContent ?? "").not.toMatch(/there is no upload/i);
  });
});

describe("uploads: ingesting a photograph", () => {
  test("choosing a file posts it to /api/uploads and selects the result", async () => {
    const upload = uploadFixture();
    vi.stubGlobal(
      "fetch",
      stubFetch({
        uploads: { uploads: [upload], count: 1, note: UPLOAD_NOTE },
        analysis: uploadAnalysis(upload.tablet),
      }),
    );
    const user = userEvent.setup();
    render(<App />);

    const mine = await screen.findByRole("region", { name: /your uploads/i });
    const input = within(mine).getByLabelText(/photograph to read/i);
    const file = new File([new Uint8Array([1, 2, 3])], "field_photo.jpg",
                          { type: "image/jpeg" });
    await user.upload(input, file);

    const calls = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
      .mock.calls.map((c) => String(c[0]));
    await waitFor(() => {
      expect(calls.some((u) => u.includes("/api/uploads?filename="))).toBe(true);
    });

    // The reading pane now shows the upload, so nobody has to find it in a list.
    const reading = await screen.findByRole("region", { name: /^reading$/i });
    await waitFor(() => {
      expect(within(reading).getByText(upload.tablet)).toBeTruthy();
    });
  });

  test("a rejected photograph shows the server's own message", async () => {
    // HEIC gets a specific message, since iPhones use it by default.
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/uploads") && url.includes("filename=")) {
        return {
          ok: false, status: 422, statusText: "Unprocessable Content",
          json: async () => ({
            detail: {
              error: "UploadRejected",
              detail: "That file could not be read as an image. JPEG, PNG and "
                    + "TIFF photographs work. HEIC, which iPhones use by "
                    + "default, does not: export or convert to JPEG first.",
              hint: "Try a JPEG or PNG photograph of a single tablet.",
            },
          }),
        } as Response;
      }
      let body: unknown = tabletList;
      if (url.includes("/api/health")) body = health;
      else if (url.includes("/api/uploads")) body = emptyUploads;
      return { ok: true, status: 200, statusText: "OK",
               json: async () => body } as Response;
    }));

    // applyAccept: false lets the test pick a HEIC file past the input's accept
    // filter, as a determined user can; the server decodes the bytes to check.
    const user = userEvent.setup({ applyAccept: false });
    render(<App />);
    const mine = await screen.findByRole("region", { name: /your uploads/i });
    await user.upload(
      within(mine).getByLabelText(/photograph to read/i),
      new File([new Uint8Array([9])], "photo.heic", { type: "image/heic" }),
    );

    expect(await within(mine).findByText(/HEIC, which iPhones use by default/i))
      .toBeTruthy();
    // And a failed upload leaves the evaluated tablets untouched.
    const evaluated = screen.getByRole("region", { name: /test tablets/i });
    expect(within(evaluated).getAllByRole("button")).toHaveLength(30);
  });

  test("no progress bar is invented while the detector runs", async () => {
    // Elapsed seconds only, no progress bar.
    const { mine } = await withOneUpload();
    expect(within(mine).queryByRole("progressbar")).toBeNull();
  });
});

describe("uploads: what the provenance pane must say", () => {
  async function openUploadProvenance() {
    const { user, mine, upload } = await withOneUpload();
    await user.click(within(mine).getByRole("button", { name: /field_photo/i }));
    const pane = await screen.findByRole("region", { name: /provenance/i });
    return { pane, upload };
  }

  test("reports the resize with the measurement that caused it", async () => {
    const { pane } = await openUploadProvenance();

    // Both passes are shown: 124 px before, 62 px after.
    expect(within(pane).getByText(/detector pass 1/i).closest("tr")?.textContent)
      .toMatch(/124 px/);
    expect(within(pane).getByText(/detector pass 2/i).closest("tr")?.textContent)
      .toMatch(/62 px/);
    // The reason, verbatim from the server.
    expect(pane.textContent ?? "").toMatch(/outside the 43 to 113 px range/i);
    // And the constants are described as measured, not chosen.
    expect(pane.textContent ?? "").toMatch(/nothing about them was chosen/i);
  });

  test("the unevaluated warning comes BEFORE the counts it disqualifies", async () => {
    // The "no evaluation" warning comes before the numbers.
    const { pane } = await openUploadProvenance();

    const warning = within(pane).getByText(/no evaluation in this project covers this image/i);
    const counts = within(pane).getByText(/boxes the detector emitted/i);
    expect(
      warning.compareDocumentPosition(counts) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test("a corpus tablet shows no upload block at all", async () => {
    // The block doesn't appear for the 30 tablets.
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );
    const pane = await screen.findByRole("region", { name: /provenance/i });
    expect(within(pane).queryByText(/your photograph, before it was read/i))
      .toBeNull();
  });
});

describe("uploads: the reading pane", () => {
  test("an upload's extra caveats are rendered ABOVE its sequence", async () => {
    const { user, mine } = await withOneUpload();
    await user.click(within(mine).getByRole("button", { name: /field_photo/i }));

    const reading = await screen.findByRole("region", { name: /^reading$/i });
    const caveat = await within(reading).findByText(
      /not part of any evaluation in this project/i);
    const firstToken = within(reading).getAllByText("I")[0];
    expect(
      caveat.compareDocumentPosition(firstToken) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    // Both upload caveats are present.
    expect(within(reading).getByText(/no ground truth exists for this photograph/i))
      .toBeTruthy();
  });
});

// Defects spotted by reading the finished page, kept here so they can't come back.

describe("what a reader caught that the gates could not", () => {
  test("an upload is not told a fact about the 30", async () => {
    // An upload gets its own no-metadata message, not the corpus one about "seven of the 30".
    const upload = uploadFixture();
    vi.stubGlobal(
      "fetch",
      stubFetch({
        uploads: { uploads: [upload], count: 1, note: UPLOAD_NOTE },
        analysis: uploadAnalysis(upload.tablet),
        summary: summaryFor(upload.tablet, {
          metadata: {
            had_metadata: false,
            had_metadata_at_call: false,
            claimed_metadata_used: false,
            catalogue_metadata: "",
          },
        }),
      }),
    );
    const user = userEvent.setup();
    render(<App />);
    const mine = await screen.findByRole("region", { name: /your uploads/i });
    await user.click(within(mine).getByRole("button", { name: /field_photo/i }));

    const pane = await screen.findByRole("region", { name: /tentative reading/i });
    await waitFor(() => {
      expect(within(pane).getByRole("button", { name: /offer a tentative reading/i }))
        .toBeTruthy();
    });
    await user.click(
      within(pane).getByRole("button", { name: /offer a tentative reading/i }));
    await within(pane).findByText(TENTATIVE_READING);

    expect(pane.textContent ?? "").not.toMatch(/Seven of the 30/i);
    const message = within(pane).getByText(/keyed on a museum P-number/i);

    // And it must not be in a <pre>, or the sentence wouldn't wrap.
    expect(message.closest("pre")).toBeNull();
    expect(message.tagName).toBe("P");
  });

  test("provenance does not label a different count as the [x] positions", async () => {
    // The ladder row counts boxes with a class outside the emit set, not the [x]
    // positions, which are the demoted restorations.
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );
    const pane = await screen.findByRole("region", { name: /provenance/i });

    expect(within(pane).queryByText(/positions the system will not name/i))
      .toBeNull();
    expect(within(pane).getByText(/positions holding a box we may not name/i))
      .toBeTruthy();

    // And the [x] explanation is on the row with that number.
    const demoted = within(pane).getByText(/below the floor and refused/i)
      .closest("tr");
    expect(demoted?.textContent ?? "").toMatch(/\[x\] marks in the reading/i);
  });

  test("a single segment is not described as 'segments'", async () => {
    const user = userEvent.setup();
    const one = analysisFor(TABLET_IDS[0]);
    one.counts = { ...one.counts, segments: 1, blocks: 1 };
    vi.stubGlobal("fetch", stubFetch({ analysis: one }));
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );

    const prov = await screen.findByRole("region", { name: /provenance/i });
    expect(prov.textContent ?? "").toMatch(/1 segment \(physical surface\)/);
    expect(prov.textContent ?? "").not.toMatch(/1 segments/);
    expect(prov.textContent ?? "").not.toMatch(/1 blocks/);

    const reading = screen.getByRole("region", { name: /^reading$/i });
    expect(reading.textContent ?? "").not.toMatch(/1 segments/);
  });
});

/* Copy rules. These read the source rather than the DOM, because the page also
 * shows text I didn't write (catalogue metadata, model prose) that might
 * legitimately contain an em dash. Comments are ignored. */

describe("copy rules", () => {
  /** Strip // and block comments so the rule applies to displayed copy only. */
  function stripComments(src: string): string {
    let out = "";
    let i = 0;
    let mode: "code" | "line" | "block" = "code";
    while (i < src.length) {
      const two = src.slice(i, i + 2);
      if (mode === "code" && two === "//") { mode = "line"; i += 2; continue; }
      if (mode === "code" && two === "/*") { mode = "block"; i += 2; continue; }
      if (mode === "line" && src[i] === "\n") { mode = "code"; out += "\n"; i += 1; continue; }
      if (mode === "block" && two === "*/") { mode = "code"; i += 2; continue; }
      if (mode === "code") out += src[i];
      else if (src[i] === "\n") out += "\n";
      i += 1;
    }
    return out;
  }

  // Read the files through Vite's raw glob, which needs no extra packages.
  const RAW = import.meta.glob(["./*.tsx", "./*.ts", "./panes/*.tsx"], {
    query: "?raw", import: "default", eager: true,
  }) as Record<string, string>;

  // This test file is skipped, since the check below quotes an em dash on purpose.
  const sources = (): { file: string; text: string }[] =>
    Object.entries(RAW)
      .filter(([f]) => !f.endsWith("App.test.tsx") && !f.endsWith("vite-env.d.ts"))
      .map(([f, text]) => ({ file: f.replace(/^\.\//, ""), text }));

  test("no em dash appears in text the visitor reads", () => {
    const offenders: string[] = [];
    for (const { file, text } of sources()) {
      stripComments(text).split("\n").forEach((line, n) => {
        if (line.includes("—")) offenders.push(`${file}:${n + 1}  ${line.trim()}`);
      });
    }
    expect(offenders, offenders.join("\n")).toHaveLength(0);
  });

  test("the gate can actually fail: it sees an em dash outside a comment", () => {
    // The check has to be able to fail.
    const withComment = 'const a = 1; // an em dash — in a comment\n';
    const withCopy = 'const b = "an em dash — in displayed copy";\n';
    expect(stripComments(withComment)).not.toContain("—");
    expect(stripComments(withCopy)).toContain("—");
  });

  test("every source file the rule covers was actually read", () => {
    // And it must really have read the pane files.
    const files = sources().map((s) => s.file);
    expect(files.length).toBeGreaterThanOrEqual(9);
    for (const pane of ["ReadingPane", "SummaryPane", "ImagePane",
                        "ProvenancePane", "HealthStrip", "TabletList"]) {
      expect(files.some((f) => f.includes(pane))).toBe(true);
    }
    for (const { text } of sources()) expect(text.length).toBeGreaterThan(100);
  });
});

/* Fixes from the cognitive walkthrough of the task script. */

describe("Part 8 walkthrough fixes", () => {
  test("W-3: the readable formats are named before the file picker opens", async () => {
    // The file picker hides HEIC files, so the page itself has to name the formats.
    vi.stubGlobal("fetch", stubFetch({}));
    render(<App />);
    const uploads = await screen.findByRole("region", { name: /your uploads/i });
    const text = uploads.textContent ?? "";
    expect(text).toMatch(/JPEG/);
    expect(text).toMatch(/HEIC/);
    expect(text).toMatch(/iPhone/i);
  });

  test("W-4: the plain-text note says something the reader can act on", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", stubFetch({}));
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );
    const reading = await screen.findByRole("region", { name: /^reading$/i });
    await user.click(within(reading).getByRole("button", { name: /plain text/i }));

    const text = within(reading).getByText(/one string per line/i).textContent ?? "";
    // The claim stays; the developer detail goes.
    expect(text).toMatch(/command-line/i);
    expect(text).not.toMatch(/cross-process/i);
    expect(text).not.toMatch(/live_pipeline/);
  });

  test("W-5: the sequence row's note accounts for every position in the row", async () => {
    // The note's numbers must add up to the row's value.
    const user = userEvent.setup();
    vi.stubGlobal("fetch", stubFetch({}));
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: new RegExp(TABLET_IDS[0]) }),
    );

    const prov = await screen.findByRole("region", { name: /provenance/i });
    const row = within(prov).getByText("positions in the sequence").closest("tr");
    expect(row).toBeTruthy();

    const value = Number((row!.querySelector("td.num")?.textContent ?? "")
      .replace(/[^0-9]/g, ""));
    const note = row!.querySelector("td.why")?.textContent ?? "";
    const parts = (note.match(/\d+/g) ?? []).map(Number);

    expect(parts.length).toBeGreaterThanOrEqual(3);
    expect(parts.reduce((a, b) => a + b, 0)).toBe(value);
  });
});
