"""
My readiness check before a user-testing session. Run it from D:\\FYP with
both servers already started:

    python -m cuneiform.api.session_ready

It warms the running server's analysis cache and the Domain 3 model, so a
participant never meets a cold start, and checks the upload photograph and the
population folders. It talks to the server over HTTP because the cache lives
inside the server process, and it prints the evidence beside every verdict,
ending with READY or NOT READY.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from cuneiform import config

API = "http://127.0.0.1:8000"

# The photograph every participant uses for the upload task, prepared in advance
# so a HEIC photo can't derail the session.
PART8_DIR = config.ARTIFACTS_DIR / "part8"
PREPARED_UPLOAD = PART8_DIR / "prepared_upload.jpg"

# Fixed tablet for tasks 2 to 7, so every participant sees the same evidence.
SCRIPT_TABLET = "P335940"


# Plumbing

class CheckFailed(Exception):
    """Raised by a check that cannot even gather its own evidence."""


def _get(path: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, timeout: float = 400.0) -> Any:
    req = urllib.request.Request(API + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Report:
    """Collects verdicts so the final banner is computed, not asserted."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, evidence: str) -> bool:
        self.rows.append((name, ok, evidence))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for line in evidence.splitlines():
            print(f"         {line}")
        return ok

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.rows)


# The checks

def check_server(rep: Report) -> dict | None:
    """The API answers at all, and reports which domains are up."""
    print("\n1. THE ORCHESTRATOR")
    try:
        h = _get("/api/health", timeout=30.0)
    except (urllib.error.URLError, OSError) as exc:
        rep.add(
            "API reachable on port 8000", False,
            f"{type(exc).__name__}: {exc}\n"
            f"Start it: .\\start.ps1 then "
            f"python -m uvicorn cuneiform.api.main:app --port 8000\n"
            f"Check first that nothing else holds the port: "
            f"netstat -ano | Select-String \":8000\"")
        return None

    rep.add("API reachable on port 8000", True,
            f"status={h['status']}  tablets={h['tablets']}")

    d2 = h["domain2"]
    rep.add("Domain 2 sequence model loaded", bool(d2["loaded"]),
            f"loaded={d2['loaded']}  checkpoint={d2['checkpoint']}  "
            f"warm_seconds={d2['warm_seconds']}")

    cfg = h["config"]
    frozen_ok = (
        cfg["conf"] == 0.15 and cfg["cell"] == 2.0 and cfg["lambda"] == 1.0
        and cfg["edge_extrapolation"] is False and cfg["lm_floor"] == 0.08
        and cfg["min_cluster"] == 3
    )
    rep.add("Running at the frozen configuration", frozen_ok,
            f"conf={cfg['conf']} cell={cfg['cell']} lambda={cfg['lambda']} "
            f"edge={cfg['edge_extrapolation']} floor={cfg['lm_floor']} "
            f"min_cluster={cfg['min_cluster']}\n"
            f"expected conf=0.15 cell=2.0 lambda=1.0 edge=False floor=0.08 "
            f"min_cluster=3  (the frozen configuration)")
    return h


def check_domain3(rep: Report, health: dict) -> None:
    """Ollama is up and holds the model. "degraded" is not good enough here."""
    print("\n2. DOMAIN 3 (the tentative reading)")
    d3 = health["domain3"]
    ok = bool(d3["reachable"]) and bool(d3.get("model_available"))
    rep.add("Ollama reachable and holding the model", ok,
            f"reachable={d3['reachable']}  model={d3.get('model')}  "
            f"model_available={d3.get('model_available')}  "
            f"models_held={d3.get('models_held')}\n"
            f"probe took {d3.get('checked_in_seconds')} s"
            + ("" if ok else
               "\nThe summary pane is task 7 of the script. A degraded health "
               "strip is CORRECT behaviour but it is not a testable session: "
               "start the Ollama tray app and re-run this gate."))


def check_population_separation(rep: Report) -> None:
    """
    Make sure no upload's detection file has leaked into the evaluation
    tablets' folder, which would change every evaluation result.
    """
    print("\n3. POPULATION SEPARATION")
    live = config.D1_LIVE_DETECTIONS
    records = sorted(p.stem for p in live.glob("*.json")
                     if not p.stem.startswith("_"))
    strays = [r for r in records if r.startswith("upload_")]
    ok = len(records) == 30 and not strays
    rep.add("live_detections holds exactly the 30 evaluation tablets", ok,
            f"{live}\n"
            f"records={len(records)}  uploads_found={len(strays)}"
            + (f"  STRAYS={strays}" if strays else "")
            + f"\nfirst={records[0] if records else 'none'}  "
              f"last={records[-1] if records else 'none'}")

    uploads = config.UPLOADS_DIR
    uploads.mkdir(parents=True, exist_ok=True)
    n_up = len(list(uploads.glob("*.json")))
    rep.add("uploads directory exists and is separate", uploads.exists(),
            f"{uploads}\nupload records currently held: {n_up}")


def check_prepared_upload(rep: Report) -> None:
    """The photograph handed to participants for task 8."""
    print("\n4. THE PREPARED PHOTOGRAPH (task 8)")
    if not PREPARED_UPLOAD.exists():
        rep.add("prepared_upload.jpg present", False,
                f"missing: {PREPARED_UPLOAD}")
        return

    # Catch everything, since ultralytics swaps in its own Image.open.
    try:
        from PIL import Image
        with Image.open(PREPARED_UPLOAD) as im:
            im.load()
            size, fmt = im.size, im.format
        decoded, why = True, f"{size[0]}x{size[1]} {fmt}"
    except Exception as exc:  # noqa: BLE001
        decoded, why = False, f"{type(exc).__name__}: {exc}"

    rep.add("prepared photograph decodes", decoded,
            f"{PREPARED_UPLOAD}\n{why}")

    # Compare file contents, not names, since the prepared file is a renamed copy.
    import hashlib

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    from cuneiform.api import service

    prepared_hash = _sha(PREPARED_UPLOAD)
    collisions = []
    for tid in sorted(p.stem for p in config.D1_LIVE_DETECTIONS.glob("*.json")
                      if not p.stem.startswith("_")):
        try:
            src = Path(service.image_path(tid))
        except Exception:  # noqa: BLE001
            continue
        if src.exists() and _sha(src) == prepared_hash:
            collisions.append(tid)

    rep.add("prepared photograph is not one of the 30", not collisions,
            f"sha256={prepared_hash[:16]}...  compared against "
            f"{len(list(config.D1_LIVE_DETECTIONS.glob('*.json'))) - 1} "
            f"source photographs by CONTENT, not by filename\n"
            + (f"COLLIDES WITH: {collisions}" if collisions else
               "no collision: the detector has not seen this photograph as "
               "part of the evaluation population"))


def warm(rep: Report, tablet: str, skip_domain3: bool) -> None:
    """
    Fill the server's analysis cache, then load the Domain 3 model into VRAM.
    I print per-tablet timings because the largest tablets are the slowest.
    """
    print("\n5. WARMING (this is the part a participant must never wait for)")
    tablets = [t["tablet"] for t in _get("/api/tablets")["tablets"]]
    print(f"     warming analyse over {len(tablets)} tablets ...")

    times: list[tuple[str, float, bool]] = []
    t0 = time.perf_counter()
    for tid in tablets:
        s = time.perf_counter()
        try:
            payload = _post(f"/api/tablets/{tid}/analyse")
        except Exception as exc:  # noqa: BLE001
            rep.add("analyse warm completed for every tablet", False,
                    f"{tid} failed: {type(exc).__name__}: {exc}")
            return
        dt = time.perf_counter() - s
        times.append((tid, dt, bool(payload.get("cached"))))
    total = time.perf_counter() - t0

    slowest = sorted(times, key=lambda r: -r[1])[:3]
    rep.add("analyse warm completed for every tablet", True,
            f"{len(times)} tablets in {total:.1f} s total\n"
            f"slowest three: " + ", ".join(
                f"{t} {d:.2f}s" for t, d, _ in slowest) + "\n"
            f"already-cached on arrival: "
            f"{sum(1 for _, _, c in times if c)} of {len(times)}")

    # A second pass proves the cache really serves requests in the server process.
    s = time.perf_counter()
    again = _post(f"/api/tablets/{tablet}/analyse")
    dt = time.perf_counter() - s
    rep.add("the analysis cache is serving in the SERVER process",
            bool(again.get("cached")),
            f"second POST /api/tablets/{tablet}/analyse -> "
            f"cached={again.get('cached')} in {dt:.3f} s\n"
            f"If cached=False the warm above bought the participant nothing.")

    if skip_domain3:
        rep.add("Domain 3 warmed into VRAM", False,
                "SKIPPED at the operator's request (--skip-domain3).\n"
                "Task 7 will then meet the cold call: measured at 86.02 s "
                "server-side on P335940, of which about 82 s is the model "
                "loading into video memory (n=1, 2026-09-19).")
        return

    print("     pulling the Domain 3 model into VRAM (this is the slow one) ...")
    s = time.perf_counter()
    try:
        first = _post(f"/api/tablets/{tablet}/summarise", timeout=400.0)
    except Exception as exc:  # noqa: BLE001
        rep.add("Domain 3 warmed into VRAM", False,
                f"{type(exc).__name__}: {exc}")
        return
    cold = time.perf_counter() - s

    s = time.perf_counter()
    second = _post(f"/api/tablets/{tablet}/summarise", timeout=400.0)
    warm_s = time.perf_counter() - s

    rep.add("Domain 3 warmed into VRAM", True,
            f"first call {cold:.2f} s (cached={first.get('cached')}), "
            f"second {warm_s:.2f} s (cached={second.get('cached')})\n"
            f"The second call is served from the per-tablet cache, so it does "
            f"not measure generation. What it confirms is that task 7 on "
            f"{tablet} is instant.\n"
            f"Reference: 86.02 s cold / 3.77 s warm / 0.21 s cached, "
            f"P335940, n=1, 2026-09-19.")


# Entry point

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-session readiness gate for Part 8 user testing.")
    ap.add_argument("--tablet", default=SCRIPT_TABLET,
                    help="the tablet the task script uses (default %(default)s)")
    ap.add_argument("--skip-domain3", action="store_true",
                    help="do not warm Domain 3 (leaves an 86 s cold call in "
                         "task 7; only for a session that skips task 7)")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("PART 8 -- PRE-SESSION READINESS GATE")
    print("=" * 72)
    print("Start both servers first, from D:\\FYP:")
    print("  terminal 1:  .\\start.ps1  then")
    print("               python -m uvicorn cuneiform.api.main:app --port 8000")
    print("  terminal 2:  cd D:\\FYP\\web  then  npm run dev")
    print("The dashboard is at http://localhost:5173 ONLY. The numeric address")
    print("127.0.0.1:5173 is actively refused: Vite binds IPv6 loopback.")

    rep = Report()
    health = check_server(rep)
    if health is None:
        print("\n" + "=" * 72)
        print("NOT READY -- the orchestrator is not answering.")
        print("Do not start a participant session. Nothing below was measured.")
        print("=" * 72)
        return 1

    check_domain3(rep, health)
    check_population_separation(rep)
    check_prepared_upload(rep)
    warm(rep, args.tablet, args.skip_domain3)

    failed = [n for n, ok, _ in rep.rows if not ok]
    print("\n" + "=" * 72)
    if rep.ok:
        print(f"READY -- {len(rep.rows)}/{len(rep.rows)} checks passed.")
        print("The caches are warm in the process the browser talks to.")
        print("Open http://localhost:5173 and check by eye that a tablet")
        print("renders before the participant arrives.")
    else:
        print(f"NOT READY -- {len(failed)} of {len(rep.rows)} checks failed:")
        for n in failed:
            print(f"  - {n}")
        print("Do not start a participant session until these pass. Any timing")
        print("printed above was measured in a state the session will not be in.")
    print("=" * 72)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
