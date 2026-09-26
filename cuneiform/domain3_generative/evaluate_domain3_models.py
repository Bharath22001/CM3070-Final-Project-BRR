"""
Compares my deployed Domain 3 model (llama3.1:8b) with rival local models on
the same 84 generations and the same scoring.

I reuse evaluate_domain3's loops and scoring unchanged, only swapping the model
name, and read the deployed model's rows from domain3_evaluation.json instead
of re-running them. Gates: M1 same tablets per arm, M2 the published rates
reproduce, M3 each rival really is a different model, M4 each rival produced
84 rows. The prompt was written against llama, which may favour it.

Run from the project root (Ollama running, rivals pulled):
    python -m cuneiform.domain3_generative.evaluate_domain3_models
    python -m cuneiform.domain3_generative.evaluate_domain3_models qwen2.5:7b
"""

from __future__ import annotations

import functools
import json
import sys
import time

from cuneiform import config
from cuneiform.domain3_generative import evaluate_domain3 as ed3
from cuneiform.domain3_generative.baseline_template import (
    audit_rates, gate_method, gate_population, published_rows,
)
from cuneiform.domain3_generative.generate import MODEL as DEPLOYED

RIVALS = ["qwen2.5:7b", "mistral:7b"]
OUT_JSON = config.D3_DIR / "domain3_model_comparison.json"
RULE = "=" * 78
N_TABLETS, N_SWEEP, REPEATS = 12, 6, 2


def digest(model: str) -> str | None:
    import ollama
    for m in ollama.list().get("models", []):
        name = m.get("model") or m.get("name")
        if name == model:
            return m.get("digest")
    return None


def run_model(model: str, tablets, sweep, catalogue) -> tuple[list[dict], float]:
    original = ed3.generate
    ed3.generate = functools.partial(original, model=model)
    try:
        t0 = time.time()
        rows = (ed3.run_arm_12(tablets, REPEATS, catalogue)
                + ed3.run_arm_3(sweep, REPEATS, catalogue))
        return rows, time.time() - t0
    finally:
        ed3.generate = original


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rivals = sys.argv[1:] or RIVALS

    print(RULE)
    print(f"DOMAIN 3 MODEL SELECTION -- {DEPLOYED} (deployed) against {rivals}")
    print(RULE)
    catalogue = ed3.load_catalogue()
    all_tabs = ed3.test_tablets()
    tablets, sweep = all_tabs[:N_TABLETS], all_tabs[:N_SWEEP]
    pub = published_rows()

    m1, m1_detail = gate_population(pub, tablets, sweep)
    m2, m2_detail = gate_method(pub)
    dep_digest = digest(DEPLOYED)
    print(f"\n  M1 population matches the published rows: {'PASS' if m1 else 'FAIL'}")
    print(m1_detail)
    print(f"  M2 aggregation reproduces the published figures: "
          f"{'PASS' if m2 else 'FAIL'}")
    print(m2_detail)
    if not (m1 and m2):
        print("\n  FAIL -- the comparison would not be the same measurement. "
              "Nothing run.")
        sys.exit(1)

    pub_ok = [r for r in pub if r["schema_ok"]]
    results = {f"deployed ({DEPLOYED})": {"digest": dep_digest,
                                           "rates": audit_rates(pub),
                                           "rates_successful_calls_only":
                                               audit_rates(pub_ok),
                                           "n_failed": len(pub) - len(pub_ok),
                                           "source": "domain3_evaluation.json (read)"}}
    all_rows, gates = [], {"M1": m1, "M2": m2}
    for model in rivals:
        dg = digest(model)
        m3 = dg is not None and dg != dep_digest
        print(f"\n  M3 {model} is a distinct pulled model: "
              f"{'PASS' if m3 else 'FAIL'}  (digest {str(dg)[:19]} vs deployed "
              f"{str(dep_digest)[:19]})")
        gates[f"M3 {model}"] = m3
        if not m3:
            continue
        rows, secs = run_model(model, tablets, sweep, catalogue)
        m4 = len(rows) == len(pub)
        gates[f"M4 {model}"] = m4
        print(f"  M4 {model} produced {len(rows)} rows against {len(pub)} "
              f"published: {'PASS' if m4 else 'FAIL'}")
        for r in rows:
            r["model"] = model
        all_rows += rows
        warm = [r["secs"] for r in rows if r["schema_ok"]]
        okrows = [r for r in rows if r["schema_ok"]]
        results[model] = {"digest": dg, "rates": audit_rates(rows),
                          # audit_rates counts a failed call as clean, so a rival
                          # that fails often would look more faithful. Report both.
                          "rates_successful_calls_only":
                              audit_rates(okrows) if okrows else None,
                          "wall_seconds": round(secs, 1),
                          "median_call_seconds":
                              sorted(warm)[len(warm) // 2] if warm else None,
                          "n_failed": sum(1 for r in rows if not r["schema_ok"]),
                          "errors": sorted({r["error"][:120] for r in rows
                                            if r.get("error")})}

    ok = all(gates.values())
    print("\n" + RULE)
    print(f"GATES: {'PASS' if ok else 'FAIL -- ignore every number below'}  {gates}")
    print("CAVEAT, BEFORE THE NUMBERS: the prompt and schema were developed "
          f"against {DEPLOYED}.")
    print(RULE)
    print(f"  {'model':26s} {'valid':>6} {'cite-len':>8} {'cite-str':>8} "
          f"{'prose':>6} {'coverage':>8} {'n_cited':>7}")
    for name, v in results.items():
        r = v["rates"]
        print(f"  {name:26s} {100*r['schema_valid']:6.1f} "
              f"{100*r['citation_lenient']:8.1f} {100*r['citation_strict']:8.1f} "
              f"{100*r['prose_clean']:6.1f} {r['mean_coverage']:8.3f} "
              f"{r['mean_n_cited']:7.1f}   failed calls {v.get('n_failed')}")
    print("\n  over SUCCESSFUL calls only (a failed call otherwise counts "
          "as clean):")
    for name, v in results.items():
        r = v.get("rates_successful_calls_only")
        if r:
            print(f"  {name:26s} n={r['n_rows']:>3}  cite-len "
                  f"{100*r['citation_lenient']:5.1f}  prose "
                  f"{100*r['prose_clean']:5.1f}  coverage {r['mean_coverage']:.3f}")
    with OUT_JSON.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump({"design": f"{N_TABLETS} tablets x {REPEATS} repeats at mask 0 "
                             f"+ {N_SWEEP} tablets x 5 mask rates x {REPEATS}",
                   "scoring": "evaluate_domain3.score_one, "
                              "baseline_template.audit_rates over every row",
                   "caveat": f"prompt and schema developed against {DEPLOYED}",
                   "gates": gates, "gates_pass": ok, "results": results,
                   "rows": all_rows}, fh, ensure_ascii=False, indent=2)
    print(f"\n  written: {OUT_JSON}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
