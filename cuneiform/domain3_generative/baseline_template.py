"""
Three non-generative baselines for Domain 3, scored by exactly the same four
audits as my deployed model.

    metadata_echo   returns the catalogue block as the reading and cites nothing,
                    to test whether the model's context is just retrieval.
    sign_list_echo  returns the shown sign tokens and cites every one. I
                    predicted before running it that it would score coverage
                    near 1.0 with no invented signs: faithful but useless.
    empty           valid schema, no content; the floor.

These are deterministic string templates, so they need no Ollama and give the
same output every time. The deployed model's rows are read from the saved
artifact, never re-run, and every row is scored by evaluate_domain3.score_one.
The design mirrors the published one: 84 rows per baseline.

Run from the project root:
    python -m cuneiform.domain3_generative.baseline_template
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from cuneiform import config
from cuneiform.domain3_generative.evaluate_domain3 import (
    MASK_RATES, score_one, test_tablets,
)
from cuneiform.domain3_generative.generate import (
    display_tokens, load_catalogue, load_tablet, metadata_block,
)

PUBLISHED = "domain3_evaluation.json"
ARTIFACT = "domain3_baselines.json"

N_ARM12_TABLETS = 12
N_SWEEP_TABLETS = 6
REPEATS = 2

# The audits. Every row counts in the denominator, including failed calls,
# because that is the only rule under which the published figures
# (98.8 / 51.2 / 9.5 / 92.9) reproduce.
AUDITS = [
    ("schema_valid", "Valid JSON schema"),
    ("citation_lenient", "Citation audit, lenient"),
    ("citation_strict", "Citation audit, strict"),
    ("prose_clean", "Prose scan clean"),
]


# The three templates. Each returns the same dict shape as generate().

def tpl_metadata_echo(t, catalogue: dict) -> dict:
    """Echo the catalogue block. Cites nothing, because it read nothing."""
    block, had = metadata_block(t.tablet, catalogue)
    body = block.split("\n", 1)[1].strip() if "\n" in block else block
    return {
        "tentative_reading": body,
        "confidence": "low",
        "signs_used": [],
        "uncertainty_notes": "",
        "historical_context": body,
        "metadata_used": had,
        "_had_metadata": had,
    }


def tpl_sign_list_echo(t, catalogue: dict) -> dict:
    """Echo the shown sign tokens in order and cite every one of them."""
    _, had = metadata_block(t.tablet, catalogue)
    tokens: list[str] = []
    for v in t.lines:
        tokens.extend(display_tokens(v))
    return {
        "tentative_reading": " ".join(tokens),
        "confidence": "low",
        "signs_used": sorted(t.displayed_names),
        "uncertainty_notes": "",
        "historical_context": "",
        "metadata_used": False,
        "_had_metadata": had,
    }


def tpl_empty(t, catalogue: dict) -> dict:
    """Schema-conformant and contentless."""
    _, had = metadata_block(t.tablet, catalogue)
    return {
        "tentative_reading": "",
        "confidence": "none",
        "signs_used": [],
        "uncertainty_notes": "",
        "historical_context": "",
        "metadata_used": False,
        "_had_metadata": had,
    }


TEMPLATES = [
    ("metadata_echo", "Emits the catalogue metadata, cites nothing",
     tpl_metadata_echo),
    ("sign_list_echo", "Emits and cites every sign it was shown",
     tpl_sign_list_echo),
    ("empty", "Schema-conformant and contentless", tpl_empty),
]


# Running and scoring

def run_template(name: str, fn, tablets: list[str], sweep: list[str],
                 catalogue: dict) -> list[dict]:
    """Mirror the published design: 24 arm 1-2 rows plus 60 arm 3 rows."""
    rows = []
    for tab in tablets:
        t = load_tablet(tab)
        for r in range(REPEATS):
            s = score_one(t, fn(t, catalogue))
            s.update(arm="1-2", tablet=tab, repeat=r, mask_rate=0.0,
                     damage=t.damage_rate, baseline=name, error="")
            rows.append(s)
    for rate in MASK_RATES:
        for tab in sweep:
            t = load_tablet(tab, mask_rate=rate)
            for r in range(REPEATS):
                s = score_one(t, fn(t, catalogue))
                s.update(arm="3", tablet=tab, repeat=r, mask_rate=rate,
                         damage=t.damage_rate, baseline=name, error="")
                rows.append(s)
    return rows


def audit_rates(rows: list[dict]) -> dict:
    """The four audit rates plus mean coverage, over every row."""
    n = len(rows)
    return {
        "n_rows": n,
        "schema_valid": sum(1 for r in rows if r["schema_ok"]) / n,
        "citation_lenient": sum(1 for r in rows if r["n_cited_bad"] == 0) / n,
        "citation_strict":
            sum(1 for r in rows if r["n_cited_bad_strict"] == 0) / n,
        "prose_clean": sum(1 for r in rows if r["n_prose_bad"] == 0) / n,
        "mean_coverage": sum(r["coverage"] for r in rows) / n,
        "mean_n_cited": sum(r["n_cited"] for r in rows) / n,
    }


def published_rows() -> list[dict]:
    """The deployed model's rows, read from disk, not recomputed."""
    path = config.D3_DIR / PUBLISHED
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Gates. Raw evidence first, verdict second.

def gate_population(pub: list[dict], tablets: list[str],
                    sweep: list[str]) -> tuple[bool, str]:
    """The baselines must run on the same tablets as the published rows."""
    pub12 = sorted({r["tablet"] for r in pub if r["arm"] == "1-2"})
    pub3 = sorted({r["tablet"] for r in pub if r["arm"] == "3"})
    ok = pub12 == sorted(tablets) and pub3 == sorted(sweep)
    detail = (f"    published arm 1-2 tablets : {pub12}\n"
              f"    comparator arm 1-2 tablets: {sorted(tablets)}\n"
              f"    published arm 3 tablets   : {pub3}\n"
              f"    comparator arm 3 tablets  : {sorted(sweep)}")
    return ok, detail


def gate_method(pub: list[dict]) -> tuple[bool, str]:
    """
    My aggregation must reproduce the published Domain 3 figures, otherwise the
    two columns of the table aren't the same measurement.
    """
    got = audit_rates(pub)
    want = {"schema_valid": 0.988, "citation_lenient": 0.512,
            "citation_strict": 0.095, "prose_clean": 0.929}
    lines = []
    ok = True
    for k, w in want.items():
        g = got[k]
        hit = abs(g - w) < 0.001
        ok = ok and hit
        lines.append("    %-18s published %.1f%%   recomputed %.1f%%   %s"
                     % (k, 100 * w, 100 * g, "match" if hit else "MISMATCH"))
    return ok, "\n".join(lines)


def gate_prediction(summary: dict) -> tuple[bool, str]:
    """
    Check the prediction I wrote before the run: sign_list_echo should reach
    coverage near 1.0 with no invented citation. If not, the template or the
    scorer is wrong.
    """
    s = summary["sign_list_echo"]
    ok = s["mean_coverage"] > 0.99 and s["citation_lenient"] > 0.99
    detail = ("    sign_list_echo mean coverage      : %.4f  (predicted > 0.99)\n"
              "    sign_list_echo citation clean rate: %.4f  (predicted > 0.99)"
              % (s["mean_coverage"], s["citation_lenient"]))
    return ok, detail



def write_artifact(summary: dict, rows: list[dict], gates: dict,
                   tablets: list[str], sweep: list[str]) -> Path:
    payload = {
        "population": (
            "Per comparator: %d tablets at mask 0.0 with %d repeats (arm 1-2) "
            "plus %d tablets across mask rates %s with %d repeats (arm 3), "
            "mirroring the published Domain 3 design exactly. The repeats are "
            "identical to one another by construction: these templates are "
            "deterministic, where the deployed stage runs at temperature 0.2 "
            "with no seed and is not reproducible."
            % (N_ARM12_TABLETS, REPEATS, N_SWEEP_TABLETS,
               MASK_RATES, REPEATS)
        ),
        "method": (
            "Every row scored by evaluate_domain3.score_one, unmodified. Rates "
            "counted as C18_domain3_audits counts them: numerator is rows "
            "passing, denominator is every row. The deployed row is READ from "
            "artifacts/domain3/domain3_evaluation.json and never recomputed."
        ),
        "tablets_arm12": sorted(tablets),
        "tablets_arm3": sorted(sweep),
        "summary": summary,
        "gates": gates,
        "rows": rows,
    }
    path = config.D3_DIR / ARTIFACT
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    print("=" * 74)
    print("DOMAIN 3 -- THREE NON-GENERATIVE COMPARATORS")
    print("=" * 74)

    catalogue = load_catalogue()
    all_tabs = test_tablets()
    tablets = all_tabs[:N_ARM12_TABLETS]
    sweep = all_tabs[:N_SWEEP_TABLETS]
    pub = published_rows()

    print(f"  catalogue records : {len(catalogue):,}")
    print(f"  published rows    : {len(pub)}")
    print(f"  comparators       : {len(TEMPLATES)}")

    # Gates that must pass before any number is printed
    print("\n" + "-" * 74)
    print("GATE 1  the comparators run on the published population")
    print("-" * 74)
    g1, d1 = gate_population(pub, tablets, sweep)
    print(d1)
    print("  -> %s" % ("PASS" if g1 else "FAIL"))

    print("\n" + "-" * 74)
    print("GATE 2  our aggregation reproduces the published Domain 3 figures")
    print("-" * 74)
    g2, d2 = gate_method(pub)
    print(d2)
    print("  -> %s" % ("PASS" if g2 else "FAIL"))

    if not (g1 and g2):
        print("\n  IF THIS FAILED, IGNORE EVERY NUMBER BELOW: the comparator "
              "and deployed columns are not the same measurement.")

    # Run
    summary = {"deployed (llama3.1:8b)": audit_rates(pub)}
    rows: list[dict] = []
    for name, _what, fn in TEMPLATES:
        r = run_template(name, fn, tablets, sweep, catalogue)
        rows.extend(r)
        summary[name] = audit_rates(r)

    print("\n" + "-" * 74)
    print("GATE 3  the pre-registered prediction about sign_list_echo")
    print("-" * 74)
    g3, d3 = gate_prediction(summary)
    print(d3)
    print("  -> %s" % ("PASS" if g3 else "FAIL"))
    if not g3:
        print("     The template or the scorer is wrong. Read nothing below "
              "as a finding about the deployed model.")

    gates = {"population_matches_published": g1,
             "aggregation_reproduces_published": g2,
             "prediction_sign_list_echo": g3,
             "all_pass": bool(g1 and g2 and g3)}

    # The panel
    print("\n" + "=" * 74)
    print("THE FOUR AUDITS, DEPLOYED MODEL AGAINST THREE TEMPLATES")
    print("=" * 74)
    order = ["deployed (llama3.1:8b)"] + [n for n, _w, _f in TEMPLATES]
    print(f"\n  {'method':<24} " + " ".join("%10s" % k for k, _l in AUDITS)
          + "   coverage")
    print("  " + "-" * 82)
    for name in order:
        s = summary[name]
        print(f"  {name:<24} "
              + " ".join("%9.1f%%" % (100 * s[k]) for k, _l in AUDITS)
              + "   %8.3f" % s["mean_coverage"])

    print("\n" + "=" * 74)
    print("INTERPRETATION")
    print("  Read the coverage column and the citation columns TOGETHER.")
    print("  sign_list_echo copies its input, so it is perfectly faithful and")
    print("  perfectly useless: the faithfulness audits cannot tell copying")
    print("  from reading, and that is a statement about the AUDITS.")
    print("  metadata_echo is the one that tests the deployed model. It emits")
    print("  the catalogue block the model was handed and nothing else.")
    print("=" * 74)

    path = write_artifact(summary, rows, gates, tablets, sweep)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"\n  wrote {path}")
    print(f"  md5   {digest}")
    print(f"  {time.time() - t0:.1f}s, no network call and no Ollama call")


if __name__ == "__main__":
    main()
