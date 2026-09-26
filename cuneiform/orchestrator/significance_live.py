"""
Runs the paired significance tests over both fusion artifacts, the simulated
fusion_results.json and the live live_fusion_results.json, in one labelled
table, and adds a studentized selection correction. I import the statistics
from significance.py unchanged so its own artifact stays reproducible.

The original selection correction flips the signs of uncentred difference rows,
so its null is set by the worst lambda's size rather than by chance (null mean
+0.0068 instead of 0). It is valid but very conservative. I report it next to
two studentized versions, one with the standard error held fixed and one
refitting it in every permutation; they are different statistics and both are
named. Every permutation test prints its null mean and p95 beside its p-value.

Arms that stay silent where vision can't name a sign emit far fewer tokens than
the reference, so comparing them with length-matched arms is meaningless in
sign. Those comparisons are printed without a star, under a banner saying why.

The live test is per tablet (n = 30): the unweighted mean of per-tablet
differences, 20,000 sign-flip permutations, seed 1729, with the pooled
difference always printed beside it. Where the interval spans zero I report
the estimate and interval, not significance.

Run from the project root:
    python -m cuneiform.orchestrator.significance_live
"""

from __future__ import annotations

import json

import numpy as np

from cuneiform import config
from cuneiform.orchestrator.significance import (
    N_BOOT, N_PERM, SEED, bootstrap_ci, paired_permutation, report, series,
)

SIM_JSON = config.ARTIFACTS_DIR / "fusion_results.json"
LIVE_JSON = config.ARTIFACTS_DIR / "live_fusion_results.json"
OUT_JSON = config.ARTIFACTS_DIR / "live_fusion_significance.json"

# Length regimes: which arms can be compared on an absolute level.
#
#   drop-uninformative  silent where vision can't name a sign, so it emits fewer
#                       tokens than the reference
#   length-matched      one label per scored slot, with a sentinel where it has no name
#
# Comparing across the two is meaningless, since a silent arm scores better by
# not answering. I declare the regimes here as data, so any new arm is covered
# automatically.
DROP_UNINFORMATIVE_ARMS = {
    "1_vision_only",
    "0a_oracle_geo", "0a_oracle_view", "0a_oracle_flat",
}


def length_regime(arm: str) -> str:
    return ("drop-uninformative" if arm in DROP_UNINFORMATIVE_ARMS
            else "length-matched")


def load(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def lambda_arms(data: dict) -> list[float]:
    return sorted(float(k.removeprefix("3_fusion_lam"))
                  for k in data["per_block"] if k.startswith("3_fusion_lam"))


def _diff_matrix(data: dict, lambdas: list[float], metric: str,
                 baseline: str = "2_threshold_mask") -> np.ndarray:
    """(n_lambda, n_unit) matrix of baseline - fusion, so positive means better."""
    base = series(data, baseline, metric)
    return np.vstack([base - series(data, f"3_fusion_lam{l}", metric)
                      for l in lambdas])


def selection_corrected_shipped(data, lambdas, metric, n_perm=N_PERM) -> dict:
    """
    The original selection correction, recomputed here only so its null mean
    and p95 can be printed beside the p-value. It uses a fresh rng, so the
    p-value can differ from the saved one by Monte Carlo error.
    """
    rng = np.random.default_rng(SEED)
    D = _diff_matrix(data, lambdas, metric)
    obs = D.mean(axis=1).max()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, D.shape[1]))
    null = (signs @ D.T / D.shape[1]).max(axis=1)
    p = (int((null >= obs - 1e-12).sum()) + 1) / (n_perm + 1)
    return {"stat": "raw max-T (shipped)", "best_lambda":
            lambdas[int(D.mean(axis=1).argmax())], "obs": float(obs),
            "p": p, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "n_perm": n_perm}


def selection_corrected_studentized(data, lambdas, metric,
                                    refit_se: bool = True,
                                    n_perm=N_PERM) -> dict:
    """
    Westfall-Young max-T on the studentized difference, so lambdas with large
    effects can't dominate the maximum just by size.

    refit_se=True recomputes the standard error in every permutation (the
    standard studentized test). refit_se=False holds it at the observed value,
    which stays somewhat conservative and gives a larger p. Both are reported.
    """
    rng = np.random.default_rng(SEED)
    D = _diff_matrix(data, lambdas, metric)
    n = D.shape[1]
    se_obs = D.std(axis=1, ddof=1) / np.sqrt(n)
    se_obs = np.where(se_obs <= 0, np.inf, se_obs)
    t_obs = D.mean(axis=1) / se_obs
    obs = float(t_obs.max())

    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    # Built one lambda at a time to keep memory to one (n_perm, n) array.
    null = np.full(n_perm, -np.inf)
    for r in range(D.shape[0]):
        flipped = signs * D[r]
        means = flipped.mean(axis=1)
        if refit_se:
            sd = flipped.std(axis=1, ddof=1)
            se = np.where(sd <= 0, np.inf, sd) / np.sqrt(n)
        else:
            se = se_obs[r]
        null = np.maximum(null, means / se)
    p = (int((null >= obs - 1e-12).sum()) + 1) / (n_perm + 1)
    return {"stat": f"studentized max-T, SE "
                    f"{'refit' if refit_se else 'held fixed'}",
            "refit_se": bool(refit_se),
            "best_lambda": lambdas[int(t_obs.argmax())],
            "obs_t": obs, "obs_mean_diff": float(D.mean(axis=1).max()),
            "p": p, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "n_perm": n_perm}


def micro_paired(data: dict, arm_a: str, arm_b: str, metric: str,
                 rng, n_perm=N_PERM) -> dict | None:
    """
    The pooled (micro) difference with the same per-unit sign flips. Always
    printed beside the unweighted headline, never chosen between. Live artifact only.
    """
    if "per_unit" not in data:
        return None
    key_d = "dist" if metric == "wer" else "cer_dist"
    key_n = "len_truth" if metric == "wer" else "cer_len"
    da = np.array([r[key_d] for r in data["per_unit"][arm_a]], float)
    db = np.array([r[key_d] for r in data["per_unit"][arm_b]], float)
    den = np.array([r[key_n] for r in data["per_unit"][arm_a]], float)
    d = da - db
    obs = d.sum() / den.sum()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs(signs @ d / den.sum())
    p = (int((null >= abs(obs) - 1e-12).sum()) + 1) / (n_perm + 1)
    return {"micro_diff": float(obs), "p": p,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95))}


def _emitted_len(data: dict, arm: str):
    """The arm's total emitted tokens, if the artifact records them."""
    try:
        return int(data["arms"][arm]["len_pred"])
    except (KeyError, TypeError, ValueError):
        return None


def report_uninterpretable(name_a: str, name_b: str, a: np.ndarray,
                           b: np.ndarray, metric: str, rng,
                           data: dict) -> dict:
    """
    Print a cross-regime comparison without a significance star, under a banner
    explaining why its sign can't be read. The numbers stay visible; only the
    presentation differs from report().
    """
    d = a - b
    mean_d = d.mean()
    lo, hi = bootstrap_ci(d, rng)
    p = paired_permutation(d, rng)
    la, lb = _emitted_len(data, name_a), _emitted_len(data, name_b)
    ref = data.get("population", {}).get("scored_reference")

    print(f"\n  {metric.upper()}: {name_b} vs {name_a}")
    print("    !! LENGTH-MISMATCHED -- THE SIGN OF THIS DIFFERENCE IS "
          "UNINTERPRETABLE !!")
    print(f"       {name_a:<24} {length_regime(name_a):<18}"
          + (f" emits {la:,} tokens" if la else ""))
    print(f"       {name_b:<24} {length_regime(name_b):<18}"
          + (f" emits {lb:,} tokens" if lb else ""))
    if ref:
        print(f"       reference{'':<20} {'':<18} {ref:,} tokens")
    print("       A drop-uninformative arm stays SILENT where vision cannot")
    print("       name a sign, so it avoids the insertions the length-matched")
    print("       arms pay for answering everywhere. It scores better by")
    print("       DECLINING TO ANSWER, not by reading better.")
    if "1b_vision_only_padded" in data.get("per_block", {}):
        print("       --> Use 1b_vision_only_padded for this question. It keeps")
        print("           every position and marks the unnamed ones, so the two")
        print("           sides are the same length and the difference means")
        print("           what it appears to mean.")
    else:
        print("       --> This artifact has NO length-matched vision arm, so")
        print("           the question cannot be answered from it at all. The")
        print("           live artifact's 1b_vision_only_padded is the arm that")
        print("           answers it. Do not substitute this row for it.")
    print(f"    {name_a:<26} {a.mean():.4f}")
    print(f"    {name_b:<26} {b.mean():.4f}")
    print(f"    mean difference           {mean_d:+.4f}   "
          f"(sign NOT interpretable)")
    print(f"    95% CI (bootstrap)        [{lo:+.4f}, {hi:+.4f}]")
    print(f"    permutation p (2-sided)   {p:.4f}   "
          f"(deliberately UNSTARRED -- see above)")

    return {"mean_diff": mean_d, "ci": [lo, hi], "p": p,
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "tied": int((d == 0).sum()),
            "length_mismatched": True,
            "regime_a": length_regime(name_a),
            "regime_b": length_regime(name_b),
            "len_a": la, "len_b": lb,
            "interpretable": False,
            "note": "cross-regime comparison; sign uninterpretable. Use "
                    "1b_vision_only_padded instead."}


def report_pair(name_a: str, name_b: str, a: np.ndarray, b: np.ndarray,
                metric: str, rng, data: dict) -> dict:
    """Send a comparison to report() or report_uninterpretable() based on the arms' declared regimes."""
    if length_regime(name_a) == length_regime(name_b):
        return report(name_a, name_b, a, b, metric, rng)
    return report_uninterpretable(name_a, name_b, a, b, metric, rng, data)


def mde(n: int) -> dict:
    root = n ** 0.5
    return {"n": n, "d_detectable_p05": 1.96 / root,
            "d_80pct_power": (1.96 + 0.8416) / root}


def run_one(label: str, data: dict, rng) -> dict:
    lambdas = lambda_arms(data)
    n_units = len(data["per_block"]["2_threshold_mask"])
    unit = data.get("unit", "block")
    deployed = data.get("config", {}).get("lambda_deployed",
                                          data.get("best_lambda"))
    arm_dep = f"3_fusion_lam{deployed}"

    print("\n" + "=" * 74)
    print(f"{label}   unit = {unit}, n = {n_units}")
    print("=" * 74)
    print(f"  {N_PERM:,} permutations, {N_BOOT:,} bootstrap resamples, "
          f"seed {SEED}")
    m = mde(n_units)
    print(f"  minimum detectable effect: d >= {m['d_detectable_p05']:.3f} at "
          f"p<0.05 | 80% power at d = {m['d_80pct_power']:.3f}")
    print(f"  deployed lambda {deployed}; sweep {lambdas}")

    out: dict = {"unit": unit, "n_units": n_units, "mde": m,
                 "deployed_lambda": deployed,
                 "simulated_vision": data.get("simulated_vision")}

    for metric in ("wer", "cer"):
        print("\n" + "-" * 74)
        thr = series(data, "2_threshold_mask", metric)
        dep = series(data, arm_dep, metric)
        vis = series(data, "1_vision_only", metric)

        out[f"{metric}_fusion_vs_threshold"] = report_pair(
            "2_threshold_mask", arm_dep, thr, dep, metric, rng, data)
        # Cross-regime: printed without a star.
        out[f"{metric}_fusion_vs_vision"] = report_pair(
            "1_vision_only", arm_dep, vis, dep, metric, rng, data)

        # The clean test: both sides length-matched, so the difference is only
        # the signs the language model recovered.
        if "1b_vision_only_padded" in data["per_block"]:
            pad = series(data, "1b_vision_only_padded", metric)
            out[f"{metric}_fusion_vs_padded"] = report_pair(
                "1b_vision_only_padded", arm_dep, pad, dep, metric, rng, data)
            for extra in ("3b_lmfill_fusion_ctx", "3f_deployed_lam1_floor"):
                if extra in data["per_block"]:
                    out[f"{metric}_{extra}_vs_threshold"] = report_pair(
                        "2_threshold_mask", extra, thr,
                        series(data, extra, metric), metric, rng, data)

        shipped = selection_corrected_shipped(data, lambdas, metric)
        stud_refit = selection_corrected_studentized(
            data, lambdas, metric, refit_se=True)
        stud_fixed = selection_corrected_studentized(
            data, lambdas, metric, refit_se=False)
        out[f"{metric}_corrected_shipped"] = shipped
        out[f"{metric}_corrected_studentized"] = stud_refit
        out[f"{metric}_corrected_studentized_fixed_se"] = stud_fixed

        print(f"\n  {metric.upper()}: selection correction over "
              f"{len(lambdas)} lambdas, {N_PERM:,} permutations")
        print(f"    {'statistic':<34} {'best lam':>9} {'obs':>10} "
              f"{'p':>8} {'null mean':>11} {'null p95':>10}")
        print(f"    {shipped['stat']:<34} {shipped['best_lambda']:>9} "
              f"{shipped['obs']:>10.4f} {shipped['p']:>8.4f} "
              f"{shipped['null_mean']:>11.4f} {shipped['null_p95']:>10.4f}")
        for s in (stud_refit, stud_fixed):
            print(f"    {s['stat']:<34} {s['best_lambda']:>9} "
                  f"{s['obs_t']:>10.4f} {s['p']:>8.4f} "
                  f"{s['null_mean']:>11.4f} {s['null_p95']:>10.4f}")
        print("    A paired-difference null must sit at 0. The shipped row's")
        print("    null mean is the whole problem: it is driven by the WORST")
        print("    lambda's magnitude, not by chance near the winner.")
        print("    The two studentized rows are DIFFERENT STATISTICS, not a")
        print("    figure and its error bar. Report the pair; the gap between")
        print("    them is itself evidence about the correction, not the data.")

        mic = micro_paired(data, "2_threshold_mask", arm_dep, metric, rng)
        if mic:
            out[f"{metric}_micro_fusion_vs_threshold"] = mic
            print(f"\n    micro/pooled {metric.upper()} difference "
                  f"(secondary, never selected between):")
            print(f"      {mic['micro_diff']:+.4f}   p {mic['p']:.4f}   "
                  f"null mean {mic['null_mean']:+.4f}   "
                  f"null p95 {mic['null_p95']:.4f}")
    return out


def main() -> None:
    print("=" * 74)
    print("SIGNIFICANCE -- SIMULATED and LIVE side by side")
    print("=" * 74)
    print("  significance.py is NOT modified; its artifact stays reproducible.")
    print("  Every test below prints its own null mean and p95 beside its")
    print("  p-value. A paired-difference null centred anywhere but 0 is a")
    print("  defective test, however plausible its p-value looks.")

    out = {}
    for label, path in (("SIMULATED  (artifacts/fusion_results.json)", SIM_JSON),
                        ("LIVE       (artifacts/live_fusion_results.json)",
                         LIVE_JSON)):
        data = load(path)
        if data is None:
            print(f"\n  {path.name} not found -- skipped. Run "
                  f"evaluate_live.py first.")
            continue
        rng = np.random.default_rng(SEED)
        out["live" if path is LIVE_JSON else "simulated"] = run_one(
            label, data, rng)

    print("\n" + "=" * 74)
    print("HOW TO STATE THIS")
    print("=" * 74)
    sim = out.get("simulated", {})
    if sim:
        sh = sim.get("wer_corrected_shipped", {})
        st = sim.get("wer_corrected_studentized", {})
        sf = sim.get("wer_corrected_studentized_fixed_se", {})
        print(f"  SIMULATED WER, selection-corrected: shipped p = "
              f"{sh.get('p', float('nan')):.4f}; studentized p = "
              f"{st.get('p', float('nan')):.4f} (SE refit) / "
              f"{sf.get('p', float('nan')):.4f} (SE fixed).")
        print("  Report the shipped figure AND a studentized one, and say")
        print("  which studentization. The shipped figure is valid family-wise")
        print("  error control and is what fusion_significance.json carries")
        print("  (p = 0.0554 there; a fresh permutation draw of the same test")
        print("  lands a few thousandths away, which is Monte Carlo error, not")
        print("  a disagreement). The studentized figures are what the same")
        print("  data support once the correction is not dominated by the")
        print("  worst lambda -- and they differ from each other by far more")
        print("  than Monte Carlo error, so the variant must be named.")
        cs = sim.get("cer_corrected_studentized", {})
        print(f"  SIMULATED CER is NOT rescued: the best arm is still lambda "
              f"{cs.get('best_lambda')}")
        print("  -- the arm where the language model contributes nothing.")
    live = out.get("live", {})
    if live:
        print(f"\n  LIVE is scored per TABLET at n = {live['n_units']}, so it")
        print("  sidesteps significance.py's block-clustering limitation. Read")
        print("  the minimum-detectable-effect line before reading any p-value:")
        print("  at this n an interval spanning zero is weak evidence of")
        print("  nothing, not evidence of no effect.")
        print("  NEVER compare a LIVE absolute level with a SIMULATED one --")
        print("  different reference, support, length regime and scored")
        print("  population. Only the PAIRED, within-artifact differences are")
        print("  comparable across the two tables.")
    else:
        print("\n  No live artifact yet.")

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {OUT_JSON}")
    print("=" * 74)


if __name__ == "__main__":
    main()
