"""
Tests whether the fusion vs threshold difference on the simulated results is
more than noise, reading the per-block scores saved by evaluate_fusion.py.

I use a paired permutation test (flipping the sign of each block's difference)
because the differences are skewed with a spike at zero, so no normality is
assumed. Lambda was picked as the best of ten on this same data, so I report a
naive test of the winner and a corrected max-statistic test that recomputes
the best of ten in every permutation.

Blocks from the same tablet aren't independent, so these p-values are, if
anything, a little optimistic. Scores are per-block WER and CER, the same
quantity as the results table.

Run from the project root:
    python -m cuneiform.orchestrator.significance
"""

from __future__ import annotations

import json

import numpy as np

from cuneiform import config

RESULTS_JSON = config.ARTIFACTS_DIR / "fusion_results.json"
N_PERM = 20000
N_BOOT = 20000
SEED = 1729


def load_results() -> dict:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            f"{RESULTS_JSON} not found. Run evaluate_fusion.py first."
        )
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "per_block" not in data:
        raise KeyError(
            "fusion_results.json has no 'per_block' field. Apply the one-line "
            "edit to evaluate_fusion.py and re-run it."
        )
    return data


def series(data: dict, arm: str, metric: str) -> np.ndarray:
    return np.array([b[metric] for b in data["per_block"][arm]], dtype=float)


def paired_permutation(d: np.ndarray, rng, n_perm: int = N_PERM) -> float:
    """
    Two-sided paired permutation test on the mean of d. The +1 on both sides
    of the ratio stops it ever reporting p = 0.
    """
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs(signs @ d / len(d))
    return (int((null >= obs - 1e-12).sum()) + 1) / (n_perm + 1)


def bootstrap_ci(d: np.ndarray, rng, n_boot: int = N_BOOT) -> tuple:
    """Percentile bootstrap 95% interval for the mean difference."""
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def report(name_a: str, name_b: str, a: np.ndarray, b: np.ndarray,
           metric: str, rng) -> dict:
    """d = a - b, so a positive mean means b has the lower error."""
    d = a - b
    mean_d = d.mean()
    lo, hi = bootstrap_ci(d, rng)
    p = paired_permutation(d, rng)
    better = int((d > 0).sum())
    worse = int((d < 0).sum())
    tied = int((d == 0).sum())
    rel = 100 * mean_d / a.mean() if a.mean() else 0.0

    print(f"\n  {metric.upper()}: {name_b} vs {name_a}")
    print(f"    {name_a:<26} {a.mean():.4f}")
    print(f"    {name_b:<26} {b.mean():.4f}")
    print(f"    mean improvement          {mean_d:+.4f}  ({rel:+.1f}% relative)")
    print(f"    95% CI (bootstrap)        [{lo:+.4f}, {hi:+.4f}]")
    print(f"    blocks better/worse/tied  {better} / {worse} / {tied}")
    print(f"    permutation p (2-sided)   {p:.4f}"
          f"{'   *' if p < 0.05 else ''}")

    return {"mean_diff": mean_d, "ci": [lo, hi], "p": p,
            "better": better, "worse": worse, "tied": tied}


def selection_corrected(data: dict, lambdas: list[float], metric: str,
                        rng) -> dict:
    """
    Max-statistic permutation test over the lambda sweep. Each permutation
    flips each block's sign for every lambda at once, which keeps the strong
    correlation between lambdas.
    """
    thr = series(data, "2_threshold_mask", metric)
    D = np.vstack([thr - series(data, f"3_fusion_lam{l}", metric)
                   for l in lambdas])          # (n_lambda, n_blocks)

    obs = D.mean(axis=1).max()
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, D.shape[1]))
    null = (signs @ D.T / D.shape[1]).max(axis=1)
    p = (int((null >= obs - 1e-12).sum()) + 1) / (N_PERM + 1)

    best_idx = int(D.mean(axis=1).argmax())
    print(f"\n  {metric.upper()}: selection-corrected (best of "
          f"{len(lambdas)} lambdas)")
    print(f"    best lambda               {lambdas[best_idx]}")
    print(f"    its mean improvement      {obs:+.4f}")
    print(f"    max-statistic p (1-sided) {p:.4f}"
          f"{'   *' if p < 0.05 else ''}")
    return {"best_lambda": lambdas[best_idx], "obs": obs, "p": p}


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = load_results()

    lambdas = sorted(
        float(k.removeprefix("3_fusion_lam"))
        for k in data["per_block"] if k.startswith("3_fusion_lam")
    )
    best = data["best_lambda"]
    arm3 = f"3_fusion_lam{best}"

    print("=" * 68)
    print("SIGNIFICANCE  (SIMULATED vision input)")
    print("=" * 68)
    print(f"  {data['n_blocks']} blocks | {data['n_scored']:,} scored positions")
    print(f"  {N_PERM:,} permutations, {N_BOOT:,} bootstrap resamples, "
          f"seed {SEED}")
    print(f"  best lambda from the sweep: {best}")

    out = {}
    for metric in ("wer", "cer"):
        thr = series(data, "2_threshold_mask", metric)
        fus = series(data, arm3, metric)
        vis = series(data, "1_vision_only", metric)

        print("\n" + "-" * 68)
        out[f"{metric}_fusion_vs_threshold"] = report(
            "2_threshold_mask", f"3_fusion(lam={best})", thr, fus, metric, rng)
        out[f"{metric}_fusion_vs_vision"] = report(
            "1_vision_only", f"3_fusion(lam={best})", vis, fus, metric, rng)
        out[f"{metric}_threshold_vs_vision"] = report(
            "1_vision_only", "2_threshold_mask", vis, thr, metric, rng)
        out[f"{metric}_corrected"] = selection_corrected(
            data, lambdas, metric, rng)

    print("\n" + "=" * 68)
    print("HOW TO STATE THIS")
    print("=" * 68)
    naive = out["wer_fusion_vs_threshold"]["p"]
    corr = out["wer_corrected"]["p"]
    if corr < 0.05:
        print("  The WER improvement survives correction for lambda selection.")
        print("  Safe to state as a significant improvement over the threshold")
        print("  design, with the lambda sweep reported alongside it.")
    elif naive < 0.05:
        print("  The WER improvement is significant at a pre-specified lambda")
        print("  but does NOT survive correction for having chosen lambda on")
        print("  this same data. State it that way -- an honest split result is")
        print("  far more defensible than an uncorrected claim.")
    else:
        print("  The WER improvement is NOT distinguishable from noise at n =")
        print(f"  {data['n_blocks']} blocks. Report the direction and the")
        print("  interval, not a significance claim. Raising MAX_BLOCKS in")
        print("  evaluate_fusion.py would tighten the interval.")
    print("\n  In every case, remember: blocks within a tablet are correlated,")
    print("  so these p-values are mildly optimistic. Say so in the report.")

    path = config.ARTIFACTS_DIR / "fusion_significance.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  written: {path}")
    print("=" * 68)


if __name__ == "__main__":
    main()