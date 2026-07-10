"""Statistical analysis over per-question scores.

- bootstrap 95% CIs (percentile, 10k resamples) for every metric mean
- paired GRPO-vs-DPO tests at each size on the same 200 questions:
  McNemar's exact test for binary metrics, Wilcoxon signed-rank for
  continuous ones, Holm correction within each metric across sizes
"""

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config

BINARY_METRICS = ["greedy_correct", "pass1_correct", "sc_correct"]
CONTINUOUS_METRICS = [
    "consistency_ratio",
    "nli_score",
    "judge_score_first",
    "judge_score_mean",
    "judge_score_greedy",
]
ALL_METRICS = BINARY_METRICS + CONTINUOUS_METRICS

RNG = np.random.default_rng(config.MASTER_SEED)
N_BOOT = 10_000


def load_all_scores():
    frames = []
    for (method, size) in config.ADAPTER_CHECKPOINTS:
        path = config.scores_path(method, size)
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(values, n_boot=N_BOOT):
    values = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def mcnemar_exact(a, b):
    """Exact McNemar test on paired binary outcomes."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return min(1.0, 2 * sps.binom.cdf(min(n01, n10), n, 0.5))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def aggregate_table(df):
    rows = []
    for (method, size), grp in df.groupby(["method", "size"]):
        row = {"method": method, "size": size, "n": len(grp)}
        for m in ALL_METRICS:
            if m not in grp.columns:
                continue
            mean, lo, hi = bootstrap_ci(grp[m].astype(float))
            row[m] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(df):
    """GRPO vs DPO on identical question sets, per size and metric."""
    results = []
    for metric in ALL_METRICS:
        per_size = []
        for size in ["1.5b", "3b", "7b", "14b"]:
            g = df[(df.method == "grpo") & (df["size"] == size)]
            d = df[(df.method == "dpo") & (df["size"] == size)]
            if metric not in g.columns or g.empty or d.empty:
                continue
            merged = g[["question_id", metric]].merge(
                d[["question_id", metric]], on="question_id",
                suffixes=("_grpo", "_dpo"),
            ).dropna()
            if merged.empty:
                continue
            x = merged[f"{metric}_grpo"].astype(float)
            y = merged[f"{metric}_dpo"].astype(float)
            if metric in BINARY_METRICS:
                p = mcnemar_exact(x.astype(bool), y.astype(bool))
            else:
                diff = x - y
                if np.allclose(diff, 0):
                    p = 1.0
                else:
                    p = sps.wilcoxon(x, y, zero_method="pratt").pvalue
            per_size.append(
                {
                    "metric": metric,
                    "size": size,
                    "n_pairs": len(merged),
                    "grpo_mean": x.mean(),
                    "dpo_mean": y.mean(),
                    "diff": x.mean() - y.mean(),
                    "p_raw": p,
                }
            )
        if per_size:
            adj = holm([r["p_raw"] for r in per_size])
            for r, pa in zip(per_size, adj):
                r["p_holm"] = pa
            results.extend(per_size)
    return pd.DataFrame(results)


def run_aggregate_stage():
    df = load_all_scores()
    agg = aggregate_table(df)
    tests = paired_tests(df)

    config.AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(config.AGGREGATE_DIR / "aggregate_new.csv", index=False)
    tests.to_csv(config.AGGREGATE_DIR / "significance_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== Aggregate (mean [95% CI]) ===")
    for _, r in agg.iterrows():
        cells = []
        for m in ALL_METRICS:
            if m in r and not pd.isna(r.get(m, np.nan)):
                cells.append(f"{m}={r[m]:.3f} [{r[f'{m}_ci_lo']:.3f},{r[f'{m}_ci_hi']:.3f}]")
        print(f"{r['method']:>4} {r['size']:>4} (n={r['n']}): " + " | ".join(cells))
    print("\n=== Paired GRPO vs DPO (Holm-corrected) ===")
    if not tests.empty:
        print(
            tests[
                ["metric", "size", "n_pairs", "grpo_mean", "dpo_mean", "diff",
                 "p_raw", "p_holm"]
            ].to_string(index=False)
        )
    return agg, tests
