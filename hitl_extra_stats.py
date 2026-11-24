"""
Test missing from al_hitl.py: majority vote (3 doctors) vs a single doctor.

Compares 'majority' with 'single' in two ways, for each strategy (hybrid and random):
  1) SAME IMAGES: TEST macro F1 at the last cycle (both have the same labelled images,
     but 'majority' costs 3x more annotations).
  2) SAME HUMAN COST: VALIDATION macro F1 at the same number of annotations. Since the curve
     only has one point per cycle, it is linearly interpolated (per seed) and 3 costs are
     compared within the range where both curves overlap. It uses validation because the
     test F1 is only computed at the last cycle.

Paired comparison per seed (same seed = same initial set of images), two-sided Wilcoxon,
with the Holm correction over all the comparisons in the table.
Difference = majority - single: POSITIVE values favour the majority vote.

Reads results/tables/hitl_runs.csv (from al_hitl.py) and writes results/tables/hitl_majority_vs_single.csv
Usage:  python3 hitl_extra_stats.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RUNS = "results/tables/hitl_runs.csv"
OUT = "results/tables/hitl_majority_vs_single.csv"


def holm(p):
    """Holm correction for multiple comparisons; returns the adjusted p-values in the original order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def safe_wilcoxon(a, b):
    """Wilcoxon signed-rank p-value, or NaN when the test cannot be run."""
    try:
        return wilcoxon(a, b).pvalue
    except ValueError:                       # all the differences are zero
        return np.nan


def make_row(strategy, basis, a, b):
    """One row of the table: means, paired differences (majority - single) and the Wilcoxon p-value."""
    diff = a.values - b.values
    return dict(strategy=strategy, basis=basis, mean_majority=a.mean(), mean_single=b.mean(),
                mean_diff=diff.mean(), std_diff=diff.std(ddof=1),
                seeds_majority_better=int((diff > 0).sum()), n_seeds=len(diff),
                p=safe_wilcoxon(a.values, b.values))


def main():
    """Runs both comparisons for each strategy and saves the table."""
    res = pd.read_csv(RUNS)
    for o in ("majority", "single"):
        if o not in set(res.oracle):
            raise SystemExit(f"There are no runs with the '{o}' oracle in {RUNS}")
    rows = []
    for strategy in sorted(res.strategy.unique()):
        g = res[res.strategy == strategy]
        maj, sin = g[g.oracle == "majority"], g[g.oracle == "single"]
        seeds = sorted(set(maj.seed) & set(sin.seed))

        # 1) same images, test F1 at the last cycle
        def last(df):
            # Test F1 at the last cycle, one value per seed
            d = df[df.cycle == df.cycle.max()].set_index("seed")
            return d.test_f1.loc[seeds]
        rows.append(make_row(strategy, "test F1, same images (last cycle)", last(maj), last(sin)))

        # 2) same human cost, validation F1 interpolated per seed
        def curve(df, s):
            # Learning curve of one seed: number of annotations (x) and validation F1 (y)
            d = df[df.seed == s].sort_values("n_annotations")
            return d.n_annotations.values.astype(float), d.val_f1.values
        # Range of annotation costs covered by both oracles
        lo = max(maj.n_annotations.min(), sin.n_annotations.min())
        hi = min(maj.n_annotations.max(), sin.n_annotations.max())
        if lo >= hi:
            print(f"[{strategy}] the curves do not overlap in cost; no comparison at equal cost.")
            continue
        for c in np.linspace(lo, hi, 3):
            a = pd.Series({s: np.interp(c, *curve(maj, s)) for s in seeds})
            b = pd.Series({s: np.interp(c, *curve(sin, s)) for s in seeds})
            rows.append(make_row(strategy, f"val F1, same cost ({c:.0f} annotations)", a, b))

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.fillna(1.0))
    out = out.round(4)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print("majority (3 doctors) vs single (1 doctor); difference = majority - single\n")
    print(out.to_string(index=False))
    print(f"\nSaved in {OUT}")


if __name__ == "__main__":
    main()
