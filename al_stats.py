"""Paired statistical tests on the active learning curves (validation only).

For each seed, every strategy starts from the same labelled set and the same model,
so strategies can be compared seed by seed (paired design).
Two summaries per run:  mean val F1 over the AL cycles (how fast it learns)
                        val F1 at the last cycle (where it ends)
Test: Wilcoxon signed-rank (two-sided), with Holm correction for multiple comparisons.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("results/tables/al_curves.csv")
al = df[df.cycle > 0]                       # cycle 0 is identical for all strategies
per_seed = {
    "mean over cycles": al.groupby(["strategy", "seed"]).val_f1.mean().unstack("strategy"),
    "last cycle": al[al.cycle == al.cycle.max()].set_index(["seed", "strategy"]).val_f1.unstack("strategy"),
}
print("Seeds:", len(per_seed["last cycle"]))


def holm(p):
    """Holm step-down correction: returns adjusted p-values in the original order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(table, reference, others, summary):
    rows = []
    for s in others:
        diff = table[s] - table[reference]              # one difference per seed
        rows.append(dict(summary=summary, comparison=f"{s} vs {reference}",
                         mean_diff=diff.mean(), std_diff=diff.std(),
                         seeds_better=int((diff > 0).sum()), n_seeds=len(diff),
                         p_wilcoxon=wilcoxon(table[s], table[reference]).pvalue))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_wilcoxon)
    return out


strategies = [s for s in per_seed["last cycle"].columns if s != "random"]
best = per_seed["mean over cycles"][strategies].mean().idxmax()

results = []
for summary, table in per_seed.items():
    results.append(compare(table, "random", strategies, summary))            # AL vs random
results.append(compare(per_seed["mean over cycles"], best,                   # best AL vs other AL
                       [s for s in strategies if s != best], "mean over cycles"))

res = pd.concat(results, ignore_index=True).round(4)
res.to_csv("results/tables/al_stats.csv", index=False)
print(res.to_string(index=False))