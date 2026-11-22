"""
Dermoscopy Copilot - 04. Statistical comparison of the query strategies

Compares the query strategies (uncertainty sampling, the hybrid strategy that combines uncertainty and
diversity sampling, and random sampling) with paired statistical tests on the validation macro F1. For
each seed, every strategy starts from the same labelled data and the same model, so the strategies can
be compared seed by seed (paired design). Each run is summarised in two ways, the mean validation macro
F1 over the active learning cycles, which shows how fast the strategy learns, and the validation macro
F1 at the last cycle, which shows the performance reached with the full labelling budget. The test is the
two-sided Wilcoxon signed-rank test, with the Holm correction for multiple comparisons.

The table is saved in results/tables/al_stats.csv.

Prerequisite: results/tables/al_curves.csv produced by active_learning.py
"""

# Library imports
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon # paired, non-parametric statistical test


# Data
df = pd.read_csv("results/tables/al_curves.csv")  # results of the 50 experiments
al = df[df.cycle > 0]  # cycle 0 is identical for all strategies, since no image has been queried yet
# Two tables with one row per seed and one column per query strategy
per_seed = {"mean over cycles": al.groupby(["strategy", "seed"]).val_f1.mean().unstack("strategy"), "last cycle": al[al.cycle == al.cycle.max()].set_index(["seed", "strategy"]).val_f1.unstack("strategy")}
print("Seeds:", len(per_seed["last cycle"]))


# Auxiliary functions
def holm(p):
    # Holm correction for multiple comparisons. 
    # Adjusts the p-values so that performing several tests does not increase the chance of a false positive, and returns them in the original order.
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

def compare(table, reference, others, summary):
    # Compares each query strategy with a reference strategy, seed by seed. 
    # For each comparison it reports the mean and standard deviation of the paired differences, the number of seeds in which the strategy was better, and the Wilcoxon p-value before and after the Holm correction.
    rows = []
    for s in others:
        diff = table[s] - table[reference]  # one paired difference per seed
        rows.append(dict(summary=summary, comparison=f"{s} vs {reference}", mean_diff=diff.mean(), std_diff=diff.std(), seeds_better=int((diff > 0).sum()), n_seeds=len(diff), p_wilcoxon=wilcoxon(table[s], table[reference]).pvalue))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_wilcoxon)
    return out


# Comparisons
strategies = [s for s in per_seed["last cycle"].columns if s != "random"]  # the four active learning strategies
best = per_seed["mean over cycles"][strategies].mean().idxmax()            # best strategy, chosen on validation

results = []
for summary, table in per_seed.items():
    results.append(compare(table, "random", strategies, summary))  # each active learning strategy against random sampling
results.append(compare(per_seed["mean over cycles"], best, [s for s in strategies if s != best], "mean over cycles"))  # best strategy against the other three

# Save the table and print it
res = pd.concat(results, ignore_index=True).round(4)
res.to_csv("results/tables/al_stats.csv", index=False)
print(res.to_string(index=False))