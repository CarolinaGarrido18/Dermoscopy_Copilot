import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from active_learning import (load_data, run_experiment, LinearHead,
                             ActiveLearner, entropy_sampling, test_metrics)

SEEDS = [0, 1, 2]
NOISE = 0.10

# 1) Best strategy, chosen on VALIDATION only (mean val F1 over all cycles, random excluded)
curves = pd.read_csv("results/tables/al_curves.csv")
mean_f1 = curves[curves.strategy != "random"].groupby("strategy").val_f1.mean()
best = mean_f1.idxmax()
print("Best strategy on validation (mean over cycles):", best)

d = load_data()

# 2) Final runs: best strategy, random, and best strategy with a noisy expert
rows = []
for name, noise in [(best, 0.0), ("random", 0.0), (best, NOISE)]:
    for seed in SEEDS:
        print("running", name, "| noise", noise, "| seed", seed, flush=True)
        rows += run_experiment(name, seed, d, noise=noise, test=True)
res = pd.DataFrame(rows)
res.to_csv("results/tables/al_final_runs.csv", index=False)

# 3) Test results at the last cycle (mean and std over 3 seeds)
final = res[res.cycle == res.cycle.max()].copy()
final["setting"] = np.where(final.noise > 0, best + " (10% noisy expert)", final.strategy)
summ = final.groupby("setting").agg(
    test_f1=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
    test_bal_acc=("test_bal_acc", "mean"), test_bal_acc_std=("test_bal_acc", "std"),
    test_recall_mel=("test_recall_mel", "mean"), test_recall_mel_std=("test_recall_mel", "std"))

# 4) References: our model on 100% of the training set, and the team baseline
ref = LinearHead(seed=0).fit(d["Xtr"], d["ytr"], epochs=100, X_val=d["Xva"], y_val=d["yva"])
rm = test_metrics(ActiveLearner(ref, entropy_sampling), d)
summ.loc["100% of training set (ours, 1 run)"] = [
    rm["test_f1"], np.nan, rm["test_bal_acc"], np.nan, rm["test_recall_mel"], np.nan]
b = json.load(open("results/baseline_metrics.json"))
summ.loc["team baseline (100%, 1 run)"] = [
    b["f1_macro"], np.nan, b["balanced_acc"], np.nan, b["recall_melanoma"], np.nan]
summ = summ.round(4)
summ.to_csv("results/tables/al_test_results.csv")
print("\nTest results (final cycle, ~52% of the images labelled):")
print(summ.to_string())

# 5) Figure: clean vs noisy expert (validation curves)
fig, ax = plt.subplots(figsize=(7, 4.5))
settings = [("random", res[res.strategy == "random"], "gray"),
            (best, res[(res.strategy == best) & (res.noise == 0)], "tab:blue"),
            (best + " (10% noisy expert)", res[(res.strategy == best) & (res.noise > 0)], "tab:red")]
for label, g, color in settings:
    m = g.groupby("n_labelled").val_f1.agg(["mean", "std"])
    ax.plot(m.index, m["mean"], marker="o", ms=3, label=label, color=color)
    ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.12, color=color)
ax.set_xlabel("Labelled images")
ax.set_ylabel("Macro F1 (validation)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/al_noise.png", dpi=200)