"""
Dermoscopy Copilot - 04. Labelling cost of each query strategy

Expresses the active learning results in terms of annotation cost. For each query strategy, computes
the number of labelled images needed to reach 90% and 95% of the validation macro F1 of a reference
model trained on 100% of the training images (the same reference as in al_results.py). Only the
validation results are used.

The table is saved in results/tables/al_labels_to_target.csv.

Prerequisite: results/tables/al_curves.csv produced by active_learning.py
"""

# Library imports
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from active_learning import load_data, LinearHead  # data and model from active_learning.py


# Reference model
# Model trained on 100% of the training set, evaluated on the validation set
d = load_data()
ref = LinearHead(seed=0).fit(d["Xtr"], d["ytr"], epochs=100, X_val=d["Xva"], y_val=d["yva"])
ref_f1 = f1_score(d["yva"], ref.predict_proba(d["Xva"]).argmax(1), average="macro")
print(f"Validation macro F1 with 100% of the training set: {ref_f1:.4f}")


# Labelled images needed to reach each target
df = pd.read_csv("results/tables/al_curves.csv")  # results of the 50 experiments
max_n = df.n_labelled.max()  # largest number of labelled images (labelling budget)
rows = []
for frac in [0.90, 0.95]:
    target = frac * ref_f1  # 90% and 95% of the reference macro F1
    for strategy, g in df.groupby("strategy"):
        # For each seed, the first number of labelled images at which the target is reached (inf if it is never reached)
        first = pd.Series({seed: s.loc[s.val_f1 >= target, "n_labelled"].min() for seed, s in g.groupby("seed")}).fillna(np.inf)
        # The same quantity read on the mean learning curve over the 10 seeds                   
        mean_curve = g.groupby("n_labelled").val_f1.mean()
        reached = mean_curve[mean_curve >= target]
        med = first.median()  # median over the seeds
        rows.append(dict(target=f"{int(frac * 100)}% of full model (F1 >= {target:.3f})", strategy=strategy, seeds_reached=f"{int(np.isfinite(first).sum())}/{len(first)}", median_labels=int(med) if np.isfinite(med) else f">{max_n}", mean_curve_labels=int(reached.index.min()) if len(reached) else f">{max_n}"))

# Save the table and print it
res = pd.DataFrame(rows)
res.to_csv("results/tables/al_labels_to_target.csv", index=False)
print("\n" + res.to_string(index=False))