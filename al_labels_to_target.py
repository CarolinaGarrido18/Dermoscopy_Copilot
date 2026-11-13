"""Labels needed to reach a fraction of the full model's validation macro F1.

Reads results/tables/al_curves.csv (validation only, no test data).
Full model = linear head trained on 100% of the training set (same as al_results.py).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from active_learning import load_data, LinearHead

d = load_data()
ref = LinearHead(seed=0).fit(d["Xtr"], d["ytr"], epochs=100, X_val=d["Xva"], y_val=d["yva"])
ref_f1 = f1_score(d["yva"], ref.predict_proba(d["Xva"]).argmax(1), average="macro")
print(f"Validation macro F1 with 100% of the training set: {ref_f1:.4f}")

df = pd.read_csv("results/tables/al_curves.csv")
max_n = df.n_labelled.max()
rows = []
for frac in [0.90, 0.95]:
    target = frac * ref_f1
    for strategy, g in df.groupby("strategy"):
        # per seed: first number of labelled images at which the target is reached (inf if never)
        first = pd.Series({seed: s.loc[s.val_f1 >= target, "n_labelled"].min()
                           for seed, s in g.groupby("seed")}).fillna(np.inf)
        mean_curve = g.groupby("n_labelled").val_f1.mean()
        reached = mean_curve[mean_curve >= target]
        med = first.median()
        rows.append(dict(target=f"{int(frac * 100)}% of full model (F1 >= {target:.3f})",
                         strategy=strategy,
                         seeds_reached=f"{int(np.isfinite(first).sum())}/{len(first)}",
                         median_labels=int(med) if np.isfinite(med) else f">{max_n}",
                         mean_curve_labels=int(reached.index.min()) if len(reached) else f">{max_n}"))
res = pd.DataFrame(rows)
res.to_csv("results/tables/al_labels_to_target.csv", index=False)
print("\n" + res.to_string(index=False))