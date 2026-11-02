import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from active_learning import load_data, LinearHead

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
df = pd.read_csv("results/tables/al_curves.csv")
d = load_data()

# Reference: model trained on 100% of the training set, evaluated on the same validation set
ref = LinearHead(seed=0).fit(d["Xtr"], d["ytr"], epochs=100,
                             X_val=d["Xva"], y_val=d["yva"])
f1_ref = f1_score(d["yva"], ref.predict_proba(d["Xva"]).argmax(1), average="macro")
print("Validation macro F1 with 100% of the training set:", round(f1_ref, 4))

# Learning curves (mean and std over 3 seeds)
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, g in df.groupby("strategy"):
    m = g.groupby("n_labelled").val_f1.agg(["mean", "std"])
    color = "gray" if name == "random" else None
    line, = ax.plot(m.index, m["mean"], marker="o", ms=3, label=name, color=color)
    ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"],
                    alpha=0.12, color=line.get_color())
ax.axhline(f1_ref, ls="--", color="k", label="100% of training set")
ax.set_xlabel("Labelled images")
ax.set_ylabel("Macro F1 (validation)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/al_curves.png", dpi=200)

# Which classes each strategy requested (% of all requested images)
dc = df.dropna(subset=["requested_classes"]).copy()
mat = np.array(dc.requested_classes.apply(ast.literal_eval).tolist())
for i, c in enumerate(CLASSES):
    dc[c] = mat[:, i]
tab = dc.groupby("strategy")[CLASSES].sum()
perc = (tab.div(tab.sum(axis=1), axis=0) * 100).round(1)
perc.loc["training set (all images)"] = (
    np.bincount(d["y_orig"], minlength=7) / len(d["y_orig"]) * 100).round(1)
perc.to_csv("results/tables/al_requested_classes.csv")
print("\nRequested classes (% of the total):")
print(perc)