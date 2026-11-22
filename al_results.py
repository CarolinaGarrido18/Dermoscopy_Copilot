"""
Dermoscopy Copilot - 04. Active learning results: learning curves and requested classes

Reads the results of the 50 active learning experiments and trains a reference model on 100% of
the training images. Draws the learning curves (validation macro F1 against the number of labelled
images, mean and standard deviation over the 10 seeds) and computes the percentage of each class
among the images requested by each query strategy.

The figure is saved in results/figures/al_curves.png and the table in 
results/tables/al_requested_classes.csv.

Prerequisite: results/tables/al_curves.csv produced by active_learning.py
"""

# Library imports
import ast  # converts the saved text back into a list
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # save the figures to a file without opening a window
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from active_learning import load_data, LinearHead  # data and model from active_learning.py


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # class names (in the order of the labels 0 to 6)


# Data
df = pd.read_csv("results/tables/al_curves.csv")   # results of the 50 experiments
d = load_data()


# Reference model
# Model trained on 100% of the training set, evaluated on the same validation set
ref = LinearHead(seed=0).fit(d["Xtr"], d["ytr"], epochs=100, X_val=d["Xva"], y_val=d["yva"])
f1_ref = f1_score(d["yva"], ref.predict_proba(d["Xva"]).argmax(1), average="macro")
print("Validation macro F1 with 100% of the training set:", round(f1_ref, 4))


# Learning curves
# Mean and standard deviation of the validation macro F1 over the 10 seeds, for each strategy
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, g in df.groupby("strategy"):
    m = g.groupby("n_labelled").val_f1.agg(["mean", "std"])  # mean and std at each number of labelled images
    color = "gray" if name == "random" else None             # random sampling in grey, to stand out
    line, = ax.plot(m.index, m["mean"], marker="o", ms=3, label=name, color=color)
    ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.12, color=line.get_color()) # band with the same colour as the line
ax.axhline(f1_ref, ls="--", color="k", label="100% of training set")
ax.set_xlabel("Labelled images")
ax.set_ylabel("Macro F1 (validation)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/al_curves.png", dpi=200)


# Requested classes
# Which classes each strategy requested (% of all requested images), compared with the training set
dc = df.dropna(subset=["requested_classes"]).copy()  # cycle 0 does not request any image
mat = np.array(dc.requested_classes.apply(ast.literal_eval).tolist())
for i, c in enumerate(CLASSES):
    dc[c] = mat[:, i]  # one column per class
tab = dc.groupby("strategy")[CLASSES].sum()  # total requested images of each class
perc = (tab.div(tab.sum(axis=1), axis=0) * 100).round(1)  # percentages, each row adds up to 100
perc.loc["training set (all images)"] = (np.bincount(d["y_orig"], minlength=7) / len(d["y_orig"]) * 100).round(1)
perc.to_csv("results/tables/al_requested_classes.csv")
print("\nRequested classes (% of the total):")
print(perc)