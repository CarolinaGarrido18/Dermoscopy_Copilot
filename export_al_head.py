"""
Dermoscopy Copilot - Export of the final active learning model

Loads the linear head saved by al_hitl.py at the last active learning cycle (50% of the training
images labelled), calibrates its probabilities on the validation set with temperature scaling, and
saves it in the same format as baseline_head.pt, so that the referral rule and the interface can
use it in the same way. The test metrics and the labelled images of this run are also saved.

The model is saved in results/results_active_learning/al_head.pt, the metrics in
results/results_active_learning/al_head_metrics.json and the labelled images in
results/results_active_learning/al_labelled.npz.

Prerequisite: results/heads/ produced by al_hitl.py
"""

# Library imports
import argparse
import json
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from active_learning import BACKBONE, load_split  # features of active_learning.py


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # class names (in the order of the labels 0 to 6)
MEL = CLASSES.index("mel")  # class 4 is melanoma
OUT = Path("results") / "results_active_learning"  # output folder


def logits_views(X, W, b):
    # Applies the linear layer to every view. X: (N, V, D) -> logits (N, V, C)
    return torch.tensor(X) @ W.T + b


def probs_tta(logits, T=1.0):
    # Mean of the probabilities of the views (softmax with temperature T), as in train_baseline.py
    return F.softmax(logits / T, dim=2).mean(dim=1)


def fit_temperature(logits, y):
    # Temperature scaling, finds the T that minimises the NLL on the validation set (same method as train_baseline.py)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) stays positive
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.nll_loss(torch.log(probs_tta(logits, log_T.exp()) + 1e-12), y)  # 1e-12 avoids log(0)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().detach())


# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("--strategy", default="hybrid")
ap.add_argument("--oracle", default="perfect", choices=["perfect", "single", "majority"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


# Linear head at the last active learning cycle
name = f"results/heads/{args.strategy}__{args.oracle}__seed{args.seed}"
heads = torch.load(f"{name}.pt", map_location="cpu")
n_labelled = max(heads)  # number of labelled images at the last cycle
W, b = heads[n_labelled]["weight"].float(), heads[n_labelled]["bias"].float()
print("Model:", name, "| labelled images:", n_labelled)


# Calibration on the validation set and evaluation on the test set
Xva, yva, _, _ = load_split("val")
Xte, yte, _, _ = load_split("test")
with torch.no_grad():
    logits_va, logits_te = logits_views(Xva, W, b), logits_views(Xte, W, b)
T = fit_temperature(logits_va, torch.tensor(yva, dtype=torch.long))
probs = probs_tta(logits_te, T).detach().numpy()
pred = probs.argmax(1)
metrics = {"strategy": args.strategy, "oracle": args.oracle, "seed": args.seed, "n_labelled": int(n_labelled), "f1_macro": f1_score(yte, pred, average="macro"), "balanced_acc": balanced_accuracy_score(yte, pred), "recall_melanoma": float(recall_score(yte, pred, labels=[MEL], average=None)[0]), "auc_macro": roc_auc_score(yte, probs, multi_class="ovr", average="macro", labels=list(range(7))), "temperature": T, "backbone": BACKBONE, "n_views": int(Xte.shape[1])}
print(json.dumps(metrics, indent=2))


# Saving in the same format as baseline_head.pt
torch.save({"state_dict": {"weight": W, "bias": b}, "T": T, "classes": CLASSES, "backbone": BACKBONE, "in_features": int(W.shape[1]), "n_views": int(Xte.shape[1])}, OUT / "al_head.pt")
with open(OUT / "al_head_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
shutil.copy(f"{name}_labelled.npz", OUT / "al_labelled.npz")  # labelled images of this run (to retrain with the corrections of the clinician)
print("Saved in:", OUT)