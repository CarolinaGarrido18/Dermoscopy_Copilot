"""
Dermoscopy Copilot - 03. Passive learning baseline with Test-Time Augmentation (TTA)

Trains a linear classifier (D -> 7) on frozen features, using 100% of the labelled training data.
It is the reference for the active learning strategies. It uses class-weighted loss, early stopping
on validation macro F1, TTA (mean over 8 views), temperature scaling, and also reports results without TTA.

Prerequisite: the features_<BACKBONE>/ folder produced by extract_features.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # só grava a figura em ficheiro
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight


BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"   # must match BACKBONE in extract_features.py

BASE = Path(__file__).parent
FEATS = BASE / "features"/ f"features_{BACKBONE}"  # input: features extracted for this backbone
OUT = BASE / "results" / f"results_{BACKBONE}"  # output: head weights, metrics and confusion matrix
OUT.mkdir(exist_ok=True)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = CLASSES.index("mel")   # index of melanoma, whose recall is reported separately
SEED = 42
EPOCHS = 100   # maximum number of training epochs
PATIENCE = 10  # early stopping: epochs without improvement before stopping
BATCH = 256    # large batches are cheap because the model works on embeddings
LR = 1e-3      # Adam learning rate
WEIGHT_DECAY = 1e-4 # L2 regularization

torch.manual_seed(SEED)
np.random.seed(SEED)

# Auxiliary functions
def carregar(split):
    # Loads the features (X), integer labels (y) and augmentation indices (aug_idx) of a split
    d = np.load(FEATS / f"{split}_features.npz", allow_pickle=True)
    return (torch.tensor(d["X"], dtype=torch.float32),
            torch.tensor(d["y"], dtype=torch.long),
            d["aug_idx"])


def logits_vistas(head, X):
    # Applies the linear head to every view. X: (N, V, D) -> logits (N, V, C)
    N, V, D = X.shape
    return head(X.reshape(N * V, D)).reshape(N, V, -1)


def probs_tta(logits, T=1.0):
    # Mean of the probabilities of the V views (softmax with temperature T).
    # logits: (N, V, C) -> (N, C)
    return F.softmax(logits / T, dim=2).mean(dim=1)


def ajustar_temperatura(logits, y):
    # Temperature scaling: finds the T that minimizes the NLL on the validation set,
    # applied on the mean probability of the views
    # Optimize log_T so that T = exp(log_T) remains strictly positive
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        p = probs_tta(logits, log_T.exp())
        loss = F.nll_loss(torch.log(p + 1e-12), y) # 1e-12 avoids log(0)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp())


def ece(probs, y, n_bins=15):
    # Expected Calibration Error: weighted gap between confidence and accuracy over 15 bins
    conf, pred = probs.max(1) # confidence and predicted class of each sample
    acerto = (pred == y).float() # 1 if the prediction is correct, else 0
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) #  samples whose confidence falls in this bin
        if m.any():
            total += m.float().mean() * (conf[m].mean() - acerto[m].mean()).abs()
    return float(total)


def avaliar(y_true, probs):
    # Computes the evaluation metrics: balanced accuracy, macro precision / recall
    # (sensitivity) / F1 / AUC, plus the recall of melanoma and the recall per class
    y_pred = probs.argmax(1)
    rec_cls = recall_score(y_true, y_pred, labels=range(7), average=None, zero_division=0)
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_macro": roc_auc_score(y_true, probs, multi_class="ovr",
                                   average="macro", labels=list(range(7))),
        "recall_melanoma": float(rec_cls[MEL]),
        "recall_por_classe": {c: float(r) for c, r in zip(CLASSES, rec_cls)},
    }


def avaliar_config(logits_va, logits_te, y_va, y_te):
    # Fits T on validation and evaluates on test (with whichever views are passed in)
    T = ajustar_temperatura(logits_va, y_va)
    probs = probs_tta(logits_te, T)
    res = avaliar(y_te.numpy(), probs.numpy())
    res["temperatura"] = T
    res["ece_antes"] = ece(probs_tta(logits_te, 1.0), y_te) # calibration error with T = 1
    res["ece_depois"] = ece(probs, y_te)  # calibration error after scaling
    return res, probs


# Data and class weigths
X_tr, y_tr, aug_tr = carregar("train")
X_va, y_va, _ = carregar("val")
X_te, y_te, _ = carregar("test")
D = X_tr.shape[1] # feature dimension of the backbone
print(f"Backbone: {BACKBONE} | features: {D}")
print("Treino:", tuple(X_tr.shape), "| originais:", int((aug_tr == 0).sum()),
      "| val:", tuple(X_va.shape), "| teste:", tuple(X_te.shape))

# "balanced" weights computed only on the original images (aug_idx == 0), so that the
# augmented copies do not distort the class frequencies
pesos = compute_class_weight("balanced", classes=np.arange(7), y=y_tr[aug_tr == 0].numpy())
print("Pesos das classes:", {c: round(float(p), 2) for c, p in zip(CLASSES, pesos)})
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32))

# Training the classification layer (D -> 7) 
head = nn.Linear(D, 7)
opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

melhor_f1, melhor_estado, sem_melhoria = -1.0, None, 0
for ep in range(1, EPOCHS + 1):
    # Train on shuffled mini-batches of feature vectors
    head.train()
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(head(X_tr[idx]), y_tr[idx])
        loss.backward()
        opt.step()

    # Evaluate on validation with TTA, so model selection mirrors test-time inference
    head.eval()
    with torch.no_grad():
        p_va = probs_tta(logits_vistas(head, X_va))          # validação já com TTA
    f1_val = f1_score(y_va, p_va.argmax(1), average="macro", zero_division=0)
    print(f"Época {ep:3d} | loss {loss.item():.4f} | F1-macro val (TTA) {f1_val:.4f}")

    # Keep the weights with the best validation macro F1 (chosen over accuracy so that
    # minority malignant classes weigh equally); stop after PATIENCE epochs without improvement
    if f1_val > melhor_f1:
        melhor_f1, sem_melhoria = f1_val, 0
        melhor_estado = {k: v.clone() for k, v in head.state_dict().items()}
    else:
        sem_melhoria += 1
        if sem_melhoria >= PATIENCE:
            print("Early stopping.")
            break

# Restore the best weights
head.load_state_dict(melhor_estado)
head.eval()

# Temperature scaling and evaluation on the test set, with and without TTA. 
with torch.no_grad():
    logits_va, logits_te = logits_vistas(head, X_va), logits_vistas(head, X_te)

# With TTA: mean of the probabilities over all 8 views
res_tta, probs_te = avaliar_config(logits_va, logits_te, y_va, y_te)
# Without TTA = only view 0 (original image)
# Slice ':1' (instead of index 0) to keep the 3D shape (N, 1, C) expected by probs_tta
res_sem, _ = avaliar_config(logits_va[:, :1], logits_te[:, :1], y_va, y_te)

# Comparing both quantifies the gain brought by TTA
chaves = ["balanced_acc", "precision_macro", "recall_macro", "f1_macro", "auc_macro",
          "recall_melanoma"]
print(f"\n===== RESULTADOS NO TESTE ({BACKBONE}) =====")
print(f"{'métrica':16s} {'sem TTA':>9s} {'com TTA':>9s}")
for k in chaves:
    print(f"{k:16s} {res_sem[k]:9.4f} {res_tta[k]:9.4f}")
print(f"{'T':16s} {res_sem['temperatura']:9.3f} {res_tta['temperatura']:9.3f}")
print(f"{'ECE':16s} {res_sem['ece_depois']:9.4f} {res_tta['ece_depois']:9.4f}")
print("Recall por classe (com TTA):",
      {c: round(r, 3) for c, r in res_tta["recall_por_classe"].items()})

# Confusion matrix on the test set, with TTA. 
# The rows are the true classes and the columns are the predicted classes.
cm = confusion_matrix(y_te, probs_te.argmax(1), labels=range(7))
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES, rotation=45)
ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black")
ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusão - {BACKBONE} + TTA (teste)")
fig.colorbar(im, label="Fração da classe real")
fig.tight_layout()
fig.savefig(OUT / "baseline_confusion_matrix.png", dpi=150)

# Saving the results
# baseline_head.pt: weights of the linear layer, fitted temperature, classes, backbone name,
# feature dimension and number of views
torch.save({"state_dict": head.state_dict(), "T": res_tta["temperatura"], "classes": CLASSES,
            "backbone": BACKBONE, "in_features": D, "n_views": int(X_te.shape[1])},
           OUT / "baseline_head.pt")
# baseline_metrics.json: the top level holds the result WITH TTA; the result without TTA is
# stored under "sem_tta".
with open(OUT / "baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump({**res_tta, "sem_tta": res_sem, "backbone": BACKBONE,
               "n_views": int(X_te.shape[1])}, f, indent=2)
print("\nGuardado em:", OUT)