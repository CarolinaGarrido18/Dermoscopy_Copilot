"""
Prerequisites: export_al_head.py (al_head.pt, al_labelled.npz, al_head_metrics.json) and corrections saved in the app.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from active_learning import LinearHead              # same model and training function as the active learning loop
from copilot_common import (AL_DIR, AL_LABELLED, AL_METRICS, BACKBONE, CLASSES, FEATS, HEAD_PATH, IMG_DIR,
                            LOG_CSV, N_AUG, device, load_backbone)

SEED, EPOCHS = 42, 3                                # EPOCHS = epochs per cycle in active_learning.py
MEL = CLASSES.index("mel")                          # index of the melanoma class, used for the melanoma recall
torch.manual_seed(SEED); np.random.seed(SEED)       # fixed seeds so that the retraining is reproducible


def carregar(split):
    """Loads the pre-extracted features of one split ("train", "val" or "test").

    Returns:
        X: features as a float tensor (train is (N, D); val/test are (N, 8, D), one row per TTA view).
        y: true labels as a long tensor.
        aug_idx: 0 for the original image, 1..N_AUG for its augmented copies (train only).
        image_ids: id of the source image of each row.
    """
    d = np.load(FEATS / f"{split}_features.npz", allow_pickle=True)
    # image_ids added to the return value, to select the images labelled by the active learning loop
    return (torch.tensor(d["X"], dtype=torch.float32), torch.tensor(d["y"], dtype=torch.long), d["aug_idx"], d["image_ids"])


def probs_tta(logits, T=1.0):                     # logits (N, V, C) -> (N, C)
    """Converts the logits of the TTA views into one probability vector per image.

    Each view is divided by the temperature T and passed through a softmax, and the
    resulting probabilities are averaged over the V views.
    """
    return F.softmax(logits / T, dim=2).mean(dim=1)


def logits_vistas(head, X):
    """Applies the linear head to every TTA view of every image.

    The views are flattened into the batch dimension, passed through the head and
    reshaped back, so X of shape (N, V, D) gives logits of shape (N, V, n_classes).
    """
    N, V, D = X.shape
    return head(X.reshape(N * V, D)).reshape(N, V, -1)


def ajustar_temperatura(logits, y):               # same as train_baseline.py
    """Fits the temperature scaling parameter T on the validation set.

    T is optimised (in log space, which keeps it positive) to minimise the negative
    log-likelihood of the TTA-averaged probabilities. It only rescales the confidence
    of the predictions; the predicted class stays the same.

    Returns:
        The fitted temperature T as a float.
    """
    log_T = torch.zeros(1, requires_grad=True)    
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    def closure():
        # LBFGS may evaluate the loss several times per step, so it needs a closure
        opt.zero_grad()
        loss = F.nll_loss(torch.log(probs_tta(logits, log_T.exp()) + 1e-12), y)  # +1e-12 avoids log(0)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_T.exp())


def metricas(y, probs):
    """Computes the evaluation metrics from the true labels and the predicted probabilities.

    Returns a dict with the macro F1-score, the balanced accuracy (both robust to the
    class imbalance of HAM10000) and the recall of the melanoma class, which is the
    most critical one clinically.
    """
    p = probs.argmax(1)                           # predicted class = most probable one
    return {"f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, p),
            "recall_melanoma": float(recall_score(y, p, labels=[MEL], average=None, zero_division=0)[0])}


# Original data (train 2D: (N, D); val/test 3D: (N, 8, D))
X_tr, y_tr, aug_tr, ids_tr = carregar("train")
X_va, y_va, _, _ = carregar("val")
X_te, y_te, _, _ = carregar("test")

# Keep only the images labelled by the active learning loop (original + augmented copies), with the labels
# given by the annotators (they can differ from the true labels when the oracle is noisy)
al = np.load(AL_LABELLED, allow_pickle=True)      # allow_pickle: the ids are saved as an object array
al_ids = al["image_ids"].astype(str)
rows = np.isin(ids_tr.astype(str), al_ids)        # boolean mask of the training rows that were labelled
# Sanity check: every labelled image must be present in the features, otherwise the files are out of sync
assert len(np.unique(ids_tr.astype(str)[rows])) == len(np.unique(al_ids)), \
    "al_labelled.npz does not match the features: run extract_features.py and al_hitl.py with the same data"
given = pd.Series(al["labels"], index=al_ids)     # image id -> label given by the annotator
X_tr, aug_tr = X_tr[torch.from_numpy(rows)], aug_tr[rows]
# Replace the true labels with the annotators' labels (the augmented copies inherit the label of their image)
y_tr = torch.tensor(given.reindex(ids_tr.astype(str)[rows]).values.astype(int), dtype=torch.long)

# Features of the clinician's images: original + N_AUG augmented copies
# If the same image was saved more than once, only the latest decision is kept
log = pd.read_csv(LOG_CSV).drop_duplicates("image_file", keep="last")
if log.empty:
    raise SystemExit("feedback_log.csv is empty.")
bb, base_tf, aug_tf, _ = load_backbone()
labels = log["final_label"].map({c: i for i, c in enumerate(CLASSES)}).values  # class name -> class index
feats, ys, aug_idx = [], [], []
for f, lab in zip(log["image_file"], labels):
    img = Image.open(IMG_DIR / f).convert("RGB")
    # k = 0 is the original view (deterministic transform); k = 1..N_AUG are random augmented copies
    for k, tf in [(0, base_tf)] + [(k, aug_tf) for k in range(1, N_AUG + 1)]:
        with torch.no_grad():
            feats.append(bb(tf(img).unsqueeze(0).to(device)).float()[0].cpu())
        ys.append(int(lab)); aug_idx.append(k)
X_new, y_new = torch.stack(feats), torch.tensor(ys, dtype=torch.long)
print(f"{len(log)} new images | {int(log['disagreement'].sum())} corrections | "
      f"{int(log['sent_to_human'].sum())} sent to the human | {len(np.unique(al_ids))} images from the active learning loop")

# Training set of the new cycle = images labelled by the active learning loop + the clinician's images
X_all, y_all = torch.cat([X_tr, X_new]), torch.cat([y_tr, y_new])

# One more active learning cycle: continue training the active learning head (not random weights) on the
# labelled images + the clinician's images, exactly as ActiveLearner.teach does (fit with 3 epochs, no early stopping)
model = LinearHead(seed=SEED, n_features=X_all.shape[1])
model.net.load_state_dict(torch.load(HEAD_PATH, map_location="cpu")["state_dict"])  # warm start from al_head.pt
model.fit(X_all.numpy(), y_all.numpy(), epochs=EPOCHS)
head = model.net.eval()

# Calibration and evaluation on the test set (with TTA)
with torch.no_grad():
    lv, lt = logits_vistas(head, X_va), logits_vistas(head, X_te)
T = ajustar_temperatura(lv, y_va)                 # T is fitted on validation only, never on test
res = metricas(y_te.numpy(), probs_tta(lt, T).numpy())
antes = json.load(open(AL_METRICS))               # active learning head before the retraining
# Side-by-side comparison of the test metrics before and after the retraining
print(f"\n{'metric':16s} {'before':>8s} {'after':>8s}")
for k in res:
    print(f"{k:16s} {antes[k]:8.4f} {res[k]:8.4f}")

# al_head.pt is not overwritten; to use the retrained head in the app, point HEAD_PATH (copilot_common.py) to it
torch.save({"state_dict": head.state_dict(), "T": T, "classes": CLASSES, "backbone": BACKBONE,
            "in_features": X_all.shape[1], "n_views": int(X_te.shape[1])},
           AL_DIR / "al_head_retrained.pt")
json.dump({**res, "n_new_images": len(log)}, open(AL_DIR / "retrained_metrics.json", "w"), indent=2)
print("Saved in:", AL_DIR)
