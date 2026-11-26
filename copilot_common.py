"""Shared by the app.py and by the retrain_with_corrections.py script. Mirrors extract_features.py, referral.py and gradcam_eval.py.
The model used by the app is the final active learning head (al_head.pt, created by export_al_head.py) retrained with the
clinician's corrections (al_head_retrained.pt, created by retrain_with_corrections.py).
"""
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_data_config
from torchvision import transforms

BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"   # pretrained timm model used as a frozen feature extractor
BASE = Path(__file__).parent                       # repository root (folder where this file lives)
FEATS = BASE / "features" / f"features_{BACKBONE}"  # pre-extracted features (train/val/test .npz files)
RESULTS = BASE / "results" / "results_baseline" / f"results_{BACKBONE}"   # passive baseline (kept only for comparison)
AL_DIR = BASE / "results" / "results_active_learning"                      # output folder of export_al_head.py
HEAD_PATH = AL_DIR / "al_head_retrained.pt"       # model of the app: active learning head retrained with the clinician's corrections (al_head.pt is the original)
AL_LABELLED = AL_DIR / "al_labelled.npz"          # images labelled by the active learning loop (ids + labels given)
AL_METRICS = AL_DIR / "al_head_metrics.json"      # test metrics of the active learning head
APP_DIR = BASE / "app_data"                       # everything the app produces at runtime
IMG_DIR = APP_DIR / "images"                      # copies of the images reviewed by the clinician
LOG_CSV = APP_DIR / "feedback_log.csv"            # one row per clinician decision (accept / correct)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # HAM10000 classes; the order matches the head outputs
N_AUG = 2                                         # = extract_features.py
device = "cuda" if torch.cuda.is_available() else "cpu"  # use the GPU when available, otherwise fall back to CPU


def load_backbone():
    """Frozen backbone and transforms identical to those of extract_features.py.

    Loads the pretrained EfficientNetV2-S without its classification layer
    (num_classes=0), so it outputs a feature vector per image. It also builds the
    preprocessing pipelines, which must match the ones used to extract the training
    features; otherwise the head would receive inputs from a different distribution.

    Returns:
        bb: the backbone in eval mode, already on `device`.
        base_tf: deterministic transform (resize + tensor + normalisation), used at inference.
        aug_tf: random augmentation transform (flips, rotation, brightness), used for training.
        size: input resolution expected by the backbone (images are resized to size x size).
    """
    bb = timm.create_model(BACKBONE, pretrained=True, num_classes=0).eval().to(device)
    # Read the resolution and normalisation statistics the backbone was trained with
    cfg = resolve_data_config({}, model=bb)
    size = cfg["input_size"][1]
    norm = transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    # Inference pipeline: no randomness, so the same image always gives the same prediction
    base_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    # Training pipeline: lesions have no canonical orientation, hence flips and full rotations
    aug_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3), transforms.ToTensor(), norm])
    return bb, base_tf, aug_tf, size


def vistas_tta(x):
    """8 TTA views (4 rotations x horizontal flip), as in extract_features.py. 

    Test-time augmentation: the same image is shown to the model in 8 different
    orientations and the predictions are later averaged, which makes the output
    more stable. Returns a list of 8 tensors, each with the same shape as x.
    """
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))            
        out += [r, torch.flip(r, dims=(3,))]          # keep the rotated view and its horizontal mirror
    return out


def load_head(path=HEAD_PATH):
    """Loads the trained linear classification head from a checkpoint.

    The head is a single linear layer that maps the backbone features to one logit
    per class. The checkpoint stores the input dimension and the weights, so the
    layer is rebuilt with the right shape before the weights are loaded.

    Args:
        path: checkpoint file (by default, the final active learning head).

    Returns:
        The head in eval mode, on `device`.
    """
    ckpt = torch.load(path, map_location="cpu")       # load on CPU first so it works without a GPU
    head = nn.Linear(ckpt["in_features"], len(CLASSES))
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval()


def fit_temperature(logits_val, y_val):
    """Same as referral.py: T fitted on the validation set over the mean of the logits of the 8 views.

    Temperature scaling is a post-hoc calibration method: the logits are divided by
    a single scalar T so that the softmax confidences better reflect the real
    probability of being correct. T is found by minimising the cross-entropy on the
    validation set; it changes the confidence values but never the predicted class.

    Args:
        logits_val: (N, n_classes) tensor with the TTA-averaged validation logits.
        y_val: true labels of the validation images.

    Returns:
        The fitted temperature T (T > 1 softens overconfident predictions).
    """
    T = nn.Parameter(torch.ones(1))                   
    y_t = torch.tensor(y_val, dtype=torch.long)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        # LBFGS may evaluate the loss several times per step, so it needs a closure
        opt.zero_grad()
        loss = F.cross_entropy(logits_val / T.clamp(min=1e-3), y_t)  # clamp avoids dividing by ~0
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.clamp(min=1e-3).item())


def calibrate(head):
    """Reproduces the calibration of referral.py for the given head.

    Uses the pre-extracted validation features (8 TTA views per image), passes them
    through the head, averages the logits over the views (exactly as the app does at
    inference) and fits the temperature on those averaged logits.

    Args:
        head: the linear classification head to calibrate.

    Returns:
        The temperature T to divide the logits by before the softmax.
    """
    d = np.load(FEATS / "val_features.npz", allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)             # (N, 8, D)
    N, V, D = X.shape
    with torch.no_grad():
        # Flatten the views into the batch, apply the head, then average the 8 views per image
        logits = head.cpu()(X.reshape(N * V, D)).reshape(N, V, -1).mean(1)
    head.to(device)                                           # move the head back to the original device
    return fit_temperature(logits, d["y"])
