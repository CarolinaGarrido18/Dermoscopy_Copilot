"""
Grad-CAM vs lesion masks: does the model look at the lesion? And does that change with the number of labels?

Pipeline:
  1. frozen EfficientNetV2-S backbone (timm) -> activation map A (B, 1280, h, w) of the last
     convolutional layer, computed only once per image;
  2. for each saved linear head (passive baseline + AL heads at several labelling budgets,
     from al_hitl.py): CAM of the PREDICTED class, ReLU, upsampling, normalisation by the maximum;
  3. localisation metrics against the mask (cam_metrics.py) over the whole test set.

Note on Grad-CAM. The classifier is average pooling + a linear layer, so
dlogit_c/dA[k,h,w] = W[c,k]/(h*w) and the Grad-CAM weights (alpha_k = mean of the gradient) are
proportional to W[c,k]. Grad-CAM therefore has this closed form (it matches the CAM of Zhou et al.),
which is what is computed here. `--verify` checks it against Grad-CAM computed with autograd, on one batch.
The map is computed over the ORIGINAL view (without TTA).

Usage:
  python gradcam_eval.py --verify --max-images 64      # quick test
  python gradcam_eval.py                               # full test set
Prerequisites: splits/test.csv, baseline_head.pt, results/heads/*.pt (al_hitl.py), images and
masks (by default obtained with kagglehub; or --img-dir / --mask-dir).
"""
import argparse
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cam_metrics import localisation_metrics

BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"          # same as in extract_features.py
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
METRICS = ["energy_in_mask", "energy_ratio", "pointing_game", "iou", "border_ratio"]


# ------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------
class TestSet(Dataset):
    # Test images with their lesion masks; returns (transformed image, boolean mask, row index)
    def __init__(self, df, img_paths, mask_paths, tf, size):
        self.df, self.img_paths, self.mask_paths = df.reset_index(drop=True), img_paths, mask_paths
        self.tf, self.size = tf, size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        iid = self.df.image_id[i]
        img = Image.open(self.img_paths[iid]).convert("RGB")
        m = Image.open(self.mask_paths[iid]).convert("L").resize((self.size, self.size), Image.NEAREST)
        return self.tf(img), torch.from_numpy(np.asarray(m) > 127), i


def find_files(args):
    """Builds the dictionaries {image_id: path} for the images and for the lesion masks."""
    img_dir, mask_dir = args.img_dir, args.mask_dir
    if img_dir is None or mask_dir is None:
        import kagglehub                                   # returns the cache if it was already downloaded
        img_dir = img_dir or kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        mask_dir = mask_dir or kagglehub.dataset_download("tschandl/ham10000-lesion-segmentations")
    imgs = {os.path.splitext(os.path.basename(f))[0]: f
            for f in glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)}
    masks = {os.path.basename(f)[:-len("_segmentation.png")]: f
             for f in glob(os.path.join(mask_dir, "**", "*_segmentation.png"), recursive=True)}
    if not masks:
        raise SystemExit(f"No '*_segmentation.png' masks found in {mask_dir}")
    return imgs, masks


# ------------------------------------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def compute_cam(A, W, b, size):
    """A (B,C,h,w), W (7,C), b (7). CAM of the predicted class, upsampled and normalised by the maximum."""
    pred = (A.mean(dim=(2, 3)) @ W.T + b).argmax(1)
    cam = torch.relu(torch.einsum("bchw,bc->bhw", A, W[pred]))
    cam = F.interpolate(cam[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    mx = cam.flatten(1).max(1).values.clamp_min(1e-8)
    return cam / mx[:, None, None], pred


def verify_equivalence(backbone, W, b, x, device):
    """Compares the closed formula with Grad-CAM computed with autograd (gradient with respect to A)."""
    with torch.no_grad():
        A = backbone.forward_features(x).float()
    A = A.detach().requires_grad_(True)
    logits = A.mean(dim=(2, 3)) @ W.T + b
    c = logits.argmax(1)
    logits.gather(1, c[:, None]).sum().backward()
    alpha = A.grad.mean(dim=(2, 3))                                    # (B, C)
    gc = torch.relu((alpha[:, :, None, None] * A.detach()).sum(1))
    cf = torch.relu(torch.einsum("bchw,bc->bhw", A.detach(), W[c]))
    gc, cf = gc / gc.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None], \
        cf / cf.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
    err = (gc - cf).abs().max().item()
    print(f"[verify] maximum error Grad-CAM (autograd) vs closed formula: {err:.2e}")
    assert err < 1e-4, "The closed formula does not match Grad-CAM computed with autograd"


# ------------------------------------------------------------------------------------------
# Saved heads
# ------------------------------------------------------------------------------------------
def collect_heads(args, n_train, device):
    """Loads the passive baseline head and the AL heads closest to each labelling budget."""
    heads = []
    base = args.baseline
    if not os.path.exists(base):
        alt = f"results_{BACKBONE}/baseline_head.pt"
        base = alt if os.path.exists(alt) else base
    st = torch.load(base, map_location="cpu")["state_dict"]
    heads.append(dict(kind="passive", strategy="passive (100% of training set)", oracle="perfect",
                      seed=0, n_labelled=n_train, W=st["weight"].float().to(device),
                      b=st["bias"].float().to(device)))
    for path in sorted(glob(os.path.join(args.heads_dir, "*.pt"))):
        m = re.match(r"(.+)__(.+)__seed(\d+)\.pt$", os.path.basename(path))
        if not m:
            continue
        strategy, oracle, seed = m.group(1), m.group(2), int(m.group(3))
        if oracle not in args.oracles or seed >= args.seeds:
            continue
        ck = torch.load(path, map_location="cpu")                       # {n_labelled: state_dict}
        keys, used = sorted(ck), set()
        for frac in args.budgets:
            k = min(keys, key=lambda x: abs(x - frac * n_train))
            if k in used:
                continue
            used.add(k)
            heads.append(dict(kind="AL", strategy=strategy, oracle=oracle, seed=seed, n_labelled=k,
                              W=ck[k]["weight"].float().to(device), b=ck[k]["bias"].float().to(device)))
    return heads


# ------------------------------------------------------------------------------------------
def outcome(true, pred):
    """Type of result of one prediction (correct, the two mel <-> nv errors, or another error)."""
    t, p = CLASSES[true], CLASSES[pred]
    if t == p:
        return "correct"
    if (t, p) == ("nv", "mel"):
        return "nv -> mel (false melanoma)"
    if (t, p) == ("mel", "nv"):
        return "mel -> nv (missed melanoma)"
    return "other error"


def main():
    """Computes the CAMs and the localisation metrics of every head, then saves the tables and figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir"), ap.add_argument("--mask-dir")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--heads-dir", default="results/heads")
    ap.add_argument("--baseline", default=f"results/results_baseline/results_{BACKBONE}/baseline_head.pt")
    ap.add_argument("--oracles", nargs="+", default=["perfect", "majority"])
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (saved heads) to evaluate")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.30, 0.50],
                    help="fraction of the training set labelled (uses the closest saved cycle)")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model and preprocessing: exactly the same as in extract_features.py
    backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0).eval().to(device)
    cfg = resolve_data_config({}, model=backbone)
    size = cfg["input_size"][1]
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(mean=cfg["mean"], std=cfg["std"])])

    imgs, masks = find_files(args)
    df = pd.read_csv(os.path.join(args.splits, "test.csv"))
    n_train = len(pd.read_csv(os.path.join(args.splits, "train.csv")))
    have = df.image_id.isin(masks) & df.image_id.isin(imgs)
    if not have.all():
        print(f"WARNING: {int((~have).sum())} test images without image/mask; skipped.")
    df = df[have]
    if args.max_images:
        df = df.sample(min(args.max_images, len(df)), random_state=0)
    df = df.reset_index(drop=True)
    y_true = df.dx.map(CLASS_TO_IDX).values
    ds = TestSet(df, imgs, masks, tf, size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    heads = collect_heads(args, n_train, device)
    print(f"{len(df)} test images | {len(heads)} heads | input {size}x{size}")

    if args.verify:
        x, _, _ = next(iter(loader))
        verify_equivalence(backbone, heads[0]["W"], heads[0]["b"], x.to(device), device)

    # ---- metrics for all the heads (the backbone is run only once per batch) ----
    acc = [{k: [] for k in ["idx", "pred"] + METRICS + ["mask_area", "cam_empty"]} for _ in heads]
    for bi, (x, m, idx) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            A = backbone.forward_features(x)
        A = A.float()
        for h, store in zip(heads, acc):
            cam, pred = compute_cam(A, h["W"], h["b"], size)
            met = localisation_metrics(cam.cpu().numpy(), m.numpy())
            store["idx"].append(idx.numpy()); store["pred"].append(pred.cpu().numpy())
            for k in METRICS + ["mask_area", "cam_empty"]:
                store[k].append(met[k])
        print(f"  batch {bi + 1}/{len(loader)}", end="\r")
    print()

    frames = []
    for h, store in zip(heads, acc):
        f = pd.DataFrame({k: np.concatenate(v) for k, v in store.items()})
        f["image_id"] = df.image_id.values[f.idx]
        f["true"] = y_true[f.idx]
        f["outcome"] = [outcome(t, p) for t, p in zip(f.true, f.pred)]
        for k in ("kind", "strategy", "oracle", "seed", "n_labelled"):
            f[k] = h[k]
        frames.append(f.drop(columns="idx"))
    per_img = pd.concat(frames, ignore_index=True)
    per_img.to_csv("results/tables/gradcam_per_image.csv", index=False)

    # ---- 1) interpretability vs number of labels (mean per head, then over seeds) ----
    keys = ["kind", "strategy", "oracle", "n_labelled"]
    per_head = per_img.groupby(keys + ["seed"])[METRICS + ["cam_empty"]].mean().reset_index()
    summ = per_head.groupby(keys)[METRICS + ["cam_empty"]].agg(["mean", "std"]).round(4)
    summ.to_csv("results/tables/gradcam_summary.csv")
    print("\nGrad-CAM localisation (mean over images and then over seeds):")
    print(summ[[(m, "mean") for m in METRICS]].to_string())

    # ---- 2) by class and by type of result (includes mel <-> nv), for the passive model and for the final AL model (hybrid, perfect, seed 0, last cycle), which is the one used by the referral and by the app ----
    base = per_img[per_img.kind == "passive"]
    al_all = per_img[per_img.kind == "AL"]
    final_al = al_all[(al_all.strategy == "hybrid") & (al_all.oracle == "perfect") & (al_all.seed == 0) & (al_all.n_labelled == al_all.n_labelled.max())]
    rows = []
    for model_name, sub in (("passive (100%)", base), ("active learning (hybrid, 50%)", final_al)):
        for gtype, col in (("class (true)", sub.true.map(lambda i: CLASSES[i])), ("outcome", sub.outcome)):
            t = sub.groupby(col)[METRICS].mean().round(4)
            t["n"] = sub.groupby(col).size()
            t.insert(0, "group_type", gtype)
            t.insert(0, "model", model_name)
            rows.append(t)
    byg = pd.concat(rows)
    byg.to_csv("results/tables/gradcam_by_group.csv")
    print("\nBy group (passive model and final AL model):\n", byg.to_string())

    # ---- 3) figure: metrics vs labels ----
    al = per_head[per_head.kind == "AL"].groupby(["strategy", "oracle", "n_labelled"])[METRICS].agg(
        ["mean", "std"])
    ref = per_head[per_head.kind == "passive"][METRICS].mean()
    panels = [("energy_ratio", "Energy inside the lesion / lesion area\n(1 = chance)"),
              ("pointing_game", "Pointing game (peak inside the lesion)"),
              ("border_ratio", "Energy on the border / border area\n(1 = chance, lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (met, title) in zip(axes, panels):
        for (strategy, oracle), g in al.groupby(level=[0, 1]):
            ls = "--" if strategy == "random" else "-"
            ax.errorbar(g.index.get_level_values(2), g[(met, "mean")], yerr=g[(met, "std")].fillna(0),
                        marker="o", ms=3, capsize=2, ls=ls, label=f"{strategy} ({oracle})")
        ax.axhline(ref[met], color="k", ls=":", label="passive (100% of training set)")
        if met != "pointing_game":
            ax.axhline(1.0, color="gray", lw=0.6)
        ax.set_xlabel("Labelled images")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/figures/gradcam_vs_labels.png", dpi=200)

    # ---- 4) qualitative examples ----
    bidx = base.set_index("image_id")
    cases = [("correct melanoma", (base.true == 4) & (base.pred == 4)),
        ("correct nevus", (base.true == 5) & (base.pred == 5)),
        ("nevus -> melanoma", base.outcome.str.startswith("nv ->")),
        ("melanoma -> nevus", base.outcome.str.startswith("mel ->"))]
    chosen = [(n, base[c].sample(1, random_state=0).image_id.iloc[0]) for n, c in cases if c.any()]
    al_heads = sorted([h for h in heads if h["kind"] == "AL" and h["strategy"] != "random"
                       and h["seed"] == 0 and h["oracle"] == args.oracles[0]],
                      key=lambda h: h["n_labelled"])
    show = [heads[0]] + ([al_heads[0], al_heads[-1]] if len(al_heads) > 1 else al_heads)
    mean, std = np.array(cfg["mean"])[:, None, None], np.array(cfg["std"])[:, None, None]
    fig, axes = plt.subplots(len(chosen), 1 + len(show), figsize=(3.1 * (1 + len(show)), 3.1 * len(chosen)),
                             squeeze=False)
    for r, (name, iid) in enumerate(chosen):
        x, m, _ = ds[int(df.index[df.image_id == iid][0])]
        img = np.clip(x.numpy() * std + mean, 0, 1).transpose(1, 2, 0)
        mk = m.numpy()
        with torch.no_grad():
            A = backbone.forward_features(x[None].to(device)).float()
        axes[r, 0].imshow(img); axes[r, 0].contour(mk, levels=[0.5], colors="w", linewidths=1)
        axes[r, 0].set_title(f"{name}\n(true: {CLASSES[int(bidx.loc[iid, 'true'])]})", fontsize=8)
        for c, h in enumerate(show, start=1):
            cam, pred = compute_cam(A, h["W"], h["b"], size)
            axes[r, c].imshow(img); axes[r, c].imshow(cam[0].cpu().numpy(), cmap="jet", alpha=0.45)
            axes[r, c].contour(mk, levels=[0.5], colors="w", linewidths=1)
            tag = "passive" if h["kind"] == "passive" else f"{h['strategy']} @ {h['n_labelled']}"
            axes[r, c].set_title(f"{tag}\npredicted: {CLASSES[int(pred[0])]}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/figures/gradcam_examples.png", dpi=170)
    print("\nSaved: results/tables/gradcam_*.csv, results/figures/gradcam_*.png")


if __name__ == "__main__":
    main()
