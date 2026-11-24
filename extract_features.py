"""
Dermoscopy Copilot - Feature extraction with a pre-trained backbone and Test-Time Augmentation

- Training: original image (aug_idx = 0) + N_AUG augmented copies -> X shape (N * (1 + N_AUG), D)
- Validation/Test: original image with TTA, 8 views (4 rotations x horizontal flip) -> X shape (N, 8, D)

(N = number of images in the split, D = feature dimension of the backbone)

Prerequisite: the "splits" folder produced by prepare_data.py
"""

import os
from glob import glob
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Location of the HAM10000 images: kagglehub returns the local cache folder (it only downloads the
# dataset if it is not there yet)
PATH_IMG = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

# Pre-trained network from timm; it also names the output folder, so the same value must be
# used in the baseline script to load the correct features
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"

BASE = Path(__file__).parent
SPLITS_DIR = BASE / "splits"
OUT_DIR = BASE / "features" / f"features_{BACKBONE}"    # must match BACKBONE in the train_baseline script
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16        
N_AUG = 2   # augmented copies generated per training image      
SEED = 42   # fixed seed for reproducibility
MAX_IMAGES_POR_SPLIT = None   # limit images per split for quick tests (None = use all)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Dataset and future extraction functions

class HAMDataset(Dataset):
    # Loads each dermoscopic image in RGB, applies the transform, returns (image, class index)
    def __init__(self, df, img_paths, transform):
        self.df = df.reset_index(drop=True)
        self.img_paths = img_paths  # {image_id: path of the .jpg file}
        self.transform = transform

    # Returns the number of images in the split
    def __len__(self):
        return len(self.df)

    # Loads the image at position i as RGB, applies the transform and returns it with its class index
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_paths[row["image_id"]]).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[row["dx"]]


def vistas_tta(x):
    # Generate the 8 TTA views of a batch: rotations of 0/90/180/270 degrees, each with and
    # without horizontal flip
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        out += [r, torch.flip(r, dims=(3,))]
    return out


@torch.no_grad()
def extrair(model, loader, device, tta):
    # Run the backbone over a data loader; return the feature matrix and the labels.
    # tta=True -> all 8 views are processed, X has shape (N, 8, D); tta=False -> X has shape (N, D)
    feats, ys = [], []
    for b, (x, y) in enumerate(loader):
        x = x.to(device)
        views = vistas_tta(x) if tta else [x]
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            # Stack the V views along dim=1 -> shape (B, V, D)
            f = torch.stack([model(v).float() for v in views], dim=1)  
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
        print(f"  batch {b + 1}/{len(loader)}", end="\r")
    X = np.concatenate(feats)
    # Squeeze the view dimension (V=1) when TTA is disabled -> (N, D)
    return (X if tta else X[:, 0]), np.concatenate(ys)


def main():
    # image_id -> path of the .jpg file
    img_paths = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob(os.path.join(PATH_IMG, "*", "*.jpg"))
    }
    print("Images found:", len(img_paths))

    # Model and preprocessing
    # The backbone is created without its final classification layer (num_classes=0),
    # so it outputs the feature vector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| Backbone:", BACKBONE)
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.eval().to(device)

    # Input size, mean and std come from the backbone's own pre-training configuration
    cfg = resolve_data_config({}, model=model)
    SIZE = cfg["input_size"][1]
    print(f"Input size: {SIZE} | features: {model.num_features}")

    norm = transforms.Normalize(mean=cfg["mean"], std=cfg["std"])

    # base_tf: resize + normalization (original training images and validation/test sets)
    base_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        norm,
    ])
    # aug_tf: stochastic augmentation for the N_AUG training copies. Flips and full rotations
    # (degrees=180) exploit the rotational invariance of skin lesions
    aug_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        norm,
    ])

    torch.manual_seed(SEED)

    # Feature extraction, split by split
    for split in ["train", "val", "test"]:
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        if MAX_IMAGES_POR_SPLIT:
            df = df.head(MAX_IMAGES_POR_SPLIT)

        if split == "train":
            # Original (aug_idx = 0) + N_AUG augmented copies, without TTA -> ((1 + N_AUG) * N, D)
            variantes = [(0, base_tf)] + [(k, aug_tf) for k in range(1, N_AUG + 1)]
            all_X, all_y, all_ids, all_aug = [], [], [], []
            for aug_idx, tf in variantes:
                loader = DataLoader(HAMDataset(df, img_paths, tf),
                                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                print(f"[{split}] aug {aug_idx}")
                X_a, y_a = extrair(model, loader, device, tta=False)
                all_X.append(X_a); all_y.append(y_a)
                all_ids.append(df["image_id"].values)
                all_aug.append(np.full(len(df), aug_idx)) # aug_idx records the variant of each row
            X, y = np.concatenate(all_X), np.concatenate(all_y)
            ids, aug = np.concatenate(all_ids), np.concatenate(all_aug)
        else:
            # Val/test: original image only, with the 8 TTA views -> (N, 8, D)
            # shuffle=False preserves row order, so features align with df["image_id"]
            loader = DataLoader(HAMDataset(df, img_paths, base_tf),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            print(f"[{split}] TTA x8")
            X, y = extrair(model, loader, device, tta=True)
            ids, aug = df["image_id"].values, np.zeros(len(df), dtype=int)
        
        # Save as compressed .npz with X (features), y (labels), image_ids and aug_idx
        np.savez_compressed(OUT_DIR / f"{split}_features.npz",
                            X=X, y=y, image_ids=ids, aug_idx=aug)
        print(f"\n[{split}] X: {X.shape} | y: {y.shape} -> saved")

    print("\nDone. Files in:", OUT_DIR)


if __name__ == "__main__":
    main()