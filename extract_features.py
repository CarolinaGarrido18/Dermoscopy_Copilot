"""
Extração de features com backbone pré-treinado (timm) + Test-Time Augmentation.

- Treino: imagem original (aug_idx 0) + N_AUG cópias aumentadas -> X shape (N, D)
- Val/Teste: TTA com 8 vistas (4 rotações x espelhamento)       -> X shape (N, 8, D)

Pré-requisito: pasta "splits" com train.csv, val.csv e test.csv (image_id, lesion_id, dx).
"""

import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# CONFIGURAÇÃO

PATH_IMG = "C:\\Users\\carol\\.cache\\kagglehub\\datasets\\kmader\\skin-cancer-mnist-ham10000\\versions\\2"

# "efficientnet_b3" (1536 feats) | "convnext_tiny" (768) | "convnext_small" (768) | "convnext_base" (1024)
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"

BASE = Path(__file__).parent
SPLITS_DIR = BASE / "splits"
OUT_DIR = BASE / f"features_{BACKBONE}"     # tem de coincidir com BACKBONE do baseline_passivo.py
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16        # as 8 vistas multiplicam a memória; baixa para 8 se faltar memória GPU
N_AUG = 2              # cópias aumentadas por imagem de TREINO
SEED = 42
MAX_IMAGES_POR_SPLIT = None

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# DATASET

class HAMDataset(Dataset):
    def __init__(self, df, img_paths, transform):
        self.df = df.reset_index(drop=True)
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_paths[row["image_id"]]).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[row["dx"]]


def vistas_tta(x):
    """8 vistas: rotações 0/90/180/270, cada uma com e sem espelhamento."""
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        out += [r, torch.flip(r, dims=(3,))]
    return out


@torch.no_grad()
def extrair(model, loader, device, tta):
    feats, ys = [], []
    for b, (x, y) in enumerate(loader):
        x = x.to(device)
        views = vistas_tta(x) if tta else [x]
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            f = torch.stack([model(v).float() for v in views], dim=1)  # (B, V, D)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
        print(f"  batch {b + 1}/{len(loader)}", end="\r")
    X = np.concatenate(feats)
    return (X if tta else X[:, 0]), np.concatenate(ys)


def main():
    # 1) image_id -> ficheiro .jpg
    img_paths = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob(os.path.join(PATH_IMG, "*", "*.jpg"))
    }
    print("Imagens encontradas:", len(img_paths))

    # 2) Modelo (sem camada final) e configuração de entrada do próprio backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device, "| Backbone:", BACKBONE)
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.eval().to(device)

    cfg = resolve_data_config({}, model=model)
    SIZE = cfg["input_size"][1]
    print(f"Tamanho de entrada: {SIZE} | features: {model.num_features}")

    norm = transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    base_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        norm,
    ])
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

    # 3) Extração
    for split in ["train", "val", "test"]:
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        if MAX_IMAGES_POR_SPLIT:
            df = df.head(MAX_IMAGES_POR_SPLIT)

        if split == "train":
            # original + cópias aumentadas, sem TTA -> (N, D)
            variantes = [(0, base_tf)] + [(k, aug_tf) for k in range(1, N_AUG + 1)]
            all_X, all_y, all_ids, all_aug = [], [], [], []
            for aug_idx, tf in variantes:
                loader = DataLoader(HAMDataset(df, img_paths, tf),
                                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                print(f"[{split}] aug {aug_idx}")
                X_a, y_a = extrair(model, loader, device, tta=False)
                all_X.append(X_a); all_y.append(y_a)
                all_ids.append(df["image_id"].values)
                all_aug.append(np.full(len(df), aug_idx))
            X, y = np.concatenate(all_X), np.concatenate(all_y)
            ids, aug = np.concatenate(all_ids), np.concatenate(all_aug)
        else:
            # val/teste: só a imagem original, com 8 vistas TTA -> (N, 8, D)
            loader = DataLoader(HAMDataset(df, img_paths, base_tf),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            print(f"[{split}] TTA x8")
            X, y = extrair(model, loader, device, tta=True)
            ids, aug = df["image_id"].values, np.zeros(len(df), dtype=int)

        np.savez_compressed(OUT_DIR / f"{split}_features.npz",
                            X=X, y=y, image_ids=ids, aug_idx=aug)
        print(f"\n[{split}] X: {X.shape} | y: {y.shape} -> guardado")

    print("\nConcluído. Ficheiros em:", OUT_DIR)


if __name__ == "__main__":
    main()