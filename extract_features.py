#path_img = "C:\\Users\\carol\\.cache\\kagglehub\\datasets\\kmader\\skin-cancer-mnist-ham10000\\versions\\2"
#path_seg = "C:\\Users\\carol\\.cache\\kagglehub\\datasets\\tschandl\\ham10000-lesion-segmentations\\versions\\1"
"""
Extração de features com EfficientNet-B0 (pré-treinada no ImageNet).

Passa cada imagem uma só vez pela CNN e guarda os vetores (1280 valores por imagem)
em disco (mais cópias aumentadas das imagens de treino). Depois, o ciclo de Active Learning só treina a camada de classificação
sobre estes vetores, o que é muito rápido.

Pré-requisito: já ter corrido o preparar_dados.py e ter a pasta "splits" com
train.csv, val.csv e test.csv (colunas image_id, lesion_id, dx).
"""

import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# CONFIGURAÇÃO

# Mudar este caminho para o mesmo path_img da preparação de dados
PATH_IMG = "C:\\Users\\carol\\.cache\\kagglehub\\datasets\\kmader\\skin-cancer-mnist-ham10000\\versions\\2"

BASE = Path(__file__).parent   # Pasta onde este script está guardado
SPLITS_DIR = BASE / "splits"   # Caminho para a pasta com os CSVs (treino, teste, validação)
OUT_DIR = BASE / "features"    # Caminho para a pasta onde os resultados vão ficar
OUT_DIR.mkdir(exist_ok=True)   # Cria a pasta "features" se ela ainda não existir

BATCH_SIZE = 32                # Processa 32 imagens de cada vez para não esgotar a RAM.
MODEL_NAME = "efficientnet_b0"

# Nº de cópias aumentadas (data augmentation) por imagem de TREINO.
# 0 = sem augmentation. Com 2, cada imagem de treino fica com 3 versões:
# original + 2 variações (rotação, espelhamento, brilho). Val e teste nunca são aumentados.
N_AUG = 2
SEED = 42


MAX_IMAGES_POR_SPLIT = None

# Ordem fixa das classes
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


def main():
    # 1) Ligar image_id -> ficheiro .jpg (as imagens estão em 2 pastas)
    img_paths = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob(os.path.join(PATH_IMG, "*", "*.jpg"))
    }
    print("Imagens encontradas:", len(img_paths))

    # 2) Transformações
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    # Versão original: só resize + normalização
    base_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm,
    ])

    # Versão aumentada: espelhar, rodar e mudar o brilho (aleatório em cada cópia)
    aug_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        norm,
    ])

    torch.manual_seed(SEED)  # reprodutibilidade das variações aleatórias

    # 3) Modelo pré-treinado, sem a camada final (num_classes=0 -> devolve as features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval().to(device)

    # 4) Extrair features para cada split
    for split in ["train", "val", "test"]:
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        if MAX_IMAGES_POR_SPLIT:
            df = df.head(MAX_IMAGES_POR_SPLIT)

        # Só o treino tem cópias aumentadas; aug_idx 0 = imagem original
        variantes = [(0, base_tf)]
        if split == "train":
            variantes += [(k, aug_tf) for k in range(1, N_AUG + 1)]

        all_X, all_y, all_ids, all_aug = [], [], [], []
        for aug_idx, tf in variantes:
            # num_workers=0 evita problemas no Windows
            loader = DataLoader(HAMDataset(df, img_paths, tf),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            with torch.no_grad():
                for b, (x, y) in enumerate(loader):
                    f = model(x.to(device))
                    all_X.append(f.cpu().numpy())
                    all_y.append(y.numpy())
                    print(f"  [{split}] aug {aug_idx} batch {b + 1}/{len(loader)}", end="\r")
            all_ids.append(df["image_id"].values)
            all_aug.append(np.full(len(df), aug_idx))

        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        np.savez_compressed(
            OUT_DIR / f"{split}_features.npz",
            X=X, y=y,
            image_ids=np.concatenate(all_ids),
            aug_idx=np.concatenate(all_aug),
        )
        print(f"\n[{split}] X: {X.shape} | y: {y.shape} -> guardado")

    print("\nConcluído. Ficheiros em:", OUT_DIR)


if __name__ == "__main__":
    main()