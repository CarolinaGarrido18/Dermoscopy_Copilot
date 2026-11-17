"""
Baseline PASSIVO com backbone configurável + Test-Time Augmentation (TTA).

Lê as features geradas pelo extract_features.py:
  - train: X (N, D)       (original + cópias aumentadas, aug_idx 0..N_AUG)
  - val/test: X (N, V, D) (V vistas TTA da mesma imagem)

Treina uma camada linear (D -> 7) com class weights. Em validação e teste a previsão é a
MÉDIA DAS PROBABILIDADES das V vistas. O temperature scaling é ajustado na validação já
sobre essa probabilidade média. Mostra também o resultado SEM TTA (só a vista original),
para quantificar o ganho do TTA.

Pré-requisito: features_<BACKBONE>/ (extract_features.py).
Instalação: pip install scikit-learn matplotlib
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

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"   # tem de ser igual ao BACKBONE do extract_features.py

BASE = Path(__file__).parent
FEATS = BASE / f"features_{BACKBONE}"
OUT = BASE / f"results_{BACKBONE}"
OUT.mkdir(exist_ok=True)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = CLASSES.index("mel")
SEED = 42
EPOCHS = 100
PATIENCE = 10
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------------
def carregar(split):
    d = np.load(FEATS / f"{split}_features.npz", allow_pickle=True)
    return (torch.tensor(d["X"], dtype=torch.float32),
            torch.tensor(d["y"], dtype=torch.long),
            d["aug_idx"])


def logits_vistas(head, X):
    """X: (N, V, D) -> logits (N, V, C)."""
    N, V, D = X.shape
    return head(X.reshape(N * V, D)).reshape(N, V, -1)


def probs_tta(logits, T=1.0):
    """Média das probabilidades das V vistas. logits: (N, V, C) -> (N, C)."""
    return F.softmax(logits / T, dim=2).mean(dim=1)


def ajustar_temperatura(logits, y):
    """Temperature scaling sobre a probabilidade média das vistas (NLL na validação)."""
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        p = probs_tta(logits, log_T.exp())
        loss = F.nll_loss(torch.log(p + 1e-12), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp())


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1)
    acerto = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - acerto[m].mean()).abs()
    return float(total)


def avaliar(y_true, probs):
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
    """Ajusta T na validação e avalia no teste (com as vistas que forem passadas)."""
    T = ajustar_temperatura(logits_va, y_va)
    probs = probs_tta(logits_te, T)
    res = avaliar(y_te.numpy(), probs.numpy())
    res["temperatura"] = T
    res["ece_antes"] = ece(probs_tta(logits_te, 1.0), y_te)
    res["ece_depois"] = ece(probs, y_te)
    return res, probs


# ----------------------------------------------------------------------------
# 1) DADOS E PESOS DAS CLASSES
# ----------------------------------------------------------------------------
X_tr, y_tr, aug_tr = carregar("train")
X_va, y_va, _ = carregar("val")
X_te, y_te, _ = carregar("test")
D = X_tr.shape[1]
print(f"Backbone: {BACKBONE} | features: {D}")
print("Treino:", tuple(X_tr.shape), "| originais:", int((aug_tr == 0).sum()),
      "| val:", tuple(X_va.shape), "| teste:", tuple(X_te.shape))

pesos = compute_class_weight("balanced", classes=np.arange(7), y=y_tr[aug_tr == 0].numpy())
print("Pesos das classes:", {c: round(float(p), 2) for c, p in zip(CLASSES, pesos)})
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32))

# ----------------------------------------------------------------------------
# 2) TREINO DA CAMADA DE CLASSIFICAÇÃO (D -> 7)
# ----------------------------------------------------------------------------
head = nn.Linear(D, 7)
opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

melhor_f1, melhor_estado, sem_melhoria = -1.0, None, 0
for ep in range(1, EPOCHS + 1):
    head.train()
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(head(X_tr[idx]), y_tr[idx])
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        p_va = probs_tta(logits_vistas(head, X_va))          # validação já com TTA
    f1_val = f1_score(y_va, p_va.argmax(1), average="macro", zero_division=0)
    print(f"Época {ep:3d} | loss {loss.item():.4f} | F1-macro val (TTA) {f1_val:.4f}")

    if f1_val > melhor_f1:
        melhor_f1, sem_melhoria = f1_val, 0
        melhor_estado = {k: v.clone() for k, v in head.state_dict().items()}
    else:
        sem_melhoria += 1
        if sem_melhoria >= PATIENCE:
            print("Early stopping.")
            break

head.load_state_dict(melhor_estado)
head.eval()

# ----------------------------------------------------------------------------
# 3) TEMPERATURE SCALING + AVALIAÇÃO NO TESTE (com e sem TTA)
# ----------------------------------------------------------------------------
with torch.no_grad():
    logits_va, logits_te = logits_vistas(head, X_va), logits_vistas(head, X_te)

res_tta, probs_te = avaliar_config(logits_va, logits_te, y_va, y_te)
# Sem TTA = só a vista 0 (imagem original)
res_sem, _ = avaliar_config(logits_va[:, :1], logits_te[:, :1], y_va, y_te)

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

# Referência: baseline anterior (EfficientNet-B0 sem TTA), se existir
ref = BASE / "results" / "baseline_metrics.json"
if ref.exists():
    b0 = json.load(open(ref, encoding="utf-8"))
    print("\nReferência EfficientNet-B0 (v1, sem TTA):",
          {k: round(b0[k], 4) for k in chaves})

# Matriz de confusão (com TTA)
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

# Guardar (o topo do JSON é o resultado COM TTA, no mesmo formato do baseline anterior)
torch.save({"state_dict": head.state_dict(), "T": res_tta["temperatura"], "classes": CLASSES,
            "backbone": BACKBONE, "in_features": D, "n_views": int(X_te.shape[1])},
           OUT / "baseline_head.pt")
with open(OUT / "baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump({**res_tta, "sem_tta": res_sem, "backbone": BACKBONE,
               "n_views": int(X_te.shape[1])}, f, indent=2)
print("\nGuardado em:", OUT)