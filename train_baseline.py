"""
Baseline PASSIVO: treina a camada de classificação sobre as features da EfficientNet-B0
usando todas as imagens de treino rotuladas.

  1. Transfer learning (features pré-treinadas + camada de classificação nova)
  2. Data augmentation (já nas features) + pesos para as classes raras
  3. Temperature scaling (calibração da confiança)
Métricas: F1-macro, recall do melanoma e matriz de confusão (+ Balanced Accuracy,
Precision, Recall e AUC macro, como na proposta).

Pré-requisito: ter corrido extract_features.py (pasta "features").

"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # só grava a figura em ficheiro, não bloqueia o script
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight


# CONFIGURAÇÃO

BASE = Path(__file__).parent
FEATS = BASE / "features"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # mesma ordem do extract_features
MEL = CLASSES.index("mel")
SEED = 42
EPOCHS = 100
PATIENCE = 10          # early stopping: para se o F1-macro da validação não melhorar
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


# FUNÇÕES AUXILIARES

# Importa as representações numéricas (features) das imagens para os conjuntos de treino, validação e teste
def carregar(split):
    d = np.load(FEATS / f"{split}_features.npz", allow_pickle=True)
    return (torch.tensor(d["X"], dtype=torch.float32),
            torch.tensor(d["y"], dtype=torch.long),
            d["aug_idx"])

# Calcula o Erro Esperado de Calibração (ECE), medindo se a confiança que o modelo tem na previsão 
# corresponde à sua taxa real de acerto
def ece(probs, y, n_bins=15):
    """Expected Calibration Error: diferença média entre confiança e acerto real."""
    conf, pred = probs.max(1)
    acerto = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - acerto[m].mean()).abs()
    return float(total)

# Calcula as várias métricas de desempenho final do modelo, comparando as previsões com os dados reais
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

# Encontra o valor ideal de Temperatura (T) nos dados de validação para suavizar a confiança exagerada do modelo
def ajustar_temperatura(logits, y):
    """Temperature scaling: aprende um único T que divide os logits (na validação)."""
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp())


# 1) DADOS E PESOS DAS CLASSES

X_tr, y_tr, aug_tr = carregar("train")
X_va, y_va, _ = carregar("val")
X_te, y_te, _ = carregar("test")
print("Treino (com cópias aumentadas):", tuple(X_tr.shape),
      "| originais:", int((aug_tr == 0).sum()))

# Pesos "balanced": inversamente proporcionais à frequência (calculados nas originais)
# Obriga o modelo a penalizar com mais força os erros cometidos nas doenças mais raras, 
# evitando que o sistema ignore lesões minoritárias
pesos = compute_class_weight("balanced", classes=np.arange(7), y=y_tr[aug_tr == 0].numpy())
print("Pesos das classes:", {c: round(float(p), 2) for c, p in zip(CLASSES, pesos)})
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32))


# 2) TREINO DA CAMADA DE CLASSIFICAÇÃO (1280 -> 7)

modelo = nn.Linear(X_tr.shape[1], 7)
opt = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

melhor_f1, melhor_estado, sem_melhoria = -1.0, None, 0
for ep in range(1, EPOCHS + 1):
    modelo.train()
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(modelo(X_tr[idx]), y_tr[idx])
        loss.backward()
        opt.step()

    modelo.eval()
    with torch.no_grad():
        f1_val = f1_score(y_va, modelo(X_va).argmax(1), average="macro", zero_division=0)
    print(f"Época {ep:3d} | loss {loss.item():.4f} | F1-macro val {f1_val:.4f}")

    if f1_val > melhor_f1:
        melhor_f1, sem_melhoria = f1_val, 0
        melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
    else:
        sem_melhoria += 1
        if sem_melhoria >= PATIENCE:
            print("Early stopping.")
            break

modelo.load_state_dict(melhor_estado)
modelo.eval()


# 3) TEMPERATURE SCALING (ajustado na validação)

with torch.no_grad():
    logits_va, logits_te = modelo(X_va), modelo(X_te)

T = ajustar_temperatura(logits_va, y_va)
print(f"\nTemperatura T = {T:.3f}")

probs_te_antes = F.softmax(logits_te, dim=1)
probs_te_depois = F.softmax(logits_te / T, dim=1)
ece_antes, ece_depois = ece(probs_te_antes, y_te), ece(probs_te_depois, y_te)
print(f"ECE no teste: antes {ece_antes:.4f} -> depois {ece_depois:.4f}")


# 4) AVALIAÇÃO NO TESTE (só aqui se usa o teste)

res = avaliar(y_te.numpy(), probs_te_depois.numpy())
res["temperatura"] = T
res["ece_antes"], res["ece_depois"] = ece_antes, ece_depois

print("\nRESULTADOS NO TESTE (baseline passivo)")
for k in ["balanced_acc", "precision_macro", "recall_macro", "f1_macro", "auc_macro",
          "recall_melanoma"]:
    print(f"{k:16s}: {res[k]:.4f}")
print("Recall por classe:", {c: round(r, 3) for c, r in res["recall_por_classe"].items()})

# Matriz de confusão (normalizada por linha = recall por classe)
cm = confusion_matrix(y_te, probs_te_depois.argmax(1), labels=range(7))
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
ax.set_title("Matriz de confusão - baseline passivo (teste)")
fig.colorbar(im, label="Fração da classe real")
fig.tight_layout()
fig.savefig(OUT / "baseline_confusion_matrix.png", dpi=150)

# Guardar modelo e resultados
torch.save({"state_dict": modelo.state_dict(), "T": T, "classes": CLASSES},
           OUT / "baseline_head.pt")
with open(OUT / "baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("\nGuardado em:", OUT)