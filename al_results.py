import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from active_learning import carregar_dados, CabecaLinear

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
df = pd.read_csv("results/tables/al_curvas.csv")
d = carregar_dados()

# Referência: modelo treinado com 100% do treino, avaliado na mesma validação
ref = CabecaLinear(seed=0).fit(d["Xtr"], d["ytr"], epochs=100,
                               X_val=d["Xva"], y_val=d["yva"])
f1_ref = f1_score(d["yva"], ref.predict_proba(d["Xva"]).argmax(1), average="macro")
print("F1 val com 100% do treino:", round(f1_ref, 4))

# Passo 28: gráfico das curvas (média e desvio-padrão de 3 seeds)
fig, ax = plt.subplots(figsize=(7, 4.5))
for nome, g in df.groupby("estrategia"):
    m = g.groupby("n_rotuladas").f1_val.agg(["mean", "std"])
    cor = "gray" if nome == "random" else None
    linha, = ax.plot(m.index, m["mean"], marker="o", ms=3, label=nome, color=cor)
    ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"],
                    alpha=0.12, color=linha.get_color())
ax.axhline(f1_ref, ls="--", color="k", label="100% of training set")
ax.set_xlabel("Labelled images")
ax.set_ylabel("Macro F1 (validation)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/al_curvas.png", dpi=200)

# Passo 30: que classes cada estratégia pediu (em % do total pedido)
dc = df.dropna(subset=["classes_pedidas"]).copy()
mat = np.array(dc.classes_pedidas.apply(ast.literal_eval).tolist())
for i, c in enumerate(CLASSES):
    dc[c] = mat[:, i]
tab = dc.groupby("estrategia")[CLASSES].sum()
perc = (tab.div(tab.sum(axis=1), axis=0) * 100).round(1)
perc.loc["treino (todas as imagens)"] = (
    np.bincount(d["y_orig"], minlength=7) / len(d["y_orig"]) * 100).round(1)
perc.to_csv("results/tables/al_classes_pedidas.csv")
print("\nClasses pedidas (% do total):")
print(perc)