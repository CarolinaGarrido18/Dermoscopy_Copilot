import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SEED = 0
INIT_FRAC = 0.10    # 10% do treino começa rotulado
BATCH = 300         # imagens pedidas em cada ciclo
MAX_FRAC = 0.50     # paramos quando 50% do treino estiver rotulado


def carregar(split):
    d = np.load(f"features/{split}_features.npz", allow_pickle=True)
    return d["X"].astype("float32"), d["y"], d["image_ids"], d["aug_idx"]


def separar_inicial(y_orig, seed):
    """Devolve uma máscara: True = imagem já rotulada (10%), False = por rotular."""
    idx = np.arange(len(y_orig))
    rotulado, _ = train_test_split(idx, train_size=INIT_FRAC,
                                   stratify=y_orig, random_state=seed)
    mascara = np.zeros(len(y_orig), dtype=bool)
    mascara[rotulado] = True
    return mascara


class CabecaLinear:
    """Camada linear 1280 -> 7 sobre as features. Interface igual à do sklearn."""
    def __init__(self, seed=0, n_features=1280, n_classes=7):
        torch.manual_seed(seed)
        self.net = nn.Linear(n_features, n_classes)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)

    def fit(self, X, y, epochs=3, X_val=None, y_val=None, patience=10):
        n = np.bincount(y, minlength=7).clip(min=1)
        w = torch.tensor(len(y) / (7 * n), dtype=torch.float32)   # pesos por classe
        loss_fn = nn.CrossEntropyLoss(weight=w)
        X_t, y_t = torch.tensor(X), torch.tensor(y, dtype=torch.long)
        melhor, estado, espera = -1, None, 0
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                self.opt.zero_grad()
                loss_fn(self.net(X_t[b]), y_t[b]).backward()
                self.opt.step()
            if X_val is not None:                       # early stopping (só no treino inicial)
                f1 = f1_score(y_val, self.predict_proba(X_val).argmax(1), average="macro")
                if f1 > melhor:
                    melhor, estado, espera = f1, copy.deepcopy(self.net.state_dict()), 0
                else:
                    espera += 1
                    if espera >= patience:
                        break
        if estado is not None:
            self.net.load_state_dict(estado)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X)), 1).numpy()


def least_confidence(model, X):
    p = model.predict_proba(X)
    scores = 1 - p.max(axis=1)
    return np.argsort(-scores), scores            # mais incertas primeiro


def margin_sampling(model, X):
    p = np.sort(model.predict_proba(X), axis=1)
    scores = 1 - (p[:, -1] - p[:, -2])            # margem pequena = incerteza grande
    return np.argsort(-scores), scores


def entropy_sampling(model, X):
    p = model.predict_proba(X)
    scores = -(p * np.log(p + 1e-12)).sum(axis=1)
    return np.argsort(-scores), scores


def random_sampling(model, X):
    """Baseline do plano (passo 22): 300 ao acaso."""
    return np.random.permutation(len(X)), None


def hybrid_sampling(model, X, n=BATCH, frac=0.10):
    """Das 10% mais incertas (entropia), escolhe n diferentes entre si com k-means."""
    ordem, _ = entropy_sampling(model, X)
    k = max(n, int(frac * len(X)))
    cand = ordem[:k]
    km = KMeans(n_clusters=n, n_init=1, random_state=0).fit(X[cand])
    dist = km.transform(X[cand])                      # distância de cada candidata a cada centro
    escolhidos = list(dict.fromkeys(cand[dist.argmin(axis=0)]))   # a mais próxima de cada centro
    if len(escolhidos) < n:                           # se dois centros escolheram a mesma imagem
        usados = set(escolhidos)
        resto = [i for i in cand if i not in usados]
        escolhidos += resto[:n - len(escolhidos)]
    return np.array(escolhidos), None


class ActiveLearner:
    """Mesma estrutura do tutorial do professor: query, teach, score."""
    def __init__(self, estimator, query_strategy):
        self.estimator = estimator
        self.query_strategy = query_strategy

    def query(self, X_pool, n=BATCH):
        order, _ = self.query_strategy(self.estimator, X_pool)
        return order[:n]                              # as n mais informativas

    def teach(self, X, y, epochs=3):
        self.estimator.fit(X, y, epochs=epochs)       # continua o treino (passo 24)

    def score(self, X, y):
        return f1_score(y, self.estimator.predict_proba(X).argmax(1), average="macro")


ESTRATEGIAS = {
    "random": random_sampling,
    "least_confidence": least_confidence,
    "margin": margin_sampling,
    "entropy": entropy_sampling,
    "hybrid": hybrid_sampling,
}


def carregar_dados():
    Xtr, ytr, idtr, augtr = carregar("train")
    Xva, yva, _, _ = carregar("val")
    orig = np.where(augtr == 0)[0]
    return dict(Xtr=Xtr, ytr=ytr, idtr=idtr, orig=orig,
                ids_orig=idtr[orig], y_orig=ytr[orig], Xva=Xva, yva=yva)


def correr(nome, seed, d):
    """Um experimento completo: uma estratégia, uma seed (passos 19 a 26)."""
    np.random.seed(seed)                              # controla o random_sampling
    rotulada = separar_inicial(d["y_orig"], seed)     # passo 19

    def linhas_rotuladas():                           # as 3 versões de cada imagem rotulada
        return np.isin(d["idtr"], d["ids_orig"][rotulada])

    l = linhas_rotuladas()                            # passo 20
    modelo = CabecaLinear(seed=seed).fit(d["Xtr"][l], d["ytr"][l], epochs=100,
                                         X_val=d["Xva"], y_val=d["yva"])
    learner = ActiveLearner(modelo, ESTRATEGIAS[nome])
    hist = [dict(estrategia=nome, seed=seed, ciclo=0, n_rotuladas=int(rotulada.sum()),
                 f1_val=learner.score(d["Xva"], d["yva"]), classes_pedidas=None)]

    ciclo = 0
    while rotulada.sum() < MAX_FRAC * len(d["y_orig"]):     # passo 26
        ciclo += 1
        pool = np.where(~rotulada)[0]                       # passo 21
        escolhidas = pool[learner.query(d["Xtr"][d["orig"][pool]])]   # passo 22
        rotulada[escolhidas] = True                         # passo 23
        l = linhas_rotuladas()
        learner.teach(d["Xtr"][l], d["ytr"][l], epochs=3)   # passo 24
        hist.append(dict(estrategia=nome, seed=seed, ciclo=ciclo,
                         n_rotuladas=int(rotulada.sum()),
                         f1_val=learner.score(d["Xva"], d["yva"]),   # passo 25
                         classes_pedidas=np.bincount(d["y_orig"][escolhidas],
                                                     minlength=7).tolist()))
    return hist


if __name__ == "__main__":
    import os
    os.makedirs("results/tables", exist_ok=True)
    d = carregar_dados()

    linhas = []
    for nome in ESTRATEGIAS:
        for seed in [0, 1, 2]:                        # passo 27
            print("a correr", nome, "| seed", seed, flush=True)
            linhas += correr(nome, seed, d)

    df = pd.DataFrame(linhas)
    df.to_csv("results/tables/al_curvas.csv", index=False)

    ultimo = df[df.ciclo == df.ciclo.max()]
    print("\nF1 val no último ciclo (média e desvio-padrão de 3 seeds):")
    print(ultimo.groupby("estrategia").f1_val.agg(["mean", "std"]).round(4))
    print("\nF1 val médio ao longo de todos os ciclos:")
    print(df.groupby("estrategia").f1_val.mean().round(4))