import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

SEED = 0
INIT_FRAC = 0.10    # 10% of the training images start labelled
BATCH = 300         # images requested in each cycle
MAX_FRAC = 0.50     # stop when 50% of the training images are labelled


def load_split(split):
    d = np.load(f"features/{split}_features.npz", allow_pickle=True)
    return d["X"].astype("float32"), d["y"], d["image_ids"], d["aug_idx"]


def initial_split(y_orig, seed):
    """Boolean mask over images: True = already labelled (10%), False = in the pool."""
    idx = np.arange(len(y_orig))
    labelled, _ = train_test_split(idx, train_size=INIT_FRAC,
                                   stratify=y_orig, random_state=seed)
    mask = np.zeros(len(y_orig), dtype=bool)
    mask[labelled] = True
    return mask


class LinearHead:
    """Linear layer 1280 -> 7 on top of frozen EfficientNet-B0 features (sklearn-like API)."""
    def __init__(self, seed=0, n_features=1280, n_classes=7):
        torch.manual_seed(seed)
        self.net = nn.Linear(n_features, n_classes)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)

    def fit(self, X, y, epochs=3, X_val=None, y_val=None, patience=10):
        n = np.bincount(y, minlength=7).clip(min=1)
        w = torch.tensor(len(y) / (7 * n), dtype=torch.float32)   # balanced class weights
        loss_fn = nn.CrossEntropyLoss(weight=w)
        X_t, y_t = torch.tensor(X), torch.tensor(y, dtype=torch.long)
        best, state, wait = -1, None, 0
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                self.opt.zero_grad()
                loss_fn(self.net(X_t[b]), y_t[b]).backward()
                self.opt.step()
            if X_val is not None:                       # early stopping (initial training only)
                f1 = f1_score(y_val, self.predict_proba(X_val).argmax(1), average="macro")
                if f1 > best:
                    best, state, wait = f1, copy.deepcopy(self.net.state_dict()), 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if state is not None:
            self.net.load_state_dict(state)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X)), 1).numpy()


# Query strategies: each returns (order, scores), most informative first (as in the tutorial)
def least_confidence(model, X):
    p = model.predict_proba(X)
    scores = 1 - p.max(axis=1)
    return np.argsort(-scores), scores


def margin_sampling(model, X):
    p = np.sort(model.predict_proba(X), axis=1)
    scores = 1 - (p[:, -1] - p[:, -2])            # small margin = high uncertainty
    return np.argsort(-scores), scores


def entropy_sampling(model, X):
    p = model.predict_proba(X)
    scores = -(p * np.log(p + 1e-12)).sum(axis=1)
    return np.argsort(-scores), scores


def random_sampling(model, X):
    """Passive baseline: random images."""
    return np.random.permutation(len(X)), None


def hybrid_sampling(model, X, n=BATCH, frac=0.10):
    """Top 10% most uncertain (entropy), then k-means to pick n diverse images."""
    order, _ = entropy_sampling(model, X)
    k = max(n, int(frac * len(X)))
    cand = order[:k]
    km = KMeans(n_clusters=n, n_init=1, random_state=0).fit(X[cand])
    dist = km.transform(X[cand])                  # distance of each candidate to each centre
    chosen = list(dict.fromkeys(cand[dist.argmin(axis=0)]))   # closest candidate to each centre
    if len(chosen) < n:                           # two centres picked the same image
        used = set(chosen)
        rest = [i for i in cand if i not in used]
        chosen += rest[:n - len(chosen)]
    return np.array(chosen), None


class ActiveLearner:
    """Same structure as the course tutorial: query, teach, score."""
    def __init__(self, estimator, query_strategy):
        self.estimator = estimator
        self.query_strategy = query_strategy

    def query(self, X_pool, n=BATCH):
        order, _ = self.query_strategy(self.estimator, X_pool)
        return order[:n]

    def teach(self, X, y, epochs=3):
        self.estimator.fit(X, y, epochs=epochs)   # continue training for a few epochs

    def score(self, X, y):
        return f1_score(y, self.estimator.predict_proba(X).argmax(1), average="macro")


STRATEGIES = {
    "random": random_sampling,
    "least_confidence": least_confidence,
    "margin": margin_sampling,
    "entropy": entropy_sampling,
    "hybrid": hybrid_sampling,
}


def load_data():
    Xtr, ytr, idtr, augtr = load_split("train")
    Xva, yva, _, _ = load_split("val")
    Xte, yte, _, _ = load_split("test")
    orig = np.where(augtr == 0)[0]                # rows of the original (non-augmented) images
    ids_orig = idtr[orig]
    # image (position in ids_orig) that each of the feature rows belongs to
    row_to_img = pd.Series(np.arange(len(ids_orig)), index=ids_orig).reindex(idtr).values
    return dict(Xtr=Xtr, ytr=ytr, idtr=idtr, orig=orig, ids_orig=ids_orig,
                y_orig=ytr[orig], row_to_img=row_to_img,
                Xva=Xva, yva=yva, Xte=Xte, yte=yte)


def test_metrics(learner, d):
    pred = learner.estimator.predict_proba(d["Xte"]).argmax(1)
    return dict(test_f1=f1_score(d["yte"], pred, average="macro"),
                test_bal_acc=balanced_accuracy_score(d["yte"], pred),
                test_recall_mel=recall_score(d["yte"], pred, labels=[4], average=None)[0])


def run_experiment(name, seed, d, noise=0.0, test=False):
    """One full experiment: one strategy, one seed.
    noise > 0: the simulated expert gives a wrong label for that fraction of requested images.
    test=True: also evaluate on the test set at the end (final model only)."""
    np.random.seed(seed)
    rng = np.random.default_rng(1000 + seed)
    labelled = initial_split(d["y_orig"], seed)
    y_given = d["y_orig"].copy()                  # label the expert gives for each image

    def labelled_data():                          # the 3 versions of every labelled image
        rows = np.isin(d["idtr"], d["ids_orig"][labelled])
        return d["Xtr"][rows], y_given[d["row_to_img"][rows]]

    X_l, y_l = labelled_data()
    model = LinearHead(seed=seed).fit(X_l, y_l, epochs=100,
                                      X_val=d["Xva"], y_val=d["yva"])
    learner = ActiveLearner(model, STRATEGIES[name])
    hist = [dict(strategy=name, seed=seed, noise=noise, cycle=0,
                 n_labelled=int(labelled.sum()),
                 val_f1=learner.score(d["Xva"], d["yva"]), requested_classes=None)]

    cycle = 0
    while labelled.sum() < MAX_FRAC * len(d["y_orig"]):
        cycle += 1
        pool = np.where(~labelled)[0]
        chosen = pool[learner.query(d["Xtr"][d["orig"][pool]])]
        labelled[chosen] = True                   # the simulated expert labels them
        if noise > 0:                             # noisy expert: wrong label for a fraction
            wrong = chosen[rng.random(len(chosen)) < noise]
            y_given[wrong] = (d["y_orig"][wrong] + rng.integers(1, 7, len(wrong))) % 7
        X_l, y_l = labelled_data()
        learner.teach(X_l, y_l, epochs=3)
        hist.append(dict(strategy=name, seed=seed, noise=noise, cycle=cycle,
                         n_labelled=int(labelled.sum()),
                         val_f1=learner.score(d["Xva"], d["yva"]),
                         requested_classes=np.bincount(d["y_orig"][chosen],
                                                       minlength=7).tolist()))
    if test:
        hist[-1].update(test_metrics(learner, d))
    return hist


if __name__ == "__main__":
    import os
    os.makedirs("results/tables", exist_ok=True)
    d = load_data()

    rows = []
    for name in STRATEGIES:
        for seed in [0, 1, 2]:
            print("running", name, "| seed", seed, flush=True)
            rows += run_experiment(name, seed, d)

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/al_curves.csv", index=False)

    last = df[df.cycle == df.cycle.max()]
    print("\nValidation macro F1 at the last cycle (mean and std over 3 seeds):")
    print(last.groupby("strategy").val_f1.agg(["mean", "std"]).round(4))
    print("\nMean validation macro F1 over all cycles:")
    print(df.groupby("strategy").val_f1.mean().round(4))