"""
Dermoscopy Copilot - 04. Active learning with frozen EfficientNetV2-S features

The model starts with 10% of the training images labelled. In each cycle, a query strategy selects
300 images from the unlabelled pool, a simulated oracle returns their labels (the diagnoses stored
in the dataset), and the model is retrained and evaluated on the validation set with the macro F1.
The cycle is repeated until 50% of the training images are labelled. Five query strategies are
compared (least confidence, margin, entropy, hybrid and random), each with 10 seeds.

In the training set each image has three versions (the original and two augmented copies).
In the validation and test sets each image has 8 views (rotated and mirrored), and the
prediction for the image is the average of the predictions for its 8 views.

The results are saved in results/tables/al_curves.csv and used by al_results.py, al_stats.py,
al_labels_to_target.py and al_final.py. The functions in this file are also used by al_hitl.py
and referral.py.

Prerequisite: the features/features_<BACKBONE>/ folder produced by extract_features.py
"""

# Library imports
import copy                         # to save a copy of the model weights (early stopping)
import numpy as np
import pandas as pd
import torch                        # tensors and model training
import torch.nn as nn               # linear layer and loss function
from sklearn.cluster import KMeans  # k-means clustering, used by the hybrid query strategy
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score 
from sklearn.model_selection import train_test_split # stratified sampling of the initial labelled set


BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"   # must match BACKBONE in extract_features.py
FEATS = f"features/features_{BACKBONE}"          # directory

INIT_FRAC = 0.10    # 10% of the training images are labelled at the start
BATCH = 300         # number of images requested in each cycle
MAX_FRAC = 0.50     # stop when 50% of the training images are labelled


# Data loading
def load_split(split):
    # Loads the features, labels, image ids and augmentation indices of a split
    d = np.load(f"{FEATS}/{split}_features.npz", allow_pickle=True)
    return d["X"].astype("float32"), d["y"], d["image_ids"], d["aug_idx"]

def initial_split(y_orig, seed):
    # Selects 10% of the training images, stratified by class (True = labelled)
    idx = np.arange(len(y_orig))
    labelled, _ = train_test_split(idx, train_size=INIT_FRAC, stratify=y_orig, random_state=seed)
    mask = np.zeros(len(y_orig), dtype=bool)
    mask[labelled] = True
    return mask


# Model
class LinearHead:
    # The model receives the 1280 features of each image and gives one score for each of the 7 classes.
    # The EfficientNetV2-S is frozen, so only this final layer is trained.
    def __init__(self, seed=0, n_features=1280, n_classes=7):
        # Creates the layer and the optimiser, with the same values as the baseline, so that the comparison between both is fair.
        torch.manual_seed(seed) # running again gives the same initial weights
        self.net = nn.Linear(n_features, n_classes)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)

    def fit(self, X, y, epochs=3, X_val=None, y_val=None, patience=10):
        # Trains the model, giving more importance to the rare classes. 
        # In the initial training, stops when the validation F1 no longer improves (early stopping).
        n = np.bincount(y, minlength=7).clip(min=1)  # class weights inversely proportional to class frequency
        w = torch.tensor(len(y) / (7 * n), dtype=torch.float32)   
        loss_fn = nn.CrossEntropyLoss(weight=w)
        X_t, y_t = torch.tensor(X), torch.tensor(y, dtype=torch.long)
        best, state, wait = -1, None, 0
        for _ in range(epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(perm), 256): # train on mini-batches of 256 images at a time
                b = perm[i:i + 256]
                self.opt.zero_grad()
                loss_fn(self.net(X_t[b]), y_t[b]).backward()
                self.opt.step()
            if X_val is not None: # only with the initial 10%
                f1 = f1_score(y_val, self.predict_proba(X_val).argmax(1), average="macro")
                if f1 > best:
                    best, state, wait = f1, copy.deepcopy(self.net.state_dict()), 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if state is not None:  # restore the version with the best weights
            self.net.load_state_dict(state)
        return self

    def predict_proba(self, X):
        # Returns the class probabilities. 
        # For validation and test images, they are the average of the 8 views.
        with torch.no_grad():
            X = torch.tensor(X)
            if X.ndim == 3:  # 3 dimensions (images, views and features)
                N, V, D = X.shape
                p = torch.softmax(self.net(X.reshape(N * V, D)), 1).reshape(N, V, -1)
                return p.mean(1).numpy()   # average of the 8 views
            return torch.softmax(self.net(X), 1).numpy()


# Query strategies
# Each strategy returns the pool images ordered from the most to the least informative, together with their scores
def least_confidence(model, X):
    # Chooses the images where the model is least sure about its predicted class
    p = model.predict_proba(X)
    scores = 1 - p.max(axis=1)  # a low top probability means high uncertainty
    return np.argsort(-scores), scores  # highest scores first

def margin_sampling(model, X):
    # Chooses the images where the two most likely classes have very close probabilities.
    p = np.sort(model.predict_proba(X), axis=1)
    scores = 1 - (p[:, -1] - p[:, -2])  # a small gap between the top two means high uncertainty
    return np.argsort(-scores), scores

def entropy_sampling(model, X):
    # Chooses the images whose probability is spread over many classes (highest entropy)
    p = model.predict_proba(X)
    scores = -(p * np.log(p + 1e-12)).sum(axis=1) # the tiny constant avoids taking the log of zero (error)
    return np.argsort(-scores), scores

def random_sampling(model, X):
    # Chooses the images at random, without using the model. 
    # This is the baseline that represents labelling without active learning, used for comparison.
    return np.random.permutation(len(X)), None

def hybrid_sampling(model, X, n=BATCH, frac=0.10):
    # Chooses images that are both uncertain and different from each other. 
    # The 10% most uncertain images (by entropy) are grouped into n clusters with k-means, and the image closest to the centre of each cluster is selected.
    order, _ = entropy_sampling(model, X)
    k = max(n, int(frac * len(X)))
    cand = order[:k]  # the most uncertain images are the candidates
    km = KMeans(n_clusters=n, n_init=1, random_state=0).fit(X[cand])
    dist = km.transform(X[cand])  # distance of each candidate to each cluster centre
    chosen = list(dict.fromkeys(cand[dist.argmin(axis=0)]))  # closest image to each centre
    if len(chosen) < n:  # if two centres picked the same image, fill the rest with the next most uncertain images
        used = set(chosen)
        rest = [i for i in cand if i not in used]
        chosen += rest[:n - len(chosen)]
    return np.array(chosen), None


# Active learner
# Combines the model with a query strategy. 
# In each cycle it chooses the images to label (query), learns from their labels (teach) and is evaluated (score).
class ActiveLearner:
    def __init__(self, estimator, query_strategy):
        # Keeps the model and the query strategy that will be used in every cycle
        self.estimator = estimator
        self.query_strategy = query_strategy

    def query(self, X_pool, n=BATCH):
        # Returns the indices of the n most informative images in the pool 
        order, _ = self.query_strategy(self.estimator, X_pool)
        return order[:n]

    def teach(self, X, y, epochs=3):
        # Trains the model with all the labelled images, including the new ones
        self.estimator.fit(X, y, epochs=epochs)  # continue training for a few epochs

    def score(self, X, y):
        # Returns the macro F1 score
        return f1_score(y, self.estimator.predict_proba(X).argmax(1), average="macro")


# Links the name of each strategy to its function
STRATEGIES = {"random": random_sampling, "least_confidence": least_confidence, "margin": margin_sampling, "entropy": entropy_sampling, "hybrid": hybrid_sampling}


# Data and evaluation
def load_data():
    # Loads the three splits and links each row of the training features to its original image.
    # Each training image appears three times (original and two augmented copies).
    # All three versions must receive the same label when the image is labelled.
    Xtr, ytr, idtr, augtr = load_split("train")
    Xva, yva, _, _ = load_split("val")
    Xte, yte, _, _ = load_split("test")
    orig = np.where(augtr == 0)[0]  # rows of the original images
    ids_orig = idtr[orig]
    # Link each row of the training features to its original image
    row_to_img = pd.Series(np.arange(len(ids_orig)), index=ids_orig).reindex(idtr).values
    # Return all the data in a dictionary
    return dict(Xtr=Xtr, ytr=ytr, idtr=idtr, orig=orig, ids_orig=ids_orig, y_orig=ytr[orig], row_to_img=row_to_img, Xva=Xva, yva=yva, Xte=Xte, yte=yte)

def test_metrics(learner, d):
    # Computes the macro F1, the balanced accuracy and the melanoma sensitivity on the test set
    pred = learner.estimator.predict_proba(d["Xte"]).argmax(1)
    return dict(test_f1=f1_score(d["yte"], pred, average="macro"), test_bal_acc=balanced_accuracy_score(d["yte"], pred), test_recall_mel=recall_score(d["yte"], pred, labels=[4], average=None)[0])  # class 4 is melanoma


# One experiment
# One complete active learning run for one strategy and one seed
def run_experiment(name, seed, d, noise=0.0, test=False):
    # Runs the active learning cycle from 10% to 50% of labelled images, and records the validation macro F1 of each cycle. 
    # The noise parameter is the fraction of queried images that the oracle labels wrongly. 
    # If test is True, the final model is also evaluated on the test set.
    np.random.seed(seed)                        # the same seed always gives the same result
    rng = np.random.default_rng(1000 + seed)    # separate random numbers for the oracle mistakes
    labelled = initial_split(d["y_orig"], seed) # True for the images that are already labelled
    y_given = d["y_orig"].copy()                # labels returned by the oracle

    def labelled_data():           
        # Returns the three versions of every image in the labelled set, with the oracle labels
        rows = np.isin(d["idtr"], d["ids_orig"][labelled])
        return d["Xtr"][rows], y_given[d["row_to_img"][rows]]

    # Cycle 0, the initial model is trained on the initial labelled set
    X_l, y_l = labelled_data()
    model = LinearHead(seed=seed).fit(X_l, y_l, epochs=100, X_val=d["Xva"], y_val=d["yva"])
    learner = ActiveLearner(model, STRATEGIES[name])
    hist = [dict(strategy=name, seed=seed, noise=noise, cycle=0,n_labelled=int(labelled.sum()), val_f1=learner.score(d["Xva"], d["yva"]), requested_classes=None)]

    # Active learning cycles, repeated until 50% of the images are labelled
    cycle = 0
    while labelled.sum() < MAX_FRAC * len(d["y_orig"]):
        cycle += 1
        pool = np.where(~labelled)[0]   # unlabelled pool
        chosen = pool[learner.query(d["Xtr"][d["orig"][pool]])]   # query 300 images
        labelled[chosen] = True         # the oracle labels them
        if noise > 0:  # noisy oracle, some queried images receive a wrong class, always different from the true one
            wrong = chosen[rng.random(len(chosen)) < noise]
            y_given[wrong] = (d["y_orig"][wrong] + rng.integers(1, 7, len(wrong))) % 7
        X_l, y_l = labelled_data()
        learner.teach(X_l, y_l, epochs=3)   # retrain for 3 epochs
        hist.append(dict(strategy=name, seed=seed, noise=noise, cycle=cycle, n_labelled=int(labelled.sum()), val_f1=learner.score(d["Xva"], d["yva"]), requested_classes=np.bincount(d["y_orig"][chosen], minlength=7).tolist()))
    if test:
        hist[-1].update(test_metrics(learner, d))
    return hist


# Main
# Runs the full experiment
if __name__ == "__main__":
    import os
    os.makedirs("results/tables", exist_ok=True)
    d = load_data()

    # Five query strategies with ten seeds each, 50 experiments in total
    rows = []
    for name in STRATEGIES:
        for seed in range(10):
            print("running", name, "| seed", seed, flush=True)
            rows += run_experiment(name, seed, d)

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/al_curves.csv", index=False)

    last = df[df.cycle == df.cycle.max()]
    print("\nValidation macro F1 at the last cycle (mean and std over 10 seeds):")
    print(last.groupby("strategy").val_f1.agg(["mean", "std"]).round(4))
    print("\nMean validation macro F1 over all cycles:")
    print(df.groupby("strategy").val_f1.mean().round(4))