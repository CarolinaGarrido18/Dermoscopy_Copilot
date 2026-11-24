"""
Active learning with simulated annotators (Human-in-the-Loop part).

Reuses the code of active_learning.py and only changes the oracle. By default it uses 3 oracles:
  perfect  : always the correct label (= what run_experiment already did with noise=0)
  single   : each image is seen by ONE dermatologist of the panel (structured errors)
  majority : each image is seen by k=3 dermatologists and the majority vote wins
There is also the optional 'uniform' oracle (10% random errors), which is not used by default
because it is the same experiment that al_final.py already runs with noise=0.10
(enable it with --oracles ... uniform).

For each run it records:
  - the validation macro F1 and the number of labelled IMAGES and of ANNOTATIONS (human cost);
  - the accuracy of the collected labels against the ground truth;
  - the disagreement rate between doctors on the images queried by the AL (majority only);
  - Krippendorff's alpha and the accuracy of each doctor at the end;
  - the linear head at each cycle (for the Grad-CAM, in gradcam_eval.py).

Prerequisites: features/ (extract_features.py), results/tables/al_curves.csv (active_learning.py).
Usage:  python al_hitl.py                (10 seeds, 3 oracles, best strategy vs random)
        python al_hitl.py --seeds 3      (quick test)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from active_learning import (STRATEGIES, ActiveLearner, LinearHead, MAX_FRAC, initial_split,
                             load_data, test_metrics)
from annotators import MISSING, Oracle, quality_report

ORACLES = ["perfect", "uniform", "single", "majority"]          # all the available oracles
DEFAULT_ORACLES = ["perfect", "single", "majority"]           # the ones used in the saved results


def run_hitl(strategy, oracle_kind, seed, d, k=3, head_path=None):
    """One complete run: one strategy, one oracle, one seed. Returns (history, quality)."""
    np.random.seed(seed)                              # random_sampling uses the global np.random
    rng = np.random.default_rng(1000 + seed)
    oracle = Oracle(oracle_kind, rng, k=k)
    y_true = d["y_orig"]
    n_img = len(y_true)

    labelled = initial_split(y_true, seed)
    y_given = y_true.copy()                           # final label given by the "panel" to each image
    votes_all = np.full((n_img, oracle.P), MISSING, dtype=int)
    cost = {"n": 0}                                   # number of annotations requested so far (human cost)

    def ask(idx):
        """Asks the annotators to label the images idx (indices of original images)."""
        y, votes, c = oracle.annotate(y_true[idx])
        y_given[idx], votes_all[idx] = y, votes
        cost["n"] += c
        return votes

    def labelled_data():                              # the 3 versions of every labelled image
        rows = np.isin(d["idtr"], d["ids_orig"][labelled])
        return d["Xtr"][rows], y_given[d["row_to_img"][rows]]

    ask(np.where(labelled)[0])                        # the initial labels also come from the annotators
    X_l, y_l = labelled_data()
    model = LinearHead(seed=seed).fit(X_l, y_l, epochs=100, X_val=d["Xva"], y_val=d["yva"])
    learner = ActiveLearner(model, STRATEGIES[strategy])

    heads = {}

    def snapshot():
        # Saves a copy of the linear head, indexed by the number of labelled images
        heads[int(labelled.sum())] = {n: p.clone() for n, p in model.net.state_dict().items()}

    def record(cycle, chosen=None, votes=None):
        # One row of the history: validation F1, cost and label quality at this cycle
        row = dict(strategy=strategy, oracle=oracle_kind, seed=seed, cycle=cycle,
                   n_labelled=int(labelled.sum()), n_annotations=cost["n"],
                   val_f1=learner.score(d["Xva"], d["yva"]),
                   label_acc=float((y_given[labelled] == y_true[labelled]).mean()),
                   disagree_rate=np.nan, requested_classes=None)
        if chosen is not None:
            row["requested_classes"] = np.bincount(y_true[chosen], minlength=7).tolist()
            if oracle_kind == "majority":             # % of queried images on which the doctors disagree
                row["disagree_rate"] = float(np.mean(
                    [len(set(r[r != MISSING])) > 1 for r in votes]))
        return row

    hist = [record(0)]
    snapshot()
    cycle = 0
    while labelled.sum() < MAX_FRAC * n_img:
        cycle += 1
        pool = np.where(~labelled)[0]
        chosen = pool[learner.query(d["Xtr"][d["orig"][pool]])]
        labelled[chosen] = True
        votes = ask(chosen)
        X_l, y_l = labelled_data()
        learner.teach(X_l, y_l, epochs=3)
        hist.append(record(cycle, chosen, votes))
        snapshot()
    hist[-1].update(test_metrics(learner, d))

    quality = dict(strategy=strategy, oracle=oracle_kind, seed=seed)
    quality.update(quality_report(votes_all[labelled], y_true[labelled], y_given[labelled]))
    quality["n_annotations"] = cost["n"]              # real cost (perfect/uniform do not record votes)
    if head_path is not None:
        os.makedirs(os.path.dirname(head_path), exist_ok=True)
        torch.save(heads, head_path)
        # Also saves the labelled images of the last cycle and the labels given by the annotators (used to retrain the model with the corrections of the clinician)
        np.savez(head_path.replace(".pt", "_labelled.npz"), image_ids=d["ids_orig"][labelled], labels=y_given[labelled])
    return hist, quality

def holm(p):
    """Holm correction for multiple comparisons; returns the adjusted p-values in the original order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def safe_wilcoxon(a, b):
    """Wilcoxon signed-rank p-value, or NaN when the test cannot be run."""
    try:
        return wilcoxon(a, b).pvalue
    except ValueError:                                # all the differences are zero
        return np.nan


def main():
    """Runs all the experiments, then saves the tables, the statistical tests and the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None,
                    help="AL strategy to compare with random (default: the best one on validation)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--head-seeds", type=int, default=3, help="number of seeds whose heads are saved")
    ap.add_argument("--oracles", nargs="+", default=DEFAULT_ORACLES, choices=ORACLES)
    ap.add_argument("--k", type=int, default=3, help="annotators per image in the 'majority' oracle")
    args = ap.parse_args()

    for p in ("results/tables", "results/figures", "results/heads"):
        os.makedirs(p, exist_ok=True)

    best = args.strategy
    if best is None:                                  # chosen on VALIDATION only, as in al_final.py
        curves = pd.read_csv("results/tables/al_curves.csv")
        best = (curves[curves.strategy != "random"].groupby("strategy").val_f1.mean().idxmax())
    print("AL strategy:", best)

    d = load_data()
    rows, qual = [], []
    for strategy in (best, "random"):
        for oracle in args.oracles:
            for seed in range(args.seeds):
                print(f"running {strategy} | {oracle} | seed {seed}", flush=True)
                hp = (f"results/heads/{strategy}__{oracle}__seed{seed}.pt"
                      if seed < args.head_seeds else None)
                h, q = run_hitl(strategy, oracle, seed, d, k=args.k, head_path=hp)
                rows += h
                qual.append(q)
    res = pd.DataFrame(rows)
    res.to_csv("results/tables/hitl_runs.csv", index=False)
    pd.DataFrame(qual).to_csv("results/tables/hitl_annotator_quality.csv", index=False)

    # ---- 1) summary at the last cycle (mean and standard deviation over the seeds) ----
    final = res[res.cycle == res.cycle.max()]
    summ = final.groupby(["strategy", "oracle"]).agg(
        n_labelled=("n_labelled", "mean"), n_annotations=("n_annotations", "mean"),
        label_acc=("label_acc", "mean"),
        test_f1=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        test_bal_acc=("test_bal_acc", "mean"), test_recall_mel=("test_recall_mel", "mean"),
        test_recall_mel_std=("test_recall_mel", "std")).round(4)
    summ.to_csv("results/tables/hitl_summary.csv")
    print("\nResults at the last cycle (test, mean over seeds):\n", summ.to_string())

    # ---- 2) does the AL query images on which the doctors disagree more? ----
    # WARNING: the simulated annotators make mistakes based on the CLASS, not on how hard the image is.
    # This rate only reflects the mix of queried classes; it does not prove that the AL picks "hard" cases.
    if "majority" in args.oracles:
        dis = (res[(res.oracle == "majority") & (res.cycle > 0)]
               .groupby("strategy").disagree_rate.agg(["mean", "std"]).round(4))
        dis.to_csv("results/tables/hitl_disagreement.csv")
        print("\nDisagreement between doctors on the queried images (majority oracle):\n", dis)

    # ---- 3) quality of the collected annotations ----
    qdf = pd.DataFrame(qual)
    qsum = qdf.groupby(["strategy", "oracle"]).mean(numeric_only=True).drop(columns="seed").round(4)
    qsum.to_csv("results/tables/hitl_quality_summary.csv")
    print("\nAnnotation quality (mean over seeds):\n", qsum.to_string())

    # ---- 4) paired tests (same seed = same initial set) with the Holm correction ----
    pf = final.pivot_table(index="seed", columns=["strategy", "oracle"], values="test_f1")
    mean_val = (res[res.cycle > 0].groupby(["strategy", "oracle", "seed"]).val_f1.mean()
                .unstack(["strategy", "oracle"]))
    tests = []
    for oracle in args.oracles:                       # AL vs random, for each oracle
        for name, tab in (("test F1 (last cycle)", pf), ("mean val F1 (cycles)", mean_val)):
            a, b = tab[(best, oracle)], tab[("random", oracle)]
            tests.append(dict(family="AL vs random", summary=name, oracle=oracle,
                              comparison=f"{best} vs random", mean_diff=(a - b).mean(),
                              seeds_better=int(((a - b) > 0).sum()), n_seeds=len(a),
                              p=safe_wilcoxon(a, b)))
    if "perfect" in args.oracles:                     # cost of the label quality: each oracle vs perfect
        for oracle in [o for o in args.oracles if o != "perfect"]:
            a, b = pf[(best, oracle)], pf[(best, "perfect")]
            tests.append(dict(family="oracle vs perfect", summary="test F1 (last cycle)",
                              oracle=oracle, comparison=f"{oracle} vs perfect ({best})",
                              mean_diff=(a - b).mean(), seeds_better=int(((a - b) > 0).sum()),
                              n_seeds=len(a), p=safe_wilcoxon(a, b)))
    tdf = pd.DataFrame(tests)
    tdf["p_holm"] = holm(tdf.p.fillna(1.0))
    tdf.round(4).to_csv("results/tables/hitl_stats.csv", index=False)
    print("\nWilcoxon tests (Holm over all the comparisons):\n", tdf.round(4).to_string(index=False))

    # ---- 5) curves: F1 vs labelled images and F1 vs ANNOTATIONS (human cost) ----
    colors = dict(zip(ORACLES, ["black", "tab:orange", "tab:red", "tab:blue"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for strategy, ls in ((best, "-"), ("random", "--")):
        for oracle in args.oracles:
            g = res[(res.strategy == strategy) & (res.oracle == oracle)]
            m = g.groupby("n_labelled").agg(f=("val_f1", "mean"), s=("val_f1", "std"),
                                            cost=("n_annotations", "mean"))
            for ax, x in zip(axes, (m.index, m.cost)):
                ax.plot(x, m.f, ls=ls, color=colors[oracle], marker="o", ms=2.5,
                        label=f"{oracle} ({strategy})")
                ax.fill_between(x, m.f - m.s, m.f + m.s, color=colors[oracle], alpha=0.08)
    axes[0].set_xlabel("Labelled images")
    axes[1].set_xlabel("Annotations requested from dermatologists (annotation cost)")
    axes[0].set_ylabel("Macro F1 (validation)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/figures/hitl_curves.png", dpi=200)
    print("\nSaved: results/tables/hitl_*.csv, results/figures/hitl_curves.png, results/heads/")


if __name__ == "__main__":
    main()
