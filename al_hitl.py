"""
Active Learning com anotadores simulados (parte Human-in-the-Loop).

Reutiliza o código das colegas (active_learning.py) e só troca o oráculo:
  perfect  : etiqueta sempre correta (= o que o run_experiment já fazia com noise=0)
  uniform  : 10% de erros ao acaso (= noise=0.10 do al_final.py)
  single   : cada imagem é vista por UM dermatologista do painel (erros estruturados)
  majority : cada imagem é vista por k=3 dermatologistas e vale o voto de maioria

Por cada execução regista:
  - macro F1 em validação e o nº de IMAGENS etiquetadas e de ANOTAÇÕES (custo humano);
  - exatidão das etiquetas recolhidas face ao ground truth;
  - taxa de desacordo entre médicos nas imagens que o AL pediu (só majority);
  - alpha de Krippendorff e exatidão por médico no fim;
  - a cabeça linear em cada ciclo (para o Grad-CAM, em gradcam_eval.py).

Pré-requisitos: features/ (extract_features.py), results/tables/al_curves.csv (active_learning.py).
Uso:  python al_hitl.py                (10 seeds, 4 oráculos, melhor estratégia vs random)
      python al_hitl.py --seeds 3      (teste rápido)
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

ORACLES = ["perfect", "uniform", "single", "majority"]


def run_hitl(strategy, oracle_kind, seed, d, k=3, head_path=None):
    """Uma execução completa: uma estratégia, um oráculo, uma seed. Devolve (histórico, qualidade)."""
    np.random.seed(seed)                              # random_sampling usa o np.random global
    rng = np.random.default_rng(1000 + seed)
    oracle = Oracle(oracle_kind, rng, k=k)
    y_true = d["y_orig"]
    n_img = len(y_true)

    labelled = initial_split(y_true, seed)
    y_given = y_true.copy()                           # etiqueta final que o "painel" deu a cada imagem
    votes_all = np.full((n_img, oracle.P), MISSING, dtype=int)
    cost = {"n": 0}

    def ask(idx):
        """Pede etiquetas aos anotadores para as imagens idx (índices de imagens originais)."""
        y, votes, c = oracle.annotate(y_true[idx])
        y_given[idx], votes_all[idx] = y, votes
        cost["n"] += c
        return votes

    def labelled_data():                              # as 3 versões de cada imagem etiquetada
        rows = np.isin(d["idtr"], d["ids_orig"][labelled])
        return d["Xtr"][rows], y_given[d["row_to_img"][rows]]

    ask(np.where(labelled)[0])                        # as etiquetas iniciais também vêm dos anotadores
    X_l, y_l = labelled_data()
    model = LinearHead(seed=seed).fit(X_l, y_l, epochs=100, X_val=d["Xva"], y_val=d["yva"])
    learner = ActiveLearner(model, STRATEGIES[strategy])

    heads = {}

    def snapshot():
        heads[int(labelled.sum())] = {n: p.clone() for n, p in model.net.state_dict().items()}

    def record(cycle, chosen=None, votes=None):
        row = dict(strategy=strategy, oracle=oracle_kind, seed=seed, cycle=cycle,
                   n_labelled=int(labelled.sum()), n_annotations=cost["n"],
                   val_f1=learner.score(d["Xva"], d["yva"]),
                   label_acc=float((y_given[labelled] == y_true[labelled]).mean()),
                   disagree_rate=np.nan, requested_classes=None)
        if chosen is not None:
            row["requested_classes"] = np.bincount(y_true[chosen], minlength=7).tolist()
            if oracle_kind == "majority":             # % de imagens pedidas em que os médicos discordam
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
    quality["n_annotations"] = cost["n"]              # custo real (perfect/uniform não registam votos)
    if head_path is not None:
        os.makedirs(os.path.dirname(head_path), exist_ok=True)
        torch.save(heads, head_path)
    return hist, quality


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def safe_wilcoxon(a, b):
    try:
        return wilcoxon(a, b).pvalue
    except ValueError:                                # diferenças todas nulas
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None,
                    help="estratégia de AL a comparar com random (por omissão: a melhor em validação)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--head-seeds", type=int, default=3, help="nº de seeds cujas cabeças se guardam")
    ap.add_argument("--oracles", nargs="+", default=ORACLES, choices=ORACLES)
    ap.add_argument("--k", type=int, default=3, help="anotadores por imagem no oráculo 'majority'")
    args = ap.parse_args()

    for p in ("results/tables", "results/figures", "results/heads"):
        os.makedirs(p, exist_ok=True)

    best = args.strategy
    if best is None:                                  # escolhida só em VALIDAÇÃO, como no al_final.py
        curves = pd.read_csv("results/tables/al_curves.csv")
        best = (curves[curves.strategy != "random"].groupby("strategy").val_f1.mean().idxmax())
    print("Estratégia de AL:", best)

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

    # ---- 1) resumo no último ciclo (média e desvio-padrão sobre as seeds) ----
    final = res[res.cycle == res.cycle.max()]
    summ = final.groupby(["strategy", "oracle"]).agg(
        n_labelled=("n_labelled", "mean"), n_annotations=("n_annotations", "mean"),
        label_acc=("label_acc", "mean"),
        test_f1=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        test_bal_acc=("test_bal_acc", "mean"), test_recall_mel=("test_recall_mel", "mean"),
        test_recall_mel_std=("test_recall_mel", "std")).round(4)
    summ.to_csv("results/tables/hitl_summary.csv")
    print("\nResultados no último ciclo (teste, média sobre seeds):\n", summ.to_string())

    # ---- 2) o AL pede imagens em que os médicos discordam mais? ----
    # ATENÇÃO: os anotadores simulados erram em função da CLASSE, não da dificuldade da imagem.
    # Esta taxa só reflete a mistura de classes pedida; não prova que o AL escolhe casos "difíceis".
    if "majority" in args.oracles:
        dis = (res[(res.oracle == "majority") & (res.cycle > 0)]
               .groupby("strategy").disagree_rate.agg(["mean", "std"]).round(4))
        dis.to_csv("results/tables/hitl_disagreement.csv")
        print("\nDesacordo entre médicos nas imagens pedidas (oráculo majority):\n", dis)

    # ---- 3) qualidade das anotações recolhidas ----
    qdf = pd.DataFrame(qual)
    qsum = qdf.groupby(["strategy", "oracle"]).mean(numeric_only=True).drop(columns="seed").round(4)
    qsum.to_csv("results/tables/hitl_quality_summary.csv")
    print("\nQualidade das anotações (média sobre seeds):\n", qsum.to_string())

    # ---- 4) testes pareados (mesma seed = mesmo conjunto inicial) com correção de Holm ----
    pf = final.pivot_table(index="seed", columns=["strategy", "oracle"], values="test_f1")
    mean_val = (res[res.cycle > 0].groupby(["strategy", "oracle", "seed"]).val_f1.mean()
                .unstack(["strategy", "oracle"]))
    tests = []
    for oracle in args.oracles:                       # AL vs random, para cada oráculo
        for name, tab in (("test F1 (último ciclo)", pf), ("val F1 médio (ciclos)", mean_val)):
            a, b = tab[(best, oracle)], tab[("random", oracle)]
            tests.append(dict(family="AL vs random", summary=name, oracle=oracle,
                              comparison=f"{best} vs random", mean_diff=(a - b).mean(),
                              seeds_better=int(((a - b) > 0).sum()), n_seeds=len(a),
                              p=safe_wilcoxon(a, b)))
    if "perfect" in args.oracles:                     # custo da qualidade: cada oráculo vs perfeito
        for oracle in [o for o in args.oracles if o != "perfect"]:
            a, b = pf[(best, oracle)], pf[(best, "perfect")]
            tests.append(dict(family="oráculo vs perfeito", summary="test F1 (último ciclo)",
                              oracle=oracle, comparison=f"{oracle} vs perfect ({best})",
                              mean_diff=(a - b).mean(), seeds_better=int(((a - b) > 0).sum()),
                              n_seeds=len(a), p=safe_wilcoxon(a, b)))
    tdf = pd.DataFrame(tests)
    tdf["p_holm"] = holm(tdf.p.fillna(1.0))
    tdf.round(4).to_csv("results/tables/hitl_stats.csv", index=False)
    print("\nTestes de Wilcoxon (Holm sobre todas as comparações):\n", tdf.round(4).to_string(index=False))

    # ---- 5) curvas: F1 vs imagens etiquetadas e F1 vs ANOTAÇÕES (custo humano) ----
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
    axes[0].set_xlabel("Imagens etiquetadas")
    axes[1].set_xlabel("Anotações pedidas aos médicos (custo humano)")
    axes[0].set_ylabel("Macro F1 (validação)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/figures/hitl_curves.png", dpi=200)
    print("\nGuardado: results/tables/hitl_*.csv, results/figures/hitl_curves.png, results/heads/")


if __name__ == "__main__":
    main()
