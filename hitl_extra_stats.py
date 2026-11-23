"""
Teste que faltava em al_hitl.py: voto de maioria (3 médicos) vs um só médico.

Compara 'majority' com 'single' de duas formas, para cada estratégia (hybrid e random):
  1) MESMAS IMAGENS: macro F1 de TESTE no último ciclo (as duas têm as mesmas imagens
     etiquetadas, mas 'majority' custa 3x mais anotações).
  2) MESMO CUSTO HUMANO: macro F1 de VALIDAÇÃO ao mesmo nº de anotações. Como a curva só
     tem pontos em cada ciclo, interpola-se linearmente (por seed) e comparam-se 3 custos
     dentro da zona em que as duas curvas se sobrepõem. É em validação porque o F1 de teste
     só é calculado no último ciclo.

Comparação pareada por seed (mesma seed = mesmo conjunto inicial de imagens), Wilcoxon
bilateral, com correção de Holm sobre todas as comparações da tabela.
Diferença = majority - single: valores POSITIVOS favorecem o voto de maioria.

Lê results/tables/hitl_runs.csv (de al_hitl.py) e escreve results/tables/hitl_majority_vs_single.csv
Uso:  python3 hitl_extra_stats.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RUNS = "results/tables/hitl_runs.csv"
OUT = "results/tables/hitl_majority_vs_single.csv"


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
    except ValueError:                       # diferenças todas nulas
        return np.nan


def make_row(strategy, basis, a, b):
    diff = a.values - b.values
    return dict(strategy=strategy, basis=basis, mean_majority=a.mean(), mean_single=b.mean(),
                mean_diff=diff.mean(), std_diff=diff.std(ddof=1),
                seeds_majority_better=int((diff > 0).sum()), n_seeds=len(diff),
                p=safe_wilcoxon(a.values, b.values))


def main():
    res = pd.read_csv(RUNS)
    for o in ("majority", "single"):
        if o not in set(res.oracle):
            raise SystemExit(f"Não há execuções com o oráculo '{o}' em {RUNS}")
    rows = []
    for strategy in sorted(res.strategy.unique()):
        g = res[res.strategy == strategy]
        maj, sin = g[g.oracle == "majority"], g[g.oracle == "single"]
        seeds = sorted(set(maj.seed) & set(sin.seed))

        # 1) mesmas imagens, F1 de teste no último ciclo
        def last(df):
            d = df[df.cycle == df.cycle.max()].set_index("seed")
            return d.test_f1.loc[seeds]
        rows.append(make_row(strategy, "test F1, same images (last cycle)", last(maj), last(sin)))

        # 2) mesmo custo humano, F1 de validação interpolado por seed
        def curve(df, s):
            d = df[df.seed == s].sort_values("n_annotations")
            return d.n_annotations.values.astype(float), d.val_f1.values
        lo = max(maj.n_annotations.min(), sin.n_annotations.min())
        hi = min(maj.n_annotations.max(), sin.n_annotations.max())
        if lo >= hi:
            print(f"[{strategy}] as curvas não se sobrepõem em custo; sem comparação a igual custo.")
            continue
        for c in np.linspace(lo, hi, 3):
            a = pd.Series({s: np.interp(c, *curve(maj, s)) for s in seeds})
            b = pd.Series({s: np.interp(c, *curve(sin, s)) for s in seeds})
            rows.append(make_row(strategy, f"val F1, same cost ({c:.0f} annotations)", a, b))

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.fillna(1.0))
    out = out.round(4)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print("majority (3 médicos) vs single (1 médico); diferença = majority - single\n")
    print(out.to_string(index=False))
    print(f"\nGuardado em {OUT}")


if __name__ == "__main__":
    main()
