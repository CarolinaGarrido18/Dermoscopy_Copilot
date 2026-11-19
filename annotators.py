"""
Anotadores (dermatologistas) simulados e métricas de qualidade das anotações.

Parte "Human-in-the-Loop" do projeto. Em vez de um oráculo perfeito (ou com ruído uniforme),
simulamos um PAINEL de dermatologistas, cada um com a sua matriz de confusão, e medimos a
qualidade das anotações com as métricas da folha "Human-in-the-Loop Machine Learning:
annotation quality frameworks" (Munro): exatidão face ao ground truth, acordo entre
anotadores, acordo com a maioria, alpha de Krippendorff e matriz de confusão por anotador.

Só depende de NumPy. Para validar contra a folha de cálculo: python annotators.py
"""
import numpy as np

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
N_CLASSES = len(CLASSES)
MISSING = -1          # "este anotador não anotou esta imagem"

# ----------------------------------------------------------------------------------------
# PRESSUPOSTOS DA SIMULAÇÃO (editar aqui; têm de ser justificados/calibrados no relatório)
# ----------------------------------------------------------------------------------------
# Exatidão global de cada dermatologista do painel. Valores ILUSTRATIVOS, não vêm da
# literatura: substituir por valores citados (ou fazer análise de sensibilidade).
PANEL_ACCURACY = [0.92, 0.88, 0.85, 0.80, 0.75]
PANEL_NAMES = [f"Derm {i + 1}" for i in range(len(PANEL_ACCURACY))]

# Estrutura dos erros: quando um dermatologista erra, para que classe tende a ir?
# Pesos relativos (linhas = classe verdadeira). Inspirada nos erros da matriz de confusão do
# modelo (mel<->nv, bkl<->mel, akiec<->bcc/bkl) e em confusões dermatoscópicas comuns.
# É uma hipótese de modelação, não um resultado empírico.
_CONFUSION_WEIGHTS = {
    "akiec": {"bcc": 2, "bkl": 2, "mel": 1},
    "bcc":   {"akiec": 2, "bkl": 1, "mel": 1},
    "bkl":   {"akiec": 1, "bcc": 1, "mel": 2, "nv": 2},
    "df":    {"bkl": 1, "nv": 2, "mel": 0.5},
    "mel":   {"nv": 3, "bkl": 2, "akiec": 0.5},
    "nv":    {"mel": 3, "bkl": 2, "df": 0.5},
    "vasc":  {"bcc": 1, "nv": 1, "df": 0.5},
}


def error_structure():
    """Matriz (7x7) com a distribuição dos erros por classe verdadeira (diagonal = 0)."""
    S = np.zeros((N_CLASSES, N_CLASSES))
    for true_c, targets in _CONFUSION_WEIGHTS.items():
        for wrong_c, w in targets.items():
            S[CLASSES.index(true_c), CLASSES.index(wrong_c)] = w
    return S / S.sum(axis=1, keepdims=True)


def annotator_confusion(accuracy):
    """Matriz de confusão de um anotador: P(etiqueta dada | classe verdadeira)."""
    C = (1 - accuracy) * error_structure()
    C[np.diag_indices(N_CLASSES)] = accuracy
    return C


def make_panel(accuracies=None):
    """Lista de matrizes de confusão, uma por dermatologista do painel."""
    accuracies = PANEL_ACCURACY if accuracies is None else accuracies
    return [annotator_confusion(a) for a in accuracies]


# ----------------------------------------------------------------------------------------
# ORÁCULOS
# ----------------------------------------------------------------------------------------
def _sample_labels(C, y_true, rng):
    """Uma etiqueta por imagem, amostrada da matriz de confusão C."""
    cdf = C[y_true].cumsum(axis=1)
    u = rng.random(len(y_true))[:, None]
    return np.minimum((u > cdf).sum(axis=1), N_CLASSES - 1)


class Oracle:
    """
    Responde aos pedidos do Active Learning. kind:
      perfect      : etiqueta correta (o que o código original fazia)
      uniform      : erra uma fração `p` das imagens, para uma classe ao acaso (ruído uniforme)
      single       : cada imagem é vista por UM dermatologista do painel, escolhido ao acaso
      majority     : cada imagem é vista por `k` dermatologistas distintos e vale o voto de maioria
    annotate() devolve (etiquetas finais, votos (n, n_painel) com MISSING, nº de anotações).
    O custo humano é o nº de anotações (uma imagem vista por 3 médicos custa 3).
    """

    def __init__(self, kind, rng, panel=None, k=3, p=0.10):
        assert kind in ("perfect", "uniform", "single", "majority")
        self.kind, self.rng, self.k, self.p = kind, rng, k, p
        self.panel = make_panel() if panel is None else panel
        self.P = len(self.panel)

    def annotate(self, y_true):
        n = len(y_true)
        votes = np.full((n, self.P), MISSING, dtype=int)
        if self.kind == "perfect":
            return y_true.copy(), votes, n
        if self.kind == "uniform":
            y = y_true.copy()
            wrong = self.rng.random(n) < self.p
            y[wrong] = (y_true[wrong] + self.rng.integers(1, N_CLASSES, wrong.sum())) % N_CLASSES
            return y, votes, n
        k = 1 if self.kind == "single" else self.k
        # k anotadores distintos por imagem, escolhidos ao acaso
        chosen = np.argsort(self.rng.random((n, self.P)), axis=1)[:, :k]
        for a in range(self.P):
            rows = np.where((chosen == a).any(axis=1))[0]
            if len(rows):
                votes[rows, a] = _sample_labels(self.panel[a], y_true[rows], self.rng)
        return majority_vote(votes, self.rng), votes, n * k


def majority_vote(votes, rng):
    """Voto de maioria por imagem; empates resolvidos ao acaso entre as classes empatadas."""
    counts = np.stack([(votes == c).sum(axis=1) for c in range(N_CLASSES)], axis=1)
    return (counts + rng.random(counts.shape) * 0.5).argmax(axis=1)   # ruído < 1: só desempata


# ----------------------------------------------------------------------------------------
# MÉTRICAS DE QUALIDADE DAS ANOTAÇÕES (equivalentes às abas da folha de cálculo)
# ----------------------------------------------------------------------------------------
def accuracy_vs_truth(votes, y_true):
    """Exatidão de cada anotador face ao ground truth (aba 'Accuracy Ground Truth')."""
    out = np.full(votes.shape[1], np.nan)
    for a in range(votes.shape[1]):
        m = votes[:, a] != MISSING
        if m.any():
            out[a] = (votes[m, a] == y_true[m]).mean()
    return out


def per_annotator_agreement(votes):
    """
    Acordo de cada anotador com os OUTROS anotadores da mesma tarefa (excluindo-se a si próprio),
    média sobre as tarefas em que há pelo menos 2 anotadores (abas 'Partial Krippendorff's').
    """
    P = votes.shape[1]
    ok = (votes != MISSING).sum(axis=1) >= 2
    sums, cnts = np.zeros(P), np.zeros(P)
    for a in range(P):
        rows = np.where(ok & (votes[:, a] != MISSING))[0]
        for i in rows:
            others = np.delete(votes[i], a)
            others = others[others != MISSING]
            sums[a] += (others == votes[i, a]).mean()
            cnts[a] += 1
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)


def majority_agreement(votes, rng=None):
    """Fração das anotações de cada anotador que coincide com o voto de maioria da tarefa."""
    rng = np.random.default_rng(0) if rng is None else rng
    ok = (votes != MISSING).sum(axis=1) >= 2
    maj = majority_vote(votes[ok], rng)
    out = np.full(votes.shape[1], np.nan)
    for a in range(votes.shape[1]):
        m = votes[ok, a] != MISSING
        if m.any():
            out[a] = (votes[ok][m, a] == maj[m]).mean()
    return out


def krippendorff_alpha(votes):
    """
    Alpha de Krippendorff para etiquetas nominais com anotações em falta, com a mesma fórmula
    da folha (aba 'Krippendorffs alpha', pesos de etiqueta = identidade):
        p_i  = sum_k r_ik (r_ik - 1) / (r_medio (r_i - 1))        acordo ponderado da tarefa i
        p_a  = media(p_i) (1 - eps) + eps,   eps = 1 / (n r_medio)
        p_e  = sum_k pi_k^2,                 pi_k = contagem_k / (n r_medio)
        alpha = (p_a - p_e) / (1 - p_e)
    Só entram tarefas com >= 2 anotações. Devolve (alpha, p_a, p_e).
    """
    ok = (votes != MISSING).sum(axis=1) >= 2
    v = votes[ok]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan
    R = np.stack([(v == c).sum(axis=1) for c in range(N_CLASSES)], axis=1)   # (n, K) contagens
    r = R.sum(axis=1)
    r_bar = r.mean()
    p_i = (R * (R - 1)).sum(axis=1) / (r_bar * (r - 1))
    eps = 1.0 / (n * r_bar)
    p_a = p_i.mean() * (1 - eps) + eps
    pi = R.sum(axis=0) / (n * r_bar)
    p_e = (pi ** 2).sum()
    return (p_a - p_e) / (1 - p_e), p_a, p_e


def confusion_per_annotator(votes, y_true, a):
    """Matriz de confusão de um anotador: linhas = real, colunas = etiqueta dada."""
    m = votes[:, a] != MISSING
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(cm, (y_true[m], votes[m, a]), 1)
    return cm


def quality_report(votes, y_true, final_labels=None, names=None):
    """Resumo único das métricas, pronto para ir para uma tabela do relatório."""
    names = PANEL_NAMES if names is None else names
    alpha, p_a, p_e = krippendorff_alpha(votes)
    rep = dict(krippendorff_alpha=alpha, observed_agreement=p_a, expected_agreement=p_e,
               n_annotations=int((votes != MISSING).sum()))
    for nm, v in zip(names, accuracy_vs_truth(votes, y_true)):
        rep[f"acc_{nm}"] = v
    if final_labels is not None:
        rep["label_accuracy_final"] = float((final_labels == y_true).mean())
    return rep


# ----------------------------------------------------------------------------------------
# VALIDAÇÃO CONTRA A FOLHA DE CÁLCULO DO PROFESSOR
# ----------------------------------------------------------------------------------------
def _sheet_example():
    """Dados da folha (Alex, Blake, Cameron, Dancer/Drew, Evan; tarefas 1 a 11)."""
    lab = {"Pedestrian": 0, "Sign": 1, "Cyclist": 2, "Animal": 3}
    rows = [
        ["Pedestrian", "Pedestrian", "Pedestrian", None, None],
        [None, "Sign", "Sign", "Sign", None],
        ["Pedestrian", "Pedestrian", "Cyclist", "Cyclist", "Pedestrian"],
        [None, "Cyclist", "Cyclist", "Cyclist", None],
        ["Pedestrian", "Pedestrian", "Pedestrian", "Pedestrian", None],
        ["Cyclist", "Cyclist", None, None, "Cyclist"],
        ["Pedestrian", "Pedestrian", None, "Pedestrian", None],
        ["Animal", "Animal", None, "Animal", None],
        ["Sign", None, "Animal", "Animal", "Animal"],
        [None, "Sign", "Sign", None, "Sign"],
        [None, "Animal", None, None, None],      # tarefa 11: 1 só anotação (fora do alpha)
    ]
    gt = ["Pedestrian", "Sign", "Cyclist", "Cyclist", "Pedestrian",
          "Cyclist", "Pedestrian", "Animal", "Animal", "Sign", "Animal"]
    votes = np.array([[MISSING if x is None else lab[x] for x in r] for r in rows])
    return votes, np.array([lab[g] for g in gt])


def selftest():
    global N_CLASSES
    old, N_CLASSES = N_CLASSES, 4       # a folha usa 4 etiquetas
    try:
        votes, gt = _sheet_example()
        alpha, p_a, p_e = krippendorff_alpha(votes)
        acc = accuracy_vs_truth(votes, gt)
        agr = per_annotator_agreement(votes)
        print(f"alpha = {alpha:.6f}   (folha: 0.803103)")
        print(f"p_a   = {p_a:.6f}   (folha: 0.857266)   p_e = {p_e:.6f}   (folha: 0.275087)")
        print("exatidão vs GT:", np.round(acc, 6), "  (folha: 0.714286 0.9 1 1 0.75)")
        print("acordo (self-excluded):", np.round(agr, 6))
        assert abs(alpha - 0.803103) < 1e-6 and abs(p_a - 0.857266) < 1e-6
        assert abs(p_e - 0.275087) < 1e-6
        assert np.allclose(acc, [0.714286, 0.9, 1, 1, 0.75], atol=1e-6)
        assert abs(agr[0] - 0.785714) < 1e-6        # 'Annotator Average Agreement' do Alex
        print("OK: coincide com a folha de cálculo.")
    finally:
        N_CLASSES = old


if __name__ == "__main__":
    selftest()
