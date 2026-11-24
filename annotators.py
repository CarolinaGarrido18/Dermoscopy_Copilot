"""
Simulated annotators (dermatologists) and annotation quality metrics.

"Human-in-the-Loop" part of the project. Instead of a perfect oracle (or one with uniform noise),
we simulate a PANEL of dermatologists, each with their own confusion matrix, and measure the
quality of the annotations with the metrics of the spreadsheet "Human-in-the-Loop Machine Learning:
annotation quality frameworks" (Munro): accuracy against the ground truth, agreement between
annotators, agreement with the majority, Krippendorff's alpha and confusion matrix per annotator.

Depends only on NumPy. To validate against the spreadsheet: python annotators.py
"""
import numpy as np

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
N_CLASSES = len(CLASSES)
MISSING = -1          # "this annotator did not label this image"

# ----------------------------------------------------------------------------------------
# SIMULATION ASSUMPTIONS (edit here; they must be justified/calibrated in the report)
# ----------------------------------------------------------------------------------------
# Overall accuracy of each dermatologist of the panel. ILLUSTRATIVE values, not taken from the
# literature: replace with cited values (or run a sensitivity analysis).
PANEL_ACCURACY = [0.92, 0.88, 0.85, 0.80, 0.75]
PANEL_NAMES = [f"Derm {i + 1}" for i in range(len(PANEL_ACCURACY))]

# Error structure: when a dermatologist makes a mistake, which class do they tend to pick?
# Relative weights (rows = true class). Inspired by the errors in the confusion matrix of the
# model (mel<->nv, bkl<->mel, akiec<->bcc/bkl) and by common dermoscopic confusions.
# It is a modelling assumption, not an empirical result.
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
    """7x7 matrix with the distribution of the errors for each true class (diagonal = 0)."""
    S = np.zeros((N_CLASSES, N_CLASSES))
    for true_c, targets in _CONFUSION_WEIGHTS.items():
        for wrong_c, w in targets.items():
            S[CLASSES.index(true_c), CLASSES.index(wrong_c)] = w
    return S / S.sum(axis=1, keepdims=True)


def annotator_confusion(accuracy):
    """Confusion matrix of one annotator: P(given label | true class)."""
    C = (1 - accuracy) * error_structure()
    C[np.diag_indices(N_CLASSES)] = accuracy
    return C


def make_panel(accuracies=None):
    """List of confusion matrices, one per dermatologist of the panel."""
    accuracies = PANEL_ACCURACY if accuracies is None else accuracies
    return [annotator_confusion(a) for a in accuracies]


# ----------------------------------------------------------------------------------------
# ORACLES
# ----------------------------------------------------------------------------------------
def _sample_labels(C, y_true, rng):
    """One label per image, sampled from the confusion matrix C."""
    cdf = C[y_true].cumsum(axis=1)
    u = rng.random(len(y_true))[:, None]
    return np.minimum((u > cdf).sum(axis=1), N_CLASSES - 1)


class Oracle:
    """
    Answers the requests of the active learning loop. kind:
      perfect      : correct label (what the original code did)
      uniform      : a fraction `p` of the images gets a random wrong class (uniform noise)
      single       : each image is seen by ONE dermatologist of the panel, chosen at random
      majority     : each image is seen by `k` different dermatologists and the majority vote wins
    annotate() returns (final labels, votes (n, n_panel) with MISSING, number of annotations).
    The human cost is the number of annotations (an image seen by 3 doctors costs 3).
    """

    def __init__(self, kind, rng, panel=None, k=3, p=0.10):
        assert kind in ("perfect", "uniform", "single", "majority")
        self.kind, self.rng, self.k, self.p = kind, rng, k, p
        self.panel = make_panel() if panel is None else panel
        self.P = len(self.panel)

    def annotate(self, y_true):
        """Labels the images whose true classes are y_true, according to the type of oracle."""
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
        # k different annotators per image, chosen at random
        chosen = np.argsort(self.rng.random((n, self.P)), axis=1)[:, :k]
        for a in range(self.P):
            rows = np.where((chosen == a).any(axis=1))[0]
            if len(rows):
                votes[rows, a] = _sample_labels(self.panel[a], y_true[rows], self.rng)
        return majority_vote(votes, self.rng), votes, n * k


def majority_vote(votes, rng):
    """Majority vote per image; ties are broken at random between the tied classes."""
    counts = np.stack([(votes == c).sum(axis=1) for c in range(N_CLASSES)], axis=1)
    return (counts + rng.random(counts.shape) * 0.5).argmax(axis=1)   # noise < 1: it only breaks ties


# ----------------------------------------------------------------------------------------
# ANNOTATION QUALITY METRICS (equivalent to the tabs of the spreadsheet)
# ----------------------------------------------------------------------------------------
def accuracy_vs_truth(votes, y_true):
    """Accuracy of each annotator against the ground truth (tab 'Accuracy Ground Truth')."""
    out = np.full(votes.shape[1], np.nan)
    for a in range(votes.shape[1]):
        m = votes[:, a] != MISSING
        if m.any():
            out[a] = (votes[m, a] == y_true[m]).mean()
    return out


def per_annotator_agreement(votes):
    """
    Agreement of each annotator with the OTHER annotators of the same task (excluding themselves),
    averaged over the tasks with at least 2 annotators (tabs 'Partial Krippendorff's').
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
    """Fraction of the annotations of each annotator that match the majority vote of the task."""
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
    Krippendorff's alpha for nominal labels with missing annotations, with the same formula as
    the spreadsheet (tab 'Krippendorffs alpha', label weights = identity):
        p_i  = sum_k r_ik (r_ik - 1) / (r_mean (r_i - 1))         weighted agreement of task i
        p_a  = mean(p_i) (1 - eps) + eps,    eps = 1 / (n r_mean)
        p_e  = sum_k pi_k^2,                 pi_k = count_k / (n r_mean)
        alpha = (p_a - p_e) / (1 - p_e)
    Only tasks with >= 2 annotations are used. Returns (alpha, p_a, p_e).
    """
    ok = (votes != MISSING).sum(axis=1) >= 2
    v = votes[ok]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan
    R = np.stack([(v == c).sum(axis=1) for c in range(N_CLASSES)], axis=1)   # (n, K) counts
    r = R.sum(axis=1)
    r_bar = r.mean()
    p_i = (R * (R - 1)).sum(axis=1) / (r_bar * (r - 1))
    eps = 1.0 / (n * r_bar)
    p_a = p_i.mean() * (1 - eps) + eps
    pi = R.sum(axis=0) / (n * r_bar)
    p_e = (pi ** 2).sum()
    return (p_a - p_e) / (1 - p_e), p_a, p_e


def confusion_per_annotator(votes, y_true, a):
    """Confusion matrix of one annotator: rows = true class, columns = given label."""
    m = votes[:, a] != MISSING
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(cm, (y_true[m], votes[m, a]), 1)
    return cm


def quality_report(votes, y_true, final_labels=None, names=None):
    """Single summary of the metrics, ready to go into a table of the report."""
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
# VALIDATION AGAINST THE PROFESSOR'S SPREADSHEET
# ----------------------------------------------------------------------------------------
def _sheet_example():
    """Data of the spreadsheet (Alex, Blake, Cameron, Dancer/Drew, Evan; tasks 1 to 11)."""
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
        [None, "Animal", None, None, None],      # task 11: only 1 annotation (left out of alpha)
    ]
    gt = ["Pedestrian", "Sign", "Cyclist", "Cyclist", "Pedestrian",
          "Cyclist", "Pedestrian", "Animal", "Animal", "Sign", "Animal"]
    votes = np.array([[MISSING if x is None else lab[x] for x in r] for r in rows])
    return votes, np.array([lab[g] for g in gt])


def selftest():
    """Checks the metrics against the values of the spreadsheet."""
    global N_CLASSES
    old, N_CLASSES = N_CLASSES, 4       # the spreadsheet uses 4 labels
    try:
        votes, gt = _sheet_example()
        alpha, p_a, p_e = krippendorff_alpha(votes)
        acc = accuracy_vs_truth(votes, gt)
        agr = per_annotator_agreement(votes)
        print(f"alpha = {alpha:.6f}   (sheet: 0.803103)")
        print(f"p_a   = {p_a:.6f}   (sheet: 0.857266)   p_e = {p_e:.6f}   (sheet: 0.275087)")
        print("accuracy vs GT:", np.round(acc, 6), "  (sheet: 0.714286 0.9 1 1 0.75)")
        print("agreement (self-excluded):", np.round(agr, 6))
        assert abs(alpha - 0.803103) < 1e-6 and abs(p_a - 0.857266) < 1e-6
        assert abs(p_e - 0.275087) < 1e-6
        assert np.allclose(acc, [0.714286, 0.9, 1, 1, 0.75], atol=1e-6)
        assert abs(agr[0] - 0.785714) < 1e-6        # 'Annotator Average Agreement' of Alex
        print("OK: matches the spreadsheet.")
    finally:
        N_CLASSES = old


if __name__ == "__main__":
    selftest()
