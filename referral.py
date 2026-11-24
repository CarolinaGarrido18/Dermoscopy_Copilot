"""
Phase 4: Who decides (Referral / Selective Prediction).

Loads the final active learning model (`al_head.pt`, from export_al_head.py), calibrates the
probabilities on the validation set (temperature scaling) and evaluates the clinical referral rule:
  confidence < X  OR  P(melanoma) >= 0.20  ->  sent to the human (in the loop)
  otherwise                                ->  automatic decision under supervision (on the loop)

Usage: python3 referral.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from active_learning import load_split

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX = CLASSES.index("mel")  # index 4
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]   # confidence thresholds X that are evaluated
MEL_RISK_THR = 0.20   # melanoma probability from which the case is always referred


def load_baseline_head(path="baseline_head.pt"):
    """Loads the weights (W) and bias (b) of a saved linear head (here it is used for al_head.pt).

    If the given path does not exist, it falls back to the baseline head of the backbone folder.
    """
    if not os.path.exists(path):
        alt = "results_tf_efficientnetv2_s.in21k_ft_in1k/baseline_head.pt"
        path = alt if os.path.exists(alt) else path
    ckpt = torch.load(path, map_location="cpu")
    st = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    W = st["weight"].float()
    b = st["bias"].float()
    return W, b


def get_logits(X, W, b):
    """Computes the logits (supports TTA if X has 3 dimensions: N, V, D, by averaging the views)."""
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        if Xt.ndim == 3:
            N, V, D = Xt.shape
            logits = (Xt.reshape(N * V, D) @ W.T + b).reshape(N, V, -1).mean(1)
        else:
            logits = Xt @ W.T + b
    return logits


def fit_temperature(logits_val, y_val):
    """Calibration with temperature scaling (minimises the NLL on the validation set)."""
    T = nn.Parameter(torch.ones(1))
    y_t = torch.tensor(y_val, dtype=torch.long)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_val / T.clamp(min=1e-3), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.clamp(min=1e-3).item())


def compute_ece(probs, y_true, n_bins=15):
    """Expected Calibration Error (ECE): weighted gap between confidence and accuracy over the bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def evaluate_referral(probs, y_true, split_name):
    """Applies the referral rule for every threshold X and counts the workload and the errors."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    p_mel = probs[:, MEL_IDX]
    is_mel = y_true == MEL_IDX
    total_mel = int(is_mel.sum())

    rows = []
    for X_thr in THRESHOLDS:
        to_human = (conf < X_thr) | (p_mel >= MEL_RISK_THR)
        to_ai = ~to_human

        n_total = len(y_true)
        n_human = int(to_human.sum())
        n_ai = int(to_ai.sum())

        # Errors in the cases kept by the AI (on the loop)
        ai_errors = int((pred[to_ai] != y_true[to_ai]).sum())
        ai_err_rate_on_kept = (ai_errors / n_ai) if n_ai > 0 else 0.0
        ai_err_rate_overall = ai_errors / n_total

        # Melanomas missed by the AI (real melanoma kept by the AI and not predicted as mel)
        missed_mel = int((to_ai & is_mel & (pred != MEL_IDX)).sum())
        missed_mel_pct = (missed_mel / total_mel) if total_mel > 0 else 0.0

        rows.append({
            "split": split_name,
            "conf_threshold_X": X_thr,
            "pct_sent_human": round(100 * n_human / n_total, 2),
            "n_sent_human": n_human,
            "n_kept_ai": n_ai,
            "ai_errors": ai_errors,
            "ai_error_rate_kept_pct": round(100 * ai_err_rate_on_kept, 2),
            "ai_error_rate_total_pct": round(100 * ai_err_rate_overall, 2),
            "missed_melanomas": missed_mel,
            "total_melanomas": total_mel,
            "missed_mel_pct": round(100 * missed_mel_pct, 2),
        })
    return pd.DataFrame(rows)


def main():
    """Loads the model, calibrates it, evaluates the rule on validation and test and saves the results."""
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Load the final active learning model (the same one used by the app) and the splits
    W, b = load_baseline_head("results/results_active_learning/al_head.pt")
    Xva, yva, _, _ = load_split("val")
    Xte, yte, _, _ = load_split("test")

    logits_va = get_logits(Xva, W, b)
    logits_te = get_logits(Xte, W, b)

    # 2. Calibration on the validation set
    probs_va_raw = torch.softmax(logits_va, dim=1).numpy()
    probs_te_raw = torch.softmax(logits_te, dim=1).numpy()

    T = fit_temperature(logits_va, yva)
    probs_va = torch.softmax(logits_va / T, dim=1).numpy()
    probs_te = torch.softmax(logits_te / T, dim=1).numpy()

    print(f"Temperature fitted on validation: T = {T:.4f}")
    print(f"ECE Validation: before = {compute_ece(probs_va_raw, yva):.4f} -> after = {compute_ece(probs_va, yva):.4f}")
    print(f"ECE Test      : before = {compute_ece(probs_te_raw, yte):.4f} -> after = {compute_ece(probs_te, yte):.4f}\n")

    # 3. Evaluate the thresholds X on validation and test
    df_va = evaluate_referral(probs_va, yva, "val")
    df_te = evaluate_referral(probs_te, yte, "test")
    res = pd.concat([df_va, df_te], ignore_index=True)
    res.to_csv("results/tables/referral_results.csv", index=False)

    print("=== VALIDATION ===")
    print(df_va.to_string(index=False))
    print("\n=== TEST ===")
    print(df_te.to_string(index=False))

    # 4. Plot "% sent to the human vs errors"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for df_sub, title in [(df_va, "Validation"), (df_te, "Test")]:
        ax = axes[0] if title == "Validation" else axes[1]
        ax2 = ax.twinx()

        l1 = ax.plot(
            df_sub["pct_sent_human"], df_sub["ai_error_rate_total_pct"],
            "o-", color="tab:blue", label="Total AI error rate (%)"
        )
        l2 = ax2.plot(
            df_sub["pct_sent_human"], df_sub["missed_melanomas"],
            "s--", color="tab:red", label="Melanomas missed by the AI (n)"
        )

        for _, r in df_sub.iterrows():
            ax.annotate(
                f"X={int(r['conf_threshold_X']*100)}%",
                (r["pct_sent_human"], r["ai_error_rate_total_pct"]),
                textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8
            )

        ax.set_title(f"Referral: human workload vs. risk ({title})")
        ax.set_xlabel("% of cases referred to the human (in the loop)")
        ax.set_ylabel("AI errors on the cases it keeps (% of total)", color="tab:blue")
        ax2.set_ylabel("Melanomas missed by the AI (n)", color="tab:red")
        ax.grid(alpha=0.3)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("results/figures/referral_tradeoff.png", dpi=200)
    print("\nSaved: results/tables/referral_results.csv and results/figures/referral_tradeoff.png")


if __name__ == "__main__":
    main()