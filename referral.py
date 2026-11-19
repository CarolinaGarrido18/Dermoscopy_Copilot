"""
Fase 4: Quem decide (Referral / Selective Prediction).

Carrega o modelo de referência (`baseline_head.pt`), calibra as probabilidades na
validação (Temperature Scaling) e avalia a regra de encaminhamento clínico:
  confiança < X  OU  P(melanoma) >= 0.20  ->  enviado para o humano (in the loop)
  caso contrário                          ->  decisão automática com supervisão (on the loop)

Uso: python3 referral.py
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
MEL_IDX = CLASSES.index("mel")  # índice 4
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
MEL_RISK_THR = 0.20


def load_baseline_head(path="baseline_head.pt"):
    if not os.path.exists(path):
        alt = "results_tf_efficientnetv2_s.in21k_ft_in1k/baseline_head.pt"
        path = alt if os.path.exists(alt) else path
    ckpt = torch.load(path, map_location="cpu")
    st = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    W = st["weight"].float()
    b = st["bias"].float()
    return W, b


def get_logits(X, W, b):
    """Calcula logits (suporta TTA se X tiver 3 dimensões: N, V, D)."""
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        if Xt.ndim == 3:
            N, V, D = Xt.shape
            logits = (Xt.reshape(N * V, D) @ W.T + b).reshape(N, V, -1).mean(1)
        else:
            logits = Xt @ W.T + b
    return logits


def fit_temperature(logits_val, y_val):
    """Calibração por Temperature Scaling (otimiza NLL na validação)."""
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
    """Expected Calibration Error (ECE)."""
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

        # Erros nos casos que ficam com a IA (on the loop)
        ai_errors = int((pred[to_ai] != y_true[to_ai]).sum())
        ai_err_rate_on_kept = (ai_errors / n_ai) if n_ai > 0 else 0.0
        ai_err_rate_overall = ai_errors / n_total

        # Melanomas falhados pela IA (melanoma real que ficou na IA e não foi previsto como mel)
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
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Carregar modelo de referência e splits
    W, b = load_baseline_head()
    Xva, yva, _, _ = load_split("val")
    Xte, yte, _, _ = load_split("test")

    logits_va = get_logits(Xva, W, b)
    logits_te = get_logits(Xte, W, b)

    # 2. Calibração na validação
    probs_va_raw = torch.softmax(logits_va, dim=1).numpy()
    probs_te_raw = torch.softmax(logits_te, dim=1).numpy()

    T = fit_temperature(logits_va, yva)
    probs_va = torch.softmax(logits_va / T, dim=1).numpy()
    probs_te = torch.softmax(logits_te / T, dim=1).numpy()

    print(f"Temperatura aprendida na validação: T = {T:.4f}")
    print(f"ECE Validação : antes = {compute_ece(probs_va_raw, yva):.4f} -> depois = {compute_ece(probs_va, yva):.4f}")
    print(f"ECE Teste     : antes = {compute_ece(probs_te_raw, yte):.4f} -> depois = {compute_ece(probs_te, yte):.4f}\n")

    # 3. Avaliar limiares X na validação e no teste
    df_va = evaluate_referral(probs_va, yva, "val")
    df_te = evaluate_referral(probs_te, yte, "test")
    res = pd.concat([df_va, df_te], ignore_index=True)
    res.to_csv("results/tables/referral_results.csv", index=False)

    print("=== VALIDAÇÃO ===")
    print(df_va.to_string(index=False))
    print("\n=== TESTE ===")
    print(df_te.to_string(index=False))

    # 4. Gráfico "% enviados ao humano vs erros"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for df_sub, title in [(df_va, "Validação"), (df_te, "Teste")]:
        ax = axes[0] if title == "Validação" else axes[1]
        ax2 = ax.twinx()

        l1 = ax.plot(
            df_sub["pct_sent_human"], df_sub["ai_error_rate_total_pct"],
            "o-", color="tab:blue", label="Taxa de erro total da IA (%)"
        )
        l2 = ax2.plot(
            df_sub["pct_sent_human"], df_sub["missed_melanomas"],
            "s--", color="tab:red", label="Melanomas falhados pela IA (nº)"
        )

        for _, r in df_sub.iterrows():
            ax.annotate(
                f"X={int(r['conf_threshold_X']*100)}%",
                (r["pct_sent_human"], r["ai_error_rate_total_pct"]),
                textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8
            )

        ax.set_title(f"Referral: Custo Humano vs. Risco ({title})")
        ax.set_xlabel("% de casos enviados ao humano (in the loop)")
        ax.set_ylabel("Erros nos casos mantidos na IA (% do total)", color="tab:blue")
        ax2.set_ylabel("Melanomas falhados pela IA (nº absoluto)", color="tab:red")
        ax.grid(alpha=0.3)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("results/figures/referral_tradeoff.png", dpi=200)
    print("\nGuardado: results/tables/referral_results.csv e results/figures/referral_tradeoff.png")


if __name__ == "__main__":
    main()