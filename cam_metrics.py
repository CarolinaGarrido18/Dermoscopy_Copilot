"""
Métricas para comparar um heatmap (Grad-CAM) com a máscara de segmentação da lesão.
Só depende de NumPy. Autoteste:  python cam_metrics.py

Convenção: `cam` (B, H, W) com valores >= 0 e máximo 1 por imagem (já após ReLU, upsample e
normalização); `mask` (B, H, W) booleano, True = lesão.

Métricas (por imagem):
  energy_in_mask : fração da energia do CAM que cai dentro da lesão (0..1)
  energy_ratio   : energy_in_mask / fração de área da lesão. 1 = o CAM não distingue a lesão
                   do resto da imagem (nível do acaso); > 1 = concentra-se na lesão. Corrige o
                   facto de lesões grandes terem energy_in_mask alto "de graça".
  pointing_game  : 1 se o pico do CAM cai dentro da lesão
  iou            : IoU entre o CAM binarizado (cam >= thr) e a máscara
  border_ratio   : (energia na moldura exterior) / (fração de área da moldura). Aponta para
                   artefactos do bordo do dermatoscópio; > 1 = foca mais o bordo do que o acaso.
  cam_empty      : 1 se o CAM é todo zero (a ReLU anulou tudo); as métricas ficam NaN
"""
import numpy as np


def border_mask(H, W, frac=0.10):
    b = max(1, int(round(frac * min(H, W))))
    m = np.ones((H, W), dtype=bool)
    m[b:H - b, b:W - b] = False
    return m


def localisation_metrics(cam, mask, thr=0.5, border_frac=0.10):
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    B, H, W = cam.shape
    flat, mflat = cam.reshape(B, -1), mask.reshape(B, -1)
    total = flat.sum(1)
    empty = total <= 1e-8
    safe = np.maximum(total, 1e-8)

    energy = (flat * mflat).sum(1) / safe
    area = mflat.mean(1)
    ratio = np.where(area > 0, energy / np.maximum(area, 1e-8), np.nan)

    point = mflat[np.arange(B), flat.argmax(1)].astype(float)

    binm = cam >= thr
    inter = (binm & mask).reshape(B, -1).sum(1)
    union = (binm | mask).reshape(B, -1).sum(1)
    iou = inter / np.maximum(union, 1)

    bm = border_mask(H, W, border_frac)
    e_border = (flat * bm.reshape(1, -1)).sum(1) / safe
    b_ratio = e_border / bm.mean()

    out = dict(energy_in_mask=energy, energy_ratio=ratio, pointing_game=point, iou=iou,
               border_ratio=b_ratio, mask_area=area, cam_empty=empty.astype(float))
    for k in ("energy_in_mask", "energy_ratio", "pointing_game", "iou", "border_ratio"):
        out[k] = np.where(empty, np.nan, out[k])
    return out


# ------------------------------------------------------------------------------------------
def _blob(H, W, cy, cx, s):
    y, x = np.mgrid[:H, :W]
    g = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * s ** 2)))
    return g / g.max()


def selftest():
    H = W = 100
    mask = np.zeros((H, W), bool)
    mask[30:70, 30:70] = True                                    # lesão central: 16% da imagem
    masks = np.stack([mask] * 3)
    cams = np.stack([
        _blob(H, W, 50, 50, 8),                                  # 0) foca a lesão
        np.ones((H, W)),                                         # 1) uniforme (acaso)
        _blob(H, W, 4, 4, 5),                                    # 2) foca o canto/bordo
    ])
    m = localisation_metrics(cams, masks)
    for k in ("energy_in_mask", "energy_ratio", "pointing_game", "iou", "border_ratio"):
        print(f"{k:15s}", np.round(m[k], 3))
    assert m["energy_in_mask"][0] > 0.95 and m["pointing_game"][0] == 1
    assert abs(m["energy_ratio"][1] - 1.0) < 1e-9, "um CAM uniforme tem de dar razão 1"
    assert abs(m["energy_in_mask"][1] - 0.16) < 1e-9
    assert m["pointing_game"][2] == 0 and m["energy_in_mask"][2] < 0.01
    assert m["border_ratio"][2] > 1.5 > m["border_ratio"][0]
    assert m["iou"][0] > 0.05 and m["iou"][2] == 0
    empty = localisation_metrics(np.zeros((1, H, W)), mask[None])
    assert np.isnan(empty["energy_ratio"][0]) and empty["cam_empty"][0] == 1
    # Fórmula fechada do Grad-CAM com pooling médio + camada linear: o gradiente do logit c em
    # relação a A[k,h,w] é W[c,k]/(h*w) (constante) => alpha_k proporcional a W[c,k].
    rng = np.random.default_rng(0)
    C, h, w = 6, 5, 5
    A, Wt, b = rng.random((C, h, w)), rng.normal(size=(3, C)), rng.normal(size=3)
    logit = lambda A_: Wt[1] @ A_.mean(axis=(1, 2)) + b[1]
    grad = np.zeros_like(A)
    for idx in np.ndindex(A.shape):                              # derivada numérica
        e = np.zeros_like(A); e[idx] = 1e-6
        grad[idx] = (logit(A + e) - logit(A - e)) / 2e-6
    alpha = grad.mean(axis=(1, 2))
    assert np.allclose(alpha, Wt[1] / (h * w), atol=1e-6)        # alpha_k = W[c,k] / (h*w)
    pre_gc = (alpha[:, None, None] * A).sum(0)                   # Grad-CAM antes da ReLU
    pre_cf = np.einsum("chw,c->hw", A, Wt[1])                    # fórmula fechada (CAM)
    assert np.allclose(pre_gc * (h * w), pre_cf, atol=1e-5)      # iguais a menos de 1/(h*w)
    print("OK: métricas e fórmula fechada do Grad-CAM verificadas.")


if __name__ == "__main__":
    selftest()
