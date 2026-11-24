"""
Metrics to compare a heat map (Grad-CAM) with the segmentation mask of the lesion.
Depends only on NumPy. Self-test:  python cam_metrics.py

Convention: `cam` (B, H, W) with values >= 0 and a maximum of 1 per image (already after ReLU,
upsampling and normalisation); `mask` (B, H, W) boolean, True = lesion.

Metrics (per image):
  energy_in_mask : fraction of the CAM energy that falls inside the lesion (0..1)
  energy_ratio   : energy_in_mask / fraction of the image covered by the lesion. 1 = the CAM does
                   not separate the lesion from the rest of the image (chance level); > 1 = it
                   focuses on the lesion. Corrects the fact that large lesions get a high
                   energy_in_mask "for free".
  pointing_game  : 1 if the peak of the CAM falls inside the lesion
  iou            : IoU between the binarised CAM (cam >= thr) and the mask
  border_ratio   : (energy on the outer frame) / (fraction of the image covered by the frame).
                   Points to artefacts on the border of the dermatoscope; > 1 = focuses on the
                   border more than chance.
  cam_empty      : 1 if the CAM is all zeros (the ReLU removed everything); the metrics become NaN
"""
import numpy as np


def border_mask(H, W, frac=0.10):
    """Boolean mask of the outer frame of the image (a band of `frac` of the smallest side)."""
    b = max(1, int(round(frac * min(H, W))))
    m = np.ones((H, W), dtype=bool)
    m[b:H - b, b:W - b] = False
    return m


def localisation_metrics(cam, mask, thr=0.5, border_frac=0.10):
    """Computes all the localisation metrics described above, one value per image of the batch."""
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    B, H, W = cam.shape
    flat, mflat = cam.reshape(B, -1), mask.reshape(B, -1)
    total = flat.sum(1)
    empty = total <= 1e-8                                       # CAMs with no energy at all
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
    """Gaussian blob centred at (cy, cx) with maximum 1, used as a synthetic CAM in the self-test."""
    y, x = np.mgrid[:H, :W]
    g = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * s ** 2)))
    return g / g.max()


def selftest():
    """Checks the metrics on synthetic CAMs and the closed formula of Grad-CAM."""
    H = W = 100
    mask = np.zeros((H, W), bool)
    mask[30:70, 30:70] = True                                    # central lesion: 16% of the image
    masks = np.stack([mask] * 3)
    cams = np.stack([
        _blob(H, W, 50, 50, 8),                                  # 0) focuses on the lesion
        np.ones((H, W)),                                         # 1) uniform (chance)
        _blob(H, W, 4, 4, 5),                                    # 2) focuses on the corner/border
    ])
    m = localisation_metrics(cams, masks)
    for k in ("energy_in_mask", "energy_ratio", "pointing_game", "iou", "border_ratio"):
        print(f"{k:15s}", np.round(m[k], 3))
    assert m["energy_in_mask"][0] > 0.95 and m["pointing_game"][0] == 1
    assert abs(m["energy_ratio"][1] - 1.0) < 1e-9, "a uniform CAM must give a ratio of 1"
    assert abs(m["energy_in_mask"][1] - 0.16) < 1e-9
    assert m["pointing_game"][2] == 0 and m["energy_in_mask"][2] < 0.01
    assert m["border_ratio"][2] > 1.5 > m["border_ratio"][0]
    assert m["iou"][0] > 0.05 and m["iou"][2] == 0
    empty = localisation_metrics(np.zeros((1, H, W)), mask[None])
    assert np.isnan(empty["energy_ratio"][0]) and empty["cam_empty"][0] == 1
    # Closed formula of Grad-CAM with average pooling + linear layer: the gradient of logit c with
    # respect to A[k,h,w] is W[c,k]/(h*w) (constant) => alpha_k is proportional to W[c,k].
    rng = np.random.default_rng(0)
    C, h, w = 6, 5, 5
    A, Wt, b = rng.random((C, h, w)), rng.normal(size=(3, C)), rng.normal(size=3)
    logit = lambda A_: Wt[1] @ A_.mean(axis=(1, 2)) + b[1]
    grad = np.zeros_like(A)
    for idx in np.ndindex(A.shape):                              # numerical derivative
        e = np.zeros_like(A); e[idx] = 1e-6
        grad[idx] = (logit(A + e) - logit(A - e)) / 2e-6
    alpha = grad.mean(axis=(1, 2))
    assert np.allclose(alpha, Wt[1] / (h * w), atol=1e-6)        # alpha_k = W[c,k] / (h*w)
    pre_gc = (alpha[:, None, None] * A).sum(0)                   # Grad-CAM before the ReLU
    pre_cf = np.einsum("chw,c->hw", A, Wt[1])                    # closed formula (CAM)
    assert np.allclose(pre_gc * (h * w), pre_cf, atol=1e-5)      # equal up to the factor 1/(h*w)
    print("OK: metrics and closed formula of Grad-CAM verified.")


if __name__ == "__main__":
    selftest()
