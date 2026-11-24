"""
Dermoscopy Copilot - Streamlit       
Run (from the repository root): streamlit run app.py
Prediction = final active learning model (al_head.pt), calibrated, with TTA x8 and the same calibration as referral.py.
Heat map = CAM of the predicted class of that same model over the original view (closed formula of gradcam_eval.py).
Prerequisite: results/results_active_learning/al_head.pt (export_al_head.py).
"""
import csv
import time

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from copilot_common import (CLASSES, IMG_DIR, LOG_CSV, calibrate, device, load_backbone,
                            load_head, vistas_tta)

# referral.py: low confidence OR melanoma risk -> human in the loop
CONF_X = 0.80  # minimum confidence for the model to be trusted on its own    
MEL_RISK_THR = 0.20  # melanoma probability from which the case is always referred
MEL_IDX = CLASSES.index("mel") # melanoma probability from which the case is always referred
# Columns of the feedback log (one row per clinician decision)
LOG_COLS = ["timestamp", "image_file", "pred_class", "confidence", "mel_prob",
            "sent_to_human", "action", "final_label", "disagreement"]
IMG_DIR.mkdir(parents=True, exist_ok=True) # folder where the images reviewed by the clinician are saved

@st.cache_resource
def load_all():
    #Loads the frozen backbone, the active learning head and its temperature (T).
    # Cached by Streamlit, so the models are loaded only once and not at every interaction with the page.
    bb, base_tf, _, size = load_backbone()
    head = load_head()
    return bb, base_tf, size, head, calibrate(head)


@torch.no_grad()
def predict(pil):
    # Returns the calibrated class probabilities and the feature maps used for the heat map.
    # The probabilities come from the 8 TTA views (4 rotations x horizontal flip): the logits of the views are
    # averaged and then divided by the temperature T (temperature scaling) before the softmax.
    x = base_tf(pil).unsqueeze(0).to(device)
    feats = bb(torch.cat(vistas_tta(x))).float()                  # (8, D)
    logits = head(feats).mean(0)                                  # mean of the logits of the 8 views
    probs = torch.softmax(logits / T, dim=0).cpu().numpy()
    A = bb.forward_features(x).float()                            # (1, C, h, w), original view
    return probs, A


@torch.no_grad()
def cam_overlay(A, pred, pil):
    # Builds the heat map (CAM) of the predicted class and overlays it on the original image.
    # The CAM is the weighted sum of the backbone feature maps, using the head weights of the predicted class.
    # It is then resized to the image size, normalised to [0, 1] and blended with the image (55% image, 45% colour map).
    cam = torch.relu(torch.einsum("bchw,c->bhw", A, head.weight[pred]))
    cam = F.interpolate(cam[:, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam / cam.max().clamp_min(1e-8)).cpu().numpy()
    img = np.asarray(pil.resize((size, size))).astype(np.float32) / 255.0
    return (0.55 * img + 0.45 * cm.jet(cam)[..., :3]).clip(0, 1)


def log_row(row):
    # Appends one clinician decision to feedback_log.csv (the header is written only when the file is created).
    new = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            wr.writeheader()
        wr.writerow(row)

# Page layout
st.set_page_config(page_title="Dermoscopy Copilot", layout="wide")
st.title("Dermoscopy Copilot")
st.caption("Decision support tool: the final decision always belongs to the clinician.")

bb, base_tf, size, head, T = load_all()

# Image upload: nothing else is shown until an image is provided
up = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Upload an image to get started.")
    st.stop()

pil = Image.open(up).convert("RGB")
# Streamlit reruns the script at every click, so the current image is tracked in the session state:
# when a new image is uploaded, the "already saved" flag is reset and the buttons are enabled again
key = f"{up.name}-{up.size}"
if st.session_state.get("current") != key:
    st.session_state.current, st.session_state.saved = key, False

# Prediction and referral rule: the case goes to the human if the model is not confident enough
# or if the melanoma probability is high (missing a melanoma is the most costly error)
probs, A = predict(pil)
pred = int(probs.argmax())
human = bool(probs.max() < CONF_X or probs[MEL_IDX] >= MEL_RISK_THR)

# Original image and heat map side by side
c1, c2 = st.columns(2)
c1.subheader("Image")
c1.image(pil, use_container_width=True)
c2.subheader("Heat map (Grad-CAM)")
c2.image(cam_overlay(A, pred, pil), use_container_width=True)

# Predicted class, confidence and referral message
st.markdown(f"### Predicted class: **{CLASSES[pred]}** — confidence {probs[pred]:.1%}")
st.caption(f"P(melanoma) = {probs[MEL_IDX]:.1%}")
if human:
    st.warning("⚠️ Uncertain case, review needed. Sent to a human (human in the loop).")
else:
    st.info("Model suggestion, under clinician supervision (human on the loop).")
st.bar_chart(pd.DataFrame({"prob": probs}, index=CLASSES)) # probability of each of the 7 classes

# Clinician decision

st.divider()
st.subheader("Clinician decision")
corrected = st.selectbox("Correct class (if you disagree):", CLASSES, index=pred)


def save(action, final_label):
    # Saves the reviewed image and logs the decision.
    # action: "accepted" (clinician agrees with the model) or "corrected" (clinician chose another class).
    # final_label: the class confirmed by the clinician, used as the label when the model is retrained.
    # disagreement is 1 when the final label differs from the model prediction.
    fname = f"{int(time.time())}_{up.name}" # timestamp prefix avoids overwriting images with the same name
    pil.save(IMG_DIR / fname)
    log_row({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "image_file": fname,
             "pred_class": CLASSES[pred], "confidence": round(float(probs[pred]), 4),
             "mel_prob": round(float(probs[MEL_IDX]), 4), "sent_to_human": int(human),
             "action": action, "final_label": final_label,
             "disagreement": int(final_label != CLASSES[pred])})
    st.session_state.saved = True         # disables the buttons so the image is saved only once


b1, b2 = st.columns(2)
if b1.button("✅ Accept prediction", disabled=st.session_state.saved):
    save("accepted", CLASSES[pred])
if b2.button("✏️ Correct", disabled=st.session_state.saved):
    save("corrected", corrected)
if st.session_state.saved:
    st.success(f"Saved to {LOG_CSV}")

# Table with all the decisions registered so far
if LOG_CSV.exists():
    with st.expander("Log (feedback_log.csv)"):
        st.dataframe(pd.read_csv(LOG_CSV))
