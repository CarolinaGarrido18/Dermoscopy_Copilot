# Dermoscopy Copilot

**Course:** Human-Centered Artificial Intelligence (HCAI) · MSc in Biomedical Engineering · University of Coimbra
· **Group 13:** Carolina Garrido, Sara Santos, Sara Sêco

A human-centered decision-support tool for the classification of dermoscopic images into the 7 diagnostic categories of [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). The project combines:

- **Active learning**, to reduce the number of images that dermatologists have to label;
- **Human-in-the-loop annotation**, with a simulated panel of dermatologists who make realistic mistakes;
- **Explainability**, with Grad-CAM heatmaps checked against the lesion masks;
- **A referral rule**, which decides which cases are sent to the clinician and which are only supervised;
- **A Streamlit app**, where the clinician sees the prediction, its confidence and the heatmap, accepts or corrects it, and the corrections can be used to retrain the model.

The final decision always belongs to the clinician: the system is a decision-support tool, not an autonomous diagnostic tool.

---

## Pipeline

| Phase | What it does | Scripts | Notebook |
|---|---|---|---|
| 1. Data and splits | Downloads HAM10000 and the lesion masks, and splits the images into train / validation / test (~70/15/15), stratified by diagnosis and grouped by lesion to avoid data leakage | `prepare_data.py` | `01_data_and_splits` |
| 2. Feature extraction | Extracts features with a frozen pre-trained backbone (`tf_efficientnetv2_s.in21k_ft_in1k`): original + 2 augmented copies per training image, and 8 TTA views per validation/test image | `extract_features.py` | `02_feature_extraction` |
| 3. Passive baseline | Trains a linear layer on 100% of the labels, with class weights, early stopping, TTA and temperature scaling | `train_baseline.py` | `03_passive_baseline` |
| 4. Active learning | Compares 5 query strategies (least confidence, margin, entropy, hybrid and random) from 10% to ~52% of the labels, with 10 seeds, statistical tests and a noisy oracle | `active_learning.py`, `al_results.py`, `al_stats.py`, `al_labels_to_target.py`, `al_final.py` | `04_active_learning` |
| 5. Human-in-the-loop | Replaces the oracle with a simulated panel of 5 dermatologists (one dermatologist or majority vote of 3), measures annotation quality and cost, exports the final model and evaluates the referral rule | `annotators.py`, `al_hitl.py`, `hitl_extra_stats.py`, `export_al_head.py`, `referral.py` | `05_human_in_the_loop` |
| 6. Explainability | Compares the Grad-CAM heatmaps of the passive and active learning models with the lesion masks on the test set | `cam_metrics.py`, `gradcam_eval.py` | `06_gradcam` |
| 7. Copilot app | Streamlit app with the active learning model retrained with the clinician's corrections, and the retraining script | `copilot_common.py`, `app.py`, `retrain_with_corrections.py` | `07_copilot_app` |

The notebooks document each phase and load the saved results in `results/`, so they run in a few seconds without training anything. Each notebook has a `RETRAIN` / `RECOMPUTE` option to run the experiments again.

---

## Repository structure

```
├── prepare_data.py                 # 1. download HAM10000 and create the splits
├── extract_features.py             # 2. frozen backbone features (+ TTA)
├── train_baseline.py               # 3. passive baseline
├── active_learning.py              # 4. active learning loop and query strategies
├── al_results.py / al_stats.py / al_labels_to_target.py / al_final.py
├── annotators.py                   # 5. simulated dermatologists and annotation quality metrics
├── al_hitl.py / hitl_extra_stats.py
├── export_al_head.py               #    final active learning model (starting point of the retraining)
├── referral.py                     #    referral rule (human-in-the-loop / human-on-the-loop)
├── cam_metrics.py / gradcam_eval.py  # 6. Grad-CAM vs lesion masks
├── copilot_common.py               # 7. code shared by the app and the retraining script
├── app.py                          #    Streamlit app
├── retrain_with_corrections.py     #    retraining with the clinician's corrections
├── notebooks/                      # 01 to 07, one per phase
├── splits/                         # train.csv, val.csv, test.csv
├── app_data/                       # images reviewed in the app and feedback_log.csv (used to retrain the app model)
├── results/
│   ├── results_baseline/           # baseline head and metrics for each backbone tested
│   ├── results_active_learning/    # final active learning model and retrained model used by the app
│   ├── heads/                      # active learning heads saved at every cycle
│   ├── tables/                     # all result tables (CSV)
│   └── figures/                    # all result figures
├── requirements.txt
└── README.md
```

Not included in the repository (created when the code is run):

- `features/`: extracted features (created by `extract_features.py`, listed in `.gitignore`).

`app_data/` is created by `app.py`. The version in the repository holds the 21 decisions used to retrain the model of the app (see *Main results*); new decisions made in the app are appended to `feedback_log.csv`.

---

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The HAM10000 images and lesion masks are downloaded automatically with `kagglehub` the first time they are needed. The backbone weights are downloaded by `timm` on the first run.

---

## How to run

All scripts must be run **from the root of the repository**, because they use paths relative to it.

**Minimum to run the notebooks and the app** (the results are already in `results/`):

```bash
python extract_features.py          # features/ is not in the repository
streamlit run app.py
```

**Full reproduction**, in this order:

```bash
python prepare_data.py              # splits/
python extract_features.py          # features/
python train_baseline.py            # results/results_baseline/
python active_learning.py           # results/tables/al_curves.csv (50 runs, slow)
python al_results.py
python al_stats.py
python al_labels_to_target.py
python al_final.py                  # test set and noisy oracle
python al_hitl.py                   # simulated dermatologists (60 runs, slow), results/heads/
python hitl_extra_stats.py
python export_al_head.py            # results/results_active_learning/al_head.pt
python referral.py
python gradcam_eval.py              # needs the images and the lesion masks
```

**Using the app:**

```bash
streamlit run app.py
```

The clinician uploads a dermoscopic image and sees the predicted category, its calibrated confidence, the probability of melanoma, the Grad-CAM heatmap and the category probabilities. A case is **sent to the clinician** (*human-in-the-loop*) when the confidence is below 80% (the highest threshold that refers less than half of the validation cases to the clinician) or the probability of melanoma is at least 20%; otherwise the prediction is shown as a suggestion under supervision (*human-on-the-loop*). The clinician then accepts or corrects the prediction, and every decision is saved in `app_data/feedback_log.csv`.

**Retraining with the clinician's corrections:**

```bash
python retrain_with_corrections.py
```

The images validated or corrected in the app are added to the images labelled by the active learning loop, and the model is trained for one more active learning cycle, always starting from the original active learning model `al_head.pt`. The new model is saved as `results/results_active_learning/al_head_retrained.pt` (overwritten at every run; `al_head.pt` is never overwritten) and its test metrics are compared with those of `al_head.pt`. The app currently uses this retrained model (`HEAD_PATH` in `copilot_common.py`); to go back to the original active learning model, point `HEAD_PATH` to `al_head.pt`.

---

## Main results

All results are on the test set (1431 images), unless stated otherwise.

**Backbone (passive baseline, 100% of the labels, with TTA)**

| Backbone | Macro F1 | Balanced accuracy | Melanoma recall |
|---|---|---|---|
| `tf_efficientnetv2_s.in21k_ft_in1k` (used) | **0.623** | **0.637** | **0.692** |
| `convnext_tiny.fb_in22k_ft_in1k` | 0.582 | 0.639 | 0.572 |
| `efficientnet_b0` | 0.544 | 0.595 | 0.491 |
| `efficientnet_b3` | 0.511 | 0.620 | 0.547 |

TTA raises the macro F1 of the baseline from 0.583 (original view only) to 0.623.

**Active learning (~52% of the labels, mean ± std over 10 seeds)**

| Setting | Macro F1 | Balanced accuracy | Melanoma recall |
|---|---|---|---|
| Hybrid (uncertainty + diversity) | **0.608 ± 0.011** | 0.614 ± 0.013 | 0.569 ± 0.037 |
| Random sampling | 0.574 ± 0.015 | 0.593 ± 0.014 | 0.581 ± 0.044 |
| Hybrid, 10% noisy oracle | 0.509 ± 0.020 | 0.585 ± 0.014 | 0.537 ± 0.039 |
| Random, 10% noisy oracle | 0.429 ± 0.027 | 0.561 ± 0.020 | 0.540 ± 0.054 |
| Reference: same model with 100% of the labels | 0.639 | 0.622 | 0.610 |

The hybrid strategy (the best one on validation) reaches 90% of the validation macro F1 of the full model with 1615 labelled images, against 2215 with random sampling.

**Human-in-the-loop (hybrid strategy, ~52% of the labels)**

| Oracle | Annotations | Label accuracy | Macro F1 |
|---|---|---|---|
| Perfect | 3715 | 100% | 0.608 |
| One dermatologist per image | 3715 | 84% | 0.522 |
| Majority vote of 3 dermatologists | 11145 | 95% | 0.583 |

The hybrid strategy beats random sampling with every oracle (10/10 seeds, Holm-corrected p = 0.016). At the same number of annotations, the majority vote and a single dermatologist give similar results for the hybrid strategy.

**Referral rule (final model, confidence threshold 0.80):** 45.8% of the test cases are sent to the clinician, and 15 of the 159 test melanomas (9.4%) are missed among the cases kept by the model.

**Grad-CAM:** all models focus on the lesion (energy inside the lesion about 2 to 2.4 times the chance level, peak inside the lesion in 78–86% of the images). With the labels of the simulated dermatologists, the localisation gets worse as more images are labelled, which does not happen with correct labels.

**Model used by the app:** the final head of the run hybrid / perfect oracle / seed 0 (3715 labelled images; macro F1 0.593, balanced accuracy 0.599, melanoma recall 0.591, macro AUC 0.926), retrained for one more active learning cycle with the decisions made in the app.

**Test of the app:** 19 training images not yet labelled by the active learning loop (about 3 per category) were reviewed in the app, with the true label taken from `splits/train.csv`. The model was right in 14 of the 19 images and wrong in 5, which were corrected. The 2 missed melanomas (predicted as `nv`) were both sent to the clinician by the referral rule; the other 3 errors (one `df` and two `akiec`) were predicted with high confidence (86–99.6%) and were not referred. These images were then used to retrain the model of the app; with so few images, this is a test of the feedback loop and not a measurable improvement of the model.

More details, tables and figures are in the notebooks and in `results/`.

---

## Limitations

- The dermatologists of the human-in-the-loop experiments are **simulated**. Their accuracies (0.92 to 0.75) and error structure are modelling assumptions, not values from the literature, and their mistakes depend on the category of the image, not on how difficult the image is.
- The models are trained and evaluated only on HAM10000, with a single test set.
- The app is a research prototype for the course and is **not a medical device**.
