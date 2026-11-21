# Dermoscopy Copilot - 01. Data preparation and train/validation/test split
# Downloads HAM10000 (images + metadata) and the lesion masks, explores the data,
# builds a lesion-grouped, diagnosis-stratified split (~70/15/15)
# and saves it as CSV files to be reused by the remaining notebooks.

# Library imports
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from sklearn.model_selection import StratifiedGroupKFold

# Dataset download
# Images + metadata (10,015 dermoscopic images + HAM10000_metadata.csv)
path_img = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Imagens:", path_img)

# Lesion segmentation masks
path_seg = kagglehub.dataset_download("tschandl/ham10000-lesion-segmentations")
print("Máscaras:", path_seg)

# Inspect the folder structure
for p in [path_img, path_seg]:
    print("\n", p)
    for item in os.listdir(p):
        print("  ", item)

# Exploratory analysis of the metadata
df = pd.read_csv(os.path.join(path_img, "HAM10000_metadata.csv"))
print(df.shape)   # expected: (10015, 7)
# Images per diagnosis: class imbalance (motivates class weights and macro-averaged metrics)
print(df["dx"].value_counts())   
# Images per lesion (how many lesions have 1 image, 2 images, ...)
# The split must be done by lesion, not by image, to avoid data leakage
print(df["lesion_id"].value_counts().value_counts()) 

# Linking each image to its file
# Dictionary {image_id: path of the .jpg file}, built by searching all subfolders
img_paths = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob(os.path.join(path_img, "*", "*.jpg"))}
# New "path" column with the file path of each image
df["path"] = df["image_id"].map(img_paths)
# Sanity check: every image must have a file associated with it
print("Images without an associated file:", df["path"].isna().sum()) # expected: 0

# Example images per class
# Visual check that images were correctly linked to their diagnoses (one per class)
classes = df["dx"].unique() # "dx" = diagnosis
fig, axes = plt.subplots(1, len(classes), figsize=(18, 3))
for ax, c in zip(axes, classes): # ax: where the image is drawn, c: class name
    row = df[df["dx"] == c].iloc[0]
    ax.imshow(Image.open(row["path"]))
    ax.set_title(c)
    ax.axis("off")
plt.show()

# Train / validation / test split
# StratifiedGroupKFold keeps similar class proportions across splits (stratified by "dx")
# and keeps all images of the same lesion in the same split (grouped by "lesion_id").

# Separate the test set (1 out of 7 folds ~ 15%)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
# next() keeps only the first fold of the split
trainval_idx, test_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

# Separate the validation set from the remainder (1 out of 6 folds)
sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
train_idx, val_idx = next(sgkf2.split(trainval, trainval["dx"], groups=trainval["lesion_id"]))
train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]

# Split verification
print(len(train), len(val), len(test))
# No lesion may be shared between splits (expected: set())
print("Lesions shared train/test:", set(train.lesion_id) & set(test.lesion_id))  
print("Lesions shared train/val:", set(train.lesion_id) & set(val.lesion_id))     
print("Lesions shared val/test:  ", set(val.lesion_id) & set(test.lesion_id))

# All classes must be represented in the test set
print("\nClass distribution in the test set:")
print(test["dx"].value_counts())

# Saving the splits
os.makedirs("splits", exist_ok=True)
cols = ["image_id", "lesion_id", "dx"]
train[cols].to_csv("splits/train.csv", index=False)
val[cols].to_csv("splits/val.csv", index=False)
test[cols].to_csv("splits/test.csv", index=False)
print("Splits saved to 'splits/'.")