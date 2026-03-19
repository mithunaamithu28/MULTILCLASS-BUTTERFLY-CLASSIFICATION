import pandas as pd
import os
import shutil

# CSV files
TRAIN_CSV = "Training_set.csv"
TEST_CSV = "Testing_set.csv"

# Image folders
TRAIN_IMG_DIR = "train"
TEST_IMG_DIR = "test"

# Output folder
OUTPUT_DIR = "dataset_ready"

# Create output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

# ----------- TRAIN DATA -----------
train_df = pd.read_csv(TRAIN_CSV)

for _, row in train_df.iterrows():
    img_name = row["filename"]
    label = row["label"]

    src = os.path.join(TRAIN_IMG_DIR, img_name)
    class_dir = os.path.join(OUTPUT_DIR, "train", label)

    os.makedirs(class_dir, exist_ok=True)

    if os.path.exists(src):
        shutil.copy(src, os.path.join(class_dir, img_name))

print("✅ Training data organized")

# ----------- TEST DATA ------------
test_df = pd.read_csv(TEST_CSV)

for _, row in test_df.iterrows():
    img_name = row["filename"]
    src = os.path.join(TEST_IMG_DIR, img_name)

    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUTPUT_DIR, "test", img_name))

print("✅ Testing data organized")
print("🎉 Dataset preparation completed!")
