import os
import shutil
import random

# === YOUR PATHS ===
source_dir = "C:/Users/Siwar/OneDrive/Desktop/My research paper topic/EGAN-TopoNet/Topomaps MultiBand"
target_dir = "C:/Users/Siwar/OneDrive/Desktop/My research paper topic/EGAN-TopoNet/Topomap GAN Split"
train_split = 0.8
bands = ["theta", "alpha", "beta"]

# === CREATE OUTPUT FOLDERS ===
for band in bands:
    os.makedirs(os.path.join(target_dir, "train", band), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test", band), exist_ok=True)

# === SPLIT & COPY FILES ===
for band in bands:
    band_files = [f for f in os.listdir(source_dir) if f.endswith(f"{band}.png")]
    random.shuffle(band_files)

    split_index = int(len(band_files) * train_split)
    train_files = band_files[:split_index]
    test_files = band_files[split_index:]

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, "train", band, f))

    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, "test", band, f))

    print(f"{band}: {len(train_files)} train, {len(test_files)} test")

print("✅ Train/test split complete for all bands.")
