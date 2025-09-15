import os
import shutil

# === LOCAL PATHS ===
original_dir = r"C:\Users\Siwar\OneDrive\Desktop\My research paper topic\EGAN-TopoNet\Topomaps MultiBand"
fixed_dir = r"C:\Users\Siwar\OneDrive\Desktop\My research paper topic\EGAN-TopoNet\Topomap GAN Split"
bands = ["theta", "alpha", "beta"]

# === CREATE TRAIN FOLDERS PER BAND ===
for band in bands:
    os.makedirs(os.path.join(fixed_dir, "train", band), exist_ok=True)

# === COPY FILES TO MATCH GAN EXPECTATION ===
for file in os.listdir(original_dir):
    for band in bands:
        if file.endswith(f"{band}.png"):
            src = os.path.join(original_dir, file)
            dst = os.path.join(fixed_dir, "train", band, file)
            shutil.copy(src, dst)

# === PRINT SUMMARY ===
for band in bands:
    folder = os.path.join(fixed_dir, "train", band)
    print(f"{band}: {len(os.listdir(folder))} files in {folder}")

print("✅ Folder structure is now compatible with GAN training.")
