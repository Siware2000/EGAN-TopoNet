import os
import shutil

# === Paths ===
real_dir = r"C:\Users\Siwar\OneDrive\Desktop\My research paper topic\EGAN-TopoNet\Topomap GAN Split\train"
synth_dir = r"C:\Users\Siwar\OneDrive\Desktop\My research paper topic\EGAN-TopoNet\Synthetic Topomaps"
merged_dir = r"C:\Users\Siwar\OneDrive\Desktop\My research paper topic\EGAN-TopoNet\Merged Topomaps"

bands = ["theta", "alpha", "beta"]

# === Create merged folders ===
for band in bands:
    out_band_dir = os.path.join(merged_dir, band)
    os.makedirs(out_band_dir, exist_ok=True)

    # === Copy real images ===
    real_band_dir = os.path.join(real_dir, band)
    for i, fname in enumerate(sorted(os.listdir(real_band_dir))):
        if fname.endswith(".png"):
            src = os.path.join(real_band_dir, fname)
            dst = os.path.join(out_band_dir, f"real_{i:04d}.png")
            shutil.copyfile(src, dst)

    # === Copy synthetic images ===
    synth_band_dir = os.path.join(synth_dir, band)
    for i, fname in enumerate(sorted(os.listdir(synth_band_dir))):
        if fname.endswith(".png"):
            src = os.path.join(synth_band_dir, fname)
            dst = os.path.join(out_band_dir, f"synth_{i:04d}.png")
            shutil.copyfile(src, dst)

    print(f"✅ Merged {band}: {len(os.listdir(out_band_dir))} files")

print("🎯 All bands merged successfully.")
