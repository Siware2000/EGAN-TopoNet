import os
import pandas as pd

# === SET YOUR LOCAL MERGED FOLDER PATH HERE ===
merged_dir = r"C:\Users\Siwar\OneDrive\Desktop\My research paper topic\EGAN-TopoNet\Merged Topomaps"
bands = ["theta", "alpha", "beta"]
label_map = {"low": 0, "medium": 1, "high": 2}

data = []

for band in bands:
    band_path = os.path.join(merged_dir, band)
    if not os.path.exists(band_path):
        continue
    for fname in os.listdir(band_path):
        if not fname.endswith(".png"):
            continue

        full_path = os.path.join(band_path, fname)
        is_synthetic = 1 if fname.startswith("synth") else 0

        label_str = "unknown"
        for key in label_map:
            if key in fname.lower():
                label_str = key
                break

        data.append({
            "filename": fname,
            "path": full_path,
            "band": band,
            "synthetic": is_synthetic,
            "label_str": label_str,
            "label": label_map.get(label_str, -1)
        })

df = pd.DataFrame(data)
output_path = os.path.join(merged_dir, "merged_topomap_metadata.csv")
df.to_csv(output_path, index=False)
print(f"✅ Metadata saved to: {output_path}")
