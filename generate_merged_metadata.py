import os
import pandas as pd

# Path to your topomap directory
topomap_dir = "C:/Users/Siwar/OneDrive/Desktop/My research paper topic/EGAN-TopoNet/Topomaps MultiBand"

# Path to the label file
label_csv_path = "C:/Users/Siwar/PycharmProjects/EGAN-TopoNet/mapped_cognitive_load_labels.csv"

# Load labels
df_labels = pd.read_csv(label_csv_path)

# Extract image paths
image_records = []
for root, _, files in os.walk(topomap_dir):
    for fname in files:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_id = os.path.splitext(fname)[0]
            full_path = os.path.join(root, fname).replace("\\", "/")  # Normalize path
            image_records.append({"image_id": image_id, "image_path": full_path})

# Create DataFrame from image records
df_images = pd.DataFrame(image_records)

# Option 1: Extract numeric part (e.g., "sub01_0_label0_alpha" → 01)
df_images['image_id'] = df_images['image_id'].str.extract('(\d+)').astype(int)
df_labels['image_id'] = df_labels['image_id'].astype(int)

# Option 2: Keep as strings (if labels also have text IDs)
# df_images['image_id'] = df_images['image_id'].astype(str)
# df_labels['image_id'] = df_labels['image_id'].astype(str)

# Merge with labels on image_id
df_merged = pd.merge(df_images, df_labels, on="image_id", how="inner")

# Show class distribution
print("✅ Label Distribution:")
print(df_merged["label"].value_counts())

# Save to CSV
output_path = "merged_metadata.csv"
df_merged.to_csv(output_path, index=False)
print(f"\n📁 Merged metadata saved to: {output_path}")