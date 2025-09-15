import pandas as pd

# Step 1: Load ratings.txt into a DataFrame
ratings_path = r"C:\Users\Siwar\OneDrive\Desktop\My research paper topic\EGAN-TopoNet\Data set\STEW Dataset\STEW Dataset\ratings.txt"
ratings_df = pd.read_csv(ratings_path, header=None, names=["image_id", "user_id", "rating"])

# Step 2: Map ratings to cognitive load
def map_rating_to_label(r):
    if r <= 4:
        return "low"
    elif r <= 6:
        return "medium"
    else:
        return "high"

ratings_df["label"] = ratings_df["rating"].apply(map_rating_to_label)

# Step 3: Preview
print(ratings_df.head())

# Optional: Save as CSV if needed
ratings_df.to_csv("mapped_cognitive_load_labels.csv", index=False)
