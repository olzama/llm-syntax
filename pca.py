import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import ast

# Load the JSON file
with open("analysis/cosine-pairs/models/all-data-normalized-by-num-sen.json", "r") as f:
    raw_data = json.load(f)

# Parse the flat dictionary into a list of tuples
entries = [(ast.literal_eval(k), float(v)) for k, v in raw_data.items()]

# Get unique labels
labels = sorted(set(i for pair in entries for i in pair[0]))

# Initialize symmetric matrix
similarity_matrix = pd.DataFrame(1.0, index=labels, columns=labels)

# Fill in cosine similarity values
for (a, b), val in entries:
    similarity_matrix.loc[a, b] = val
    similarity_matrix.loc[b, a] = val  # ensure symmetry

# Rename labels for clarity
rename_map = {
    "original": "Original NYT",
    "llama_07": "LLaMA 7B",
    "llama_13": "LLaMA 13B",
    "llama_30": "LLaMA 30B",
    "llama_65": "LLaMA 65B",
    "mistral_07": "Mistral 7B",
    "falcon_07": "Falcon 7B",
    "wsj": "WSJ",
    "wikipedia": "Wikipedia"
}
similarity_matrix.rename(index=rename_map, columns=rename_map, inplace=True)

# Compute distance matrix
distance_matrix = 1 - similarity_matrix

# Run PCA
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(distance_matrix)
explained_variance = pca.explained_variance_ratio_ * 100

# Prepare DataFrame for plotting
pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
pca_df["Label"] = similarity_matrix.index

# Plot PCA projection
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=200, color='black', marker='o')
for _, row in pca_df.iterrows():
    plt.text(row["PC1"] + 0.002, row["PC2"], row["Label"], fontsize=10, ha='left', va='center')

plt.title("Cosine Similarity of Construction Frequencies Across Human and Model-Generated Texts (PCA Projection)", fontsize=12)
plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.2f}% variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.2f}% variance)")
plt.grid(True)
plt.tight_layout()
plt.savefig('analysis/plots/cosines-pairwise-pca-norm-by-num-sen.png')
