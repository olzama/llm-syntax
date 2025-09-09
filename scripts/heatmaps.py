import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import ast

# Load JSON
with open("analysis/cosine-pairs/models/norm-by-constr-count/all-data.json", "r") as f:
    raw_data = json.load(f)

# Parse the flat dict
entries = [(ast.literal_eval(k), float(v)) for k, v in raw_data.items()]

# Build full symmetric matrix
labels = sorted(set(i for pair, _ in entries for i in pair))
similarity_matrix = pd.DataFrame(1.0, index=labels, columns=labels)

for (a, b), val in entries:
    similarity_matrix.loc[a, b] = val
    similarity_matrix.loc[b, a] = val

# Rename for clarity
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

# Custom colormap: purple gradient without lightest tones
original_cmap = plt.get_cmap("Purples")
colors = original_cmap(np.linspace(0.3, 1.0, 256))
custom_purples = LinearSegmentedColormap.from_list("CustomPurples", colors)

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(
    similarity_matrix,
    annot=False,
    cmap=custom_purples,
    vmin=0.98,
    vmax=1.0,
    linewidths=0.5
)
plt.title("Cosine Similarity of Construction Frequencies (0.98–1.00, Custom Purples)", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('analysis/plots/all-cosines-heatmap-norm-by-constr-count.png')
