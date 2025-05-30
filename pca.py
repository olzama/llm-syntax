import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ast
import os
import matplotlib.ticker as ticker
from sklearn.metrics import pairwise_distances

# Define input/output file mapping and titles
datasets = [
    {
        "title": "Syntactic Types",
        "input": "analysis/cosine-pairs/models/norm-by-constr-count/syntax-only.json",
        "output": "analysis/plots/pca_syntax.png"
    },
    {
        "title": "Lexical Types",
        "input": "analysis/cosine-pairs/models/norm-by-constr-count/lextype-only.json",
        "output": "analysis/plots/pca_lextype.png"
    },
    {
        "title": "Lexical Rules",
        "input": "analysis/cosine-pairs/models/norm-by-constr-count/lexrule-only.json",
        "output": "analysis/plots/pca_lexrule.png"
    }
]

# Label renaming
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

cmap = plt.cm.get_cmap("tab10")

# Loop through all datasets
for ds in datasets:
    with open(ds["input"], "r") as f:
        raw_data = json.load(f)

    entries = [(ast.literal_eval(k), float(v)) for k, v in raw_data.items()]
    labels = sorted(set(i for pair in entries for i in pair[0]))

    similarity_matrix = pd.DataFrame(1.0, index=labels, columns=labels)
    for (a, b), val in entries:
        similarity_matrix.loc[a, b] = val
        similarity_matrix.loc[b, a] = val

    similarity_matrix.rename(index=rename_map, columns=rename_map, inplace=True)
    distance_matrix = 1 - similarity_matrix

    pca = PCA(n_components=2)
    coords = pca.fit_transform(distance_matrix)
    explained_variance = pca.explained_variance_ratio_ * 100

    df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df["Label"] = similarity_matrix.index

    if df.loc[df["Label"] == "Original NYT", "PC1"].values[0] > 0:
        df["PC1"] *= -1

    # Define color and shape maps
    color_map = {}
    marker_map = {}
    for i, label in enumerate(df["Label"]):
        if label == "Original NYT":
            color_map[label] = "red"
            marker_map[label] = "*"
        elif label == "Wikipedia" or label == "WSJ":
            color_map[label] = "black"
            marker_map[label] = "s"
        elif "Falcon" in label:
            color_map[label] = cmap(len(color_map) % 10)
            marker_map[label] = "^"
        elif "Mistral" in label:
            color_map[label] = cmap(len(color_map) % 10)
            marker_map[label] = "v"
        else:
            color_map[label] = cmap(len(color_map) % 10)
            marker_map[label] = "o"

    fig, ax = plt.subplots(figsize=(10, 8))

    x_min, x_max = df["PC1"].min(), df["PC1"].max()
    y_min, y_max = df["PC2"].min(), df["PC2"].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Compute distance to nearest neighbor for label placement decision
    dist_matrix = pairwise_distances(df[["PC1", "PC2"]])
    for i in range(len(dist_matrix)):
        dist_matrix[i, i] = float("inf")
    nearest_dists = dist_matrix.min(axis=1)
    threshold = pd.Series(nearest_dists).quantile(0.2)

    legend_labels = []
    for i, row in df.iterrows():
        label = row["Label"]
        x, y = row["PC1"], row["PC2"]
        color = color_map[label]
        marker = marker_map[label]
        ax.scatter(x, y, marker=marker, color=color, s=150)

        label_fits = (x_min + x_pad < x < x_max - x_pad) and (y_min + y_pad < y < y_max - y_pad)

        if nearest_dists[i] > threshold:
            ax.text(x + 0.004, y, label, fontsize=10, ha='left', va='center', color=color)
        else:
            legend_labels.append(label)

    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1f}% variance)")
    ax.set_title(f"PCA of Cosine Distances: {ds['title']}")
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

    # Always show full legend
    legend_labels = df['Label'].tolist()
    if legend_labels:
        handles = [
            plt.Line2D([0], [0], marker=marker_map[lbl], color='w', label=lbl,
                       markerfacecolor=color_map[lbl], markersize=10)
            for lbl in legend_labels
        ]
        ax.legend(handles=handles, title="Sources", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(ds["output"])
    plt.close()
