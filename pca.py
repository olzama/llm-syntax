import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ast
import os
import matplotlib.ticker as ticker

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

    # Prepare colors for legend
    color_map = {}
    cmap = plt.cm.get_cmap("tab10")
    for i, label in enumerate(df["Label"]):
        if label in {"Original NYT", "Wikipedia", "WSJ"}:
            color_map[label] = "black" if label != "Original NYT" else "red"
        else:
            color_map[label] = cmap(len(color_map) % 10)

    x_min, x_max = df["PC1"].min(), df["PC1"].max()
    y_min, y_max = df["PC2"].min(), df["PC2"].max()
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15

    aspect_ratio = (y_max - y_min + y_pad * 2) / (x_max - x_min + x_pad * 2)
    aspect_ratio = max(aspect_ratio, 0.75)

    fig_width = 10
    fig_height = fig_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Plot points only, with legend colors
    for _, row in df.iterrows():
        label = row["Label"]
        marker = "*" if label == "Original NYT" else "s" if label in {"Wikipedia", "WSJ"} else "o"
        ax.scatter(row["PC1"], row["PC2"], marker=marker, color=color_map[label], s=150)

    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1f}% variance)")
    ax.set_title(f"PCA of Cosine Distances: {ds['title']}")
    ax.grid(True)

    # Use fixed y-tick step
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.001))

    # Always include legend
    handles = [plt.Line2D([0], [0], marker="o" if label not in {"Original NYT", "Wikipedia", "WSJ"} else ("*" if label == "Original NYT" else "s"),
                          color="w", label=label,
                          markerfacecolor=color_map[label], markersize=10)
               for label in df["Label"]]
    ax.legend(handles=handles, title="Sources", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(ds["output"])
    plt.close()
