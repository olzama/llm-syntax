"""
pca.py — PCA plots of cosine-distance matrices for syntactic and lexical types.

For each phenomenon (syntax, lextype, lexrule), reads a pairwise cosine
similarity JSON, converts to a distance matrix, runs PCA, and saves a scatter
plot annotated with model labels.

Usage (run from repo root):
    python scripts/pca.py [input_dir] [output_dir]

Arguments:
    input_dir    Directory containing syntax-only.json, lextype-only.json,
                 lexrule-only.json (pairwise cosine similarities).
                 Default: analysis/cosine-pairs/models/norm-by-constr-count
    output_dir   Directory for output PNG files.
                 Default: analysis/plots
"""

import sys, os, json, ast
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

_DEFAULT_INPUT_DIR  = os.path.join('analysis', 'cosine-pairs', 'models', 'norm-by-constr-count')
_DEFAULT_OUTPUT_DIR = os.path.join('analysis', 'plots')

_PHENOMENA = [
    ('syntax-only.json',  'pca_syntax.png',  'Syntactic Types'),
    ('lextype-only.json', 'pca_lextype.png', 'Lexical Types'),
    ('lexrule-only.json', 'pca_lexrule.png', 'Lexical Rules'),
]

_RENAME_MAP = {
    'new-original': 'NYT-2025',
    'original':     'Original NYT',
    'llama_07':     'LLaMA 7B',
    'llama_13':     'LLaMA 13B',
    'llama_30':     'LLaMA 30B',
    'llama_65':     'LLaMA 65B',
    'mistral_07':   'Mistral 7B',
    'falcon_07':    'Falcon 7B',
    'wsj':          'WSJ',
    'wikipedia':    'Wikipedia',
}


def _load_similarity_matrix(json_path):
    """Load a pairwise cosine JSON and return a symmetric DataFrame."""
    with open(json_path) as f:
        raw = json.load(f)
    entries = [(ast.literal_eval(k), float(v)) for k, v in raw.items()]
    labels = sorted({i for pair, _ in entries for i in pair})
    mat = pd.DataFrame(1.0, index=labels, columns=labels)
    for (a, b), val in entries:
        mat.loc[a, b] = val
        mat.loc[b, a] = val
    mat.rename(index=_RENAME_MAP, columns=_RENAME_MAP, inplace=True)
    return mat


def _assign_style(label, color_counter, cmap):
    """Return (color, marker) for a model label."""
    if label == 'Original NYT':
        return 'red', '*'
    if label == 'NYT-2025':
        return 'orange', '*'
    if label in ('Wikipedia', 'WSJ'):
        return 'black', 's'
    if 'Falcon' in label:
        return cmap(color_counter % 10), '^'
    if 'Mistral' in label:
        return cmap(color_counter % 10), 'v'
    return cmap(color_counter % 10), 'o'


def plot_pca(input_path, output_path, title):
    """Run PCA on the cosine-distance matrix and save a scatter plot.

    input_path:  path to pairwise cosine similarity JSON
    output_path: path for the output PNG
    title:       plot title suffix (phenomenon name)
    """
    sim = _load_similarity_matrix(input_path)
    dist = 1 - sim

    pca = PCA(n_components=2)
    coords = pca.fit_transform(dist)
    explained = pca.explained_variance_ratio_ * 100

    df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    df['Label'] = sim.index.tolist()

    # Orient so Original NYT is on the left
    if not df.loc[df['Label'] == 'Original NYT', 'PC1'].empty:
        if df.loc[df['Label'] == 'Original NYT', 'PC1'].values[0] > 0:
            df['PC1'] *= -1

    cmap = plt.colormaps['tab10']
    color_map, marker_map = {}, {}
    color_counter = 0
    for label in df['Label']:
        color, marker = _assign_style(label, color_counter, cmap)
        color_map[label] = color
        marker_map[label] = marker
        color_counter += 1

    x_min, x_max = df['PC1'].min(), df['PC1'].max()
    y_min, y_max = df['PC2'].min(), df['PC2'].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    dist_matrix = pairwise_distances(df[['PC1', 'PC2']])
    for i in range(len(dist_matrix)):
        dist_matrix[i, i] = float('inf')
    nearest_dists = dist_matrix.min(axis=1)
    threshold = pd.Series(nearest_dists).quantile(0.2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for i, row in df.iterrows():
        label = row['Label']
        x, y = row['PC1'], row['PC2']
        ax.scatter(x, y, marker=marker_map[label], color=color_map[label], s=150)
        if nearest_dists[i] > threshold:
            ax.text(x + 0.004, y, label, fontsize=10, ha='left', va='center', color=color_map[label])

    handles = [
        plt.Line2D([0], [0], marker=marker_map[lbl], color='w', label=lbl,
                   markerfacecolor=color_map[lbl], markersize=10)
        for lbl in df['Label']
    ]
    ax.legend(handles=handles, title='Sources', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% variance)')
    ax.set_title(f'PCA of Cosine Distances: {title}')
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(input_dir=_DEFAULT_INPUT_DIR, output_dir=_DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    for input_file, output_file, title in _PHENOMENA:
        input_path  = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        plot_pca(input_path, output_path, title)
        print(f'Written: {output_path}')


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print(__doc__)
        sys.exit(1)
    in_dir  = sys.argv[1] if len(sys.argv) >= 2 else _DEFAULT_INPUT_DIR
    out_dir = sys.argv[2] if len(sys.argv) == 3 else _DEFAULT_OUTPUT_DIR
    main(in_dir, out_dir)
