"""
compare_authors_llms.py — boxplot comparison of LLM vs. human-author cosine similarity distributions.

Loads pre-computed pairwise cosine-similarity JSONs for LLM models and for human author
pairs, then produces a boxplot comparing the two distributions and reports Levene's test.

Usage (run from repo root):
    python scripts/compare_authors_llms.py <model_cosines_json> <author_cosines_json>
        [--output-dir <dir>]

Arguments:
    model_cosines_json    JSON with pairwise cosine similarities between LLM models
                          (produced by construction_frequencies.py __main__).
    author_cosines_json   JSON with pairwise cosine similarities between human authors
                          (produced by sentences2authors.py or author_llm_pair_compare.py).

Options:
    --output-dir          Directory to write the output PNG
                          (default: analysis/cosine-pairs).
"""

import argparse
import json
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from construction_frequencies import is_human

_DEFAULT_OUTPUT_DIR = os.path.join('analysis', 'cosine-pairs')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model_cosines_json',
                    help='Pairwise cosine similarities between LLM models.')
    ap.add_argument('author_cosines_json',
                    help='Pairwise cosine similarities between human authors.')
    ap.add_argument('--output-dir', default=_DEFAULT_OUTPUT_DIR,
                    help=f'Directory to write output PNG (default: {_DEFAULT_OUTPUT_DIR}).')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.model_cosines_json, 'r') as f:
        all_cosines = json.load(f)
    with open(args.author_cosines_json, 'r') as f:
        author_cosines = json.load(f)

    # Exclude pairs where either name is a human corpus baseline
    llm_cosines = {k: v for k, v in all_cosines.items() if not is_human(k)}
    llm_cosine_list    = [float(v) for v in llm_cosines.values()]
    human_cosine_list  = [float(v) for v in author_cosines.values()]

    llm_mean   = np.mean(llm_cosine_list)
    human_mean = np.mean(human_cosine_list)
    llm_var    = np.var(llm_cosine_list)
    human_var  = np.var(human_cosine_list)
    llm_stdev   = np.std(llm_cosine_list)
    human_stdev = np.std(human_cosine_list)
    f_statistic, p_value = stats.levene(llm_cosine_list, human_cosine_list)

    print(f"LLM mean: {llm_mean:.4f}, variance: {llm_var:.6f}, stdev: {llm_stdev:.4f}")
    print(f"Human mean: {human_mean:.4f}, variance: {human_var:.6f}, stdev: {human_stdev:.4f}")
    print(f"F-statistic: {f_statistic:.4f}, p-value: {p_value:.4e}")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[llm_cosine_list, human_cosine_list])
    plt.xticks([0, 1], ['LLMs', 'Humans'])
    plt.ylabel('Cosine Similarity')
    plt.title('Boxplot Comparison of Cosine Similarities (LLMs vs Humans)')
    out_path = os.path.join(args.output_dir, 'models_vs_authors.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")
