# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
# ]
# ///

import os
import math
import re
import random
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import json
import argparse


def get_short_label_from_filepath(filepath):
    """Extracts a pattern like '1K', '3K' etc., from a filename for a concise label."""
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r'(\d+[kK])', base_name)
    if match:
        return match.group(1).upper()
    return None


def calculate_shannon_diversity(names):
    """Calculate Shannon's diversity index for a given list of names."""
    if not names: return 0
    name_counts = Counter(names)
    total = len(names)
    diversity = 0
    for count in name_counts.values():
        p = count / total
        diversity -= p * math.log(p)
    return diversity


def simpson_diversity_index(observations):
    """Calculate Simpson's Diversity Index from a list of observed species."""
    if not observations: return 0
    counts = Counter(observations).values()
    N = sum(counts)
    if N <= 1: return 0
    numerator = sum(n * n for n in counts)
    denominator = N * N
    return 1 - (numerator / denominator)


def plot_sentence_diversity(thing, model_points, output_dir, diversity_index='shannon'):
    """
    Plots diversity score as a function of the number of sentences on a single graph.
    Each model gets a line connecting its discrete data points (1K, 3K, etc.).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    model_names = sorted(model_points.keys())
    colors = plt.cm.get_cmap('tab10', len(model_names))
    model_to_color = {model: colors(i) for i, model in enumerate(model_names)}

    print("Generating plot of diversity vs. sentence count...")
    for model_name, points in model_points.items():
        if not points:
            continue

        points.sort(key=lambda p: p[0])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        ax.plot(x_vals, y_vals, marker='o', linestyle='-', label=model_name, color=model_to_color.get(model_name))

    ax.set_xlabel("Number of Sentences")
    ax.set_ylabel(f"{diversity_index.capitalize()} Diversity Score")
    ax.set_title(f"Diversity as a Function of Sentence Count ({thing.title()})", fontsize=14, fontweight='bold')
    ax.legend(title="Model Type", fontsize='medium')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(left=0)

    # MODIFIED: Set Y-axis to automatically fit the data range with some padding.
    all_y_values = [p[1] for points in model_points.values() for p in points if points]
    if all_y_values:
        min_y = min(all_y_values)
        max_y = max(all_y_values)
        y_range = max_y - min_y
        padding = y_range * 0.1 if y_range > 0 else 0.1  # 10% padding
        ax.set_ylim(min_y - padding, max_y + padding)
    else:
        ax.set_ylim(bottom=0)  # Fallback if there's no data

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x / 1000)}K'))

    plt.tight_layout()
    png_filename = os.path.join(output_dir, f"final-plot-{thing}-{diversity_index}.png")
    plt.savefig(png_filename, dpi=150)
    plt.close()
    print(f"Saved final plot to {png_filename}")


def main():
    parser = argparse.ArgumentParser(description='Plot diversity score as a function of sentence count from JSON files')
    parser.add_argument('json_files', nargs='+', help='JSON files containing linguistic data')
    parser.add_argument('--phenomena', nargs='+', choices=['lexrule', 'lextype', 'constr'], default=['constr'])
    parser.add_argument('--diversity-index', type=str, choices=['shannon', 'simpson'], default='shannon')
    parser.add_argument('--output-dir', type=str, default='out')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    phenomena_map = {'constr': 'Constructions', 'lexrule': 'Lexical_Rules', 'lextype': 'Lexical_Types'}
    calc_diversity = simpson_diversity_index if args.diversity_index == 'simpson' else calculate_shannon_diversity

    for phenomenon in args.phenomena:
        thing = phenomena_map[phenomenon]
        print(f"\n=== Analyzing {thing} ===")

        model_points = defaultdict(list)
        for json_file in args.json_files:
            k_label = get_short_label_from_filepath(json_file)
            if not k_label:
                print(f"Warning: Skipping file {json_file}, could not determine K-value.")
                continue
            k_value = int(k_label[:-1]) * 1000

            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            if phenomenon in file_data:
                for model_name, observations_dict in file_data[phenomenon].items():
                    all_observations = [item for sublist in
                                        [[name] * count for name, count in observations_dict.items()] for item in
                                        sublist]
                    if all_observations:
                        diversity_score = calc_diversity(all_observations)
                        model_points[model_name].append((k_value, diversity_score))

        if not model_points:
            print("No data found to plot. Skipping...")
            continue

        plot_sentence_diversity(thing, model_points, args.output_dir, args.diversity_index)


if __name__ == "__main__":
    main()