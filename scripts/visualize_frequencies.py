#!/usr/bin/env python3
"""
visualize_frequencies.py — generate frequency comparison bar/scatter plots.

Produces two sets of plots:

1. Per-model: one bar chart per LLM model comparing it against the three human baselines,
   normalized by sentence count.

2. Combined: a single bar chart with all LLM models aggregated into one 'llm' entry,
   compared against the three human baselines, normalized by construction count.

Usage (run from repo root):
    python scripts/visualize_frequencies.py <frequencies_json> [output_dir]

Arguments:
    frequencies_json   Path to a JSON file with structure
                       {phenomenon: {model: {type: count}}}.
                       Model names must match those in dataset_sizes and
                       LLM_GENERATED / ALL_HUMAN_AUTHORED in construction_frequencies.py.
    output_dir         Directory for output PNG files.
                       Default: analysis/plots/frequencies

Output:
    PNG files written to <output_dir>/<start>-<end>/.
    One file per (model × human baseline combination × phenomenon).
"""

import sys, os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construction_frequencies import (
    normalize_by_num_sen, build_llm_vs_human,
    ALL_HUMAN_AUTHORED, dataset_sizes,
)
from util import freq_counts_by_model, normalize_by_constr_count, sort_normalized_data

_PLOT_RANGES = [(0, 50)]
_DEFAULT_OUTPUT_DIR = os.path.join('analysis', 'plots', 'frequencies')


def visualize_per_model(frequencies, reverse_frequencies, output_dir=_DEFAULT_OUTPUT_DIR):
    """Plot top and bottom frequency comparisons for each LLM vs. the human datasets."""
    start, end = _PLOT_RANGES[0]
    present_models = set(next(iter(frequencies.values())).keys())
    for model in present_models:
        if model not in ALL_HUMAN_AUTHORED:
            freq_counts_by_model(frequencies, model, ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[2],
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, model, ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[2],
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)
            freq_counts_by_model(frequencies, ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[2], model,
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[2], model,
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)
            freq_counts_by_model(frequencies, ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[2], model,
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[2], model,
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)
            freq_counts_by_model(frequencies, ALL_HUMAN_AUTHORED[2], ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], model,
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, ALL_HUMAN_AUTHORED[2], ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], model,
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)


def visualize_combined_llm(frequencies, output_dir=_DEFAULT_OUTPUT_DIR):
    """Plot all LLMs aggregated as 'llm' vs. the human datasets.

    Combines raw counts across all LLM models, then normalizes by construction count
    so that the combined dataset does not need a sentence-count entry in dataset_sizes.
    """
    merged = build_llm_vs_human(frequencies)
    normed = normalize_by_constr_count(merged)
    ascending, descending = sort_normalized_data(normed)
    start, end = _PLOT_RANGES[0]
    freq_counts_by_model(descending, 'llm', ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[2],
                         start, end, "Top frequencies", reverse=True, output_dir=output_dir)
    freq_counts_by_model(ascending, 'llm', ALL_HUMAN_AUTHORED[0], ALL_HUMAN_AUTHORED[1], ALL_HUMAN_AUTHORED[2],
                         start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)


def main(frequencies_json, output_dir=_DEFAULT_OUTPUT_DIR):
    with open(frequencies_json, 'r') as f:
        frequencies = json.load(f)
    for start, end in _PLOT_RANGES:
        os.makedirs(os.path.join(output_dir, f'{start}-{end}'), exist_ok=True)
    normalized, reverse = normalize_by_num_sen(frequencies)
    visualize_per_model(normalized, reverse, output_dir=output_dir)
    visualize_combined_llm(frequencies, output_dir=output_dir)
    print(f"Plots written to {output_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) == 3 else _DEFAULT_OUTPUT_DIR
    main(sys.argv[1], output_dir=out)
