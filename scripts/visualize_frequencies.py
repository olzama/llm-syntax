#!/usr/bin/env python3
"""
visualize_frequencies.py — generate frequency comparison bar/scatter plots.

Produces two sets of plots:

1. Per-model: one bar chart per LLM model comparing it against the human baselines,
   normalized by construction count.

2. Combined: a single bar chart with all LLM models aggregated into one 'llm' entry,
   compared against the human baselines, normalized by construction count.

Usage (run from repo root):
    python scripts/visualize_frequencies.py <frequencies_json> [--output-dir <dir>]

Arguments:
    frequencies_json   Path to a JSON file with structure
                       {phenomenon: {model: {type: count}}}.
                       Model names must follow the '-human'/'-llm' naming convention.

Options:
    --output-dir       Directory for output PNG files.
                       Default: analysis/plots/frequencies

Output:
    PNG files written to <output_dir>/<start>-<end>/.
    One file per (model x human baseline combination x phenomenon).
"""

import argparse
import os
import json

from construction_frequencies import build_llm_vs_human, is_human
from util import freq_counts_by_model, normalize_by_constr_count, sort_normalized_data

_PLOT_RANGES = [(0, 50)]
_DEFAULT_OUTPUT_DIR = os.path.join('analysis', 'plots', 'frequencies')


def get_human_models(frequencies):
    """Return sorted list of human model names found in frequencies."""
    models = set()
    for rt in frequencies.values():
        models.update(rt.keys())
    return sorted(m for m in models if is_human(m))


def visualize_per_model(frequencies, reverse_frequencies, output_dir=_DEFAULT_OUTPUT_DIR):
    """Plot top and bottom frequency comparisons for each LLM vs. the human datasets."""
    start, end = _PLOT_RANGES[0]
    humans = get_human_models(frequencies)
    present_models = set(next(iter(frequencies.values())).keys())
    for model in present_models:
        if not is_human(model):
            freq_counts_by_model(frequencies, model, humans[0], humans[1], humans[2],
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, model, humans[0], humans[1], humans[2],
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)
            freq_counts_by_model(frequencies, humans[0], humans[1], humans[2], model,
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, humans[0], humans[1], humans[2], model,
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)
            freq_counts_by_model(frequencies, humans[1], humans[0], humans[2], model,
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, humans[1], humans[0], humans[2], model,
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)
            freq_counts_by_model(frequencies, humans[2], humans[0], humans[1], model,
                                 start, end, "Top frequencies", reverse=True, output_dir=output_dir)
            freq_counts_by_model(reverse_frequencies, humans[2], humans[0], humans[1], model,
                                 start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)


def visualize_combined_llm(frequencies, output_dir=_DEFAULT_OUTPUT_DIR):
    """Plot all LLMs aggregated as 'llm' vs. the human datasets."""
    merged = build_llm_vs_human(frequencies)
    normed = normalize_by_constr_count(merged)
    ascending, descending = sort_normalized_data(normed)
    humans = get_human_models(normed)
    start, end = _PLOT_RANGES[0]
    freq_counts_by_model(descending, 'llm', humans[0], humans[1], humans[2],
                         start, end, "Top frequencies", reverse=True, output_dir=output_dir)
    freq_counts_by_model(ascending, 'llm', humans[0], humans[1], humans[2],
                         start, end, "Bottom frequencies", reverse=False, output_dir=output_dir)


def main(frequencies_json, output_dir=_DEFAULT_OUTPUT_DIR):
    with open(frequencies_json, 'r') as f:
        frequencies = json.load(f)
    for start, end in _PLOT_RANGES:
        os.makedirs(os.path.join(output_dir, f'{start}-{end}'), exist_ok=True)
    normalized = normalize_by_constr_count(frequencies)
    ascending, descending = sort_normalized_data(normalized)
    visualize_per_model(descending, ascending, output_dir=output_dir)
    visualize_combined_llm(frequencies, output_dir=output_dir)
    print(f"Plots written to {output_dir}/")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frequencies_json', help='Path to frequency JSON file.')
    ap.add_argument('--output-dir', default=_DEFAULT_OUTPUT_DIR,
                    help=f'Directory for output PNG files (default: {_DEFAULT_OUTPUT_DIR}).')
    args = ap.parse_args()
    main(args.frequencies_json, output_dir=args.output_dir)
