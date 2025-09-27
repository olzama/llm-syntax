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
from collections import defaultdict as dd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import json
import argparse


nyt =  ['original', 'original_2025', 'nyt_2023_human',  'nyt_human-2025']
humans =  nyt + ['wsj', 'wikipedia']
series = {"nyt":{"label": "Human (NYT)",
                 "color": "#B44E3A"},  # chestnut brown
          "human":{"label": "Human (WSJ/Wiki)", "color": "#D17F4A"},  # copper
          "llm2023":{"label": "LLM (2023)",     "color": "#3A6DB4"},  # cobalt blue
          "llm2025":{"label": "LLM (2025)",     "color": "#4AA6B4"},  # teal-blue
          "llms": {"label": "LLM (all)",     "color": 'black'},  # black
          }



def get_short_label_from_filepath(filepath):
    """Extracts a pattern like '1K', '3K' etc., from a filename for a concise label."""
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r'(\d+[kK])', base_name)
    if match:
        return match.group(1).upper()
    return base_name

def get_year_from_filepath(filepath):
    """
    Extracts the year from a filename for a concise label.
    This should be in the model name
    """
    if filepath.endswith('wsj.json'):
        return "2023"
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r'(20\d\d)', base_name)
    if match:
        return match.group(1).upper()
    return base_name



def load_data_from_json(json_files, phenomena=['constr']):
    """Load data from JSON files with the specified structure."""
    data = {}
    data['LLMs'] = []

    model_to_file_map = {}

    combined_data = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            for phenomenon in file_data:
                if phenomenon not in combined_data:
                    combined_data[phenomenon] = {}
                for model in file_data[phenomenon]:
                    short_label = get_short_label_from_filepath(json_file)
                    year =  get_year_from_filepath(json_file)
                    unique_model_name = f"{model} ({year})"
                    if unique_model_name not in combined_data[phenomenon]:
                        combined_data[phenomenon][unique_model_name] = {}
                    combined_data[phenomenon][unique_model_name].update(file_data[phenomenon][model])

                    model_to_file_map[unique_model_name] = json_file

    for phenomenon in phenomena:
        if phenomenon in combined_data:
            for unique_model_name in combined_data[phenomenon]:
                if unique_model_name not in data:
                    data[unique_model_name] = []

                names = []
                for type_name, count in combined_data[phenomenon][unique_model_name].items():
                    names.extend([type_name] * count)
                    
                data[unique_model_name].extend(names)

                if 'human' not in unique_model_name.lower() and 'wsj' not in unique_model_name.lower() and 'wikipedia' not in unique_model_name.lower():
                    data['LLMs'].extend(names)

    return data, model_to_file_map



def calculate_shannon_diversity(names):
    """Calculate Shannon's diversity index for a given list of names."""
    name_counts = dd(int)
    for name in names:
        name_counts[name] += 1

    total = len(names)
    diversity = 0
    for count in name_counts.values():
        p = count / total
        diversity -= p * math.log(p)
    return diversity


def simpson_diversity_index(observations):
    """
    Calculate Simpson's Diversity Index from a list of observed species.
    """
    counts = Counter(observations).values()
    N = sum(counts)

    if N <= 1:
        raise ValueError("Total number of observations must be greater than 1.")

    numerator = sum(n * n for n in counts)
    denominator = N * N

    D = 1 - (numerator / denominator)
    return D


def analyze_diversity(thing, data, model_to_file_map, output_dir, json_files):
    """Analyze diversity metrics for the given data."""
    print(f"Analyzing diversity for: {thing}")
    print(f"Available models: {list(data.keys())}")
    if 'LLMs' in data:
        print(f"LLMs sample (last 20): {data['LLMs'][-20:]}")

    diversity = dict()
    diversity['Shannon'] = dict()
    diversity['Simpson'] = dict()

    for model in data:
        if len(data[model]) > 1:
            diversity['Shannon'][model] = calculate_shannon_diversity(data[model])
            diversity['Simpson'][model] = simpson_diversity_index(data[model])

    file_to_color = {file: plt.cm.tab10(i) for i, file in enumerate(json_files)}

    for idx in ('Shannon', 'Simpson'):
        models = list(diversity[idx].keys())
        scores = list(diversity[idx].values())

        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])
        models_sorted = [models[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]

        md_filename = os.path.join(output_dir, f"erg-llm-{thing.replace(' ', '-').lower()}-{idx.lower()}.md")
        with open(md_filename, 'w') as out:
            print("""| Model   | Diversity |
| --- | --- |""", file=out)
            for model, score in zip(models_sorted, scores_sorted):
                print(f"| {model} | {score:.3f} |", file=out)

        fig, ax = plt.subplots(figsize=(6, 4))

        for i, (score, model) in enumerate(zip(scores_sorted, models_sorted)):
            mname= model.lower().split(' (')[0]
            if mname in nyt:
                ax.scatter(score, i, color=series['nyt']['color'], s=40, marker='*')
            elif  mname  in humans:
                ax.scatter(score, i, color=series['human']['color'], s=40, marker='+')
            elif mname == 'llms':
                ax.scatter(score, i, color=series['llms']['color'], s=80, marker='o')
            else:
                if '2023' in model:
                    ax.scatter(score, i, color=series['llm2023']['color'], s=20, marker='o')
                else:
                    ax.scatter(score, i, color=series['llm2025']['color'], s=20, marker='o')
                    
        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted, fontsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.set_xlabel('Score', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')
        ax.set_title(f"Diversity: {thing} ({idx} Index)", fontsize=12, fontweight='bold')

        legend_patches = [mpatches.Patch(color=series[k]['color'],
                                         label=series[k]['label']) for k in series]
                          
                                         
        ax.legend(handles=legend_patches, loc='best', fontsize='small', title='Source File')

        plt.tight_layout()
        png_filename = os.path.join(output_dir, f"llm-erg-{thing.replace(' ', '-').lower()}-{idx.lower()}.png")
        plt.savefig(png_filename, dpi=150, bbox_inches='tight')
        plt.close()


def permutation_test(sample1, sample2, index='shannon', reps=10000, rng=None):
    """Perform permutation test to compare diversity between two samples."""
    if rng is None:
        rng = np.random.default_rng()

    calc = simpson_diversity_index if index == 'simpson' else calculate_shannon_diversity
    obs = abs(calc(sample1) - calc(sample2))

    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)

    perm_diffs = []
    for _ in range(reps):
        perm = rng.permutation(combined)
        d1, d2 = calc(perm[:n1]), calc(perm[n1:])
        perm_diffs.append(abs(d1 - d2))

    p_val = np.mean(np.array(perm_diffs) >= obs)
    return obs, p_val, np.array(perm_diffs)


def main():
    parser = argparse.ArgumentParser(description='Analyze linguistic diversity from JSON files')
    parser.add_argument('json_files', nargs='+', help='JSON files containing linguistic data')
    parser.add_argument('--phenomena', nargs='+',
                        choices=['lexrule', 'lextype', 'constr'],
                        default=['lexrule', 'lextype', 'constr'],
                        help='Phenomena to analyze (default: constr, lexrule, lextype)')
    parser.add_argument('--num-bootstrap', type=int, default=10000,
                        help='Number of bootstrap/permutation iterations (default: 10000)')
    parser.add_argument('--output-dir', type=str, default='out',
                        help='Output directory for generated files (default: out)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output files will be saved to: {args.output_dir}")

    phenomena_map = {
        'constr': 'Constructions',
        'lexrule': 'Lexical Rules',
        'lextype': 'Lexical Types'
    }

    for phenomenon in args.phenomena:
        thing = phenomena_map[phenomenon]
        print(f"\n=== Analyzing {thing} ===")

        data, model_to_file_map = load_data_from_json(args.json_files, [phenomenon])

        if not data or len(data) <= 1:
            print(f"Insufficient data for {thing}. Skipping...")
            continue
        analyze_diversity(thing, data, model_to_file_map, args.output_dir, args.json_files)
        
        # Perform statistical tests if we have both LLMs and original data
        if 'LLMs' in data and any(model in data for model in nyt):
            reference_data = []
            for model in nyt:
                if model in data:
                    reference_data.extend(data[model]) 
            if reference_data and len(data['LLMs']) > 1 and len(reference_data) > 1:
                observed_diff, p_value, diffs = permutation_test(
                    data['LLMs'], reference_data, 'shannon', reps=args.num_bootstrap
                )
                print(
                    f'Shannon {thing}; difference: {observed_diff:.4f}, p_value: {p_value:.4f} ({args.num_bootstrap} reps)')

                observed_diff, p_value, diffs = permutation_test(
                    data['LLMs'], reference_data, 'simpson', reps=args.num_bootstrap
                )
                print(
                    f'Simpson {thing}; difference: {observed_diff:.4f}, p_value: {p_value:.4f} ({args.num_bootstrap} reps)')


if __name__ == "__main__":
    main()
