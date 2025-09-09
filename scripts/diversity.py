# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
# ]
# ///

import os
import math
import random
import numpy as np
from collections import defaultdict as dd
import matplotlib.pyplot as plt
from collections import Counter
import json
import argparse

def load_data_from_json(json_files, phenomena=['constr']):
    """Load data from JSON files with the specified structure."""
    data = {}
    data['LLMs'] = []
    
    # Combine data from all JSON files
    combined_data = {}
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            # Merge data from this file
            for phenomenon in file_data:
                if phenomenon not in combined_data:
                    combined_data[phenomenon] = {}
                for model in file_data[phenomenon]:
                    if model not in combined_data[phenomenon]:
                        combined_data[phenomenon][model] = {}
                    combined_data[phenomenon][model].update(file_data[phenomenon][model])
    
    # Extract names for each model based on specified phenomena
    for phenomenon in phenomena:
        if phenomenon in combined_data:
            for model in combined_data[phenomenon]:
                if model not in data:
                    data[model] = []
                
                # Convert counts to repeated names
                names = []
                for type_name, count in combined_data[phenomenon][model].items():
                    names.extend([type_name] * count)
                
                data[model].extend(names)
                
                # Add to LLMs category if not original data
                if model not in ['original', 'original_2025', 'wsj', 'wikipedia']:
                    data['LLMs'].extend(names)
    
    return data


def load_data(directory, typs=['constr.txt']):
    """Load cc text files in the given directory (legacy function)."""
    data = {}
    data['LLMs'] = []
    for filename in os.listdir(directory):
        for typ in typs:
            filepath = os.path.join(directory, filename, typ)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as file:
                    names = []
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            name, frequency = parts[1], int(parts[0])
                            names.extend([name] * frequency)
                data[filename] = names
                if filename != 'original':
                    data['LLMs'] += names
    return data


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

    :param observations: list of species observations (e.g., ["cat", "dog", "cat", "bird"])
    :return: Simpson's Diversity Index value
    """
    counts = Counter(observations).values()
    N = sum(counts)
    
    if N <= 1:
        raise ValueError("Total number of observations must be greater than 1.")

    numerator = sum(n * n for n in counts)
    denominator = N * N
    
    D = 1 - (numerator / denominator)
    return D


def analyze_diversity(thing, data):
    """Analyze diversity metrics for the given data."""
    print(f"Analyzing diversity for: {thing}")
    print(f"Available models: {list(data.keys())}")
    if 'LLMs' in data:
        print(f"LLMs sample (last 20): {data['LLMs'][-20:]}")

    diversity = dict()
    diversity['Shannon'] = dict()
    diversity['Simpson'] = dict()

    for model in data:
        if len(data[model]) > 1:  # Ensure we have enough data
            diversity['Shannon'][model] = calculate_shannon_diversity(data[model])
            diversity['Simpson'][model] = simpson_diversity_index(data[model])

    # Create visualizations and output files
    for idx in ('Shannon', 'Simpson'):
        models = list(diversity[idx].keys())
        scores = list(diversity[idx].values())

        # Sort data for better visualization
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])
        models_sorted = [models[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]

        # Create markdown table
        with open(f"erg-llm-{thing}-{idx}.md", 'w') as out:
            print("""| Model   | Diversity |
| --- | --- |""", file=out)
            for model, score in zip(models_sorted, scores_sorted):
                print(f"| {model} | {score:.3f} |", file=out)

        # Create Tufte-style plot
        fig, ax = plt.subplots(figsize=(6, 3))

        # Create different markers for different model types
        for i, (score, model) in enumerate(zip(scores_sorted, models_sorted)):
            if model.lower().startswith('original'):
                ax.scatter(score, i, color='black', s=40, marker='*')  # Star for original
            elif model.lower() == 'llms':
                ax.scatter(score, i, color='black', s=80, marker='o')  # Large circle for LLMs
            else:
                ax.scatter(score, i, color='black', s=20, marker='o')  # Regular dots

        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted, fontsize=10)

        ax.tick_params(axis='x', labelsize=10)
        ax.set_xlabel('Score', fontsize=10)

        # Tufte style - minimal borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')

        ax.set_title(f"Diversity: {thing} ({idx} Index)", fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"llm-erg-{thing}-{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory


def permutation_test(sample1, sample2, index='shannon', reps=10000, rng=None):
    """Perform permutation test to compare diversity between two samples."""
    if rng is None:
        rng = np.random.default_rng()

    # observed statistic
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
    
    args = parser.parse_args()
    
    # Map phenomena to analysis names
    phenomena_map = {
        'constr': 'Constructions',
        'lexrule': 'Lexical Rules', 
        'lextype': 'Lexical Types'
    }
    
    for phenomenon in args.phenomena:
        thing = phenomena_map[phenomenon]
        print(f"\n=== Analyzing {thing} ===")
        
        # Load data from JSON files
        data = load_data_from_json(args.json_files, [phenomenon])
        
        if not data or len(data) <= 1:
            print(f"Insufficient data for {thing}. Skipping...")
            continue
        
        # Analyze diversity
        analyze_diversity(thing, data)
        
        # Perform statistical tests if we have both LLMs and original data
        if 'LLMs' in data and any(model in data for model in ['original', 'new-original']):
            reference_data = data.get('original', data.get('new-original'))
            if reference_data and len(data['LLMs']) > 1 and len(reference_data) > 1:
                # Shannon diversity test
                observed_diff, p_value, diffs = permutation_test(
                    data['LLMs'], reference_data, 'shannon', reps=args.num_bootstrap
                )
                print(f'Shannon {thing}; difference: {observed_diff:.4f}, p_value: {p_value:.4f} ({args.num_bootstrap} reps)')
                
                # Simpson diversity test
                observed_diff, p_value, diffs = permutation_test(
                    data['LLMs'], reference_data, 'simpson', reps=args.num_bootstrap
                )
                print(f'Simpson {thing}; difference: {observed_diff:.4f}, p_value: {p_value:.4f} ({args.num_bootstrap} reps)')


if __name__ == "__main__":
    main()
