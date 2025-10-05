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

punct = True
ignored_models = ['tinyllama-2025-llm']

# Centralized style configuration
series = {
    "nyt": {
        "label": "Human (NYT)",
        "color": "#B44E3A",  # chestnut brown
        "models": ['original', 'original_2025', 'nyt_2023_human', 'nyt_human-2025', 'nyt-2025-human'],
        "marker": "*",
        "markersize_scatter": 40,
        "markersize_line": 10,
        "linestyle": "-",
        "linewidth": 2
    },
    "human": {
        "label": "Human (WSJ/Wiki)",
        "color": "#D17F4A",  # copper
        "models": ['wsj', 'wikipedia', 'wescience'],
        "marker": "+",
        "markersize_scatter": 40,
        "markersize_line": 8,
        "linestyle": "-",
        "linewidth": 2
    },
    "llm2023": {
        "label": "LLM (2023)",
        "color": "#3A6DB4",  # cobalt blue
        "models": [],  # determined by year in name
        "marker": "o",
        "markersize_scatter": 20,
        "markersize_line": 6,
        "linestyle": "--",
        "linewidth": 1.5
    },
    "llm2025": {
        "label": "LLM (2025)",
        "color": "#4AA6B4",  # teal-blue
        "models": [],  # determined by year in name
        "marker": "o",
        "markersize_scatter": 20,
        "markersize_line": 6,
        "linestyle": "--",
        "linewidth": 1.5
    },
    "llms": {
        "label": "LLM (all)",
        "color": "black",
        "models": ['LLMs'],
        "marker": "o",
        "markersize_scatter": 80,
        "markersize_line": 8,
        "linestyle": "-",
        "linewidth": 2.5
    },
    "llms2023": {
        "label": "LLMs (2023)",
        "color": "#3A6DB4",  # cobalt blue
        "models": [],  # determined by year in name
        "marker": "o",
        "markersize_scatter": 60,
        "markersize_line": 8,
        "linestyle": "--",
        "linewidth": 1.5
    },
    "llms2025": {
        "label": "LLMs (2025)",
        "color": "#4AA6B4",  # teal-blue
        "models": [],  # determined by year in name
        "marker": "o",
        "markersize_scatter": 60,
        "markersize_line": 8,
        "linestyle": "--",
        "linewidth": 1.5
    },
    
}

# Convenience lists
nyt_models = series["nyt"]["models"]
human_models = series["nyt"]["models"] + series["human"]["models"]
all_human_models = human_models.copy()



def get_series_key(model_name):
    """Determine which series a model belongs to."""
   
    if model_name == 'LLMs (2023)':
        return 'llms2023'
    elif model_name == 'LLMs (2025)':
        return 'llms2025'

    mname = model_name.lower().split(' (')[0]
 
    
    # Check explicit model lists
    for key in ['nyt', 'human', 'llms']:
        if mname in [m.lower() for m in series[key]["models"]]:
            return key

    # Check LLM year-based categories
    if '(2023)' in model_name.lower():
        return 'llm2023'
    elif '(2025)' in model_name.lower():
        return 'llm2025'
   
     
    # Default to llm2025 for other LLMs
    return 'llm2025'


def is_human_model(model_name):
    """Check if a model is a human dataset."""
    mname = model_name.lower().split(' (')[0]
    return mname in [m.lower() for m in all_human_models]


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
    """
    Load and aggregate linguistic phenomenon data from multiple JSON files.
    
    This function processes JSON files containing linguistic analysis results for different
    language models, combining data across files and organizing it by phenomenon type first,
    then by model. It creates aggregated categories for all LLMs and year-specific LLM groups.
    
    Args:
        json_files (list of str): Paths to JSON files to load. Each file should contain
            data structured as: {phenomenon: {model: {type_name: count, ...}, ...}, ...}
        phenomena (list of str, optional): List of phenomenon types to extract from the
            JSON files. Defaults to ['constr'] (constructions).
    
    Returns:
        tuple: A tuple containing:
            - data (dict): Dictionary structured as {phenomenon: {model: [instances], ...}, ...}
              where:
                * First level keys are phenomenon names (e.g., 'constr', 'syntax')
                * Second level keys include:
                    - Individual model names in format "ModelName (year)"
                    - 'LLMs': All non-human model instances combined
                    - 'LLMs (2023)': All 2023 model instances combined
                    - 'LLMs (2025)': All 2025 model instances combined
                * Values are lists where each phenomenon type appears count times.
            - model_to_file_map (dict): Dictionary mapping unique model names 
              (format: "ModelName (year)") to their source JSON file paths.
    
    Example:
        >>> files = ['results_2023.json', 'results_2025.json']
        >>> data, file_map = load_data_from_json(files, phenomena=['constr', 'syntax'])
        >>> print(data.keys())
        dict_keys(['constr', 'syntax'])
        >>> print(data['constr'].keys())
        dict_keys(['GPT-4 (2023)', 'Claude (2025)', 'LLMs', 'LLMs (2023)', 'LLMs (2025)'])
        >>> print(len(data['constr']['LLMs']))  # Total count for this phenomenon
        1247
    
    Notes:
        - Model names are made unique by appending the year from the filename in 
          parentheses (extracted via get_year_from_filepath).
        - Human models (identified via is_human_model) are excluded from the aggregate
          'LLMs' categories.
        - Count values in the source JSON are expanded into repeated instances in the
          output lists (e.g., {"passive": 3} becomes ["passive", "passive", "passive"]).
        - If a phenomenon doesn't exist in the source data, it's silently skipped.
    """
    data = {}
    model_to_file_map = {}
    combined_data = {}
    
    # Load and combine data from all JSON files
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            for phenomenon in file_data:
                if phenomenon not in combined_data:
                    combined_data[phenomenon] = {}
                for model in file_data[phenomenon]:
                    if model in ignored_models:
                        continue
                    year = get_year_from_filepath(json_file)
                    unique_model_name = f"{model} ({year})"
                    if unique_model_name not in combined_data[phenomenon]:
                        combined_data[phenomenon][unique_model_name] = {}
                    combined_data[phenomenon][unique_model_name].update(
                        file_data[phenomenon][model]
                    )
                    model_to_file_map[unique_model_name] = json_file
    
    # Extract and expand data for requested phenomena
    for phenomenon in phenomena:
        if phenomenon in combined_data:
            # Initialize the phenomenon dict with aggregate categories
            data[phenomenon] = {
                'LLMs': [],
                'LLMs (2023)': [],
                'LLMs (2025)': []
            }
            
            for unique_model_name in combined_data[phenomenon]:
                # Initialize model's data list
                data[phenomenon][unique_model_name] = []
                
                # Expand counts into repeated instances
                for type_name, count in combined_data[phenomenon][unique_model_name].items():
                    names = [type_name] * count
                    data[phenomenon][unique_model_name].extend(names)
                    
                    # Add to aggregate LLM categories (excluding human models)
                    if not is_human_model(unique_model_name):
                        data[phenomenon]['LLMs'].extend(names)
                        if '(2023)' in unique_model_name:
                            data[phenomenon]['LLMs (2023)'].extend(names)
                        elif '(2025)' in unique_model_name:
                            data[phenomenon]['LLMs (2025)'].extend(names)
                    
    return data, model_to_file_map

def split_punct(data):
    """
    For each phenomenon, add two new ones: one with punctuation types and one without.
    
    Creates new phenomenon categories by splitting each existing phenomenon into:
    - phenomenon_punct: Contains only types identified as punctuation-related
    - phenomenon_xpunct: Contains only types NOT identified as punctuation-related
    
    Args:
        data (dict): Dictionary structured as {phenomenon: {model: [type_instances], ...}, ...}
    
    Returns:
        dict: Original data augmented with _punct and _xpunct variants for each phenomenon.
    
    Example:
        >>> data = {'constr': {'GPT-4 (2023)': ['pt_comma', 'passive', 'pct_period']}}
        >>> result = split_punct(data)
        >>> print(result.keys())
        dict_keys(['constr', 'constr_punct', 'constr_xpunct'])
        >>> print(result['constr_punct']['GPT-4 (2023)'])
        ['pt_comma', 'pct_period']
        >>> print(result['constr_xpunct']['GPT-4 (2023)'])
        ['passive']
    
    Notes:
        - Punctuation types are identified by: starting with 'pt', containing 'pct', 
          or ending with 'plr'
        - Original phenomenon data remains unchanged
        - Uses defaultdict(list) for new phenomena to handle missing models gracefully
    """
    def is_punct(type_name):
        if type_name.startswith('pt') or ('pct' in type_name) or type_name.endswith('plr'):
            return True
        else:
            return False
        
    for phenomenon in list(data.keys()):  # Use list() to avoid modifying dict during iteration
        data[f"{phenomenon}_punct"] = dd(list)
        data[f"{phenomenon}_xpunct"] = dd(list)
        for model in data[phenomenon]:
            # Filter by is_punct
            data[f"{phenomenon}_punct"][model] = [t for t in data[phenomenon][model] if is_punct(t)]
            data[f"{phenomenon}_xpunct"][model] = [t for t in data[phenomenon][model] if not is_punct(t)]
    
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
    """
    counts = Counter(observations).values()
    N = sum(counts)

    if N <= 1:
        raise ValueError("Total number of observations must be greater than 1.")

    numerator = sum(n * n for n in counts)
    denominator = N * N

    D = 1 - (numerator / denominator)
    return D


def learning_curve_analysis(thing, data, output_dir, n_bins=5):
    """
    Analyze learning curves for diversity metrics.
    
    Args:
        thing: Name of the phenomenon being analyzed
        data: Dictionary of model data
        output_dir: Directory for output files
        n_bins: Number of bins to divide data into (default 5 for ~20% each)
    """
    print(f"\n=== Learning Curve Analysis for {thing} ===")
    print(f"Using {n_bins} bins (~{100/n_bins:.0f}% each)")
    
    # Prepare data for learning curves
    learning_data = {}
    
    for model in data:
        if len(data[model]) > n_bins:  # Need enough data for bins
            # Shuffle data to avoid ordering bias
            shuffled = data[model].copy()
            random.shuffle(shuffled)
            learning_data[model] = shuffled
    
    if not learning_data:
        print("Insufficient data for learning curve analysis")
        return
    
    # Calculate diversity for each cumulative bin
    results = {
        'Shannon': {},
        'Simpson': {}
    }
    
    for model in learning_data:
        print(f"  Processing {model} ({len(learning_data[model])} samples)...")
        results['Shannon'][model] = []
        results['Simpson'][model] = []
        
        total_samples = len(learning_data[model])
        
        # Incremental counting for efficiency
        name_counts = dd(int)
        
        for i in range(1, n_bins + 1):
            end_idx = int((i / n_bins) * total_samples)
            start_idx = int(((i-1) / n_bins) * total_samples) if i > 1 else 0
            
            # Update counts incrementally
            for name in learning_data[model][start_idx:end_idx]:
                name_counts[name] += 1
            
            current_total = end_idx
            
            if current_total > 1:
                # Shannon calculation
                shannon_div = 0
                for count in name_counts.values():
                    p = count / current_total
                    shannon_div -= p * math.log(p)
                results['Shannon'][model].append(shannon_div)
                
                # Simpson calculation
                numerator = sum(n * n for n in name_counts.values())
                simpson_div = 1 - (numerator / (current_total * current_total))
                results['Simpson'][model].append(simpson_div)
    
    # Creae learning curve plots with color scheme
    for index_type in ['Shannon', 'Simpson']:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Calculate percentage labels for x-axis
        x_values = [(i / n_bins) * 100 for i in range(1, n_bins + 1)]
        
        # Organize models by category for the legend
        models_by_category = {}
        for k in ['nyt', 'human', 'llm2023', 'llm2025', 'llms']:
            models_by_category[k] = []
        
        # Plot each model's learning curve
        for model in results[index_type]:
            if results[index_type][model]:
                # Get style for this model
                series_key = get_series_key(model)
                style = series[series_key]
                models_by_category[series_key].append(model)
                
                ax.plot(x_values, results[index_type][model], 
                       marker=style["marker"], 
                       linestyle=style["linestyle"], 
                       linewidth=style["linewidth"], 
                       markersize=style["markersize_line"],
                       color=style["color"], 
                       label=model, 
                       alpha=0.8)
        
        ax.set_xlabel('Percentage of Data (%)', fontsize=11)
        ax.set_ylabel(f'{index_type} Diversity Index', fontsize=11)
        ax.set_title(f'Learning Curve: {thing} ({index_type} Index)', 
                    fontsize=12, fontweight='bold')
        
        # Tufte style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Create custom legend on the left showing models organized by category
        legend_elements = []
        legend_labels = []
        
        # Add header
        legend_labels.append("Models (by initial value):")
        legend_elements.append(plt.Line2D([0], [0], color='none'))
        
        # Sort models within each category by their initial value
        for k in ['nyt', 'human', 'llm2023', 'llm2025', 'llms']:
            if models_by_category[k]:
                # Sort by initial diversity value
                sorted_models = sorted(models_by_category[k], 
                                     key=lambda m: results[index_type][m][0] if results[index_type][m] else 0)
                
                # Add category header
                legend_labels.append(f"\n{series[k]['label']}:")
                legend_elements.append(plt.Line2D([0], [0], color='none'))
                
                # Add each model with its marker
                for model in sorted_models:
                    legend_elements.append(plt.Line2D([0], [0], 
                                                     marker=series[k]['marker'],
                                                     color=series[k]['color'],
                                                     linestyle=series[k]['linestyle'],
                                                     markersize=6,
                                                     linewidth=1.5))
                    legend_labels.append(f"  {model}")
        
        # Place legend on the left side
        ax.legend(legend_elements, legend_labels, 
                 loc='center left', bbox_to_anchor=(-0.35, 0.5),
                 fontsize=8, frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(output_dir, 
                               f"learning-curve-{thing.replace(' ', '-').lower()}-{index_type.lower()}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved learning curve: {filename}")
    
    # Print summary statistics
    print("\nLearning Curve Summary:")
    for index_type in ['Shannon', 'Simpson']:
        print(f"\n{index_type} Index:")
        for model in results[index_type]:
            if results[index_type][model]:
                initial = results[index_type][model][0]
                final = results[index_type][model][-1]
                change = final - initial
                pct_change = (change / initial) * 100 if initial > 0 else 0
                print(f"  {model}: {initial:.3f} → {final:.3f} "
                      f"(+{change:.3f}, {pct_change:+.1f}%)")


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

    for idx in ('Shannon', 'Simpson'):
        models = list(diversity[idx].keys())
        scores = list(diversity[idx].values())

        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])
        models_sorted = [models[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]

        md_filename = os.path.join(output_dir, f"llm-erg-{thing.replace(' ', '-').lower()}-{idx.lower()}.md")
        with open(md_filename, 'w') as out:
            print("""| Model   | Diversity |
| --- | --- |""", file=out)
            for model, score in zip(models_sorted, scores_sorted):
                print(f"| {model} | {score:.3f} |", file=out)

        fig, ax = plt.subplots(figsize=(6, 4))

        for i, (score, model) in enumerate(zip(scores_sorted, models_sorted)):
            series_key = get_series_key(model)
            style = series[series_key]
            
            ax.scatter(score, i, 
                      color=style["color"], 
                      s=style["markersize_scatter"], 
                      marker=style["marker"])
                    
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
                                         label=series[k]['label']) 
                         for k in ['nyt', 'human', 'llm2023', 'llm2025', 'llms']]
        ax.legend(handles=legend_patches, loc='best', fontsize='small', title='Model Type')

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
                        choices=['constr', 'lexrule', 'lextype'],
                        default=['constr', 'lextype', 'lexrule'],
                        help='Phenomena to analyze (default: constr, lextype, lexrule)')
    parser.add_argument('--num-bootstrap', type=int, default=1,
                        help='Number of bootstrap/permutation iterations (default: 1)')
    parser.add_argument('--output-dir', type=str, default='out',
                        help='Output directory for generated files (default: out)')
    parser.add_argument('--learning', type=int, metavar='N',
                        help='Generate learning curves with N bins (e.g., 5 for ~20%% each)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output files will be saved to: {args.output_dir}")

    def name_phenomenon(phenomenon):
        phenomena_map = {
            'constr': 'Constructions',
            'lexrule': 'Lexical Rules',
            'lextype': 'Lexical Types'
        }
        parts = phenomenon.split('_')
        
        name = phenomena_map[parts[0]]
        if len(parts) > 1:
            if parts[1] == 'punct':
                name += ' (punctuation only)'
            elif parts[1] == 'xpunct':
                name += ' (no punctuation)'
        return name
            

    all_data, model_to_file_map = load_data_from_json(args.json_files,
                                                      args.phenomena)
    all_data=split_punct(all_data)
     
    for phenomenon in all_data:
        thing = name_phenomenon(phenomenon)
        print(f"\n=== Analyzing {thing} ===")

        data = all_data[phenomenon]

        if not data or len(data) <= 1:
            print(f"Insufficient data for {thing}. Skipping...")
            continue
            
        # Run learning curve analysis if requested
        if args.learning:
            learning_curve_analysis(thing, data, args.output_dir, n_bins=args.learning)
            
        analyze_diversity(thing, data, model_to_file_map, args.output_dir, args.json_files)
        
        # Perform statistical tests if we have both LLMs and original data
        if 'LLMs' in data and any(model in data for model in nyt_models):
            reference_data = []
            for model in nyt_models:
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
