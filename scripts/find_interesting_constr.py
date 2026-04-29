"""
find_interesting_constr.py — identify statistically significant and hapax constructions.

Loads and merges one or more frequency JSON files, then:
  1. Runs Mann-Whitney U tests (with two-stage BH correction) to find constructions
     whose normalized frequency differs significantly between human and LLM models.
  2. Finds hapax-mismatch constructions: rare (< hapax_threshold) in NYT human models
     but common in LLMs, or vice versa.

Model classification is automatic: names containing '-human' are treated as human-authored;
all others are treated as LLM-generated.  Names also containing 'nyt' (case-insensitive)
are used as the NYT baseline for hapax analysis.

Output written to output_dir:
  {constr,lexrule,lextype}_{frequent,infrequent}.txt  — significant constructions
  significant_constr.json                             — full significant-constr dict
  hapax_constr.json                                   — full hapax-mismatch dict

Usage (run from repo root):
    python scripts/find_interesting_constr.py <frequencies_json> [<frequencies_json> ...] [options]

Example:
    python scripts/find_interesting_constr.py \\
        analysis/frequencies-json/frequencies-2023.json \\
        analysis/frequencies-json/frequencies-2025.json

Options:
    --output-dir DIR     Output directory (default: analysis/constructions)
    --threshold INT      Percentile for frequent/infrequent split (default: 80)
    --hapax-threshold INT  Max count to be considered hapax (default: 2)
"""

import sys, os, json, argparse
import scipy.stats as stats
import numpy as np
from statsmodels.stats import multitest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import normalize_by_constr_count, sort_normalized_data

_DEFAULT_OUTPUT_DIR = os.path.join('analysis', 'constructions')
_PHENOMENA = ['constr', 'lexrule', 'lextype']


def is_human(model):
    return '-human' in model.lower()


def is_nyt(model):
    return is_human(model) and 'nyt' in model.lower()


def merge_frequencies(*json_files):
    """Load and merge multiple frequency JSON files into one dict.

    Later files overwrite models with the same name.  Phenomena keys are unioned.
    Returns {phenomenon: {model: {type: count}}}.
    """
    merged = {}
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        for phenomenon, models in data.items():
            merged.setdefault(phenomenon, {}).update(models)
    return merged


def _split_by_frequency(phenom_data, threshold_percentile):
    """Split types into frequent and infrequent sets by mean frequency percentile.

    phenom_data: {model: {type: freq}}
    Returns (frequent_types, infrequent_types) as sets of type names.
    """
    first_model = next(iter(phenom_data))
    mean_freqs  = {
        t: np.mean([phenom_data[m].get(t, 0) for m in phenom_data])
        for t in phenom_data[first_model]
    }
    threshold  = np.percentile(list(mean_freqs.values()), threshold_percentile)
    frequent   = {t for t, f in mean_freqs.items() if f >= threshold}
    infrequent = {t for t, f in mean_freqs.items() if f <  threshold}
    return frequent, infrequent


def find_significant_constr(normalized_freq, threshold_percentile=80):
    """Return constructions with significantly different normalized frequencies.

    Compares human vs LLM models (detected by '-human' suffix) using Mann-Whitney U
    tests with two-stage BH multiple-testing correction (alpha=0.1).

    normalized_freq:      {phenomenon: {model: {type: freq}}}
    threshold_percentile: percentile split between 'frequent' and 'infrequent'

    Returns {phenomenon: {'frequent': {type: p_value}, 'infrequent': {type: p_value}}}.
    """
    significant = {p: {'frequent': {}, 'infrequent': {}} for p in _PHENOMENA}
    p_values    = {'frequent': [], 'infrequent': []}

    for phenomenon in _PHENOMENA:
        if phenomenon not in normalized_freq:
            continue
        human_models = [m for m in normalized_freq[phenomenon] if     is_human(m)]
        llm_models   = [m for m in normalized_freq[phenomenon] if not is_human(m)]
        frequent, infrequent = _split_by_frequency(normalized_freq[phenomenon], threshold_percentile)
        _test_tier(phenomenon, frequent,   normalized_freq, p_values, significant, 'frequent',   human_models, llm_models)
        _test_tier(phenomenon, infrequent, normalized_freq, p_values, significant, 'infrequent', human_models, llm_models)

    # Apply multiple-testing correction across all collected p-values.
    # NOTE: correction results are not currently used to re-filter significant;
    # this is a known limitation inherited from the original analysis.
    if p_values['frequent']:
        multitest.multipletests(p_values['frequent'],   alpha=0.1, method='fdr_tsbh')
    if p_values['infrequent']:
        multitest.multipletests(p_values['infrequent'], alpha=0.1, method='fdr_tsbh')
    return significant


def find_hapax_constr(raw_counts, infrequent_threshold=2):
    """Return constructions rare in one group but common in the other.

    Compares NYT human models (names containing 'nyt' and '-human') against all
    LLM models.

    raw_counts:           {phenomenon: {model: {type: count}}}
    infrequent_threshold: max count to be considered hapax (exclusive)

    Returns {phenomenon: {type: {'human count': int, 'llm count': int}}}.
    """
    special_cases = {p: {} for p in _PHENOMENA}
    for phenomenon in _PHENOMENA:
        if phenomenon not in raw_counts:
            continue
        nyt_models = [m for m in raw_counts[phenomenon] if     is_nyt(m)]
        llm_models = [m for m in raw_counts[phenomenon] if not is_human(m)]
        first_model = next(iter(raw_counts[phenomenon]))
        for type_name in raw_counts[phenomenon][first_model]:
            human_freq   = np.array([raw_counts[phenomenon][m].get(type_name, 0) for m in nyt_models])
            machine_freq = np.array([raw_counts[phenomenon][m].get(type_name, 0) for m in llm_models])
            human_hapax   = np.all(human_freq   < infrequent_threshold)
            machine_hapax = np.all(machine_freq < infrequent_threshold)
            if human_hapax != machine_hapax:
                special_cases[phenomenon][type_name] = {
                    'human count': int(human_freq.sum()),
                    'llm count':   int(machine_freq.sum()),
                }
    return special_cases


def _test_tier(phenomenon, constructions, normalized_freq, p_values, significant, k, human_models, llm_models):
    """Run Mann-Whitney U tests for one frequency tier and collect p-values."""
    for type_name in constructions:
        human_freqs = [normalized_freq[phenomenon][m].get(type_name, 0) for m in human_models]
        llm_freqs   = [normalized_freq[phenomenon][m].get(type_name, 0) for m in llm_models]
        if human_freqs and llm_freqs:
            try:
                _, p = stats.mannwhitneyu(human_freqs, llm_freqs, alternative='two-sided')
            except ValueError:
                p = 1.0  # identical values — not significant
            if p < 0.05:
                significant[phenomenon][k][type_name] = p
            p_values[k].append(p)
        else:
            p_values[k].append(1.0)


def _write_significant(significant, output_dir):
    """Write per-phenomenon text files listing significant constructions."""
    for phenomenon, tiers in significant.items():
        for tier, constructions in tiers.items():
            path = os.path.join(output_dir, f'{phenomenon}_{tier}.txt')
            with open(path, 'w') as f:
                for type_name, p in constructions.items():
                    f.write(f'{type_name} - p-value: {p}\n')


def main(json_files, output_dir=_DEFAULT_OUTPUT_DIR, threshold=80, hapax_threshold=2):
    os.makedirs(output_dir, exist_ok=True)
    frequencies = merge_frequencies(*json_files)
    normalized  = normalize_by_constr_count(frequencies)
    _, descending = sort_normalized_data(normalized)

    significant = find_significant_constr(descending, threshold)
    hapax       = find_hapax_constr(frequencies, hapax_threshold)

    _write_significant(significant, output_dir)
    with open(os.path.join(output_dir, 'significant_constr.json'), 'w') as f:
        json.dump(significant, f, ensure_ascii=False)
    with open(os.path.join(output_dir, 'hapax_constr.json'), 'w') as f:
        json.dump(hapax, f, ensure_ascii=False)
    print(f'Output written to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('frequencies_json', nargs='+',
                        help='One or more JSON files with {phenomenon: {model: {type: count}}}')
    parser.add_argument('--output-dir', default=_DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {_DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--threshold', type=int, default=80,
                        help='Percentile split for frequent/infrequent (default: 80)')
    parser.add_argument('--hapax-threshold', type=int, default=2,
                        help='Max count to be considered hapax (default: 2)')
    args = parser.parse_args()
    main(args.frequencies_json, args.output_dir, args.threshold, args.hapax_threshold)
