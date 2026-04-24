import sys, os
import re
import json
import numpy as np
from collections import defaultdict
from erg import get_n_supertypes, populate_type_defs
from count_constructions import collect_types_multidir
from util import compute_cosine, print_cosine_similarities, serialize_dict, normalize_by_constr_count
from erg import types2defs, read_lexicon, lexical_types

ALL_HUMAN_AUTHORED = ["original", "wikipedia", "wsj"]
HUMAN_NYT = ["original"]
LLM_GENERATED = ['falcon_07', 'llama_07', 'llama_13', 'llama_30', 'llama_65', 'mistral_07']
LLM_NO_FALCON = ['llama_07', 'llama_13', 'llama_30', 'llama_65', 'mistral_07']

dataset_sizes = {
    'original': 26102, 'falcon_07': 27769, 'llama_07': 37825, 'llama_13': 37800,
    'llama_30': 37568, 'llama_65': 38107, 'mistral_07': 35086,
    'wikipedia': 10726, 'wsj': 43043, 'new-original': 29339,
}


def exclusive_members(freq, my_dataset, datasets_to_compare):
    """Return types that appear exclusively in my_dataset vs. the comparison datasets.

    Returns (only_in_mine, only_in_other), each a dict with 'constr', 'lexrule',
    'lextype' keys mapping to sets of type strings with count > 0 on that side only.
    """
    mine_set  = {'constr': set(), 'lexrule': set(), 'lextype': set()}
    other_set = {'constr': set(), 'lexrule': set(), 'lextype': set()}
    for rule_type in freq:
        for model, rules in freq[rule_type].items():
            if model == my_dataset:
                target = mine_set
            elif model in datasets_to_compare:
                target = other_set
            else:
                continue
            for rule, count in rules.items():
                if count > 0:
                    target[rule_type].add(rule)
    only_in_mine  = {rt: mine_set[rt]  - other_set[rt] for rt in mine_set}
    only_in_other = {rt: other_set[rt] - mine_set[rt]  for rt in other_set}
    return only_in_mine, only_in_other


def read_freq(data_dir, lex, depth):
    """Read per-model rule frequencies from .txt files under data_dir.

    Each subdirectory of data_dir is a model name; inside it, files named
    constr.txt, lexrule.txt, lextype.txt list (count, rule) pairs one per line.
    Lextype rules are resolved to a supertype at hierarchy depth if depth > 0,
    and rare lextypes (normalized count < 1% of dataset size) are discarded.
    Returns {rule_type: {model: {rule: count}}} with zeros filled in for missing rules.
    """
    freq_by_model = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    all_rules = {'lexrule': set(), 'constr': set(), 'lextype': set()}

    for model in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, model)
        if not os.path.isdir(subdir_path):
            continue
        for f in os.listdir(subdir_path):
            if not f.endswith('.txt'):
                continue
            rule_type = f[:-len('.txt')]
            f_path = os.path.join(subdir_path, f)
            with open(f_path, 'r') as fh:
                lines = fh.readlines()
            freq_by_model[rule_type].setdefault(model, {})
            for ln in lines:
                parts = ln.strip().split(' ')
                freq = int(parts[0].strip())
                rule = parts[1].strip().strip('/\\"')
                if 'u_unknown' in rule:
                    rule = re.sub(r"([A-Z]+_u_unknown(_rel)?)(<\d+:\d+>)", r"\1\2", rule)
                model_rules = freq_by_model[rule_type][model]
                if rule_type == 'lextype':
                    if freq / dataset_sizes[model] < 0.01:
                        continue
                    if depth > 0:
                        supertypes = get_n_supertypes(lex, rule, depth)
                        if supertypes:
                            for st in supertypes[depth - 1]:
                                model_rules[st] = model_rules.get(st, 0) + freq
                    else:
                        model_rules[rule] = model_rules.get(rule, 0) + freq
                else:
                    model_rules[rule] = model_rules.get(rule, 0) + freq

    for rule_type, models in freq_by_model.items():
        for rules in models.values():
            all_rules[rule_type].update(rules)
    for rule_type in all_rules:
        for model in freq_by_model[rule_type]:
            for rule in all_rules[rule_type]:
                if rule not in freq_by_model[rule_type][model]:
                    freq_by_model[rule_type][model][rule] = 0
    return freq_by_model


def normalize_by_num_sen(freq_by_model):
    """Return (normalized_freq, reverse_freq) with counts divided by dataset size.

    normalized_freq: {rule_type: {model: {rule: normalized_count}}} sorted descending.
    reverse_freq:    same data sorted ascending (for bottom-N queries).
    The input freq_by_model is not modified.
    """
    normalized = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    reverse_freq = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    for key in ['constr', 'lexrule', 'lextype']:
        for model, rules in freq_by_model[key].items():
            normed = {rule: count / dataset_sizes[model] for rule, count in rules.items()}
            normalized[key][model] = dict(sorted(
                normed.items(), key=lambda item: (item[1], item[0]), reverse=True
            ))
            reverse_freq[key][model] = dict(sorted(
                normed.items(), key=lambda item: (item[1], item[0]), reverse=False
            ))
    return normalized, reverse_freq


def separate_dataset(all_datasets, dataset_name):
    """Extract a single dataset's rules from a combined frequencies dict.

    Returns {rule_type: {rule: count}} for the named dataset only.
    """
    dataset = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    for rule_type in all_datasets:
        if dataset_name in all_datasets[rule_type]:
            dataset[rule_type] = dict(all_datasets[rule_type][dataset_name])
    return dataset


def read_dataset(frequencies, new_dataset, dataset_name):
    """Copy rules from new_dataset (a flat {rule_type: {rule: count}} dict) into frequencies."""
    for rule_type in frequencies:
        frequencies[rule_type][dataset_name] = dict(new_dataset[rule_type])


def add_new_dataset(frequencies, new_dataset, dataset_name, model_name='original'):
    """Add a new dataset into a combined frequencies dict, zero-filling missing rules.

    Copies rules from new_dataset[rule_type][model_name] into frequencies under dataset_name.
    Rules present in existing models but absent in the new dataset are set to 0 in the new
    entry. Rules present in the new dataset but absent in existing models are set to 0 there.
    """
    for rule_type in frequencies:
        existing_rules = {r for m in frequencies[rule_type] for r in frequencies[rule_type][m]}
        frequencies[rule_type][dataset_name] = {}
        for rule in new_dataset[rule_type][model_name]:
            frequencies[rule_type][dataset_name][rule] = new_dataset[rule_type][model_name][rule]
            for model in frequencies[rule_type]:
                if rule not in frequencies[rule_type][model]:
                    frequencies[rule_type][model][rule] = 0
        for rule in existing_rules:
            if rule not in frequencies[rule_type][dataset_name]:
                frequencies[rule_type][dataset_name][rule] = 0


def build_llm_vs_human(frequencies):
    """Aggregate all LLM models into a single 'llm' entry alongside the human baselines.

    Combines raw counts from all LLM_GENERATED models using combine_datasets, then
    returns a frequencies dict containing only 'llm' and ALL_HUMAN_AUTHORED models.
    Human models absent from frequencies are silently omitted.
    """
    llm_combined = combine_datasets(frequencies, LLM_GENERATED, 'llm')
    merged = {rt: {} for rt in frequencies}
    for rt in frequencies:
        merged[rt]['llm'] = llm_combined[rt]['llm']
        for human in ALL_HUMAN_AUTHORED:
            if human in frequencies[rt]:
                merged[rt][human] = frequencies[rt][human]
    return merged


def combine_types(data, relevant_keys):
    """Merge selected rule-type categories into a single flat {dataset: {rule: count}} dict.

    relevant_keys is a list of rule-type strings (e.g. ['constr', 'lexrule']) to include.
    Returns {dataset_name: {rule: total_count}}.
    """
    combined_data = defaultdict(lambda: defaultdict(int))
    for constr_type, datasets in data.items():
        if constr_type not in relevant_keys:
            continue
        for dataset_name, constructions in datasets.items():
            for constr, count in constructions.items():
                combined_data[dataset_name][constr] += count
    return {dataset: dict(constrs) for dataset, constrs in combined_data.items()}


def combine_datasets(data, dataset_names, new_name):
    """Sum counts from multiple named datasets into a single new dataset entry.

    Returns a frequencies dict with only new_name as a dataset key.
    """
    combined = {'lexrule': {new_name: {}}, 'constr': {new_name: {}}, 'lextype': {new_name: {}}}
    for dataset_name in dataset_names:
        for rule_type in data:
            if dataset_name not in data[rule_type]:
                continue
            for rule, freq in data[rule_type][dataset_name].items():
                combined[rule_type][new_name][rule] = combined[rule_type][new_name].get(rule, 0) + freq
    return combined


def combine_lextype_datasets(data, dataset_names):
    """Merge lextype membership data across multiple datasets into one combined dict.

    data: {dataset_name: {membership_tier: {lextype: [word, ...]}}}
    Returns {membership_tier: {lextype: [word, ...]}} with word lists unioned across datasets.
    """
    combined_data = {'high membership': {}, 'low membership': {}, 'singletons': {}}
    for dataset_name in dataset_names:
        for membership in data[dataset_name]:
            for lextype in data[dataset_name][membership]:
                combined_data[membership].setdefault(lextype, set()).update(data[dataset_name][membership][lextype])
    for membership in combined_data:
        for lextype in combined_data[membership]:
            combined_data[membership][lextype] = list(combined_data[membership][lextype])
    return combined_data


def compare_with_other_datasets(selected_dataset, cosine_similarities):
    """Compare selected_dataset's cosine distance to each other dataset against the max
    distance between those other datasets.

    Returns {other_dataset: (is_more_different, (diff_with_selected, diff_between_others))}.
    """
    selected_sims = {}
    max_other_sims = {}
    for (dataset1, dataset2), similarity in cosine_similarities.items():
        if selected_dataset in (dataset1, dataset2):
            other = dataset1 if selected_dataset == dataset2 else dataset2
            selected_sims[other] = similarity
        else:
            max_other_sims.setdefault(dataset1, {})[dataset2] = similarity
            max_other_sims.setdefault(dataset2, {})[dataset1] = similarity
    comparison_results = {}
    for other, selected_sim in selected_sims.items():
        max_sim_between_others = max(max_other_sims[other].values())
        diff_with_selected   = 1 - selected_sim
        diff_between_others  = 1 - max_sim_between_others
        comparison_results[other] = (diff_with_selected > diff_between_others,
                                     (diff_with_selected, diff_between_others))
    return comparison_results


def compare_human_vs_machine(my_dataset, cosine_similarities, human_datasets, machine_datasets):
    """Summarize how my_dataset's cosine similarity to humans and LLMs compares.

    Returns (avg_machine_similarity, avg_similarity_with_machine, similarities_with_human).
    """
    similarities_with_human = {}
    similarities_with_machine = {}
    similarities_between_machine = {}
    for (dataset1, dataset2), similarity in cosine_similarities.items():
        if dataset1 == my_dataset:
            if dataset2 in human_datasets:
                similarities_with_human[dataset2] = similarity
            elif dataset2 in machine_datasets:
                similarities_with_machine[dataset2] = similarity
        elif dataset2 == my_dataset:
            if dataset1 in human_datasets:
                similarities_with_human[dataset1] = similarity
            elif dataset1 in machine_datasets:
                similarities_with_machine[dataset1] = similarity
        elif dataset1 in machine_datasets and dataset2 in machine_datasets:
            similarities_between_machine[(dataset1, dataset2)] = similarity
    avg_machine_similarity    = np.mean(list(similarities_between_machine.values())) if similarities_between_machine else 0
    avg_similarity_with_machine = np.mean(list(similarities_with_machine.values())) if similarities_with_machine else 0
    return avg_machine_similarity, avg_similarity_with_machine, similarities_with_human


def report_comparison(my_dataset, machine_datasets, human_datasets, cosines):
    """Print a human-readable similarity report for my_dataset vs. humans and LLMs."""
    avg_machine, avg_with_machine, all_human = compare_human_vs_machine(
        my_dataset, cosines, human_datasets, machine_datasets
    )
    print("Average similarity between machine-generated datasets: {:.4f}".format(avg_machine))
    print("Average similarity between {} and machine-generated datasets: {:.4f}".format(
        my_dataset, avg_with_machine))
    print("Similarities between {} and human-authored datasets:".format(my_dataset))
    for dataset, similarity in all_human.items():
        print(f"{dataset}: {similarity}")
    print('All data:')
    print_cosine_similarities(cosines)


def categorize_lexentries(lexentries_nyt_wsj_wiki, word2membership, freq_threshold, discard_top):
    """Split lexical entries into high- and low-frequency groups, then map each to its membership tier.

    The top (1/freq_threshold) fraction of entries (after skipping discard_top) are
    'high frequency'; the rest are 'low frequency'. Each group is further subdivided by
    membership tier ('high membership', 'low membership', 'singletons') via word2membership.
    Returns (high_freq_lexentries, low_freq_lexentries).
    """
    high_freq_lexentries = {}
    low_freq_lexentries = {}
    for model in lexentries_nyt_wsj_wiki:
        high_freq_lexentries[model] = {'high membership': {}, 'low membership': {}, 'singletons': {}}
        low_freq_lexentries[model]  = {'high membership': {}, 'low membership': {}, 'singletons': {}}
        threshold_n = len(lexentries_nyt_wsj_wiki[model]) // freq_threshold
        for lexentry in list(lexentries_nyt_wsj_wiki[model].keys())[discard_top:threshold_n]:
            add_membership_to_freq(high_freq_lexentries, lexentry, model, word2membership)
        for lexentry in list(lexentries_nyt_wsj_wiki[model].keys())[threshold_n:]:
            add_membership_to_freq(low_freq_lexentries, lexentry, model, word2membership)
    return high_freq_lexentries, low_freq_lexentries


_TIER_TO_BUCKET = {'high': 'high membership', 'low': 'low membership', 'singleton': 'singletons'}


def add_membership_to_freq(lexentries, lexentry, model, word2membership):
    """Append lexentry to the appropriate membership tier bucket for model in lexentries."""
    if lexentry not in word2membership:
        print("{} not in dictionary".format(lexentry))
        return
    membership, lt = word2membership[lexentry]
    lexentries[model][_TIER_TO_BUCKET[membership]].setdefault(lt, []).append(lexentry)


def map_word2membership(high_membership, low_membership, singletons):
    """Build a word-to-(membership, lextype) lookup from three membership-tier dicts.

    Returns {word: (tier_string, lextype)} where tier_string is 'high', 'low', or 'singleton'.
    """
    word2membership = {}
    for lextype, words in high_membership.items():
        for word in words:
            word2membership[word] = ('high', lextype)
    for lextype, words in low_membership.items():
        for word in words:
            word2membership[word] = ('low', lextype)
    for lextype, words in singletons.items():
        for word in words:
            word2membership[word] = ('singleton', lextype)
    return word2membership


def find_absolute_diffs_lextype(model1, model2, model1_name, model2_name):
    """Find which lextypes and words each model shares with or has exclusively over the other.

    Returns (both, only_one) where:
      both[membership][lextype] = {'same': [...], 'different': {model1_name: [...], model2_name: [...]}}
      only_one[model_name][membership][lextype] = [...]
    """
    both = {'high membership': {}, 'low membership': {}, 'singletons': {}}
    only_one = {
        model1_name: {'high membership': {}, 'low membership': {}, 'singletons': {}},
        model2_name: {'high membership': {}, 'low membership': {}, 'singletons': {}},
    }
    for membership in model1:
        for lextype in model1[membership]:
            if lextype in model2[membership]:
                s1 = set(model1[membership][lextype])
                s2 = set(model2[membership][lextype])
                both[membership][lextype] = {
                    'same':      list(s1 & s2),
                    'different': {model1_name: list(s1 - s2), model2_name: list(s2 - s1)},
                }
            else:
                only_one[model1_name][membership][lextype] = list(model1[membership][lextype])
    for membership in model2:
        for lextype in model2[membership]:
            if lextype not in model1[membership]:
                only_one[model2_name][membership][lextype] = list(model2[membership][lextype])
    return both, only_one


def compare_lexentries(data):
    """Find lexical entries present only in LLM output vs. only in human-authored text.

    data: {model_name: {lexentry: count}}
    Returns (only_in_llm_d, only_in_human_d) where each is a dict keyed by model name
    (plus 'all llms' / 'not in any llm' aggregate keys).
    """
    only_in_llm_d   = {'all llms': {}}
    only_in_human_d = {'not in any llm': {}}
    llm_lexentries  = {}
    all_llms_lexentries = set()
    human_lexentries    = set()
    for model in data:
        if model in LLM_GENERATED:
            llm_lexentries[model] = set()
            only_in_human_d[model] = {}
            only_in_llm_d[model]   = {}
            for le in data[model]:
                all_llms_lexentries.add(le)
                llm_lexentries[model].add(le)
        elif model in HUMAN_NYT:
            for le in data[model]:
                human_lexentries.add(le)
    for model in data:
        if model in LLM_GENERATED:
            for le in data[model]:
                if le not in human_lexentries:
                    only_in_llm_d[model][le] = data[model][le]
                    only_in_llm_d['all llms'][le] = only_in_llm_d['all llms'].get(le, 0) + data[model][le]
        elif model in HUMAN_NYT:
            for le in data[model]:
                if le not in all_llms_lexentries:
                    only_in_human_d['not in any llm'][le] = data[model][le]
                for llm in llm_lexentries:
                    if le not in llm_lexentries[llm]:
                        only_in_human_d[llm][le] = data[model][le]
    return only_in_llm_d, only_in_human_d


def map2lt(data, word2lt):
    """Map each word in data to its lexical type, grouped by model.

    data: {model: {word: count}}
    word2lt: {word: (membership, lextype)}
    Returns {model: {lextype: [word, ...]}}.
    """
    mapping = {}
    for model in data:
        mapping[model] = {}
        for le in data[model]:
            if le not in word2lt:
                print("{} not in dictionary".format(le))
            else:
                membership, lt = word2lt[le]
                mapping[model].setdefault(lt, []).append(le)
    return mapping


def define_percentiles(sorted_frequencies):
    """Partition sorted_frequencies values into percentile-based segments.

    Returns (segments, lt2seg) where:
      segments: {percentile: {key: value}} for percentiles [10, 25, 50, 75, 90]
      lt2seg:   {key: percentile}
    """
    values = list(sorted_frequencies.values())
    percentiles = [10, 25, 50, 75, 90]
    percentile_thresholds = np.percentile(values, percentiles)
    segments = {}
    for i, p in enumerate(percentiles):
        lower_bound = percentile_thresholds[i - 1] if i > 0 else 0
        upper_bound = percentile_thresholds[i]
        segments[p] = {k: v for k, v in sorted_frequencies.items() if lower_bound <= v < upper_bound}
    lt2seg = {lt: p for p, seg in segments.items() for lt in seg}
    return segments, lt2seg


def analyze_lextypes_by_freq(norm_freq, lextypes, dataset):
    """Group lextypes by their frequency percentile segment within dataset.

    Returns {percentile: [lextype, ...]} for each segment defined by define_percentiles.
    """
    segments, lt2seg = define_percentiles(norm_freq['lextype'][dataset])
    by_seg = {seg: [] for seg in segments}
    for lt in lextypes:
        if lt not in norm_freq['lextype'][dataset]:
            print(f"{lt} not in frequency data")
        else:
            by_seg[lt2seg[lt]].append(lt)
    return by_seg


if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    with open('analysis/frequencies-json/frequencies-models-wiki-wsj.json', 'r') as f:
        all_data = json.load(f)
    with open('analysis/frequencies-json/02-05-2025-original-frequencies.json', 'r') as f:
        new_original_data = json.load(f)
    add_new_dataset(all_data, new_original_data, 'new-original')
    normalized_frequencies = normalize_by_constr_count(all_data)
    all_data        = combine_types(normalized_frequencies, ['constr', 'lexrule', 'lextype'])
    only_syntactic  = combine_types(normalized_frequencies, ['constr'])
    only_lexical    = combine_types(normalized_frequencies, ['lexrule'])
    only_vocab      = combine_types(normalized_frequencies, ['lextype'])
    syntactic_cosines = compute_cosine(only_syntactic)
    lexical_cosines   = compute_cosine(only_lexical)
    vocab_cosines     = compute_cosine(only_vocab)
    serialize_dict(vocab_cosines,     'analysis/cosine-pairs/models/norm-by-constr-count/2025/lextype-only.json')
    serialize_dict(syntactic_cosines, 'analysis/cosine-pairs/models/norm-by-constr-count/2025/syntax-only.json')
    serialize_dict(lexical_cosines,   'analysis/cosine-pairs/models/norm-by-constr-count/2025/lexrule-only.json')
