import sys, os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from erg import get_n_supertypes, populate_type_defs
from count_constructions import collect_types_multidir
from util import (compute_cosine, print_cosine_similarities, serialize_dict, normalize_by_constr_count,
                  sort_normalized_data, freq_counts_by_model)
from erg import types2defs, read_lexicon, lexical_types

ALL_HUMAN_AUTHORED = ["original", "wescience", "wsj"]
HUMAN_NYT = ["original"]
LLM_GENERATED = ['falcon_07', 'llama_07', 'llama_13', 'llama_30', 'llama_65', 'mistral_07']
LLM_NO_FALCON = ['llama_07', 'llama_13', 'llama_30', 'llama_65', 'mistral_07']


dataset_sizes = {'original':26102,'falcon_07':27769, 'llama_07':37825, 'llama_13':37800,'llama_30':37568,
                 'llama_65':38107,'mistral_07':35086, 'wikipedia': 10726, 'wsj': 43043}

def exclusive_members(freq, my_dataset, datasets_to_compare):
    only_in_mine = {'constr': set(), 'lexrule': set(), 'lextype': set()}
    only_in_other = {'constr': set(), 'lexrule': set(), 'lextype': set()}
    mine_set = {'constr': set(), 'lexrule': set(), 'lextype': set()}
    others_set = {'constr': set(), 'lexrule': set(), 'lextype': set()}
    for rule_type in freq:
        for model in freq[rule_type]:
            if not model == my_dataset:
                if model in datasets_to_compare:
                    for rule in freq[rule_type][model]:
                        if freq[rule_type][model][rule] > 0:
                            others_set[rule_type].add(rule)
            else:
                for rule in freq[rule_type][model]:
                    if freq[rule_type][model][rule] > 0:
                        mine_set[rule_type].add(rule)
    for rule_type in mine_set:
        only_in_mine[rule_type] = mine_set[rule_type] - others_set[rule_type]
        only_in_other[rule_type] = others_set[rule_type] - mine_set[rule_type]
    return only_in_mine, only_in_other


def read_freq(data_dir, lex, depth):
    freq_by_model = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    all_rules = {'lexrule': set(), 'constr': set(), 'lextype': set()}
    for model in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, model)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                relevant_dict = f[:-len('.txt')]
                if f.endswith(".txt"):
                    f_path = os.path.join(subdir_path, f)
                    with open(f_path, 'r') as file:
                        lines = file.readlines()
                    for ln in [l.strip() for l in lines ]:
                        freq, rule = ln.split(' ')
                        freq, rule = int(freq.strip()), rule.strip()
                        rule = rule.strip('/\\\"')
                        if 'u_unknown' in rule:
                            pattern = r"([A-Z]+_u_unknown(_rel)?)(<\d+:\d+>)"
                            rule = re.sub(pattern, r"\1\2", rule)
                        if model not in freq_by_model[relevant_dict]:
                            freq_by_model[relevant_dict][model] = {}
                        if relevant_dict == 'lextype':
                            if freq/dataset_sizes[model] < 0.01:
                                continue
                            if depth > 0:
                                supertypes = get_n_supertypes(lex, rule, depth)
                                if supertypes:
                                    for st in supertypes[depth-1]:
                                        if st not in freq_by_model[relevant_dict][model]:
                                            freq_by_model[relevant_dict][model][st] = 0
                                        freq_by_model[relevant_dict][model][st] += freq
                            else:
                                if rule not in freq_by_model[relevant_dict][model]:
                                    freq_by_model[relevant_dict][model][rule] = 0
                                freq_by_model[relevant_dict][model][rule] += freq
                        else:
                            if rule not in freq_by_model[relevant_dict][model]:
                                freq_by_model[relevant_dict][model][rule] = 0
                            freq_by_model[relevant_dict][model][rule] += freq
    for rule_type in freq_by_model:
        for model in freq_by_model[rule_type]:
            for rule in freq_by_model[rule_type][model]:
                all_rules[rule_type].add(rule)
    for rule_type in all_rules:
        for model in freq_by_model[rule_type]:
            for rule in all_rules[rule_type]:
                if rule not in freq_by_model[rule_type][model]:
                    freq_by_model[rule_type][model][rule] = 0
    #reverse_freq_by_model = normalize_by_num_sen(freq_by_model)
    return freq_by_model #, reverse_freq_by_model


def normalize_by_num_sen(freq_by_model):
    reverse_freq = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    for key in ['constr', 'lexrule', 'lextype']:
        # Normalize frequency counts by dataset size:
        for model in freq_by_model[key]:
            for rule in freq_by_model[key][model]:
                freq_by_model[key][model][rule] /= dataset_sizes[model]
        # Re-sort the dict in descending order of frequency:
        for model in freq_by_model[key]:
            freq_by_model[key][model] = {k: v for k, v in sorted(freq_by_model[key][model].items(),
                                                                      key=lambda item: (item[1], item[0]), reverse=True)}
            reverse_freq[key][model] = {k: v for k, v in sorted(freq_by_model[key][model].items(),
                                                                      key=lambda item: (item[1], item[0]), reverse=False)}
    return reverse_freq

def separate_dataset(all_datasets, dataset_name):
    dataset = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    for rule_type in all_datasets:
        for dt in all_datasets[rule_type]:
            if dt == dataset_name:
                for rule in all_datasets[rule_type][dt]:
                    dataset[rule_type][rule] = all_datasets[rule_type][dt][rule]
    return dataset

def read_dataset(frequencies, new_dataset, dataset_name):
    for rule_type in frequencies:
        frequencies[rule_type][dataset_name] = {}
        for rule in new_dataset[rule_type]:
            frequencies[rule_type][dataset_name][rule] = new_dataset[rule_type][rule]

def add_new_dataset(frequencies, new_dataset, dataset_name):
    for rule_type in frequencies:
        frequencies[rule_type][dataset_name] = {}
        for rule in new_dataset[rule_type]:
            frequencies[rule_type][dataset_name][rule] = new_dataset[rule_type][rule]
            for model in frequencies[rule_type]:
                if rule not in frequencies[rule_type][model]:
                    frequencies[rule_type][model][rule] = 0
        for rule in frequencies[rule_type]['original']:
            if rule not in frequencies[rule_type][dataset_name]:
                frequencies[rule_type][dataset_name][rule] = 0

def visualize_counts(frequencies, reverse_frequencies):
    #reverse_frequencies = normalize_by_num_sen(frequencies)
    # only_in_original, not_in_original, only_in_original_per_model = exclusive_members(frequencies)
    human_authored = ['original', 'wikipedia', 'wsj']
    start, end = 0, 50
    for model in dataset_sizes.keys():
        if not model in human_authored:
            freq_counts_by_model(frequencies, model, human_authored[0], human_authored[1], human_authored[2], start,
                                 end, "Top frequencies", reverse=True)
            freq_counts_by_model(reverse_frequencies, model, human_authored[0], human_authored[1], human_authored[2],
                                 start, end, "Bottom frequencies", reverse=False)
            freq_counts_by_model(frequencies, human_authored[0], human_authored[1], human_authored[2], model, start,
                                 end, "Top frequencies", reverse=True)
            freq_counts_by_model(reverse_frequencies, human_authored[0], human_authored[1], human_authored[2], model,
                                 start, end, "Bottom frequencies", reverse=False)
            freq_counts_by_model(frequencies, human_authored[1], human_authored[0], human_authored[2], model, start,
                                 end, "Top frequencies",
                                 reverse=True)
            freq_counts_by_model(reverse_frequencies, human_authored[1], human_authored[0], human_authored[2], model,
                                 start, end,
                                 "Bottom frequencies", reverse=False)
            freq_counts_by_model(frequencies, human_authored[2], human_authored[0], human_authored[1], model, start,
                                 end, "Top frequencies", reverse=True)
            freq_counts_by_model(reverse_frequencies, human_authored[2], human_authored[0], human_authored[1], model,
                                 start, end, "Bottom frequencies", reverse=False)

def combine_types(data, relevant_keys):
    combined_data = defaultdict(lambda: defaultdict(int))
    for constr_type, datasets in data.items():
        if constr_type not in relevant_keys:
            continue
        for dataset_name, constructions in datasets.items():
            for constr, count in constructions.items():
                combined_data[dataset_name][constr] += count  # Add count to the corresponding dataset and constr
    combined_data = {dataset: dict(constrs) for dataset, constrs in combined_data.items()}
    return combined_data

def combine_datasets(data, dataset_names, new_name):
    combined_data = {'lexrule': {new_name: {}}, 'constr': {new_name: {}}, 'lextype': {new_name: {}}}
    for dataset_name in dataset_names:
        for rule_type in data:
            for rule, freq in data[rule_type][dataset_name].items():
                if rule not in combined_data[rule_type][new_name]:
                    combined_data[rule_type][new_name][rule] = 0
                combined_data[rule_type][new_name][rule] += freq
    return combined_data

def combine_lextype_datasets(data, dataset_names):
    combined_data = {'high membership': {}, 'low membership': {}, 'singletons': {}}
    for dataset_name in dataset_names:
        for membership in data[dataset_name]:
            for lextype in data[dataset_name][membership]:
                if lextype not in combined_data[membership]:
                    combined_data[membership][lextype] = set()
                combined_data[membership][lextype].update(data[dataset_name][membership][lextype])
    for membership in combined_data:
        for lextype in combined_data[membership]:
            combined_data[membership][lextype] = list(combined_data[membership][lextype])
    return combined_data

def compare_with_other_datasets(selected_dataset, cosine_similarities):
    selected_dataset_similarities = {}
    max_other_similarities = {}
    for (dataset1, dataset2), similarity in cosine_similarities.items():
        # If selected_dataset is in the pair, record the similarity with the other dataset
        if selected_dataset in (dataset1, dataset2):
            other_dataset = dataset1 if selected_dataset == dataset2 else dataset2
            selected_dataset_similarities[other_dataset] = similarity
        else:
            # For pairs excluding the selected dataset, track the max similarity between them
            if dataset1 not in max_other_similarities:
                max_other_similarities[dataset1] = {}
            if dataset2 not in max_other_similarities:
                max_other_similarities[dataset2] = {}
            max_other_similarities[dataset1][dataset2] = similarity
            max_other_similarities[dataset2][dataset1] = similarity
    # Now check if the selected dataset's differences are larger than those between other pairs
    comparison_results = {}
    for other_dataset, selected_similarity in selected_dataset_similarities.items():
        max_similarity_between_others = max(max_other_similarities[other_dataset].values())
        # Compare if the difference for selected dataset is greater
        difference_with_selected = 1 - selected_similarity  # Since cosine similarity is between 0 and 1
        difference_between_others = 1 - max_similarity_between_others
        comparison_results[other_dataset] = (difference_with_selected > difference_between_others, (difference_with_selected, difference_between_others))
    return comparison_results


def compare_human_vs_machine(my_dataset, cosine_similarities, human_datasets, machine_datasets):
    similarities_with_human = {}
    similarities_with_machine = {}
    similarities_between_machine = {}
    # Loop over all the cosine similarities to separate them based on dataset type
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
        else:
            if dataset1 in machine_datasets and dataset2 in machine_datasets:
                similarities_between_machine[(dataset1, dataset2)] = similarity
    # Average similarity between machine-generated datasets
    avg_machine_similarity = np.mean(list(similarities_between_machine.values())) if similarities_between_machine else 0
    # Average similarity between my_dataset and machine-generated datasets:
    avg_similarity_with_machine = np.mean(list(similarities_with_machine.values())) if similarities_with_machine else 0
    human_differences = [1 - sim for sim in similarities_with_machine.values()]
    machine_differences = [1 - sim for sim in similarities_between_machine.values()]
    #t_stat, p_value = stat_significance(human_differences, machine_differences)
    return avg_machine_similarity, avg_similarity_with_machine, similarities_with_human #, t_stat, p_value


def report_comparison(my_dataset, machine_datasets, human_datasets, cosines):
    avg_machine, avg_with_machine, all_human = compare_human_vs_machine(my_dataset, cosines, human_datasets,
                                                                        machine_datasets)
    # Print the results
    print("Average similarity between machine-generated datasets: {:.4f}".format(avg_machine))
    print("Average similarity between {} and machine-generated datasets: {:.4f}".format(my_dataset,
                                                                                                             avg_with_machine))
    print("Similarities between {} and human-authored datasets:".format(my_dataset))
    # Statistical significance between in all_human:

    for dataset, similarity in all_human.items():
        print(f"{dataset}: {similarity}")
    print('All data:')
    print_cosine_similarities(cosines)

def categorize_lexentries(lexentries_nyt_wsj_wiki, word2membership, freq_threshold, discard_top):
    high_freq_lexentries = {}
    low_freq_lexentries = {}
    for model in lexentries_nyt_wsj_wiki:
        high_freq_lexentries[model] = {'high membership': {}, 'low membership': {}, 'singletons': {}}
        low_freq_lexentries[model] = {'high membership': {}, 'low membership': {}, 'singletons': {}}
        threshold_n = len(lexentries_nyt_wsj_wiki[model])//freq_threshold
        for lexentry in list(lexentries_nyt_wsj_wiki[model].keys())[0+discard_top:threshold_n]:
            add_membership_to_freq(high_freq_lexentries, lexentry, model, word2membership)
        for lexentry in list(lexentries_nyt_wsj_wiki[model].keys())[threshold_n:]:
            add_membership_to_freq(low_freq_lexentries, lexentry, model, word2membership)
    return high_freq_lexentries, low_freq_lexentries


def add_membership_to_freq(lexentries, lexentry, model, word2membership):
    if lexentry not in word2membership:
        print("{} not in dictionary".format(lexentry))
    else:
        membership = word2membership[lexentry][0]
        lt = word2membership[lexentry][1]
        if membership == 'high':
            if not lt in lexentries[model]['high membership']:
                lexentries[model]['high membership'][lt] = []
            lexentries[model]['high membership'][lt].append(lexentry)
        elif membership == 'low':
            if not lt in lexentries[model]['low membership']:
                lexentries[model]['low membership'][lt] = []
            lexentries[model]['low membership'][lt].append(lexentry)
        elif membership == 'singleton':
            if not lt in lexentries[model]['singletons']:
                lexentries[model]['singletons'][lt] = []
            lexentries[model]['singletons'][lt].append(lexentry)

def map_word2membership(high_membership, low_membership, singletons):
    word2membership = {}
    for lextype in high_membership:
        for word in high_membership[lextype]:
            word2membership[word] = ('high', lextype)
    for lextype in low_membership:
        for word in low_membership[lextype]:
            word2membership[word] = ('low', lextype)
    for lextype in singletons:
        for word in singletons[lextype]:
            word2membership[word] = ('singleton', lextype)
    return word2membership

def find_absolute_diffs_lextype(model1, model2, model1_name, model2_name):
    both = {'high membership': {}, 'low membership': {}, 'singletons': {}}
    only_one = {model1_name: {'high membership': {}, 'low membership': {}, 'singletons': {}},
                model2_name: {'high membership': {}, 'low membership': {}, 'singletons': {}}}
    for membership in model1:
        for lextype in model1[membership]:
            if lextype in model2[membership]:
                both[membership][lextype] = {}
                both[membership][lextype]['different'] = {}
                both[membership][lextype]['same'] = set(model1[membership][lextype]) & set(model2[membership][lextype])
                both[membership][lextype]['different'][model1_name] = set(model1[membership][lextype]) - set(model2[membership][lextype])
                both[membership][lextype]['different'][model2_name] = set(model2[membership][lextype]) - set(
                    model1[membership][lextype])
            else:
                only_one[model1_name][membership][lextype] = set(model1[membership][lextype])
    for membership in model2:
        for lextype in model2[membership]:
            if lextype not in model1[membership]:
                only_one[model2_name][membership][lextype] = set(model2[membership][lextype])
    for membership in both:
        for lextype in both[membership]:
            both[membership][lextype]['same'] = list(both[membership][lextype]['same'])
            both[membership][lextype]['different'][model1_name] = list(both[membership][lextype]['different'][model1_name])
            both[membership][lextype]['different'][model2_name] = list(both[membership][lextype]['different'][model2_name])
    for model in only_one:
        for membership in only_one[model]:
            for lextype in only_one[model][membership]:
                only_one[model][membership][lextype] = list(only_one[model][membership][lextype])
    return both, only_one

def compare_lexentries(data):
    only_in_llm_d = {'all llms': {}}
    only_in_human_d = {'not in any llm': {}}
    llm_lexentries = {}
    all_llms_lexentries = set()
    human_lexentries = set()
    for model in data:
        if model in LLM_GENERATED:
            llm_lexentries[model] = set()
            for le in data[model]:
                all_llms_lexentries.add(le)
                llm_lexentries[model].add(le)
        elif model in HUMAN_NYT:
            for le in data[model]:
                human_lexentries.add(le)
    for model in data:
        if model in LLM_GENERATED:
            only_in_human_d[model] = {}
            only_in_llm_d[model] = {}
            for le in data[model]:
                if not le in human_lexentries:
                    only_in_llm_d[model][le] = data[model][le]
                    if not le in only_in_llm_d['all llms']:
                        only_in_llm_d['all llms'][le] = 0
                    only_in_llm_d['all llms'][le] += data[model][le]
        elif model in HUMAN_NYT:
            for le in data[model]:
                if not le in all_llms_lexentries:
                    only_in_human_d['not in any llm'][le] = data[model][le]
                for llm in llm_lexentries:
                    if not le in llm_lexentries[llm]:
                        only_in_human_d[llm][le] = data[model][le]
    return only_in_llm_d, only_in_human_d

def map2lt(data, word2lt):
    mapping = {}
    for model in data:
        mapping[model] = {}
        for le in data[model]:
            if not le in word2lt:
                print("{} not in dictionary".format(le))
            else:
                membership, lt = word2lt[le]
                if not lt in mapping:
                    mapping[model][lt] = []
                mapping[model][lt].append(le)
    return mapping

def define_percentiles(sorted_frequencies):
    values = list(sorted_frequencies.values())
    # Define the percentiles
    percentiles = [10, 25, 50, 75, 90]  # You can adjust this list based on your needs
    # Calculate the percentile thresholds
    percentile_thresholds = np.percentile(values, percentiles)
    # Create segments based on the percentiles
    segments = {}
    for i, p in enumerate(percentiles):
        lower_bound = percentile_thresholds[i - 1] if i > 0 else 0
        upper_bound = percentile_thresholds[i]
        segment = {k: v for k, v in sorted_frequencies.items() if lower_bound <= v < upper_bound}
        segments[p] = segment
    lt2seg = {}
    for p in segments:
        for lt in segments[p]:
            lt2seg[lt] = p
    return segments, lt2seg

def analyze_lextypes_by_freq(norm_freq, lextypes, dataset):
    segments, lt2seg = define_percentiles(norm_freq['lextype'][dataset])
    by_seg = {}
    for seg in segments:
        by_seg[seg] = []
    for lt in lextypes:
        if not lt in norm_freq['lextype'][dataset]:
            print(f"{lt} not in frequency data")
        else:
            by_seg[lt2seg[lt]].append(lt)
    return by_seg

if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    with open('analysis/frequencies-json/frequencies-models-wiki-wsj.json', 'r') as f:
       all_data = json.load(f)
    normalized_frequencies = normalize_by_num_sen(all_data)
    #normalized_frequencies = normalize_by_constr_count(all_data)#(frequencies)
    #ascending_freq, descending_freq = sort_normalized_data(normalized_frequencies)
    all_data = combine_types(normalized_frequencies, ['constr', 'lexrule', 'lextype'])
    no_lextype = combine_types(normalized_frequencies, ['constr', 'lexrule'])
    no_lexrule = combine_types(normalized_frequencies, ['constr', 'lextype'])
    only_syntactic = combine_types(normalized_frequencies, ['constr'])
    only_lexical = combine_types(normalized_frequencies, ['lexrule'])
    only_vocab = combine_types(normalized_frequencies, ['lextype'])
    all_cosines = compute_cosine(all_data)
    syntactic_cosines = compute_cosine(only_syntactic)
    constr_and_lexrule_cosines = compute_cosine(no_lextype)
    constr_and_lextype_cosines = compute_cosine(no_lexrule)
    lexical_cosines = compute_cosine(only_lexical)
    vocab_cosines = compute_cosine(only_vocab)
    serialize_dict(vocab_cosines, 'analysis/cosine-pairs/models/norm-by-sen-count/lextype-only.json')
