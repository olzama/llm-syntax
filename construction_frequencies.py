import sys, os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from supertypes import get_n_supertypes, populate_type_defs
from count_constructions import collect_types
from util import compute_cosine, print_cosine_similarities, serialize_dict

dataset_sizes = {'original':26102,'falcon_07':27769, 'llama_07':37825, 'llama_13':37800,'llama_30':37568,
                 'llama_65':38107,'mistral_07':35086, 'wikipedia': 10726, 'wsj': 43043}


'''
Assume that the types in dicts are already sorted in descending order of frequency
'''
def freq_counts_by_model(freq_by_model, model1, model2, model3, model4, start, end, title, reverse):
    n_constructions = {}
    for rule_type in freq_by_model:
        frequencies1 = freq_by_model[rule_type][model1]
        frequencies2 = freq_by_model[rule_type][model2]
        frequencies3 = freq_by_model[rule_type][model3]
        frequencies4 = freq_by_model[rule_type][model4]
        if rule_type == 'lextype':
            #continue
            non_zero_keys = (set(k for k in freq_by_model[rule_type][model1] if freq_by_model[rule_type][model1][k] != 0) &
                             set(k for k in freq_by_model[rule_type][model2] if freq_by_model[rule_type][model2][k] != 0) &
                             set(k for k in freq_by_model[rule_type][model3] if freq_by_model[rule_type][model3][k] != 0) &
                             set(k for k in freq_by_model[rule_type][model4] if freq_by_model[rule_type][model4][k] != 0))
            frequencies1 = {k: freq_by_model[rule_type][model1][k] for k in non_zero_keys}
            frequencies2 = {k: freq_by_model[rule_type][model2][k] for k in non_zero_keys}
            frequencies3 = {k: freq_by_model[rule_type][model3][k] for k in non_zero_keys}
            frequencies4 = {k: freq_by_model[rule_type][model4][k] for k in non_zero_keys}
            # re-sort:
            frequencies1 = {k: v for k, v in sorted(frequencies1.items(), key=lambda item: (item[1], item[0]), reverse=reverse)}
            frequencies2 = {k: v for k, v in sorted(frequencies2.items(), key=lambda item: (item[1], item[0]), reverse=reverse)}
            frequencies3 = {k: v for k, v in sorted(frequencies3.items(), key=lambda item: (item[1], item[0]), reverse=reverse)}
            frequencies4 = {k: v for k, v in sorted(frequencies4.items(), key=lambda item: (item[1], item[0]), reverse=reverse)}
        n_constructions[model1] = list(frequencies1.items())[start:end]
        n_constructions[model2] = list(frequencies2.items())[start:end]
        n_constructions[model3] = list(frequencies3.items())[start:end]
        n_constructions[model4] = list(frequencies4.items())[start:end]
        # Prepare the 'original' model data and other models separately
        m1 = {k: v for k, v in n_constructions[model1]}
        m2 = {k: v for k, v in n_constructions[model2]}
        m3 = {k: v for k, v in n_constructions[model3]}
        m4 = {k: v for k, v in n_constructions[model4]}
        if len(m1) == 0 or len(m2) == 0 or len(m3) == 0 or len(m4) == 0:
            print("No common constructions between {}, {}, {}, and {} for {}".format(model1, model2, model3, model4, rule_type))
            continue
        df1 = pd.DataFrame(list(m1.items()), columns=[rule_type, model1])
        df2 = pd.DataFrame(list(m2.items()), columns=[rule_type, model2])
        df3 = pd.DataFrame(list(m3.items()), columns=[rule_type, model3])
        df4 = pd.DataFrame(list(m4.items()), columns=[rule_type, model4])
        # Merge the two DataFrames on 'Construction' for plotting
        df = pd.merge(df1, df2, on=rule_type, how='left')
        df = pd.merge(df, df3, on=rule_type, how='left')
        df = pd.merge(df, df4, on=rule_type, how='left')
        df = df.fillna(0)  # Handle missing values
        ax = df.plot(kind='bar', x=rule_type, y=model1, figsize=(14, 8), width=0.8, color='blue', label=model1,
                     alpha=0.5, zorder=2)
        # Plotting other models with patterned or outlined bars
        df.plot(kind='scatter', x=rule_type, y=model2, ax=ax, label=model2, zorder=1, color='red', s=20)
        df.plot(kind='scatter', x=rule_type, y=model3, ax=ax, label=model3, zorder=1, color='green', s=20)
        df.plot(kind='scatter', x=rule_type, y=model4, ax=ax, label=model4, zorder=1, color='yellow', s=20)
        plt.title("Comparison of {} Frequencies".format(rule_type))
        plt.xlabel(rule_type)
        plt.ylabel("Frequency (Normalized by dataset size)")
        plt.xticks(rotation=90)
        plt.legend(title="{} vs. {}, {}, and {}".format(model1, model2, model3, model4))
        plt.tight_layout()
        plt.savefig('analysis/plots/frequencies/{}-{}/{}-{}-{}-{}-{}-{}.png'.format(start, end, title, model1, model2, model3, model4, rule_type))
        plt.close()



def exclusive_members(freq_by_model):
    original_constructions = set( [rule for rule in freq_by_model['constr']['original'].keys()
                                   if freq_by_model['constr']['original'][rule] > 0 ])
    other_models_constructions = {
        model: set([rule for rule in freq_by_model['constr'][model].keys()
                                   if freq_by_model['constr'][model][rule] > 0 ])
        for model in freq_by_model['constr'] if model != 'original'
    }
    other_models_flattened = {item for s in list(other_models_constructions.values()) for item in s}
    only_in_original = original_constructions - other_models_flattened
    only_in_original_per_model = {}
    not_in_original = other_models_flattened - original_constructions
    for model in other_models_constructions:
        only_in_original_per_model[model] = original_constructions - other_models_constructions[model]
    return only_in_original, not_in_original, only_in_original_per_model


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

def add_dataset(frequencies, new_dataset, dataset_name):
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

def visualize_counts(frequencies):
    reverse_frequencies = normalize_by_num_sen(frequencies)
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


if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    lex = populate_type_defs(erg_dir)
    frequencies = read_freq(data_dir, lex, 0)
    #wikipedia, w_size = collect_types(erg_dir+'/tsdb/llm-syntax/wikipedia', lex, 1)
    #wsj, wsj_size = collect_types(erg_dir+'/tsdb/llm-syntax/wsj', lex, 1)
    #add_dataset(frequencies, wikipedia, 'wikipedia')
    #add_dataset(frequencies, wsj, 'wsj')
    with open('analysis/frequencies-json/frequencies-models.json', 'w', encoding='utf8') as f:
        json.dump(frequencies, f)
    #visualize_counts(frequencies)
    # all_data = combine_types(frequencies, ['constr', 'lexrule', 'lextype'])
    # no_lextype = combine_types(frequencies, ['constr', 'lexrule'])
    # only_syntactic = combine_types(frequencies, ['constr'])
    # only_lexical = combine_types(frequencies, ['lexrule'])
    # only_vocab = combine_types(frequencies, ['lextype'])
    # all_cosines = compute_cosine(all_data)
    # syntactic_cosines = compute_cosine(only_syntactic)
    # constr_and_lexrule_cosines = compute_cosine(no_lextype)
    # lexical_cosines = compute_cosine(only_lexical)
    # vocab_cosines = compute_cosine(only_vocab)
    # serialize_dict(all_cosines, 'analysis/cosine-pairs/models/all-data.json')
    # serialize_dict(syntactic_cosines, 'analysis/cosine-pairs/models/syntax-only.json')
    # serialize_dict(constr_and_lexrule_cosines, 'analysis/cosine-pairs/models/no-lextype.json')
    # serialize_dict(lexical_cosines, 'analysis/cosine-pairs/models/lexrule-only.json')
    # serialize_dict(vocab_cosines, 'analysis/cosine-pairs/models/lextype-only.json')
    # #with open('analysis/cosine-pairs/authors/all-cosines.json', 'r') as file:
    # #    author_pairs_cosines_all = json.load(file)
    # my_dataset = "original"
    # human_datasets = ["wikipedia", "wsj"]
    # machine_datasets = list(set(dataset_sizes.keys())-set(human_datasets)-{my_dataset})
    # report_comparison(my_dataset, machine_datasets, human_datasets, all_cosines)
    # print('Only syntactic:')
    # report_comparison(my_dataset, machine_datasets, human_datasets, syntactic_cosines)
    # print('Constructions and lexical rules:')
    # report_comparison(my_dataset, machine_datasets, human_datasets, constr_and_lexrule_cosines)
    # print('Only lexical rules:')
    # report_comparison(my_dataset, machine_datasets, human_datasets, lexical_cosines)
    # print('Only vocabulary:')
    # report_comparison(my_dataset, machine_datasets, human_datasets, vocab_cosines)
