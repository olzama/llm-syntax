import sys, os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from supertypes import get_n_supertypes, populate_type_defs
from count_constructions import collect_types

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


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def compute_cosine(data):
    dataset_vectors = build_vectors(data)
    dataset_names = sorted(list(dataset_vectors.keys()))
    cosine_similarities = {}
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            dataset1 = dataset_names[i]
            dataset2 = dataset_names[j]
            similarity = cosine_similarity(dataset_vectors[dataset1], dataset_vectors[dataset2])
            cosine_similarities[(dataset1, dataset2)] = similarity
    return cosine_similarities

def build_vectors(data):
    constrs = set()
    for dataset in data:
        constrs.update(data[dataset].keys())
    dataset_vectors = defaultdict(lambda: np.zeros(len(constrs)))
    constrs = list(constrs)
    constr_to_index = {constr: idx for idx, constr in enumerate(constrs)}
    for dataset_name, constructions in data.items():
        # Create a vector for each dataset
        vector = np.zeros(len(constrs))
        total_constr_count = sum(constructions.values())
        for constr, count in constructions.items():
            normalized_count = count / total_constr_count if total_constr_count > 0 else 0
            vector[constr_to_index[constr]] = normalized_count
        dataset_vectors[dataset_name] += vector
    return dataset_vectors

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


def print_cosine_similarities(similarities_dict):
    print("****************************************************************************************")
    sorted_pairs = sorted(similarities_dict.keys())
    for (dataset1, dataset2) in sorted_pairs:
        similarity = similarities_dict[(dataset1, dataset2)]
        print(f"Cosine similarity between {dataset1} and {dataset2}: {similarity:.4f}")
    print("****************************************************************************************")



if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    lex = populate_type_defs(erg_dir)
    frequencies = read_freq(data_dir, lex, 0)
    wikipedia, w_size = collect_types(erg_dir+'/tsdb/llm-syntax/wikipedia', lex, 1)
    wsj, wsj_size = collect_types(erg_dir+'/tsdb/llm-syntax/wsj', lex, 1)
    add_dataset(frequencies, wikipedia, 'wikipedia')
    add_dataset(frequencies, wsj, 'wsj')
    #visualize_counts(frequencies)
    all_data = combine_types(frequencies, ['constr', 'lexrule', 'lextype'])
    no_lextype = combine_types(frequencies, ['constr', 'lexrule'])
    only_syntactic = combine_types(frequencies, ['constr'])
    only_lexical = combine_types(frequencies, ['lexrule'])
    only_vocab = combine_types(frequencies, ['lextype'])
    all_cosines = compute_cosine(all_data)
    syntactic_cosines = compute_cosine(only_syntactic)
    constr_and_lexrule_cosines = compute_cosine(no_lextype)
    lexical_cosines = compute_cosine(only_lexical)
    vocab_cosines = compute_cosine(only_vocab)
    # print the cosine similarities:
    print('All data:')
    print_cosine_similarities(all_cosines)
    print('Only syntactic:')
    print_cosine_similarities(syntactic_cosines)
    print('Constructions and lexical rules:')
    print_cosine_similarities(constr_and_lexrule_cosines)
    print('Only lexical rules:')
    print_cosine_similarities(lexical_cosines)
    print('Only vocabulary:')
    print_cosine_similarities(vocab_cosines)
    print('Done.')
