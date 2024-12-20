import sys, os
import re
import pandas as pd
import matplotlib.pyplot as plt
from supertypes import get_n_supertypes, populate_type_defs

dataset_sizes = {'original':26102,'falcon_07':27769, 'llama_07':37825, 'llama_13':37800,'llama_30':37568,
                 'llama_65':38107,'mistral_07':35086}


'''
Assume that the types in dicts are already sorted in descending order of frequency
'''
def freq_counts_by_model(freq_by_model, model1, model2, start, end, title, reverse):
    n_constructions = {}
    for rule_type in freq_by_model:
        frequencies1 = freq_by_model[rule_type][model1]
        frequencies2 = freq_by_model[rule_type][model2]
        if rule_type == 'lextype':
            #continue
            non_zero_keys = (set(k for k in freq_by_model[rule_type][model1] if freq_by_model[rule_type][model1][k] != 0) &
                             set(k for k in freq_by_model[rule_type][model2] if freq_by_model[rule_type][model2][k] != 0))
            frequencies1 = {k: freq_by_model[rule_type][model1][k] for k in non_zero_keys}
            frequencies2 = {k: freq_by_model[rule_type][model2][k] for k in non_zero_keys}
            # re-sort:
            frequencies1 = {k: v for k, v in sorted(frequencies1.items(), key=lambda item: (item[1], item[0]), reverse=reverse)}
            frequencies2 = {k: v for k, v in sorted(frequencies2.items(), key=lambda item: (item[1], item[0]), reverse=reverse)}
        n_constructions[model1] = list(frequencies1.items())[start:end]
        n_constructions[model2] = list(frequencies2.items())[start:end]
        # Prepare the 'original' model data and other models separately
        m1 = {k: v for k, v in n_constructions[model1]}
        m2 = {k: v for k, v in n_constructions[model2]}
        if len(m1) == 0 or len(m2) == 0:
            print("No common constructions between {} and {} for {}".format(model1, model2, rule_type))
            continue
        df1 = pd.DataFrame(list(m1.items()), columns=[rule_type, model1])
        df2 = pd.DataFrame(list(m2.items()), columns=[rule_type, model2])
        # Merge the two DataFrames on 'Construction' for plotting
        df = pd.merge(df1, df2, on=rule_type, how='left')
        df = df.fillna(0)  # Handle missing values
        ax = df.plot(kind='bar', x=rule_type, y=model1, figsize=(14, 8), width=0.8, color='blue', label=model1,
                     alpha=0.5, zorder=2)
        # Plotting other models with patterned or outlined bars
        df.plot(kind='bar', x=rule_type, y=model2, ax=ax, width=0.8, label=model2, zorder=1, linestyle='--',
                    edgecolor='black', hatch='//')
        plt.title("Comparison of {} Frequencies".format(rule_type))
        plt.xlabel(rule_type)
        plt.ylabel("Frequency (Normalized by dataset size)")
        plt.xticks(rotation=90)
        plt.legend(title="Human-authored vs. {}".format(model1))
        plt.tight_layout()
        plt.savefig('analysis/plots/frequencies/{}-{}/{}-{}-{}-{}.png'.format(start, end, title, model1, model2, rule_type))
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
                        if relevant_dict == 'lextype' and depth > 0:
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
    for rule_type in freq_by_model:
        for model in freq_by_model[rule_type]:
            for rule in freq_by_model[rule_type][model]:
                all_rules[rule_type].add(rule)
    for rule_type in all_rules:
        for model in freq_by_model[rule_type]:
            for rule in all_rules[rule_type]:
                if rule not in freq_by_model[rule_type][model]:
                    freq_by_model[rule_type][model][rule] = 0
    reverse_freq_by_model = normalize_by_num_sen(freq_by_model)
    return freq_by_model, reverse_freq_by_model


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


if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    lex = populate_type_defs(erg_dir)
    frequencies, reverse_frequencies = read_freq(data_dir, lex, 0)
    only_in_original, not_in_original, only_in_original_per_model = exclusive_members(frequencies)
    for model in dataset_sizes.keys():
        if not model == 'original':
            freq_counts_by_model(frequencies, model, 'original',0, 50, "Top frequencies", reverse=True)
            freq_counts_by_model(reverse_frequencies, model, 'original', 0,50, "Bottom frequencies", reverse=False)
            freq_counts_by_model(frequencies, 'original', model, 0, 50, "Top frequencies", reverse=True)
            freq_counts_by_model(reverse_frequencies, 'original', model,  0,50, "Bottom frequencies", reverse=False)
