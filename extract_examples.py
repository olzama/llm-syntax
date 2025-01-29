import json
import scipy.stats as stats
import numpy as np
from statsmodels.stats import multitest
from util import normalize_by_constr_count, sort_normalized_data, freq_counts_by_model
from boxplots import prepare_data_for_plotting, create_boxplot

def visualize_freq(data, human_models, construction_type, output_filename):
    df = prepare_data_for_plotting(data, human_models, construction_type)
    create_boxplot(df, construction_type, output_filename)


def find_significant_constr(normalized_freq, raw_counts, threshold_percentile=80, infrequent_threshold=2):
    significant_constructions = {'constr': {'frequent': {}, 'infrequent': {}, 'special_cases': {}},
                                 'lexrule': {'frequent': {}, 'infrequent': {}, 'special_cases': {}},
                                 'lextype': {'frequent': {}, 'infrequent': {}, 'special_cases': {}}}
    p_values = {'frequent': [], 'infrequent': []}

    # Create a list for infrequent constructions that are used in one set and not the other
    special_cases = {'constr': [], 'lexrule': [], 'lextype': []}

    for attribute in ['constr', 'lexrule', 'lextype']:
        # Create a list of all frequencies across all models for each construction
        all_frequencies = {}
        for constr_name in normalized_freq[attribute][
            next(iter(normalized_freq[attribute]))]:  # Get any model's constr names
            all_frequencies[constr_name] = []
            for model in normalized_freq[attribute]:
                if constr_name in normalized_freq[attribute][model]:
                    all_frequencies[constr_name].append(normalized_freq[attribute][model][constr_name])

        # Determine frequency thresholds for separating frequent and infrequent constructions
        constr_freqs = {constr: np.mean(freq) for constr, freq in all_frequencies.items()}
        threshold = np.percentile(list(constr_freqs.values()), threshold_percentile)

        # Separate constructions into frequent and infrequent based on the threshold
        frequent_constructions = {constr: freq for constr, freq in constr_freqs.items() if freq >= threshold}
        infrequent_constructions = {constr: freq for constr, freq in constr_freqs.items() if freq < threshold}

        # Check for constructions that are used in one group but not the other
        for constr_name, frequencies in all_frequencies.items():
            # Get frequency for each model (human and machine) from raw counts, not normalized frequencies
            human_freq = np.array([raw_counts[attribute][model].get(constr_name, 0) for model in human_datasets])
            machine_freq = np.array([raw_counts[attribute][model].get(constr_name, 0) for model in machine_datasets])

            # Condition 1: Construction is used infrequently (0-1 times) in human models
            # Condition 2: The same construction is used more than 2 times in machine models
            if np.sum(human_freq <= infrequent_threshold) > len(human_datasets) // 2 and np.sum(
                    machine_freq > infrequent_threshold) > 0:
                special_cases[attribute].append(
                    (constr_name, 'Human Models', human_freq.sum(), 'LLM Models', machine_freq.sum()))

            # Condition 3: Construction is used infrequently (0-1 times) in machine models
            # Condition 4: The same construction is used more than 2 times in human models
            elif np.sum(machine_freq <= infrequent_threshold) > len(machine_datasets) // 2 and np.sum(
                    human_freq > infrequent_threshold) > 0:
                special_cases[attribute].append(
                    (constr_name, 'LLM Models', machine_freq.sum(), 'Human Models', human_freq.sum()))

        compare_frequencies(attribute, frequent_constructions, normalized_freq, p_values, significant_constructions,
                            "frequent")
        compare_frequencies(attribute, infrequent_constructions, normalized_freq, p_values, significant_constructions,
                            "infrequent")
    # Apply multiple testing correction (FDR); probably too harsh
    corrected_pvals_frequent = multitest.fdrcorrection(p_values['frequent'], alpha=0.05)[1]
    corrected_pvals_infrequent = multitest.fdrcorrection(p_values['infrequent'], alpha=0.05)[1]

    write_out_interesting_constr(significant_constructions, "frequent")
    write_out_interesting_constr(significant_constructions, "infrequent")
    write_out_special_cases(infrequent_threshold, special_cases)
    return significant_constructions


def compare_frequencies(attribute, constructions, normalized_freq, p_values, significant_constructions, k):
    # Compare frequencies for frequent constructions
    for constr_name in constructions:
        human_dict = {model: normalized_freq[attribute][model] for model in human_datasets}
        machine_dict = {model: normalized_freq[attribute][model] for model in machine_datasets}
        p_value = compare_attribute_frequencies(human_dict, machine_dict, constr_name)
        if p_value is not None and p_value < 0.05:
            significant_constructions[attribute][k][constr_name] = p_value
        p_values[k].append(p_value)


def write_out_interesting_constr(significant_constructions, k):
    # Print the significant constructions for both frequent and infrequent groups
    print("Significant Constructions ({}):".format(k))
    for attribute, constructions in significant_constructions.items():
        with open(f'/mnt/kesha/llm-syntax/analysis/constructions/{attribute}_{k}.txt', 'w') as f:
            for constr, p in constructions[k].items():
                f.write(f"{constr} - p-value: {p}\n")
                print(f"  Construction: {constr}, p-value: {p}")


def write_out_special_cases(infrequent_threshold, special_cases):
    print("Special Cases (Used infrequently in one group and frequently in the other):")
    # Separate into two groups: Human frequent, LLM infrequent and vice versa
    human_frequent_llm_infrequent = []
    llm_frequent_human_infrequent = []
    for attribute, cases in special_cases.items():
        for case in cases:
            if case[1] == 'Human Models' and case[2] <= infrequent_threshold and case[4] > infrequent_threshold:
                human_frequent_llm_infrequent.append(case)
            elif case[1] == 'LLM Models' and case[2] <= infrequent_threshold and case[4] > infrequent_threshold:
                llm_frequent_human_infrequent.append(case)
    # Sort each group by count (descending)
    human_frequent_llm_infrequent.sort(key=lambda x: x[2], reverse=True)
    llm_frequent_human_infrequent.sort(key=lambda x: x[4], reverse=True)
    # Combine the two groups
    sorted_special_cases = human_frequent_llm_infrequent + llm_frequent_human_infrequent
    # Write the sorted special cases to the file in tab-separated format
    with open('/mnt/kesha/llm-syntax/analysis/constructions/special_cases.txt', 'w') as f:
        # Write headers for better understanding in Excel
        f.write("Construction Name\tGroup (Infrequent)\tCount (Infrequent)\tGroup (Frequent)\tCount (Frequent)\n")

        # Write sorted special cases
        for case in sorted_special_cases:
            f.write(f"{case[0]}\t{case[1]}\t{case[2]}\t{case[3]}\t{case[4]}\n")
            print(f"{case[0]}: {case[1]} (count: {case[2]}) vs {case[3]} (count: {case[4]})")


def compare_attribute_frequencies(human_dict, machine_dict, constr_name):
    human_frequencies = []
    machine_frequencies = []
    for model in human_dict:
        if constr_name in human_dict[model]:
            human_frequencies.append(human_dict[model][constr_name])
    for model in machine_dict:
        if constr_name in machine_dict[model]:
            machine_frequencies.append(machine_dict[model][constr_name])
    # Perform Mann-Whitney U test to compare the distributions
    if human_frequencies and machine_frequencies:
        _, p_value = stats.mannwhitneyu(human_frequencies, machine_frequencies, alternative='two-sided')
        return p_value
    return None

if __name__ == '__main__':
    # Load frequencies for human authors:
    with open('/mnt/kesha/llm-syntax/analysis/frequencies-json/frequencies-models-wiki-wsj.json', 'r') as f:
        model_frequencies = json.load(f)
    normalized_freq = normalize_by_constr_count(model_frequencies)
    ascending_norm_freq, descending_norm_freq = sort_normalized_data(normalized_freq)
    human_datasets = ["original", "wikipedia", "wsj"]
    machine_datasets = list(set(normalized_freq['constr'].keys())-set(human_datasets))
    #machine_datasets = ['llama_07']
    #visualize_freq(descending_norm_freq, human_datasets, 'constr', 'constr_boxplot.png')
    #for model in machine_datasets:
    #    to_compare = [model, 'original', 'wikipedia', 'wsj']
    #    freq_counts_by_model(descending_norm_freq, to_compare[0],to_compare[1], to_compare[2], to_compare[3],
    #                     0, 50, 'llama_07', reverse=False)
    signif_constr = find_significant_constr(descending_norm_freq, model_frequencies, 80, 2)
    print(5)