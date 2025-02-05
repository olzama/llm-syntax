import json
import scipy.stats as stats
import numpy as np
from statsmodels.stats import multitest
from util import normalize_by_constr_count, sort_normalized_data, freq_counts_by_model
from boxplots import prepare_data_for_plotting, create_boxplot

ALL_HUMAN_AUTHORED = ["original", "wikipedia", "wsj"]
HUMAN_NYT = ["original"]
LLM_GENERATED = ['falcon_07', 'llama_07', 'llama_13', 'llama_30', 'llama_65', 'mistral_07']

def visualize_freq(data, human_models, construction_type, output_filename):
    df = prepare_data_for_plotting(data, human_models, construction_type)
    create_boxplot(df, construction_type, output_filename)

def find_significant_constr(normalized_freq, threshold_percentile=80):
    significant_constructions = {'constr': {'frequent': {}, 'infrequent': {}},
                                 'lexrule': {'frequent': {}, 'infrequent': {}},
                                 'lextype': {'frequent': {}, 'infrequent': {}}}
    p_values = {'frequent': [], 'infrequent': []}
    # Create a list for constructions that are used in one set and not the other
    for attribute in ['constr', 'lexrule', 'lextype']:
        # Create a list of all frequencies across all models for each construction
        all_frequencies = {}
        for constr_name in normalized_freq[attribute][next(iter(normalized_freq[attribute]))]:  # Get any model's constr names
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
        compare_frequencies(attribute, frequent_constructions, normalized_freq, p_values, significant_constructions, "frequent")
        compare_frequencies(attribute, infrequent_constructions, normalized_freq, p_values, significant_constructions, "infrequent")
    # Apply multiple testing correction (FDR); probably too harsh, not using it for
    #corrected_pvals_frequent = multitest.fdrcorrection(p_values['frequent'], alpha=0.05)[1]
    #corrected_pvals_infrequent = multitest.fdrcorrection(p_values['infrequent'], alpha=0.05)[1]
    write_out_interesting_constr(significant_constructions, "frequent")
    write_out_interesting_constr(significant_constructions, "infrequent")
    return significant_constructions

def find_hapax_constr(raw_counts, infrequent_threshold=2):
    # Create a list for constructions that are used in one set and not the other
    special_cases = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    for attribute in ['constr', 'lexrule', 'lextype']:
        # Create a list of all frequencies across all models for each construction
        all_frequencies = {}
        for constr_name in raw_counts[attribute][next(iter(raw_counts[attribute]))]:  # Get any model's constr names
            all_frequencies[constr_name] = []
            for model in raw_counts[attribute]:
                if constr_name in raw_counts[attribute][model]:
                    all_frequencies[constr_name].append(raw_counts[attribute][model][constr_name])
        # Gather special cases where constructions are used in one group but not the other
        find_hapax_mismatch(all_frequencies, attribute, infrequent_threshold, raw_counts, special_cases)
    #write_out_special_cases(infrequent_threshold, special_cases)
    return special_cases


def find_hapax_mismatch(all_frequencies, attribute, infrequent_threshold, raw_counts, special_cases):
    # Check for constructions that are used in one group but not the other
    for constr_name, frequencies in all_frequencies.items():
        # Get frequency for each model (human and machine) from raw counts, not normalized frequencies
        human_freq = np.array([raw_counts[attribute][model].get(constr_name, 0) for model in HUMAN_NYT])
        machine_freq = np.array([raw_counts[attribute][model].get(constr_name, 0) for model in LLM_GENERATED])
        # Only include special cases where one group is infrequent and the other is frequent
        human_infrequent = np.all(human_freq < infrequent_threshold)  # All counts in human are < 2
        machine_infrequent = np.all(machine_freq < infrequent_threshold)  # All counts in machine are < 2
        # Ensure one group has low counts (less than 2) and the other has high counts (greater than or equal to 2)
        if (human_infrequent and not machine_infrequent) or (machine_infrequent and not human_infrequent):  # Human is infrequent, machine is frequent
            if constr_name not in special_cases[attribute]:
                special_cases[attribute][constr_name] = {}
            special_cases[attribute][constr_name] = {'human count': int(human_freq.sum()), 'llm count': int(machine_freq.sum())}

def compare_frequencies(attribute, constructions, normalized_freq, p_values, significant_constructions, k):
    # Compare frequencies for frequent constructions
    for constr_name in constructions:
        human_dict = {model: normalized_freq[attribute][model] for model in ALL_HUMAN_AUTHORED}
        machine_dict = {model: normalized_freq[attribute][model] for model in LLM_GENERATED}
        p_value = compute_p_val(human_dict, machine_dict, constr_name)
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
    print("Used infrequently in one group and frequently in the other:")
    # Separate into two groups based on the counts for human and LLM models
    human_frequent_llm_infrequent = []
    llm_frequent_human_infrequent = []
    # Iterate over the special_cases dictionary
    for attribute, cases in special_cases.items():
        for case in cases:
            # Determine if the case belongs to the "human frequent, LLM infrequent" group
            if case['human count'] <= infrequent_threshold and case['llm count'] > infrequent_threshold:
                human_frequent_llm_infrequent.append(case)
            # Determine if the case belongs to the "LLM frequent, human infrequent" group
            elif case['human count'] > infrequent_threshold and case['llm count'] <= infrequent_threshold:
                llm_frequent_human_infrequent.append(case)
    # Sort each group by the respective count (descending)
    human_frequent_llm_infrequent.sort(key=lambda x: x['human count'], reverse=True)
    llm_frequent_human_infrequent.sort(key=lambda x: x['llm count'], reverse=True)
    # Combine the two groups
    sorted_special_cases = human_frequent_llm_infrequent + llm_frequent_human_infrequent
    # Write the sorted special cases to the file in tab-separated format
    with open('/mnt/kesha/llm-syntax/analysis/constructions/special_cases.txt', 'w') as f:
        # Write headers for better understanding in Excel
        f.write("Construction Name\tGroup (Infrequent)\tCount (Infrequent)\tGroup (Frequent)\tCount (Frequent)\n")
        for case in sorted_special_cases:
            # Write out the construction name and its respective counts
            f.write(f"{case['type']}\tHuman Models\t{case['human count']}\tLLM Models\t{case['llm count']}\n")
            print(
                f"{case['type']}: Human Models (count: {case['human count']}) vs LLM Models (count: {case['llm count']})")


def compute_p_val(human_dict, machine_dict, constr_name):
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
    #machine_datasets = list(set(normalized_freq['constr'].keys())-set(human_datasets))
    #visualize_freq(descending_norm_freq, human_datasets, 'constr', 'constr_boxplot.png')
    #for model in machine_datasets:
    #    to_compare = [model, 'original', 'wikipedia', 'wsj']
    #    freq_counts_by_model(descending_norm_freq, to_compare[0],to_compare[1], to_compare[2], to_compare[3],
    #                     0, 50, 'llama_07', reverse=False)
    signif_constr = find_significant_constr(descending_norm_freq, 80)
    hapax_constr = find_hapax_constr(model_frequencies, 2)
    with open('/mnt/kesha/llm-syntax/analysis/constructions/significant_constr.json', 'w') as f:
        json.dump(signif_constr, f, ensure_ascii=False)
    with open('/mnt/kesha/llm-syntax/analysis/constructions/hapax_constr.json', 'w') as f:
        json.dump(hapax_constr, f, ensure_ascii=False)
    print(5)
