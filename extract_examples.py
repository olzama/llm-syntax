import json
import scipy.stats as stats
from util import normalize_by_constr_count

def find_significant_constr(normalized_freq):
    significant_constructions = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    for attribute in ['constr', 'lexrule', 'lextype']:
        # Loop through all construction names in the current attribute (e.g., constr_name)
        for constr_name in normalized_freq[attribute][
            next(iter(normalized_freq[attribute]))]:  # Take any model to get the constr names
            # Get human and machine model dictionaries
            human_dict = {model: normalized_freq[attribute][model] for model in human_datasets}
            machine_dict = {model: normalized_freq[attribute][model] for model in machine_datasets}

            # Compare frequencies for this construction name and attribute
            p_value = compare_attribute_frequencies(human_dict, machine_dict, attribute, constr_name)
            if p_value is not None and p_value < 0.05:  # Adjust threshold as needed
                significant_constructions[attribute][constr_name] = p_value

    # Print the significant constructions (those where the difference is significant)
    print("Significant Constructions for each attribute:")
    for attribute, constructions in significant_constructions.items():
        print(f"\nAttribute: {attribute}")
        for constr, p in constructions.items():
            print(f"  Construction: {constr}, p-value: {p}")
    return significant_constructions


def compare_attribute_frequencies(human_dict, machine_dict, attribute, constr_name):
    human_frequencies = []
    machine_frequencies = []

    # Collect frequencies for human models
    for model in human_dict:
        if constr_name in human_dict[model][attribute]:
            human_frequencies.append(human_dict[model][attribute][constr_name])

    # Collect frequencies for machine models
    for model in machine_dict:
        if constr_name in machine_dict[model][attribute]:
            machine_frequencies.append(machine_dict[model][attribute][constr_name])

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
    human_datasets = ["original", "wikipedia", "wsj"]
    machine_datasets = list(set(normalized_freq['constr'].keys())-set(human_datasets))
    print(5)