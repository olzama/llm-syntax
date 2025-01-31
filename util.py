from scipy import stats
import numpy as np
from collections import defaultdict
import json
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from delphin import itsdb, commands

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

def normalize_by_constr_count(data):
    normalized_data = {}
    for ctype in data:
        normalized_data[ctype] = {}
        for model in data[ctype]:
            normalized_data[ctype][model] = {}
            total_count = sum(data[ctype][model].values())
            for constr in data[ctype][model]:
                normalized_data[ctype][model][constr] = data[ctype][model][constr]/total_count
    return normalized_data

def sort_normalized_data(normalized_data):
    ascending_sorted = {}
    descending_sorted = {}
    for ctype in normalized_data:
        ascending_sorted[ctype] = {}
        descending_sorted[ctype] = {}
        for model in sorted(normalized_data[ctype].keys()):  # Sort model names alphabetically
            ascending_sorted[ctype][model] = {}
            descending_sorted[ctype][model] = {}
            constr_dict = normalized_data[ctype][model]
            # Sort constr dictionary by values in ascending order and descending order
            sorted_constr_asc = dict(sorted(constr_dict.items(), key=lambda item: item[1]))
            sorted_constr_desc = dict(sorted(constr_dict.items(), key=lambda item: item[1], reverse=True))
            ascending_sorted[ctype][model] = sorted_constr_asc
            descending_sorted[ctype][model] = sorted_constr_desc
    return ascending_sorted, descending_sorted


def serialize_dict(data, filename):
    data_to_serialize = {}
    for k in data:
        data_to_serialize[str(k)] = str(data[k])
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_to_serialize, f, ensure_ascii=False)

def generate_key(sentence: str) -> str:
    # Hash the sentence and get a unique fixed-length key
    return hashlib.sha256(sentence.encode('utf-8')).hexdigest()

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        data = json.loads(text)
    return data

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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def print_cosine_similarities(similarities_dict):
    print("****************************************************************************************")
    sorted_pairs = sorted(similarities_dict.keys())
    for (dataset1, dataset2) in sorted_pairs:
        similarity = similarities_dict[(dataset1, dataset2)]
        print(f"Cosine similarity between {dataset1} and {dataset2}: {similarity:.4f}")
    print("****************************************************************************************")


def stat_significance(differences1, differences2):
    # Perform a t-test to compare the differences
    t_stat, p_value = stats.ttest_ind(differences1, differences2, equal_var=False)
    return t_stat, p_value

