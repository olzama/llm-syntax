from scipy import stats
import numpy as np
from collections import defaultdict
import json
import hashlib
from delphin import itsdb, commands

def normalize_by_constr_count(data):
    for ctype in data:
        for model in data[ctype]:
            total_count = sum(data[ctype][model].values())
            for constr in data[ctype][model]:
                data[ctype][model][constr] /= total_count
    return data

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

def create_database_subset(data_dir, output_dir, db_schema):
    db = itsdb.TestSuite(data_dir)
    ids = [ item['i-id'] for item in db['item'][:10] ]
    q = 'i-id = ' + ' or i-id = '.join([str(i) for i in ids])
    commands.mkprof(output_dir, source=data_dir, schema=db_schema, where=q, full=True)
