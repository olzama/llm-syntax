import json
from util import normalize_by_constr_count

if __name__ == '__main__':
    # Load frequencies for human authors:
    with open('/mnt/kesha/llm-syntax/analysis/frequencies-json/frequencies-models.json', 'r') as f:
        model_frequencies = json.load(f)
    normalized_freq = normalize_by_constr_count(model_frequencies)
    print(5)