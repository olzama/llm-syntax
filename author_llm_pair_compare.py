import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def insert_zero_counts(dict1, dict2):
    all_ctypes = {'constr': set(), 'lexrule': set(), 'lextype': set()}
    for ctype in ['constr', 'lexrule', 'lextype']:
        for data in [dict1, dict2]:
            for author in data[ctype]:
                all_ctypes[ctype].update(data[ctype][author].keys())
    for ctype in all_ctypes:
        for data in [dict1, dict2]:
            for author in data[ctype]:
                for constr in all_ctypes[ctype]:
                    if constr not in data[ctype][author]:
                        data[ctype][author][constr] = 0

if __name__ == '__main__':
    # Load frequencies for human authors:
    with open('/mnt/kesha/llm-syntax/analysis/frequencies-json/frequencies-authors.json', 'r') as f:
        author_frequencies = json.load(f)
    with open('/mnt/kesha/llm-syntax/analysis/frequencies-json/frequencies-models.json', 'r') as f:
        model_frequencies = json.load(f)
    insert_zero_counts(author_frequencies, model_frequencies)
    print(5)
