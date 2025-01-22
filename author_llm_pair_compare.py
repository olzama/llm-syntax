import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from util import compute_cosine
from construction_frequencies import combine_types

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
    # Combine the two frequency dicts, excluding 'original', 'wsj', and 'wikipedia':
    authors_and_models = {}
    for ctype in ['constr', 'lexrule', 'lextype']:
        authors_and_models[ctype] = {}
        for author in author_frequencies[ctype]:
            authors_and_models[ctype][author] = author_frequencies[ctype][author]
        for model in model_frequencies[ctype]:
            if model not in ['original', 'wsj', 'wikipedia']:
                authors_and_models[ctype][model] = model_frequencies[ctype][model]
    # Compute cosine similarities:
    all_data = combine_types(authors_and_models, ['constr', 'lexrule', 'lextype'])
    all_cosines = compute_cosine(all_data)
    human_authors = author_frequencies['constr'].keys()
    llm_names = [name for name in model_frequencies['constr'].keys() if name not in ['original', 'wsj', 'wikipedia']]
    # Separate the cosines into human-human, human-llm, and ll:
    human_human_cosines = {}
    human_llm_cosines = {}
    llm_llm_cosines = {}
    for pair in all_cosines:
        if pair[0] in human_authors and pair[1] in human_authors:
            human_human_cosines[pair] = all_cosines[pair]
        elif pair[0] in human_authors and pair[1] in llm_names:
            human_llm_cosines[pair] = all_cosines[pair]
        elif pair[0] in llm_names and pair[1] in llm_names:
            llm_llm_cosines[pair] = all_cosines[pair]
    print(5)
