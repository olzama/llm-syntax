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


def report_stats(all_cosines, human_authors, llm_names, title):
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
    hh_cosine_list = [float(v) for v in list(human_human_cosines.values())]
    hl_cosine_list = [float(v) for v in list(human_llm_cosines.values())]
    ll_cosine_list = [float(v) for v in list(llm_llm_cosines.values())]
    hh_mean = np.mean(hh_cosine_list)
    hl_mean = np.mean(hl_cosine_list)
    ll_mean = np.mean(ll_cosine_list)
    hh_var = np.var(hh_cosine_list)
    hl_var = np.var(hl_cosine_list)
    ll_var = np.var(ll_cosine_list)
    hh_stdev = np.std(hh_cosine_list)
    hl_stdev = np.std(hl_cosine_list)
    ll_stdev = np.std(ll_cosine_list)
    f_statistic_hh_hl, p_value_hh_hl = stats.levene(hh_cosine_list, hl_cosine_list)
    f_statistic_hl_ll, p_value_hl_ll = stats.levene(hl_cosine_list, ll_cosine_list)
    f_statistic_hh_ll, p_value_hh_ll = stats.levene(hh_cosine_list, ll_cosine_list)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[hh_cosine_list, hl_cosine_list, ll_cosine_list])
    plt.xticks([0, 1, 2], ['Human-Human', 'Human-LLM', 'LLM-LLM'])
    plt.ylabel('Cosine Similarity')
    plt.title('Boxplot Comparison of Cosine Similarities (Human-Human, Human-LLM, LLM-LLM)')
    plt.savefig('/mnt/kesha/llm-syntax/analysis/cosine-pairs/' + title + '.png')
    print(f"Human-Human mean: {hh_mean}, variance: {hh_var}, stdev: {hh_stdev}")
    print(f"Human-LLM mean: {hl_mean}, variance: {hl_var}, stdev: {hl_stdev}")
    print(f"LLM-LLM mean: {ll_mean}, variance: {ll_var}, stdev: {ll_stdev}")
    print(f"F-statistic HH-HL: {f_statistic_hh_hl}, p-value HH-HL: {p_value_hh_hl}")
    print(f"F-statistic HL-LL: {f_statistic_hl_ll}, p-value HL-LL: {p_value_hl_ll}")


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
    syntax_only = combine_types(authors_and_models, ['constr'])
    no_lextype = combine_types(authors_and_models, ['constr', 'lexrule'])
    lexrule_only = combine_types(authors_and_models, ['lexrule'])
    lextype_only = combine_types(authors_and_models, ['lextype'])
    all_cosines = compute_cosine(all_data)
    syntax_cosines = compute_cosine(syntax_only)
    no_lextype_cosines = compute_cosine(no_lextype)
    lexrule_cosines = compute_cosine(lexrule_only)
    lextype_cosines = compute_cosine(lextype_only)
    human_authors = author_frequencies['constr'].keys()
    llm_names = [name for name in model_frequencies['constr'].keys() if name not in ['original', 'wsj', 'wikipedia']]
    print('All cosines:')
    report_stats(all_cosines, human_authors, llm_names, 'hh-hl-ll-all')
    print('Syntax cosines:')
    report_stats(syntax_cosines, human_authors, llm_names, 'hh-hl-ll-syntax')
    print('Syntax and lexical rule cosines:')
    report_stats(no_lextype_cosines, human_authors, llm_names, 'hh-hl-ll-no-lextype')
    print('Lexical rule cosines:')
    report_stats(lexrule_cosines, human_authors, llm_names, 'hh-hl-ll-lexrule')
    print('Lexical type cosines:')
    report_stats(lextype_cosines, human_authors, llm_names, 'hh-hl-ll-lextype')

