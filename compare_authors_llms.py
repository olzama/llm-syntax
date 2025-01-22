import sys, os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from supertypes import get_n_supertypes, populate_type_defs
from count_constructions import collect_types
from util import compute_cosine, print_cosine_similarities, serialize_dict

if __name__ == '__main__':
    # Load cosine pairs for all LLMs:
    with open('/mnt/kesha/llm-syntax/analysis/cosine-pairs/models/all-data.json', 'r') as f:
        all_cosines = json.load(f)
    # Load cosines computed between human author pairs:
    with open('/mnt/kesha/llm-syntax/analysis/cosine-pairs/authors/all-data.json', 'r') as f:
        author_cosines = json.load(f)
    # From the LLM cosine pairs, only take the ones where the model names are not in the list: original, wsj, wikipedia:
    llm_cosines = {k: v for k, v in all_cosines.items() if not any([name in k for name in ['original', 'wsj', 'wikipedia']])}
    llm_cosine_list = [float(v) for v in list(llm_cosines.values()) ]
    human_cosine_list = [ float(v) for v in list(author_cosines.values()) ]
    llm_mean = np.mean(llm_cosine_list)
    human_mean = np.mean(human_cosine_list)
    llm_var = np.var(llm_cosine_list)
    human_var = np.var(human_cosine_list)
    llm_stdev = np.std(llm_cosine_list)
    human_stdev = np.std(human_cosine_list)

    print(5)
