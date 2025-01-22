import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # Load frequencies for human authors:
    with open('/mnt/kesha/llm-syntax/analysis/frequencies-json/frequencies-authors.json', 'r') as f:
        author_frequencies = json.load(f)
    with open('/mnt/kesha/llm-syntax/analysis/frequencies-json/frequencies-models.json', 'r') as f:
        model_frequencies = json.load(f)
    print(5)
