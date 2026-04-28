"""
type-tok.py — print type-token ratio (TTR) statistics for each model and phenomenon.

TTR = (number of distinct types with count > 0) / (total token count).

Usage (run from repo root):
    python scripts/type-tok.py [frequencies_json] [phenomenon ...]

Arguments:
    frequencies_json   Path to a JSON file with structure
                       {phenomenon: {model: {type: count}}}.
                       Default: analysis/frequencies-json/frequencies-models-wiki-wsj.json
    phenomenon         One or more phenomena to include (e.g. constr lexrule lextype).
                       Default: constr lexrule lextype
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import compute_ttr

_DEFAULT_INPUT    = os.path.join('analysis', 'frequencies-json', 'frequencies-models-wiki-wsj.json')
_DEFAULT_PHENOMENA = ['constr', 'lexrule', 'lextype']


def main(frequencies_json, phenomena=None):
    with open(frequencies_json) as f:
        data = json.load(f)
    results = compute_ttr(data, phenomena)
    for phenomenon, model_stats in results.items():
        print(f'\nCategory: {phenomenon}')
        for model, stats in model_stats.items():
            print(f'  {model}: Types = {stats["types"]}, Tokens = {stats["tokens"]}, TTR = {stats["ttr"]:.5f}')


if __name__ == '__main__':
    json_file  = sys.argv[1] if len(sys.argv) >= 2 else _DEFAULT_INPUT
    phenomena  = sys.argv[2:] if len(sys.argv) > 2 else _DEFAULT_PHENOMENA
    main(json_file, phenomena)
