"""
sentences2authors.py — parse author-tagged sentences and collect construction-type frequencies per author.

Reads a TSDB profile (produced by parsing author sentences with the ERG), maps each sentence
back to its author(s) via the sentence-to-author index, and counts construction types per
author.  Output feeds into author_llm_pair_compare.py.

Usage (run from repo root):
    python scripts/sentences2authors.py <data_dir> <erg_dir> <sen2authors_json> <authors_json>
        [--output-dir <dir>]

Arguments:
    data_dir          Path to a TSDB profile directory with parsed author sentences.
    erg_dir           Path to the ERG grammar directory (for type definitions).
    sen2authors_json  JSON mapping sentence keys to author name(s) and text
                      (produced by extract_sentences.py).
    authors_json      JSON list of author names to include (e.g. more_than_100.json).

Options:
    --output-dir      Directory to write frequencies-authors.json
                      (default: analysis/frequencies-authors).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from delphin import itsdb, derivation
from count_constructions import traverse_derivation
from erg import populate_type_defs
from construction_frequencies import combine_types
from util import serialize_dict, generate_key, compute_cosine

_DEFAULT_OUTPUT_DIR = os.path.join('analysis', 'frequencies-authors')
_CTYPES = ['constr', 'lexrule', 'lextype']


def collect_types(data_dir, sen2authors, only_these_authors, lex, depth=1):
    """Traverse parsed items and accumulate type counts per author.

    Returns {author: {ctype: {type: count}}}.
    """
    types_by_author = {}
    db = itsdb.TestSuite(data_dir)
    for response in db.processed_items():
        if not response['results']:
            continue
        sen = response['i-input']
        sen_key = generate_key(sen)
        if sen_key not in sen2authors:
            print(sen)
            continue
        assert sen == sen2authors[sen_key]['sentence']
        deriv = derivation.from_string(response['results'][0]['derivation'])
        preterminals = {pt.entity for pt in deriv.preterminals()}
        for author in sen2authors[sen_key]['authors']:
            if author not in only_these_authors:
                continue
            types_by_author.setdefault(author, {c: {} for c in _CTYPES})
            traverse_derivation(deriv, types_by_author[author], preterminals, lex, depth)
    return types_by_author


def zero_fill(types_by_author):
    """Ensure every author has an entry (possibly 0) for every type seen across all authors."""
    universe = {c: set() for c in _CTYPES}
    for author in types_by_author:
        for ctype in _CTYPES:
            universe[ctype].update(types_by_author[author][ctype])
    for author in types_by_author:
        for ctype in _CTYPES:
            for t in universe[ctype]:
                types_by_author[author][ctype].setdefault(t, 0)


def sort_by_frequency(types_by_author):
    """Return a copy with each author's type dicts sorted descending by count."""
    return {
        author: {
            ctype: dict(sorted(types_by_author[author][ctype].items(),
                               key=lambda kv: (kv[1], kv[0]), reverse=True))
            for ctype in _CTYPES
        }
        for author in types_by_author
    }


def pivot_by_ctype(sorted_types_by_author):
    """Reorganise {author: {ctype: ...}} into {ctype: {author: ...}}."""
    by_ctype = {c: {} for c in _CTYPES}
    for author, ctypes in sorted_types_by_author.items():
        for ctype, counts in ctypes.items():
            by_ctype[ctype][author] = counts
    return by_ctype


def map_sen2authors(data_dir, sen2authors, only_these_authors, lex, depth=1):
    """Collect, zero-fill, sort, and pivot author type counts from a TSDB profile.

    Returns (by_author_ctype, by_ctype_author).
    """
    types_by_author = collect_types(data_dir, sen2authors, only_these_authors, lex, depth)
    zero_fill(types_by_author)
    sorted_types = sort_by_frequency(types_by_author)
    return sorted_types, pivot_by_ctype(sorted_types)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data_dir',         help='TSDB profile directory with parsed author sentences.')
    ap.add_argument('erg_dir',          help='ERG grammar directory.')
    ap.add_argument('sen2authors_json', help='sentences2author.json from extract_sentences.py.')
    ap.add_argument('authors_json',     help='JSON list of author names to include.')
    ap.add_argument('--output-dir', default=_DEFAULT_OUTPUT_DIR,
                    help=f'Directory to write frequencies-authors.json (default: {_DEFAULT_OUTPUT_DIR}).')
    args = ap.parse_args()

    lex, constrs = populate_type_defs(args.erg_dir)

    with open(args.sen2authors_json, 'r') as f:
        sen2authors = json.load(f)
    with open(args.authors_json, 'r') as f:
        only_these_authors = json.load(f)

    by_author_ctype, by_ctype_author = map_sen2authors(
        args.data_dir, sen2authors, only_these_authors, lex
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'frequencies-authors.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(by_ctype_author, f)
    print(f"Wrote {out_path}")
