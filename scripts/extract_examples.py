"""
extract_examples.py — collect example sentences for interesting constructions.

Given TSDB parsed data, finds example sentences for:
  1. Lextype percentile groups: types of interest from lextype_hum_by_percentile.json
     and lextype_llm_by_percentile.json, matched against preterminal parents in parses.
  2. Significant and hapax constructions from find_interesting_constr.py output,
     optionally generalizing lextypes to depth N in the type hierarchy.

Usage (run from repo root):
    python scripts/extract_examples.py <data_dir> [options]

Arguments:
    data_dir             Directory containing TSDB test suite subdirectories.

Options:
    --lextypes-dir DIR        Input/output directory for lextype percentile files.
                              Default: analysis/lextypes
    --constructions-dir DIR   Input/output directory for significant/hapax JSON files.
                              Default: analysis/constructions
    --erg-dir DIR             Path to ERG grammar directory (required for construction
                              examples; skipped if not provided).
    --depth INT               Type-hierarchy depth for lextype generalization (default: 1).
"""

import sys, os, json, argparse, itertools
from delphin import itsdb, derivation
from delphin.tokens import YYTokenLattice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erg import populate_type_defs, classify_node

_DEFAULT_LEXTYPES_DIR     = os.path.join('analysis', 'lextypes')
_DEFAULT_CONSTRUCTIONS_DIR = os.path.join('analysis', 'constructions')


def collect_lextype_percentile_examples(data_dir, examples):
    """Find example sentences for types appearing as preterminal parents in parses.

    data_dir: directory whose subdirectories are TSDB test suites.
    examples: {percentile_label: [type_name, ...]} — types to search for.

    Returns {dataset_name: {type_name: [sentence, ...]}}.
    """
    of_interest = set(itertools.chain.from_iterable(examples.values()))
    result = {}
    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        result[dataset] = {}
        db    = itsdb.TestSuite(dataset_path)
        print(f'Processing {dataset}...')
        for response in db.processed_items():
            if not response['results']:
                continue
            deriv        = derivation.from_string(response['results'][0]['derivation'])
            preterminals = list(deriv.preterminals())
            for pt in preterminals:
                if pt.parent and pt.parent.entity in of_interest:
                    entity = pt.parent.entity
                    result[dataset].setdefault(entity, []).append(response['i-input'])
    return result


def collect_examples(data_dir, significant, hapax, lex, depth):
    """Find example sentences for significant and hapax construction types.

    data_dir:    directory whose subdirectories are TSDB test suites.
    significant: {phenomenon: {'frequent': {type: p}, 'infrequent': {type: p}}}
    hapax:       {phenomenon: {type: {...}}}
    lex:         type hierarchy dict from populate_type_defs.
    depth:       hierarchy depth for lextype generalization.

    Returns (significant_examples, hapax_examples), each a
    {phenomenon: {type: {dataset: [{'sentence': str, 'constituent': str}]}}} dict.
    """
    significant_examples = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    hapax_examples       = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        db = itsdb.TestSuite(dataset_path)
        print(f'Processing {dataset}...')
        for response in db.processed_items():
            if not response['results']:
                continue
            lattice      = YYTokenLattice.from_string(response['p-input'])
            deriv        = derivation.from_string(response['results'][0]['derivation'])
            tokens       = _get_tok_list(deriv)
            preterminals = {pt.entity for pt in deriv.preterminals()}
            _traverse(deriv, significant_examples, significant,
                      hapax_examples, hapax,
                      preterminals, lex, depth,
                      response['i-input'], dataset, lattice, tokens)
    return significant_examples, hapax_examples


def find_constituent(lattice, start, end, ex_text):
    """Return the substring of ex_text spanned by token positions [start, end)."""
    return ex_text[lattice.tokens[start].lnk.data[0]:lattice.tokens[end - 1].lnk.data[1]]


def _get_tok_list(node):
    """Return a flat list of all tokens in a derivation."""
    return [tok for t in node.terminals() for tok in t.tokens]


def _traverse(deriv, interesting_ex, interesting_types, hapax_ex, hapax_types,
              preterminals, lex, depth, ex_text, dataset_name, lattice, tokens,
              visited=None):
    """Recursively collect examples for matching construction/rule/lextype nodes."""
    if visited is None:
        visited = set()
    if id(deriv) in visited:
        return
    visited.add(id(deriv))
    if not isinstance(deriv, derivation.UDFNode):
        return
    for node in deriv.daughters:
        if not isinstance(node, derivation.UDFNode):
            continue
        category, type_name = classify_node(node, preterminals, lex, depth)
        constituent = find_constituent(lattice, node.start, node.end, ex_text)
        entry = {'sentence': ex_text, 'constituent': constituent}
        if (type_name in interesting_types[category].get('frequent', {}) or
                type_name in interesting_types[category].get('infrequent', {})):
            interesting_ex[category].setdefault(type_name, {}).setdefault(dataset_name, []).append(entry)
        if type_name in hapax_types.get(category, {}):
            hapax_ex[category].setdefault(type_name, {}).setdefault(dataset_name, []).append(entry)
        _traverse(node, interesting_ex, interesting_types, hapax_ex, hapax_types,
                  preterminals, lex, depth, ex_text, dataset_name, lattice, tokens, visited)


def main(data_dir, lextypes_dir=_DEFAULT_LEXTYPES_DIR,
         constructions_dir=_DEFAULT_CONSTRUCTIONS_DIR,
         erg_dir=None, depth=1):
    # Collect lextype percentile examples
    hum_path = os.path.join(lextypes_dir, 'lextype_hum_by_percentile.json')
    llm_path = os.path.join(lextypes_dir, 'lextype_llm_by_percentile.json')
    with open(hum_path) as f:
        hum_lextypes = json.load(f)
    with open(llm_path) as f:
        llm_lextypes = json.load(f)
    hum_examples = collect_lextype_percentile_examples(data_dir, hum_lextypes)
    llm_examples = collect_lextype_percentile_examples(data_dir, llm_lextypes)
    with open(os.path.join(lextypes_dir, 'hum-lextype-percentile_examples.json'), 'w', encoding='utf-8') as f:
        json.dump(hum_examples, f, ensure_ascii=False)
    with open(os.path.join(lextypes_dir, 'llm-lextype-percentile_examples.json'), 'w', encoding='utf-8') as f:
        json.dump(llm_examples, f, ensure_ascii=False)
    print(f'Lextype examples written to {lextypes_dir}/')

    # Collect significant/hapax construction examples (requires ERG)
    if erg_dir:
        print('Reading ERG lexicon...')
        lex, _ = populate_type_defs(erg_dir)
        sig_path   = os.path.join(constructions_dir, 'significant_constr.json')
        hapax_path = os.path.join(constructions_dir, 'hapax_constr.json')
        with open(sig_path) as f:
            significant = json.load(f)
        with open(hapax_path) as f:
            hapax = json.load(f)
        sig_examples, hapax_examples = collect_examples(data_dir, significant, hapax, lex, depth)
        with open(os.path.join(constructions_dir, 'significant_examples.json'), 'w', encoding='utf-8') as f:
            json.dump(sig_examples, f, ensure_ascii=False)
        with open(os.path.join(constructions_dir, 'hapax_examples.json'), 'w', encoding='utf-8') as f:
            json.dump(hapax_examples, f, ensure_ascii=False)
        print(f'Construction examples written to {constructions_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir',
                        help='Directory containing TSDB test suite subdirectories')
    parser.add_argument('--lextypes-dir', default=_DEFAULT_LEXTYPES_DIR,
                        help=f'Input/output directory for lextype files (default: {_DEFAULT_LEXTYPES_DIR})')
    parser.add_argument('--constructions-dir', default=_DEFAULT_CONSTRUCTIONS_DIR,
                        help=f'Input/output directory for significant/hapax files (default: {_DEFAULT_CONSTRUCTIONS_DIR})')
    parser.add_argument('--erg-dir',
                        help='ERG grammar directory (required for construction examples)')
    parser.add_argument('--depth', type=int, default=1,
                        help='Type-hierarchy depth for lextype generalization (default: 1)')
    args = parser.parse_args()
    main(args.data_dir, args.lextypes_dir, args.constructions_dir, args.erg_dir, args.depth)
