import json
import sys, os
from delphin import itsdb, derivation
from erg import get_n_supertypes, populate_type_defs
from util import serialize_dict
import random


def traverse_derivation(deriv, types, preterminals, lex, depth, visited=None):
    """Recursively traverse the derivation tree and count each node into the
    appropriate category: constr, lexrule, or lextype."""
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
        resolved_type = node.entity
        if node.entity in preterminals:
            category = 'lextype'
            supertypes = get_n_supertypes(lex, node.entity, depth)
            if supertypes:
                resolved_type = list(supertypes[depth - 1])[0]
        elif node.entity.endswith('lr'):
            category = 'lexrule'
        else:
            category = 'constr'
        types[category][resolved_type] = types[category].get(resolved_type, 0) + 1
        traverse_derivation(node, types, preterminals, lex, depth, visited)


def _sort_types(types):
    """Return a copy of types with each category sorted by count descending,
    breaking ties alphabetically."""
    return {
        category: {
            k: v for k, v in sorted(
                counts.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True
            )
        }
        for category, counts in types.items()
    }


def _empty_types():
    """Return an empty types dict with all four category keys."""
    return {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}


def _print_type_stats(sorted_types):
    """Print a one-line count of distinct types found in each category."""
    print("Total syntactic types found:", len(sorted_types['constr']))
    print("Total lexical rule types found:", len(sorted_types['lexrule']))
    print("Total lexical types found:", len(sorted_types['lextype']))
    print("Total lemmas found:", len(sorted_types['lexentries']))


def collect_types_multidir(data_dir, lex, depth, sample_size=None):
    """Count construction types across all dataset subdirectories under data_dir.

    Returns a sorted types dict aggregated over all subdirectories.
    """
    types = _empty_types()
    total_sen = 0
    for dataset in os.listdir(data_dir):
        dataset_dir = os.path.join(data_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        print(f"Processing dataset {dataset}...")
        total_sen += collect_types_core(dataset_dir, depth, lex, sample_size, types)
    print("Total sentences processed:", total_sen)
    sorted_types = _sort_types(types)
    _print_type_stats(sorted_types)
    return sorted_types


def collect_types(data_dir, lex, depth, sample_size=None):
    """Count construction types in a single dataset directory.

    Returns a sorted types dict (see _sort_types).
    """
    types = _empty_types()
    total_sen = collect_types_core(data_dir, depth, lex, sample_size, types)
    print("Total sentences processed:", total_sen)
    sorted_types = _sort_types(types)
    _print_type_stats(sorted_types)
    return sorted_types


def collect_types_core(data_dir, depth, lex, sample_size, types):
    """Parse derivations from the ITSDB test suite at data_dir and accumulate counts into types.

    Returns the number of items processed (after optional sampling).
    """
    db = itsdb.TestSuite(data_dir)
    items = list(db.processed_items())
    if sample_size:
        items = random.sample(items, sample_size)
    for response in items:
        if len(response['results']) > 0:
            derivation_str = response['results'][0]['derivation']
            deriv = derivation.from_string(derivation_str)
            preterminals = [pt.entity for pt in deriv.preterminals()]
            for t in deriv.terminals():
                entity = t.parent.entity
                types['lexentries'][entity] = types['lexentries'].get(entity, 0) + 1
            traverse_derivation(deriv, types, set(preterminals), lex, depth)
    return len(items)


if __name__ == '__main__':
    data_dir = sys.argv[2]
    erg_dir = sys.argv[1]
    print("Reading in the ERG lexicon...")
    lex, constrs = populate_type_defs(erg_dir)
    types = _empty_types()
    for model in os.listdir(data_dir):
        print("Counting constructions in {}...".format(model))
        dataset_path = os.path.join(data_dir, model)
        if 'WSJ' in model or 'Wikipedia' in model:
            model_types = collect_types_multidir(dataset_path, lex, 1)
        else:
            model_types = collect_types(dataset_path, lex, 1)
        for ctype in types:
            types[ctype][model] = model_types[ctype]
    with open('./analysis/frequencies-json/frequencies-debug.json', 'w', encoding='utf8') as f:
        json.dump(types, f, ensure_ascii=False, indent=2)
