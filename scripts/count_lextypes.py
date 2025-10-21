import json
import sys, os
from delphin import itsdb, derivation
from erg import get_n_supertypes, populate_type_defs
from util import serialize_dict
import random

def traverse_derivation(deriv, types, preterminals, lex, depths, visited=None):
    """Recursively traverse the derivation and its nodes."""
    if visited is None:
        visited = set()
    if id(deriv) in visited:
        return
    visited.add(id(deriv))
    if isinstance(deriv, derivation.UDFNode):
        for node in deriv.daughters:
            if isinstance(node, derivation.UDFNode):
                type = node.entity
                if node.entity in preterminals:
                    for depth in depths:
                        supertypes = get_n_supertypes(lex, node.entity, depth)
                        if supertypes and len(supertypes) >= depth:
                            type = list(supertypes[depth-1])[0]
                            if not type in types[depth]:
                                types[depth][type] = 0
                            types[depth][type] += 1
                traverse_derivation(node, types, preterminals, lex, depths, visited)

def collect_lextypes_multidir(data_dir, lex, depths, sample_size=None):
    types = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    total_sen = 0
    for dataset in os.listdir(data_dir):
        print(f"Processing dataset {dataset}...")
        dataset_dir = os.path.join(data_dir, dataset)
        total_sen += collect_lextypes_core(dataset_dir, depths, lex, sample_size, types)
    print("Total sentences processed:", total_sen)
    sorted_types = {
        d: dict(sorted(sub.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        for d, sub in sorted(types.items())  # keeps depths in numeric order
    }
    for d in types:
        print(f"Total lexical types found at depth {d}:", len(sorted_types[d]))
    return sorted_types


def collect_lextypes(data_dir, lex, depths, sample_size=None):
    types = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    total_sen = collect_lextypes_core(data_dir, depths, lex, sample_size, types)
    print("Total sentences processed:", total_sen)
    sorted_types = {
        d: dict(sorted(sub.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        for d, sub in sorted(types.items())  # keeps depths in numeric order
    }
    for d in types:
        print(f"Total lexical types found at depth {d}:", len(sorted_types[d]))
    return sorted_types


def collect_lextypes_core(data_dir, depths, lex, sample_size, types):
    db = itsdb.TestSuite(data_dir)
    items = list(db.processed_items())
    if sample_size:
        items = random.sample(items, sample_size)
    for response in items:
        if len(response['results']) > 0:
            derivation_str = response['results'][0]['derivation']
            deriv = derivation.from_string(derivation_str)
            preterminals = [pt.entity for pt in deriv.preterminals()]
            traverse_derivation(deriv, types, set(preterminals), lex, depths)
    return len(items)

if __name__ == '__main__':
    output_fname = sys.argv[3]
    data_dir = sys.argv[2]
    erg_dir = sys.argv[1]
    print("Reading in the ERG lexicon...")
    lex,constrs = populate_type_defs(erg_dir)
    lextype_depths = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    for model in os.listdir(data_dir):
        print("Counting lexical types in {}...".format(model))
        dataset_path = os.path.join(data_dir, model)
        if 'WSJ' in model or 'Wikipedia' in model:
            model_types = collect_lextypes_multidir(dataset_path, lex, lextype_depths.keys())
        else:
            model_types = collect_lextypes(dataset_path, lex, lextype_depths.keys())
        for d in lextype_depths:
            lextype_depths[d][model] = model_types[d]
    with open(output_fname, 'w', encoding='utf8') as f:
       json.dump(lextype_depths, f, ensure_ascii=False, indent=2)


