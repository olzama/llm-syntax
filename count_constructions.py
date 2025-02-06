import json
import sys, os
from delphin import itsdb, derivation
from erg import get_n_supertypes, populate_type_defs
from util import serialize_dict
import random

def traverse_derivation(deriv, types, preterminals, lex, depth, visited=None):
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
                    relevant_dict = 'lextype'
                    supertypes = get_n_supertypes(lex, node.entity, depth)
                    if supertypes:
                        type = list(supertypes[depth-1])[0]
                elif node.entity.endswith('lr'):
                    relevant_dict = 'lexrule'
                else:
                    relevant_dict = 'constr'
                if not type in types[relevant_dict]:
                    types[relevant_dict][type] = 0
                types[relevant_dict][type] += 1
                traverse_derivation(node, types, preterminals, lex, depth, visited)

def collect_types_multidir(data_dir, lex, depth, sample_size=None):
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    for dataset in os.listdir(data_dir):
        db = itsdb.TestSuite(os.path.join(data_dir, dataset))
        items = list(db.processed_items())
        if sample_size:
            items = random.sample(items, sample_size)
        for response in items:
            if len(response['results']) > 0:
                derivation_str = response['results'][0]['derivation']
                deriv = derivation.from_string(derivation_str)
                preterminals = set([pt.entity for pt in  deriv.preterminals()])
                traverse_derivation(deriv, types, preterminals, lex, depth)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    return sorted_types


def collect_types(data_dir, lex, depth, sample_size=None):
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    db = itsdb.TestSuite(data_dir)
    items = list(db.processed_items())
    if sample_size:
        items = random.sample(items, sample_size)
    for response in items:
        if len(response['results']) > 0:
            derivation_str = response['results'][0]['derivation']
            deriv = derivation.from_string(derivation_str)
            preterminals = set([pt.entity for pt in  deriv.preterminals()])
            traverse_derivation(deriv, types, preterminals, lex, depth)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    return sorted_types

if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    print("Reading in the ERG lexicon...")
    lex = populate_type_defs(erg_dir)
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    for model in os.listdir(data_dir):
        print("Counting constructions in {}...".format(model))
        dataset_path = os.path.join(data_dir, model)
        model_types = collect_types(dataset_path, lex, 1, 150)
        for ctype in types:
            types[ctype][model] = model_types[ctype]
    with open('/mnt/kesha/llm-syntax/analysis/frequencies-json/frequencies-models-150.json', 'w', encoding='utf8') as f:
        json.dump(types, f, ensure_ascii=False)


