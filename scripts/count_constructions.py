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
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    total_sen = 0
    for dataset in os.listdir(data_dir):
        print(f"Processing dataset {dataset}...")
        dataset_dir = os.path.join(data_dir, dataset)
        total_sen += collect_types_core(dataset_dir, depth, lex, sample_size, types)
    print("Total sentences processed:", total_sen)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    print("Total syntactic types found:", len(sorted_types['constr']))
    print("Total lexical rule types found:", len(sorted_types['lexrule']))
    print("Total lexical types found:", len(sorted_types['lextype']))
    print("Total lemmas found:", len(sorted_types['lexentries']))
    return sorted_types


def collect_types(data_dir, lex, depth, sample_size=None):
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    total_sen = collect_types_core(data_dir, depth, lex, sample_size, types)
    print("Total sentences processed:", total_sen)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    print("Total syntactic types found:", len(sorted_types['constr']))
    print("Total lexical rule types found:", len(sorted_types['lexrule']))
    print("Total lexical types found:", len(sorted_types['lextype']))
    print("Total lemmas found:", len(sorted_types['lexentries']))
    return sorted_types


def collect_types_core(data_dir, depth, lex, sample_size, types):
    db = itsdb.TestSuite(data_dir)
    items = list(db.processed_items())
    if sample_size:
        items = random.sample(items, sample_size)
    for response in items:
        if len(response['results']) > 0:
            derivation_str = response['results'][0]['derivation']
            deriv = derivation.from_string(derivation_str)
            preterminals = [pt.entity for pt in deriv.preterminals()]
            terminals = [t.parent for t in deriv.terminals()]
            for t in terminals:
                #form = t.to_dict()['form'] # this would be un-lemmatized
                if not t.entity in types['lexentries']:
                    types['lexentries'][t.entity] = 0
                types['lexentries'][t.entity] += 1
            traverse_derivation(deriv, types, set(preterminals), lex, depth)
    return len(items)

if __name__ == '__main__':
    data_dir = sys.argv[2]
    erg_dir = sys.argv[1]
    print("Reading in the ERG lexicon...")
    lex,constrs = populate_type_defs(erg_dir)
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    lexentries = {}
    for model in os.listdir(data_dir):
        print("Counting constructions in {}...".format(model))
        dataset_path = os.path.join(data_dir, model)
        if 'WSJ' in model or 'Wikipedia' in model:
            model_types = collect_types_multidir(dataset_path, lex, 1)
        else:
            model_types = collect_types(dataset_path, lex, 1)
        for ctype in types:
            types[ctype][model] = model_types[ctype]
    with open('./analysis/frequencies-json/frequencies-2025.json', 'w', encoding='utf8') as f:
       json.dump(types, f, ensure_ascii=False, indent=2)


