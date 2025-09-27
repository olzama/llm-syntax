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

def collect_types_multidir(data_dir, lex, lexentries, depth, sample_size=None):
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    total_sen = 0
    for dataset in os.listdir(data_dir):
        if dataset=='.DS_Store':
            continue
        print(f"Processing dataset {dataset}...")
        db = itsdb.TestSuite(os.path.join(data_dir, dataset))
        items = list(db.processed_items())
        if sample_size and len(items) > sample_size/len(os.listdir(data_dir)):
            items = random.sample(items, int(sample_size/len(os.listdir(data_dir))))
        total_sen += len(items)
        for response in items:
            if len(response['results']) > 0:
                derivation_str = response['results'][0]['derivation']
                deriv = derivation.from_string(derivation_str)
                preterminals = set([pt.entity for pt in  deriv.preterminals()])
                terminals = set([t.parent.entity for t in deriv.terminals()])
                for t in terminals:
                    if not t in lexentries:
                        lexentries[t] = 0
                    lexentries[t] += 1
                traverse_derivation(deriv, types, preterminals, lex, depth)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    print("Total sentences processed across datasets:", total_sen)
    print("Total syntactic types found:", len(sorted_types['constr']))
    print("Total lexical rule types found:", len(sorted_types['lexrule']))
    print("Total lexical types found:", len(sorted_types['lextype']))
    return sorted_types


def collect_types(data_dir, lex, lexentries, depth, sample_size=None):
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    db = itsdb.TestSuite(data_dir)
    items = list(db.processed_items())
    if sample_size:
        items = random.sample(items, sample_size)
    print("Total sentences processed:", len(items))
    for response in items:
        if len(response['results']) > 0:
            derivation_str = response['results'][0]['derivation']
            deriv = derivation.from_string(derivation_str)
            preterminals = set([pt.entity for pt in  deriv.preterminals()])
            terminals = set([t.parent.entity for t in deriv.terminals()])
            for t in terminals:
                if not t in lexentries:
                    lexentries[t] = 0
                lexentries[t] += 1
            traverse_derivation(deriv, types, preterminals, lex, depth)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    print("Total syntactic types found:", len(sorted_types['constr']))
    print("Total lexical rule types found:", len(sorted_types['lexrule']))
    print("Total lexical types found:", len(sorted_types['lextype']))
    return sorted_types

if __name__ == '__main__':
    data_dir = sys.argv[2]
    erg_dir = sys.argv[1]
    print("Reading in the ERG lexicon...")
    lex,constrs = populate_type_defs(erg_dir)
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    lexentries = {}
    for model in os.listdir(data_dir):
        lexentries[model] = {}
        print("Counting constructions in {}...".format(model))
        dataset_path = os.path.join(data_dir, model)
        if model in ['wsj', 'wescience']:
            model_types = collect_types_multidir(dataset_path, lex, lexentries[model], 1)
        else:
            model_types = collect_types(dataset_path, lex, lexentries[model], 1)
        for ctype in types:
            types[ctype][model] = model_types[ctype]
    with open('./analysis/frequencies-json/frequencies-2025.json', 'w', encoding='utf8') as f:
       json.dump(types, f, ensure_ascii=False, indent=2)


