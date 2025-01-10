import sys, os
from delphin import itsdb, derivation
from supertypes import get_n_supertypes, populate_type_defs

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

def collect_types(data_dir, lex, depth):
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    dataset_size = 0
    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        db = itsdb.TestSuite(dataset_path)
        items = list(db.processed_items())
        for response in items:
            if len(response['results']) > 0:
                dataset_size += 1
                derivation_str = response['results'][0]['derivation']
                deriv = derivation.from_string(derivation_str)
                preterminals = set([pt.entity for pt in  deriv.preterminals()])
                traverse_derivation(deriv, types, preterminals, lex, depth)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    return sorted_types, dataset_size

if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    lex = populate_type_defs(erg_dir)
    types, num_sen = collect_types(data_dir, lex, 1)
    print(5)

