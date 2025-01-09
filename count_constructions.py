import sys, os
from delphin import itsdb, derivation

def traverse_derivation(deriv, types, preterminals, visited=None):
    """Recursively traverse the derivation and its nodes."""
    if visited is None:
        visited = set()
    if id(deriv) in visited:
        return
    visited.add(id(deriv))
    if isinstance(deriv, derivation.UDFNode):
        for node in deriv.daughters:
            if isinstance(node, derivation.UDFNode):
                if node.entity in preterminals:
                    relevant_dict = 'lextype'
                elif node.entity.endswith('lr'):
                    relevant_dict = 'lexrule'
                else:
                    relevant_dict = 'constr'
                if not node.entity in types[relevant_dict]:
                    types[relevant_dict][node.entity] = 0
                types[relevant_dict][node.entity] += 1
                traverse_derivation(node, types, preterminals, visited)

def collect_types(data_dir):
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        db = itsdb.TestSuite(dataset_path)
        items = list(db.processed_items())
        for response in items:
            if len(response['results']) > 0:
                derivation_str = response['results'][0]['derivation']
                deriv = derivation.from_string(derivation_str)
                preterminals = set([pt.entity for pt in  deriv.preterminals()])
                traverse_derivation(deriv, types, preterminals)
    for relevant_dict in sorted_types:
        sorted_types[relevant_dict] = {k: v for k, v in sorted(types[relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
    return sorted_types

if __name__ == '__main__':
    data_dir = sys.argv[1]
    types = collect_types(data_dir)
    print(5)

