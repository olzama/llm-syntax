import sys, os
import json
import re
from delphin import itsdb, derivation
from delphin.tokens import YYTokenLattice
from supertypes import get_n_supertypes, populate_type_defs

def traverse_derivation(root_deriv, deriv, interesting_ex, interesting_types, hapax_ex, hapax_types, preterminals, lex,
                        depth, ex_text, dataset_name, visited=None):
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
                if type in interesting_types[relevant_dict]['frequent'] or type in interesting_types[relevant_dict]['infrequent']:
                    if not type in interesting_ex[relevant_dict]:
                        interesting_ex[relevant_dict][type] = {}
                    if not dataset_name in interesting_ex[relevant_dict][type]:
                        interesting_ex[relevant_dict][type][dataset_name] = []
                    constituent = find_constituent(root_deriv, node.start, node.end)
                    if not constituent:
                        print(5)
                    interesting_ex[relevant_dict][type][dataset_name].append({'sentence':ex_text, 'constituent': constituent})
                traverse_derivation(root_deriv, node, interesting_ex, interesting_types, hapax_ex, hapax_types, preterminals,
                                    lex, depth, ex_text, dataset_name, visited)

def find_constituent(node, start, end):
    constituent_str = ""
    for i,t in enumerate(node.terminals()):
        if i >= start and i < end:
            constituent_str += t.form + " "
    return constituent_str.strip()
# terminals = None
# lattice = None
# char_spans = {}
# if len(item['results']) > 0:
#     terminals = item.result(0).derivation().terminals()
# if (item['p-input']):
#     lattice = YYTokenLattice.from_string(item['p-input'])
# if lattice and terminals:
#     for i, t in enumerate(terminals):
#         this_gold.append(str(lextypes.get(t.parent.entity, "None_label")))
#         terminal_span = extract_span(t)
#         tokens = find_corresponding_toks(lattice.tokens, terminal_span)
#         words.append(t.form)
#         char_spans[str(terminal_span)] = []
#         for tok in tokens:
#             char_spans[str(terminal_span)].append({'terminal-form': t.form, 'token-form': tok.form,
#                                                    'start': tok.lnk.data[0],
#                                                    'end': tok.lnk.data[1]})


def collect_examples(data_dir, significant, hapax, lex, depth):
    significant_examples = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    hapax_examples = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    for dataset in os.listdir(data_dir):
        db = itsdb.TestSuite(os.path.join(data_dir, dataset))
        print(f"Processing dataset {dataset}...")
        items = list(db.processed_items())
        for response in items:
            if len(response['results']) > 0:
                tok_lattice = YYTokenLattice.from_string(response['p-input'])
                derivation_str = response['results'][0]['derivation']
                deriv = derivation.from_string(derivation_str)
                preterminals = set([pt.entity for pt in deriv.preterminals()])
                traverse_derivation(deriv, deriv, significant_examples, significant,  hapax_examples, hapax,
                                    preterminals, lex, depth, response['i-input'], dataset)
    return significant_examples, hapax_examples

def find_corresponding_toks(toks, terminal_span):
    tokens = []
    for tok in toks:
        if tok.lnk.data[0] == terminal_span[0] or tok.lnk.data[1] == terminal_span[1]:
            tokens.append(tok)
        if tok.lnk.data[1] > terminal_span[1]:
            return tokens
    return tokens

def extract_span(terminal):
    str_tok = terminal.tokens[0][1]
    from_match = re.search(r'\+FROM\s+\\"(\d+)\\"', str_tok)
    to_match = re.search(r'\+TO\s+\\"(\d+)\\"', str_tok)

    if from_match and to_match:
        from_value = int(from_match.group(1))
        to_value = int(to_match.group(1))
        return from_value, to_value
    else:
        return None


if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    print("Reading in the ERG lexicon...")
    lex = populate_type_defs(erg_dir)
    with open('/mnt/kesha/llm-syntax/analysis/constructions/significant_constr.json', 'r') as f:
        significant = json.load(f)
    with open('/mnt/kesha/llm-syntax/analysis/constructions/hapax_constr.json', 'r') as f:
        hapax = json.load(f)
    significant_examples, hapax_examples = collect_examples(data_dir, significant, hapax, lex, 1)
    print(5)