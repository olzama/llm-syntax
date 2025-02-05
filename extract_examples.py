import sys, os
import json
import re
from delphin import itsdb, derivation
from delphin.tokens import YYTokenLattice
from supertypes import get_n_supertypes, populate_type_defs

def traverse_derivation(deriv, interesting_ex, interesting_types, hapax_ex, hapax_types, preterminals, lex,
                        depth, ex_text, dataset_name, lattice, tokens, visited=None):
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
                    constituent = find_constituent(lattice, node.start, node.end, ex_text)
                    interesting_ex[relevant_dict][type][dataset_name].append({'sentence':ex_text, 'constituent': constituent})
                if type in hapax_types[relevant_dict]:
                    if not type in hapax_ex[relevant_dict]:
                        hapax_ex[relevant_dict][type] = {}
                    if not dataset_name in hapax_ex[relevant_dict][type]:
                        hapax_ex[relevant_dict][type][dataset_name] = []
                    constituent = find_constituent(lattice, node.start, node.end, ex_text)
                    hapax_ex[relevant_dict][type][dataset_name].append({'sentence':ex_text, 'constituent': constituent})
                traverse_derivation(node, interesting_ex, interesting_types, hapax_ex, hapax_types, preterminals,
                                    lex, depth, ex_text, dataset_name, lattice, tokens, visited)

def find_constituent(lattice, start, end, ex_text):
    return ex_text[lattice.tokens[start].lnk.data[0]:lattice.tokens[end-1].lnk.data[1]]

def collect_examples(data_dir, significant, hapax, lex, depth):
    significant_examples = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    hapax_examples = {'lexrule': {}, 'constr': {}, 'lextype': {}}
    for dataset in os.listdir(data_dir):
        db = itsdb.TestSuite(os.path.join(data_dir, dataset))
        print(f"Processing dataset {dataset}...")
        items = list(db.processed_items())
        for response in items:
            if len(response['results']) > 0:
                lattice = YYTokenLattice.from_string(response['p-input'])
                derivation_str = response['results'][0]['derivation']
                deriv = derivation.from_string(derivation_str)
                tokens = get_tok_list(deriv)
                preterminals = set([pt.entity for pt in deriv.preterminals()])
                traverse_derivation(deriv, significant_examples, significant,  hapax_examples, hapax,
                                    preterminals, lex, depth, response['i-input'], dataset, lattice, tokens)
    return significant_examples, hapax_examples

def get_tok_list(node):
    toks = []
    for t in node.terminals():
        for tok in t.tokens:
            toks.append(tok)
    return toks

def find_corresponding_toks(toks, terminal_span):
    tokens = []
    for tok in toks:
        if tok.lnk.data[0] == terminal_span[0] or tok.lnk.data[1] == terminal_span[1]:
            tokens.append(tok)
        if tok.lnk.data[1] > terminal_span[1]:
            return tokens
    return tokens


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
    with open('/mnt/kesha/llm-syntax/analysis/constructions/significant_examples.json', 'w', encoding='utf8') as f:
        json.dump(significant_examples, f, ensure_ascii=False)
    with open('/mnt/kesha/llm-syntax/analysis/constructions/hapax_examples.json', 'w', encoding='utf8') as f:
        json.dump(hapax_examples, f, ensure_ascii=False)
