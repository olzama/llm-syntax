import sys, os
import json
from delphin import itsdb, derivation, commands
from count_constructions import traverse_derivation
from supertypes import get_n_supertypes, populate_type_defs
from extract_sentences import generate_key

def create_database_subset(data_dir, output_dir, db_schema):
    db = itsdb.TestSuite(data_dir)
    ids = [ item['i-id'] for item in db['item'][:10] ]
    q = 'i-id = ' + ' or i-id = '.join([str(i) for i in ids])
    commands.mkprof(output_dir, source=data_dir, schema=db_schema, where=q, full=True)


def map_sen2authors(data_dir, sen2authors, depth=1):
    types_by_author = {}
    sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    db = itsdb.TestSuite(data_dir)
    processed_items = list(db.processed_items())
    for i, response in enumerate(processed_items):
        if len(response['results']) > 0:
            sen = response['i-input']
            sen_key = generate_key(sen)
            assert sen_key in sen2authors
            assert sen == sen2authors[sen_key]['sentence']
            derivation_str = response['results'][0]['derivation']
            deriv = derivation.from_string(derivation_str)
            preterminals = set([pt.entity for pt in deriv.preterminals()])
            authors = sen2authors[sen_key]['authors']
            for author in authors:
                if author not in types_by_author:
                    types_by_author[author] = {'constr': {}, 'lexrule': {}, 'lextype': {}}
                traverse_derivation(deriv, types_by_author[author], preterminals, lex, depth)
    # Collect all possible construction type names, and insert 0 counts where necessary such that all dicts are of the same length for each author:
    all_constrs = set()
    all_lexrules = set()
    all_lextypes = set()
    for author in types_by_author:
        all_constrs.update(types_by_author[author]['constr'])
        all_lexrules.update(types_by_author[author]['lexrule'])
        all_lextypes.update(types_by_author[author]['lextype'])
    for author in types_by_author:
        for constr in all_constrs:
            if constr not in types_by_author[author]['constr']:
                types_by_author[author]['constr'][constr] = 0
        for lexrule in all_lexrules:
            if lexrule not in types_by_author[author]['lexrule']:
                types_by_author[author]['lexrule'][lexrule] = 0
        for lextype in all_lextypes:
            if lextype not in types_by_author[author]['lextype']:
                types_by_author[author]['lextype'][lextype] = 0
    # For each author, sort the items by each construction frequency, and then by construction name:
    for author in types_by_author:
        for relevant_dict in ['constr', 'lexrule', 'lextype']:
            sorted_types[relevant_dict] = {k: v for k, v in sorted(types_by_author[author][relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
        types_by_author[author] = sorted_types
    # Create a new dict which is organized by construction type (constr, lexrule, lextype), and then by author:
    by_ctype = {}
    for ctype in ['constr', 'lexrule', 'lextype']:
        by_ctype[ctype] = {}
        for author in types_by_author:
            by_ctype[ctype][author] = types_by_author[author][ctype]
    return types_by_author, by_ctype



if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    lex = populate_type_defs(erg_dir)
    #small_db = create_database_subset(data_dir, 'small_db', data_dir + '/relations')
    with open(sys.argv[3], 'r') as sen2authors_file:
        sen2authors = json.load(sen2authors_file)
    types_by_author = map_sen2authors(data_dir, sen2authors)
    print(5)