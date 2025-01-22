import sys, os
import json
from delphin import itsdb, derivation
from count_constructions import traverse_derivation
from supertypes import populate_type_defs
from construction_frequencies import combine_types
from util import serialize_dict, generate_key, compute_cosine

def map_sen2authors(data_dir, sen2authors, only_these_authors, depth=1):
    types_by_author = {}
    db = itsdb.TestSuite(data_dir)
    processed_items = list(db.processed_items())
    for i, response in enumerate(processed_items):
        if len(response['results']) > 0:
            sen = response['i-input']
            sen_key = generate_key(sen)
            if not sen_key in sen2authors:
                print(sen)
                continue
            assert sen == sen2authors[sen_key]['sentence']
            derivation_str = response['results'][0]['derivation']
            deriv = derivation.from_string(derivation_str)
            preterminals = set([pt.entity for pt in deriv.preterminals()])
            authors = sen2authors[sen_key]['authors']
            for author in authors:
                if not author in only_these_authors:
                    continue
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
    sorted_types_by_author = {}
    for author in types_by_author:
        sorted_types = {'constr': {}, 'lexrule': {}, 'lextype': {}}
        for relevant_dict in ['constr', 'lexrule', 'lextype']:
            sorted_types[relevant_dict] = {k: v for k, v in sorted(types_by_author[author][relevant_dict].items(), key=lambda item: (item[1], item[0]), reverse=True)}
        sorted_types_by_author[author] = sorted_types
    by_ctype = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    # Organize the same data by construction type first:
    for ctype in ['constr', 'lexrule', 'lextype']:
        for author in types_by_author:
            by_ctype[ctype][author] = sorted_types_by_author[author][ctype]
    return sorted_types_by_author, by_ctype


if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    lex = populate_type_defs(erg_dir)
    #small_db = create_database_subset(data_dir, 'small_db', data_dir + '/relations')
    with open(sys.argv[3], 'r') as sen2authors_file:
        sen2authors = json.load(sen2authors_file)
    with open(sys.argv[4], 'r') as f:
        only_these_authors = json.load(f)
    # Non-single-authored sentences are excluded from the analysis:
    by_author_ctype, by_ctype_author = map_sen2authors(data_dir, sen2authors, only_these_authors)
    all_data = combine_types(by_ctype_author, ['constr', 'lexrule', 'lextype'])
    syntax_only = combine_types(by_ctype_author, ['constr'])
    no_lextype = combine_types(by_ctype_author, ['constr', 'lexrule'])
    lexrule_only = combine_types(by_ctype_author, ['lexrule'])
    lextype_only = combine_types(by_ctype_author, ['lextype'])
    all_cosines = compute_cosine(all_data)
    syntax_cosines = compute_cosine(syntax_only)
    no_lextype_cosines = compute_cosine(no_lextype)
    lexrule_cosines = compute_cosine(lexrule_only)
    lextype_cosines = compute_cosine(lextype_only)
    # Serialize as json:
    serialize_dict(all_cosines, 'analysis/cosine-pairs/authors/all-data.json')
    serialize_dict(syntax_cosines, 'analysis/cosine-pairs/authors/syntax-only.json')
    serialize_dict(no_lextype_cosines, 'analysis/cosine-pairs/authors/no-lextype.json')
    serialize_dict(lexrule_cosines, 'analysis/cosine-pairs/authors/lexrule-only.json')
    serialize_dict(lextype_cosines, 'analysis/cosine-pairs/authors/lextype-only.json')
