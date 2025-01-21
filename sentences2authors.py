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

    dataset_size = 0
    db = itsdb.TestSuite(data_dir)
    # create a small database for debugging:
    processed_items = list(db.processed_items())
    items = list(db['item'])
    assert len(items) == len(sen2authors)
    for i, response in enumerate(processed_items):
        if len(response['results']) > 0:
            #id = response['i-id']
            # retrieve the item from db['i-id'] where i-id = id:
            #q = '* from item' + ' where i-id = ' + str(id)
            #selection = commands.select(q, data_dir)
            #target_text = selection.data[0][6].strip()
            #assert target_text == response['i-input']
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
            else:
                print("Sentence {} does not match sen2authors".format(sen))
                exit(1)

if __name__ == '__main__':
    data_dir = sys.argv[1]
    erg_dir = sys.argv[2]
    #lex = populate_type_defs(erg_dir)
    #small_db = create_database_subset(data_dir, 'small_db', data_dir + '/relations')
    with open(sys.argv[3], 'r') as sen2authors_file:
        sen2authors = json.load(sen2authors_file)
    map_sen2authors(data_dir, sen2authors)