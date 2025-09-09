import sys, os
from delphin import itsdb, commands
import random

def create_database_subset(data_dir, output_dir, db_schema):
    db = itsdb.TestSuite(data_dir)
    ids = [ item['i-id'] for item in db['item'][:100] ]
    q = 'i-id = ' + ' or i-id = '.join([str(i) for i in ids])
    commands.mkprof(output_dir, source=data_dir, schema=db_schema, where=q, full=True)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    db_schema = sys.argv[3]
    for dataset in os.listdir(data_dir):
        print(f"Creating subset for {dataset}...")
        create_database_subset(os.path.join(data_dir, dataset), os.path.join(output_dir, dataset), db_schema)