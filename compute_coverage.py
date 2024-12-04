import sys, os
from delphin import itsdb

def compute_coverage(dataset_path):
    dataset = itsdb.TestSuite(dataset_path)
    parsed = len(dataset['result'])
    return parsed / len(dataset['item'])

def compute_all_coverage(path_to_datasets):
    for dataset in os.listdir(path_to_datasets):
        dataset_path = os.path.join(path_to_datasets, dataset)
        print(f"Coverage for {dataset}: {compute_coverage(dataset_path)}")

if __name__ == '__main__':
    path_to_datasets = sys.argv[1]
    for subdir in os.listdir(path_to_datasets):
        compute_all_coverage(os.path.join(path_to_datasets, subdir))