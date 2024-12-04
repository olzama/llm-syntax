import sys, os
from delphin import itsdb

def compute_coverage(dataset_path):
    cov_by_len = {}
    dataset = itsdb.TestSuite(dataset_path)
    proc_items = dataset.processed_items()
    for proc_it in proc_items:
        if proc_it['i-length'] not in cov_by_len:
            cov_by_len[proc_it['i-length']] = 0
        if len(proc_it.results()) > 0:
            cov_by_len[proc_it['i-length']] += 1
    total_cov = sum(cov_by_len.values())
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