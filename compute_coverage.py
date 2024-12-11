import sys, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from delphin import itsdb
import numpy as np

# Global color map to store model colors consistently
model_colors = {}

def compute_coverage(dataset_path):
    cov_by_len = {}
    sen_by_len = {}
    dataset = itsdb.TestSuite(dataset_path)
    proc_items = dataset.processed_items()

    # Collect coverage and sentence counts
    for proc_it in proc_items:
        length = proc_it['i-length']
        if length not in cov_by_len:
            cov_by_len[length] = 0
        if length not in sen_by_len:
            sen_by_len[length] = 0
        sen_by_len[length] += 1
        if len(proc_it.results()) > 0:
            cov_by_len[length] += 1

    # Sort dictionaries by sentence length
    sorted_cov_by_len = dict(sorted(cov_by_len.items()))
    sorted_sen_by_len = dict(sorted(sen_by_len.items()))

    # Calculate total coverage ratio
    parsed = len(dataset['result'])
    total_cov = parsed / len(dataset['item'])

    return total_cov, sorted_cov_by_len, sorted_sen_by_len

def bin_data(cov_by_len, sen_by_len, bin_size=10):
    # Check if the dictionary is empty
    if not cov_by_len:
        return {}, {}

    binned_cov = {}
    binned_sen = {}

    # Group data into bins and compute average coverage for each bin
    for length in range(min(cov_by_len.keys()), max(cov_by_len.keys()) + 1, bin_size):
        bin_end = length + bin_size - 1
        coverage_sum = 0
        sentence_count = 0

        for l in range(length, bin_end + 1):
            if l in cov_by_len:
                coverage_sum += cov_by_len[l]
                sentence_count += sen_by_len.get(l, 0)

        if sentence_count > 0:
            binned_cov[length] = coverage_sum / sentence_count
            binned_sen[length] = sentence_count

    return binned_cov, binned_sen


def plot_binned_coverage_per_section(cov_by_len, sen_by_len, bin_size=5):
    # Loop through each dataset in the subdir and plot a line for each
    for section_name in cov_by_len:
        plt.figure(figsize=(10, 6))
        for model_name in cov_by_len[section_name]:
            cov_by_len_for_dataset = cov_by_len[section_name][model_name]
            sen_by_len_for_dataset = sen_by_len[section_name][model_name]
            binned_cov, binned_sen = bin_data(cov_by_len_for_dataset, sen_by_len_for_dataset, bin_size)
            if not binned_cov:  # If binned_cov is empty, skip plotting
                continue
            # Plot the binned data with smaller dots
            lengths = list(binned_cov.keys())
            coverages = [binned_cov[l] for l in lengths]
            # Use smaller dots for markers
            plt.plot(lengths, coverages, marker='o', linestyle='-', color=model_colors[model_name],
                     label=f'{model_name}-{section_name} (Binned)', markersize=5)
        plt.xlabel('Length (Binned)', fontsize=14)
        plt.ylabel('Coverage (Average in Bin)', fontsize=14)
        plt.title(f'Binned Coverage vs Length - {section_name}', fontsize=16)
        plt.grid(True)
        # Save the plot as a file
        output_filename = f'plots/coverage_vs_length_{section_name}.png'
        plt.tight_layout()
        plt.legend()
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved plot as {output_filename}")


def plot_coverage_table(cov_by_len, sen_by_len, dataset_names, model_colors):
    coverage_data = {}
    section_names = set()
    model_names = set()
    totals = {}
    lengths = {}

    # Loop through each dataset (model) and calculate coverage per section
    for dataset_name in dataset_names:
        model_name = dataset_name.split('-')[0]  # Extract the model name (e.g., 'original', 'llama_7B')
        if model_name not in totals:
            totals[model_name] = 0
            lengths[model_name] = 0
        section_name = dataset_name.split('-')[1]  # Extract section name
        model_names.add(model_name)
        cov_by_len_for_dataset = cov_by_len[section_name][model_name]
        sen_by_len_for_dataset = sen_by_len[section_name][model_name]

        # Compute true coverage (no binning) for the entire section
        total_coverage = sum(cov_by_len_for_dataset.values()) / sum(
            sen_by_len_for_dataset.values()) if sen_by_len_for_dataset else 0
        totals[model_name] += sum(cov_by_len_for_dataset.values())
        lengths[model_name] += sum(sen_by_len_for_dataset.values())

        # Store coverage data in a dictionary
        if section_name not in coverage_data:
            coverage_data[section_name] = {}
        coverage_data[section_name][model_name] = total_coverage

        # Collect section names
        section_names.add(section_name)

    # Prepare the table headers: models as columns
    header = ['Section'] + sorted(list(model_names))
    section_coverages = {section: [] for section in section_names}

    # Open the file to write the table
    with open("coverage/coverage.txt", "w") as f:
        # Write the header (section names and model names)
        f.write("\t".join(header) + "\n")

        # Collect coverage data for each section
        for section in coverage_data:
            row = [section]
            for model_name in sorted(list(model_names)):
                coverage = coverage_data[section][model_name]
                section_coverages[section].append(coverage)
                row.append(f"{coverage:.2f}")
            # Write the row to the table
            f.write("\t".join(row) + "\n")

        # Write the total coverage for each model
        f.write("\t".join(["Total"] + [f"{totals[model_name] / lengths[model_name]:.2f}" for model_name in
                                       sorted(list(model_names))]) + "\n")

    print("Coverage table saved to 'coverage.txt'.")

    total_coverages = [totals[model_name] / lengths[model_name] if lengths[model_name] > 0 else 0 for model_name in
                       sorted(list(totals.keys()))]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sorted(list(totals.keys())), total_coverages,
           color=[model_colors[model_name] for model_name in sorted(list(totals.keys()))])

    ax.set_xlabel('Model')
    ax.set_ylabel('Total Coverage')
    ax.set_title('Total Coverage by Model')
    plt.tight_layout()

    plt.savefig("plots/total_coverage.png")
    print("Total coverage plot saved to 'coverage/total_coverage.png'.")


def plot_binned_coverage_all_data(cov_by_len, sen_by_len, bin_size=5):
    # Collecting and binning data across all sections for all models
    all_cov_by_len = {}
    all_sen_by_len = {}

    for section_name in cov_by_len:
        for model_name in cov_by_len[section_name]:
            cov_by_len_for_dataset = cov_by_len[section_name][model_name]
            sen_by_len_for_dataset = sen_by_len[section_name][model_name]

            for length, coverage in cov_by_len_for_dataset.items():
                if length not in all_cov_by_len:
                    all_cov_by_len[length] = 0
                    all_sen_by_len[length] = 0
                all_cov_by_len[length] += coverage
                all_sen_by_len[length] += sen_by_len_for_dataset.get(length, 0)

    binned_cov, binned_sen = bin_data(all_cov_by_len, all_sen_by_len, bin_size)

    # Plotting the binned data for all sections and models
    fig, ax = plt.subplots(figsize=(10, 6))
    lengths = list(binned_cov.keys())
    coverages = [binned_cov[l] for l in lengths]
    ax.plot(lengths, coverages, marker='o', linestyle='-', color='blue', label='Binned Coverage', markersize=5)

    ax.set_xlabel('Length (Binned)', fontsize=14)
    ax.set_ylabel('Coverage (Average in Bin)', fontsize=14)
    ax.set_title(f'Binned Coverage vs Length (All Data)', fontsize=16)
    ax.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig('plots/coverage_vs_length_all_data.png')
    print("Saved plot as 'plots/coverage_vs_length_all_data.png'.")

def compute_coverage_per_section(path_to_datasets):
    # Generate a color map for different models (e.g., 'original', 'llama_7B')
    colors = cm.get_cmap('tab10', 10)  # Using a predefined colormap (tab10)
    # Initialize dictionaries to store data
    cov_by_len = {}
    sen_by_len = {}
    dataset_names = []
    # Loop through all subdirectories and process each
    for subdir in os.listdir(path_to_datasets):
        subdir_path = os.path.join(path_to_datasets, subdir)
        if os.path.isdir(subdir_path):  # Ensure it's a directory
            # Process each dataset (model) in the subdirectory (section)
            for dataset in os.listdir(subdir_path):
                # Extract the model name (the part before the first dash)
                model_name = dataset.split('-')[0]
                section_name = dataset.split('-')[1]
                dataset_path = os.path.join(subdir_path, dataset)
                if os.path.isdir(dataset_path):  # Ensure it's a directory
                    total_cov, dataset_cov_by_len, dataset_sen_by_len = compute_coverage(dataset_path)
                    dataset_names.append(dataset)
                    if not section_name in cov_by_len:
                        cov_by_len[section_name] = {}
                        sen_by_len[section_name] = {}
                    if not model_name in cov_by_len[section_name]:
                        cov_by_len[section_name][model_name] = {}
                        sen_by_len[section_name][model_name] = {}
                    cov_by_len[section_name][model_name] = dataset_cov_by_len
                    sen_by_len[section_name][model_name] = dataset_sen_by_len
                    print(f"Coverage for {dataset}: {total_cov:.2f}")
                    # Assign a color to the model (only once per model)
                    if model_name not in model_colors:
                        model_colors[model_name] = colors(len(model_colors))
            plot_binned_coverage_per_section(cov_by_len, sen_by_len)
    # Output a single table of coverage per model per section
    plot_binned_coverage_all_data(cov_by_len, sen_by_len)
    plot_coverage_table(cov_by_len, sen_by_len, dataset_names, model_colors)

def compute_all_coverage(path_to_datasets):
    # Generate a color map for different models (e.g., 'original', 'llama_7B')
    colors = cm.get_cmap('tab10', 10)  # Using a predefined colormap (tab10)
    cov_by_len = {}
    sen_by_len = {}
    dataset_names = []

    # Loop through all subdirectories and process each
    for dataset in os.listdir(path_to_datasets):
        dataset_path = os.path.join(path_to_datasets, dataset)
        if os.path.isdir(dataset_path):  # Ensure it's a directory
            if os.path.isdir(dataset_path):  # Ensure it's a directory
                total_cov, dataset_cov_by_len, dataset_sen_by_len = compute_coverage(dataset_path)
                dataset_names.append(dataset)
                cov_by_len[dataset] = dataset_cov_by_len
                sen_by_len[dataset] = dataset_sen_by_len
                print(f"Coverage for {dataset}: {total_cov:.2f}")
                # Extract the model name (the part before the first dash)
                model_name = dataset.split('-')[0]
                # Assign a color to the model (only once per model)
                if model_name not in model_colors:
                    model_colors[model_name] = colors(len(model_colors))

    # Generate and save the plot for all subdirectories
    plot_binned_coverage_per_section(cov_by_len, sen_by_len, "all", dataset_names)


if __name__ == '__main__':
    path_to_datasets = sys.argv[1]
    # Compute coverage and generate one plot per subdirectory
    compute_coverage_per_section(path_to_datasets)
