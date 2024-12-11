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

    # Sort dictionaries by sentence length once
    sorted_cov_by_len = dict(sorted(cov_by_len.items()))
    sorted_sen_by_len = dict(sorted(sen_by_len.items()))

    # Calculate total coverage ratio
    parsed = len(dataset['result'])
    total_cov = parsed / len(dataset['item'])

    return total_cov, sorted_cov_by_len, sorted_sen_by_len


def bin_data(cov_by_len, sen_by_len, bin_size=10):
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


def plot_binned_coverage_per_section(cov_by_len, sen_by_len, model_colors, bin_size=5):
    # Loop through each dataset in the subdir and plot a line for each
    for section_name in cov_by_len:
        plt.figure(figsize=(10, 6))
        for model_name in cov_by_len[section_name]:
            cov_by_len_for_dataset = cov_by_len[section_name][model_name]
            sen_by_len_for_dataset = sen_by_len[section_name][model_name]
            binned_cov, binned_sen = bin_data(cov_by_len_for_dataset, sen_by_len_for_dataset, bin_size)
            if not binned_cov:
                continue
            lengths = list(binned_cov.keys())
            coverages = [binned_cov[l] for l in lengths]
            plt.plot(lengths, coverages, marker='o', linestyle='-', color=model_colors[model_name],
                     label=f'{model_name}-{section_name} (Binned)', markersize=5)
        plt.xlabel('Length (Binned)', fontsize=14)
        plt.ylabel('Coverage (Average in Bin)', fontsize=14)
        plt.title(f'Binned Coverage vs Length - {section_name}', fontsize=16)
        plt.grid(True)
        output_filename = f'plots/all-lengths/coverage_vs_length_{section_name}.png'
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
                coverage = coverage_data[section].get(model_name, 0)
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
    print("Total coverage plot saved to 'plots/total_coverage.png'.")


def plot_binned_coverage_all_data(cov_by_len, sen_by_len, bin_size=5):
    # Prepare a dictionary to hold binned data for each model
    binned_coverage_by_model = {}

    for section_name in cov_by_len:
        for model_name in cov_by_len[section_name]:
            cov_by_len_for_dataset = cov_by_len[section_name][model_name]
            sen_by_len_for_dataset = sen_by_len[section_name][model_name]

            # Bin the data for each model separately
            binned_cov, binned_sen = bin_data(cov_by_len_for_dataset, sen_by_len_for_dataset, bin_size)

            # Store binned data by model
            binned_coverage_by_model[model_name] = (binned_cov, binned_sen)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop over the binned data and plot each model's data
    for model_name in binned_coverage_by_model:
        binned_cov, binned_sen = binned_coverage_by_model[model_name]
        lengths = list(binned_cov.keys())
        coverages = [binned_cov[l] for l in lengths]
        ax.plot(lengths, coverages, marker='o', linestyle='-', label=f'{model_name} (Binned)', markersize=5)

    # Add labels, title, and grid
    ax.set_xlabel('Length (Binned)', fontsize=14)
    ax.set_ylabel('Coverage (Average in Bin)', fontsize=14)
    ax.set_title(f'Binned Coverage vs Length (All Data by Model)', fontsize=16)
    ax.grid(True)

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.legend()
    plt.savefig('plots/all-lengths/coverage_vs_length_all_data_by_model.png')
    print("Saved plot as 'plots/all-lengths/coverage_vs_length_all_data_by_model.png'.")


def plot_zoomed_binned_coverage_all_data(cov_by_len, sen_by_len, min, max, bin_size=5):
    all_cov_by_len = {}
    all_sen_by_len = {}

    binned_coverage_by_model = {}

    # Loop through all sections and models, collect coverage data
    for section_name in cov_by_len:
        for model_name in cov_by_len[section_name]:
            cov_by_len_for_dataset = cov_by_len[section_name][model_name]
            sen_by_len_for_dataset = sen_by_len[section_name][model_name]

            # Aggregate data for each model
            for length, coverage in cov_by_len_for_dataset.items():
                if length not in all_cov_by_len:
                    all_cov_by_len[length] = 0
                    all_sen_by_len[length] = 0
                all_cov_by_len[length] += coverage
                all_sen_by_len[length] += sen_by_len_for_dataset.get(length, 0)

            # Bin data for each model separately
            binned_cov, binned_sen = bin_data(all_cov_by_len, all_sen_by_len, bin_size)
            binned_coverage_by_model[model_name] = (binned_cov, binned_sen)

            # Reset the dictionaries for the next model
            all_cov_by_len = {}
            all_sen_by_len = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in binned_coverage_by_model:
        binned_cov, binned_sen = binned_coverage_by_model[model_name]
        # Filter data to only include lengths between min and max
        binned_cov = {k: v for k, v in binned_cov.items() if min <= k <= max}
        if not binned_cov:  # Skip if binned_cov is empty
            continue

        lengths = list(binned_cov.keys())
        coverages = [binned_cov[l] for l in lengths]

        # Plot the binned data with zoomed-in Y-axis
        ax.plot(lengths, coverages, marker='o', linestyle='-', label=f'{model_name} (Zoomed)', markersize=5)

    ax.set_xlabel('Length (Binned)', fontsize=14)
    ax.set_ylabel('Coverage (Average in Bin)', fontsize=14)
    ax.set_title(f'Zoomed Binned Coverage vs Length ({min}-{max}) by Model', fontsize=16)

    # Set the Y-axis limits to zoom in between 80 and 100 percent
    ax.set_ylim(0.8, 1.0)

    ax.grid(True)
    plt.tight_layout()
    plt.legend()

    # Correct output filename to match the zoom range
    output_filename = f'plots/zoomed-{min}-{max}/coverage_vs_length_all_data_by_model_zoomed.png'
    plt.savefig(output_filename)
    print(f"Saved zoomed plot as {output_filename}.")


# Example usage of the function:
# plot_zoomed_binned_coverage_all_data(cov_by_len, sen_by_len, bin_size=5)

def plot_zoomed_coverage(cov_by_len, sen_by_len, model_colors, min, max, bin_size=5):
    # Create plots zoomed in for lengths between 0 and 20
    for section_name in cov_by_len:
        plt.figure(figsize=(10, 6))
        for model_name in cov_by_len[section_name]:
            cov_by_len_for_dataset = cov_by_len[section_name][model_name]
            sen_by_len_for_dataset = sen_by_len[section_name][model_name]
            binned_cov, binned_sen = bin_data(cov_by_len_for_dataset, sen_by_len_for_dataset, bin_size)

            # Filter data to only include lengths between 0 and 20
            binned_cov = {k: v for k, v in binned_cov.items() if min <= k <= max}
            if not binned_cov:  # If binned_cov is empty, skip plotting
                continue

            lengths = list(binned_cov.keys())
            coverages = [binned_cov[l] for l in lengths]

            # Plot the binned data with zoomed in Y-axis
            plt.plot(lengths, coverages, marker='o', linestyle='-', color=model_colors[model_name],
                     label=f'{model_name}-{section_name} (Zoomed)', markersize=5)

        plt.xlabel('Length (Binned)', fontsize=14)
        plt.ylabel('Coverage (Average in Bin)', fontsize=14)
        plt.title(f'Zoomed in Binned Coverage vs Length - {section_name} ({min}-{max})', fontsize=16)

        # Set the Y-axis limits to zoom in between 80 and 100 percent
        plt.ylim(0.8, 1.0)

        plt.grid(True)
        output_filename = f'plots/zoomed-10-20/coverage_vs_length_zoomed_{section_name}.png'
        plt.tight_layout()
        plt.legend()
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved zoomed plot as {output_filename}")
# Example usage
def process_datasets(path_to_datasets):
    colors = cm.get_cmap('tab10', 10)
    cov_by_len = {}
    sen_by_len = {}
    dataset_names = []

    for subdir in os.listdir(path_to_datasets):
        subdir_path = os.path.join(path_to_datasets, subdir)
        if os.path.isdir(subdir_path):
            for dataset in os.listdir(subdir_path):
                print(f"Processing {dataset}...")
                model_name = dataset.split('-')[0]
                section_name = dataset.split('-')[1]
                dataset_path = os.path.join(subdir_path, dataset)
                if os.path.isdir(dataset_path):
                    total_cov, dataset_cov_by_len, dataset_sen_by_len = compute_coverage(dataset_path)
                    dataset_names.append(dataset)
                    if section_name not in cov_by_len:
                        cov_by_len[section_name] = {}
                        sen_by_len[section_name] = {}
                    cov_by_len[section_name][model_name] = dataset_cov_by_len
                    sen_by_len[section_name][model_name] = dataset_sen_by_len
                    if model_name not in model_colors:
                        model_colors[model_name] = colors(len(model_colors))

    plot_binned_coverage_per_section(cov_by_len, sen_by_len, model_colors,5)
    plot_zoomed_coverage(cov_by_len, sen_by_len, model_colors, 10, 20, 3)
    plot_zoomed_binned_coverage_all_data(cov_by_len, sen_by_len, 10, 20, 3)
    plot_binned_coverage_all_data(cov_by_len, sen_by_len, 5)
    plot_coverage_table(cov_by_len, sen_by_len, dataset_names, model_colors)


if __name__ == '__main__':
    path_to_datasets = sys.argv[1]
    process_datasets(path_to_datasets)
