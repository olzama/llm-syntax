import sys, os
import json
import stanza


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        data = json.loads(text)
    return data


# Function to tokenize paragraphs into sentences
def tokenize_paragraph(paragraph, stz):
    doc = stz(paragraph)
    sentences = [sentence.text for sentence in doc.sentences]
    return sentences


def write_per_section(data, fname):
    total_sentences = 0
    for sec, sentences in data.items():
        clean_sec = sec.replace(" ", "_").replace("&", "and").replace(".", "")
        # Create a filename with the section key
        filename = f"{fname}-{clean_sec}.txt"
        # Create a directory if it doesn't exist
        directory = f'/mnt/kesha/llm-syntax-data/raw-sentences/per-section/{clean_sec}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Write sentences to the file
        with open(directory + filename, 'w') as file:
            for sentence in sentences:
                file.write(sentence + "\n")  # Write each sentence on a new line
        total_sentences += len(sentences)
        print(f"{len(sentences)} sentences for {sec} have been saved in {filename}")
    return total_sentences


def write_all_sentences(data, fname):
    # Create a single file containing all sentences for the model (or "original")
    all_sentences = []
    for sentences in data.values():
        all_sentences.extend(sentences)
    # Write all sentences to a single file
    with open(f'/mnt/kesha/llm-syntax-data/raw-sentences/{fname}.txt', 'w') as file:
        for sentence in all_sentences:
            file.write(sentence + "\n")
    print(f"Total {len(all_sentences)} sentences written for {fname} in {fname}.txt")


if __name__ == '__main__':
    # Initialize the pipeline for sentence segmentation
    stz = stanza.Pipeline('en', processors='tokenize')
    path_to_generated_data = sys.argv[1]
    model_names = ['falcon_7B', 'llama_7B', 'llama_13B', 'llama_30B', 'llama_65B', 'mistral_7B']
    original = load_json_data(sys.argv[2])
    original_per_section = {}

    # Process the original data
    #for i in range(10):
    for i in range(len(original)):
        print("Tokenizing paragraph %d..." % i)
        sec = original[i]['section_name']
        if sec not in original_per_section:
            if sec:
                original_per_section[sec] = []
        if original[i]['lead_paragraph']:
            original_per_section[sec].extend(tokenize_paragraph(original[i]['lead_paragraph'], stz))

    # Write out the original sentences
    write_all_sentences(original_per_section, "original")
    original_sentence_count = write_per_section(original_per_section, "original")

    # Process the generated data for each model
    generated_per_model_per_section = {}
    for model in model_names:
        print(f"Processing {model}...")
        generated_per_model_per_section[model] = {}
        generated = load_json_data(path_to_generated_data + model + '.json')
        assert len(generated) == len(original)
        #for i in range(10):
        for i in range(len(generated)):
            print("Tokenizing paragraph %d..." % i)
            sec = original[i]['section_name']
            if sec not in generated_per_model_per_section[model]:
                if sec:
                    generated_per_model_per_section[model][sec] = []
            if original[i]['lead_paragraph']:
                generated_per_model_per_section[model][sec].extend(
                    tokenize_paragraph(generated[i]['lead_paragraph'], stz))

        # Write out the model sentences
        write_all_sentences(generated_per_model_per_section[model], model)
        generated_sentence_count = write_per_section(generated_per_model_per_section[model], model)

    # Create a dictionary to store the count of sentences per section for original and generated
    orig_sentence_counts = {key: len(sentences) for key, sentences in original_per_section.items()}
    gen_sentence_counts = {
        model: {key: len(sentences) for key, sentences in generated_per_model_per_section[model].items()} for model in
        model_names}

    # Sort the dictionaries by the count of sentences in descending order
    orig_sorted_sentence_counts = dict(sorted(orig_sentence_counts.items(), key=lambda item: item[1], reverse=True))
    gen_sorted_sentence_counts = {model: dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)) for
                                  model, counts in gen_sentence_counts.items()}

    # Write statistics to a file
    with open('/mnt/kesha/llm-syntax-data/raw-sentences/per-section/statistics.txt', 'w') as file:
        file.write('\toriginal\t')
        for model in model_names:
            file.write(model + '\t')
        file.write('\n')

        # Writing out the statistics for each section
        for key in orig_sorted_sentence_counts:
            file.write(f"{key}\t{orig_sorted_sentence_counts[key]}")
            for model in model_names:
                file.write(f"\t{gen_sorted_sentence_counts[model].get(key, 0)}")
            file.write("\n")

    print('done.')
