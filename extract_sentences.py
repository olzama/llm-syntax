import sys, os
import json
import stanza
import matplotlib.pyplot as plt
import numpy as np
import hashlib

EXCLUDE_AUTHORS = {'New York Times Games', ' York Times Audio', 'New York Times Audio', 'The Learning Network',
                   'Florence Fabricant',
                   'The New York Times', 'The New York Times Cooking', 'New York Times Games', 'New York Times Opinion',
                   'ABC Australia', 'The New York Times Magazine', 'News Nation', 'NBC News', 'Nbc news',
                   'The New York Times Books Staff',
                   }


def generate_key(sentence: str) -> str:
    # Hash the sentence and get a unique fixed-length key
    return hashlib.sha256(sentence.encode('utf-8')).hexdigest()



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
        #print(f"{len(sentences)} sentences for {sec} have been saved in {filename}")
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
    #print(f"Total {len(all_sentences)} sentences written for {fname} in {fname}.txt")


if __name__ == '__main__':
    # Initialize the pipeline for sentence segmentation
    stz = stanza.Pipeline('en', processors='tokenize')
    path_to_generated_data = sys.argv[1]
    model_names = ['falcon_7B', 'llama_7B', 'llama_13B', 'llama_30B', 'llama_65B', 'mistral_7B']
    original = load_json_data(sys.argv[2])
    original_per_section = {}
    per_author = {}
    sentence2authors = {}
    sen_count = 0
    # Process the original data
    #for i in range(10):
    for i in range(len(original)):
        print("Tokenizing paragraph %d..." % i)
        sec = original[i]['section_name']
        author = original[i]['byline']['original']
        if sec not in original_per_section:
            if sec:
                original_per_section[sec] = []
        if original[i]['lead_paragraph']:
            s = tokenize_paragraph(original[i]['lead_paragraph'], stz)
            original_per_section[sec].extend(s)
            for sen in s:
                sen_key = generate_key(sen)
                if sen_count not in sentence2authors:
                    sentence2authors[sen_key] = {'authors': {}, 'sentence': None}
                if author not in sentence2authors[sen_key]['authors']:
                    sentence2authors[sen_key]['authors'][author] = 0
                sentence2authors[sen_key]['sentence'] = sen
                sentence2authors[sen_key]['authors'][author] += 1
                sen_count += 1
            if author not in per_author:
                per_author[author] = []
            per_author[author].extend(s)

    # Sort by number of sentences per author:
    per_author = {k: v for k, v in sorted(per_author.items(), key=lambda item: len(item[1]), reverse=True)}
    single_authored = {}
    # Report the number of sentences per author, including the maximum, the mean, and the median.
    for author, sentences in per_author.items():
        if (not author) or ',' in author or ' and ' in author or author[2:].strip() in EXCLUDE_AUTHORS:
            continue
        single_authored[author] = sentences
        clean_author = author.replace(" ", "_").replace("&", "and").replace(".", "")
        if len(clean_author) > 50:
            clean_author = "TooManyAuthors"
        filename = f"original-{clean_author}.txt"
        directory = f'/mnt/kesha/llm-syntax-data/by-one-author/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + filename, 'w') as file:
            for sentence in sentences:
                file.write(sentence + "\n")
    # Report maximum, mean, and median number of sentences per author:
    num_authors = len(single_authored)
    num_sentences = sum([len(sentences) for sentences in single_authored.values()])
    max_sentences = len(list(single_authored.values())[0])
    mean_sentences = num_sentences / num_authors
    median_sentences = len(list(single_authored.values())[num_authors // 2])
    print(f"Number of authors: {num_authors}")
    print(f"Number of sentences: {num_sentences}")
    print(f"Maximum number of sentences per author: {max_sentences}")
    print(f"Mean number of sentences per author: {mean_sentences}")
    print(f"Median number of sentences per author: {median_sentences}")
    print(f"Authors with more than 100 sentences: {len([1 for sentences in single_authored.values() if len(sentences) > 100])}")
    print(f"Authors with more than 50 sentences: {len([1 for sentences in single_authored.values() if len(sentences) > 50])}")
    print(f"Authors with more than 25 sentences: {len([1 for sentences in single_authored.values() if len(sentences) > 25])}")

    # Plot number of authors per number of sentences:
    num_sentences_per_author = [len(sentences) for sentences in single_authored.values()]
    plt.hist(num_sentences_per_author, bins=np.arange(0, 100, 5))
    plt.xlabel('Number of sentences')
    plt.ylabel('Number of authors')
    plt.title('Number of sentences per author')
    plt.savefig('/mnt/kesha/llm-syntax-data/num_sentences_per_author.png')
    with open('/mnt/kesha/llm-syntax-data/sentences2author.json', 'w', encoding='utf-8') as f:
        json.dump(sentence2authors, f, ensure_ascii=False)
    # write_all_sentences(original_per_section, "original")
    # original_sentence_count = write_per_section(original_per_section, "original")
    #
    # # Process the generated data for each model
    # generated_per_model_per_section = {}
    # for model in model_names:
    #     print(f"Processing {model}...")
    #     generated_per_model_per_section[model] = {}
    #     generated = load_json_data(path_to_generated_data + model + '.json')
    #     assert len(generated) == len(original)
    #     #for i in range(10):
    #     for i in range(len(generated)):
    #         print("Tokenizing paragraph %d..." % i)
    #         sec = original[i]['section_name']
    #         if sec not in generated_per_model_per_section[model]:
    #             if sec:
    #                 generated_per_model_per_section[model][sec] = []
    #         if original[i]['lead_paragraph']:
    #             generated_per_model_per_section[model][sec].extend(
    #                 tokenize_paragraph(generated[i]['lead_paragraph'], stz))
    #
    #     # Write out the model sentences
    #     write_all_sentences(generated_per_model_per_section[model], model)
    #     generated_sentence_count = write_per_section(generated_per_model_per_section[model], model)
    #
    # # Create a dictionary to store the count of sentences per section for original and generated
    # orig_sentence_counts = {key: len(sentences) for key, sentences in original_per_section.items()}
    # gen_sentence_counts = {
    #     model: {key: len(sentences) for key, sentences in generated_per_model_per_section[model].items()} for model in
    #     model_names}
    #
    # # Sort the dictionaries by the count of sentences in descending order
    # orig_sorted_sentence_counts = dict(sorted(orig_sentence_counts.items(), key=lambda item: item[1], reverse=True))
    # gen_sorted_sentence_counts = {model: dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)) for
    #                               model, counts in gen_sentence_counts.items()}
    #
    # # Write statistics to a file
    # with open('/mnt/kesha/llm-syntax-data/raw-sentences/per-section/statistics.txt', 'w') as file:
    #     file.write('\toriginal\t')
    #     for model in model_names:
    #         file.write(model + '\t')
    #     file.write('\n')
    #
    #     # Writing out the statistics for each section
    #     for key in orig_sorted_sentence_counts:
    #         file.write(f"{key}\t{orig_sorted_sentence_counts[key]}")
    #         for model in model_names:
    #             file.write(f"\t{gen_sorted_sentence_counts[model].get(key, 0)}")
    #         file.write("\n")

    print('done.')
