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


def write_per_section(data,fname):
    for sec, sentences in data.items():
        clean_sec = sec.replace(" ", "_").replace("&", "and").replace(".","")
        # Create a filename with the section key
        filename = f"{fname}-{clean_sec}.txt"
        #Create a directory:
        if not os.path.exists('/mnt/kesha/llm-syntax-data/raw-sentences/per-section/' + fname + '/'):
            os.makedirs('/mnt/kesha/llm-syntax-data/raw-sentences/per-section/' + fname + '/')
        # Write sentences to the file
        with open('/mnt/kesha/llm-syntax-data/raw-sentences/per-section/' + fname + '/' + filename, 'w') as file:
            for sentence in sentences:
                file.write(sentence + "\n")  # Write each sentence on a new line
        print(f"{len(data[sec])} sentences for {sec} have been saved in {filename}")


if __name__ == '__main__':
    # Initialize the pipeline for sentence segmentation
    stz = stanza.Pipeline('en', processors='tokenize')
    path_to_generated_data = sys.argv[1]
    model_names = ['falcon_7B', 'llama_7B', 'llama_13B', 'llama_30B', 'llama_65B', 'mistral_7B']
    original = load_json_data(sys.argv[2])
    original_per_section = {}
    #for i in range(0, 10):
    for i in range(len(original)):
        print("Tokenizing paragraph %d..." % i)
        sec = original[i]['section_name']
        if sec not in original_per_section:
            if sec:
                original_per_section[sec] = []
        if original[i]['lead_paragraph']:
            original_per_section[sec].extend(tokenize_paragraph(original[i]['lead_paragraph'], stz))
    generated_per_model_per_section = {}
    write_per_section(original_per_section, "original")
    for model in model_names:
        generated_per_model_per_section[model] = {}
        generated = load_json_data(path_to_generated_data + model + '.json')
        assert len(generated) == len(original)
        #for i in range(0,10):
        for i in range(len(generated)):
            print("Tokenizing paragraph %d..." % i)
            sec = original[i]['section_name']
            if sec not in generated_per_model_per_section[model]:
                if sec:
                    generated_per_model_per_section[model][sec] = []
            if original[i]['lead_paragraph']:
                generated_per_model_per_section[model][sec].extend(tokenize_paragraph(generated[i]['lead_paragraph'], stz))
        write_per_section(generated_per_model_per_section[model], model)
    # Create a dictionary to store the count of sentences per section
    orig_sentence_counts = {key: len(sentences) for key, sentences in original_per_section.items()}
    # Sort the dictionary by the count of sentences in descending order
    orig_sorted_sentence_counts = dict(sorted(orig_sentence_counts.items(), key=lambda item: item[1], reverse=True))
    # Print out the statistics
    with open('/mnt/kesha/llm-syntax-data/raw-sentences/per-section/' + 'statistics.txt', 'w') as file:
        file.write('\toriginal\t')
        for model in model_names:
            file.write(model + '\t')
        file.write('\n')
        for key in orig_sorted_sentence_counts:
            file.write("{}\t{}".format(key, orig_sorted_sentence_counts[key]))
            for model in model_names:
                gen_sentence_counts = {key: len(sentences) for key, sentences in
                                       generated_per_model_per_section[model].items()}
                gen_sorted_sentence_counts = dict(
                    sorted(gen_sentence_counts.items(), key=lambda item: item[1], reverse=True))
                file.write("\t{}".format(gen_sorted_sentence_counts[key]))
            file.write("\n")
    print('done.')
