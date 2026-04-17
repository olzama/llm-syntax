import argparse
import json
import logging
import os

import stanza
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_lead_paragraphs(news_data: json) -> list[str]:
    return [article.get("lead_paragraph", "") for article in news_data]


def get_generated_lead_paragraphs(generated_text_path: str) -> list[str]:
    with open(generated_text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    return lines


def tokenize_lead_paragraphs(
    lead_paragraphs: list[str], output_file: str, pipe: stanza.Pipeline
) -> list[dict]:

    indexes = []
    counter_lines = 1

    with open(output_file, "w", encoding="utf-8") as f:
        iterator = tqdm.tqdm(
            lead_paragraphs,
            desc="Tokenizing lead paragraphs",
            unit="paragraph",
            dynamic_ncols=True,
        )

        for i, lead in enumerate(iterator):
            start = counter_lines

            if lead:
                doc = pipe(lead)
                for j, sentence in enumerate(doc.sentences):
                    text = sentence.text.strip()
                    if not text:
                        continue
                    # logger.info(f" Lead Paragraph {i+1} Sentence {j+1}: {text}")
                    f.write(text + "\n")
                    counter_lines += 1

            end = counter_lines
            indexes.append(
                {"start": start, "end": end - 1, "sentences_count": end - start}
            )

    return indexes


def modify_original_json(
    indexes: list[dict], original_json_path: str, output_json_path: str, model_name: str
):

    field_name = "txt_generated_indexes_" + model_name
    with open(original_json_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    for i, idx in enumerate(indexes):
        if i < len(news_data):
            news_data[i][field_name] = idx
        else:
            logger.warning(f"Index {i} out of range for news data.")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Process news articles")
    args.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/2025_2-2025_5_modified.json",
        help="Input JSON file",
    )
    # TODO cambiar el output para que sea el archivo de texto generado
    args.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/2025_2-2025_5_modified_prueba.json",
        help="Output JSON file",
    )

    args = args.parse_args()

    pipe = stanza.Pipeline(lang="en", processors="tokenize", verbose=False)

    with open(args.input, "r", encoding="utf-8") as f:

        finals = get_generated_lead_paragraphs("src/text_generator/finals.txt")
        indexes = tokenize_lead_paragraphs(
            finals,
            f"parsed_texts/generated_lead_paragraphs_{args.input.split('/')[-1].split('.')[0]}_tokenized.txt",
            pipe,
        )
        modify_original_json(indexes, args.input, args.output, model_name)
