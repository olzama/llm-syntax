import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, List

import stanza
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BLOCK_SEPARATOR = "------------------------------"

VALID_MODELS = {
    "Qwen2.5-72B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Llama-3.3-70B-Instruct",
    "TinyLlama-1.1B-Chat-v1.0",
    "gpt-oss-20b",
    "gpt-4o",
    "MegaBeam-Mistral-7B-512k",
    "Mistral-7B-Instruct-v0.3",
    "Mistral-7B-v0.3",
}


def _strip_header(text: str) -> str:

    return "\n".join(text.splitlines()[1:])


def _split_blocks(text: str) -> List[str]:
    return [b.strip() for b in text.split(BLOCK_SEPARATOR)]


def _join_lines(block: str, max_lines: int | None = None) -> str:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if max_lines is not None:
        lines = lines[:max_lines]
    return " ".join(lines)


def _mistral_fix_token_list(match):
    inside = match.group(1)
    tokens = re.findall(r"'([^']*)'", inside)
    if not tokens:
        return ""
    sentence = " ".join(tokens)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"\s+([.,!?;:])", r"\1", sentence)
    return sentence


def _mistral_clean_token_lists(text):
    text = re.sub(r"\[([^\]\n]*)", _mistral_fix_token_list, text)
    text = re.sub(r"\[\s*(?:,\s*)+(?:\]|(?=\n|$))", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
    text = text.replace("[", "").replace("]", "")
    # Remove multiple consecutive commas
    text = re.sub(r",\s*,+", ",", text)
    # Remove any consecutive punctuation marks
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)
    # Remove leading/trailing commas
    text = re.sub(r"^\s*,\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*,\s*$", "", text, flags=re.MULTILINE)
    # Remove consecutive space and dot sequences
    text = re.sub(r"(\s*\.\s*){2,}", ". ", text)
    # Remove consecutive comma sequences
    text = re.sub(r"(\s*,\s*){2,}", ", ", text)
    # Remove 'text', but keep the text inside
    text = re.sub(r"'([^']*)'", r"\1", text)
    # Normalize spaces again
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_mistral7b(gen_texts: str) -> List[str]:
    text = _strip_header(gen_texts)
    result = _mistral_clean_token_lists(text)
    return [_join_lines(b) for b in _split_blocks(result) if b]


def preprocess_default_answer(gen_texts: str) -> List[str]:
    text = _strip_header(gen_texts)
    return [_join_lines(b) for b in _split_blocks(text) if b]


def preprocess_tinyllama(gen_texts: str) -> List[str]:
    text = _strip_header(gen_texts)
    return [_join_lines(b, max_lines=2) for b in _split_blocks(text) if b]


def preprocess_gptoss20b(gen_texts: str) -> List[str]:
    text = _strip_header(gen_texts)
    matches = re.findall(
        r"assistantfinal(.*?)" + re.escape(BLOCK_SEPARATOR), text, re.DOTALL
    )
    return [_join_lines(m) for m in matches if m.strip()]


def check_model_name(generated_text_path: Path) -> str:
    with generated_text_path.open("r", encoding="utf-8") as f:
        model_name = f.readline().strip()

    if model_name not in VALID_MODELS:
        logger.error(
            "Model name '%s' not in valid models: %s", model_name, VALID_MODELS
        )
        sys.exit(1)

    logger.info("Model name '%s' is valid.", model_name)
    return model_name


PREPROCESSORS: dict[str, Callable[[str], List[str]]] = {
    "Qwen2.5-72B-Instruct": preprocess_default_answer,
    "Qwen2.5-14B-Instruct": preprocess_default_answer,
    "Qwen2.5-32B-Instruct": preprocess_default_answer,
    "gpt-4o": preprocess_default_answer,
    "MegaBeam-Mistral-7B-512k": preprocess_default_answer,
    "Mistral-7B-Instruct-v0.3": preprocess_default_answer,
    "Mistral-7B-v0.3": preprocess_mistral7b,
    "TinyLlama-1.1B-Chat-v1.0": preprocess_tinyllama,
    "gpt-oss-20b": preprocess_gptoss20b,
}


def preprocess_generated_texts(model_name: str, raw_text: str) -> List[str]:
    fn = PREPROCESSORS.get(model_name)
    if fn is None:
        logger.error("Model '%s' not supported for preprocessing.", model_name)
        sys.exit(1)
    return fn(raw_text)


def tokenize_lead_paragraphs(
    lead_paragraphs: list[str],
    output_file: str,
    pipe: stanza.Pipeline,
    max_sentences: int = -1,
) -> List[dict]:

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

                if max_sentences == -1:
                    for j, sentence in enumerate(doc.sentences):
                        text = sentence.text.strip()
                        if not text:
                            continue
                        # logger.info(f" Lead Paragraph {i+1} Sentence {j+1}: {text}")
                        f.write(text + "\n")
                        counter_lines += 1
                else:
                    for j, sentence in enumerate(doc.sentences):
                        text = sentence.text.strip()
                        if not text:
                            continue
                        # logger.info(f" Lead Paragraph {i+1} Sentence {j+1}: {text}")
                        f.write(text + "\n")
                        counter_lines += 1
                        if j >= max_sentences - 1:
                            break

            end = counter_lines
            indexes.append(
                {"start": start, "end": end - 1, "sentences_count": end - start}
            )

    return indexes


def modify_original_json(
    indexes: list[dict], original_json_path: str, output_json_path: str, model_name: str
):
    with open(original_json_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    for i, idx in enumerate(indexes):
        if i < len(news_data):
            news_data[i][f"{model_name}_txt_indexes"] = idx
        else:
            logger.warning(f"Index {i} out of range for news data.")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(news_data, f, indent=4, ensure_ascii=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process news articles")
    p.add_argument(
        "--input_json",
        "-ij",
        type=Path,
        default=Path("data/2025_2-2025_5_modified_prueba.json"),
        help="Input JSON file",
    )
    p.add_argument(
        "--input_gen_texts",
        "-igt",
        type=Path,
        default=Path("generated_texts/TinyLlama/TinyLlama-1.1B-Chat-v1.0.txt"),
        help="Input generated texts file",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/2025_2-2025_5_modified_prueba.json"),
        help="Output JSON file",
    )
    p.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for Stanza tokenizer (default: en)",
    )
    p.add_argument(
        "--max_sentences",
        "-ms",
        type=int,
        default=-1,
        help="Maximum number of sentences to extract from each lead paragraph (default: -1 for all)",
    )
    return p


def main():
    args = build_argparser().parse_args()

    input_json_path = args.input_json
    input_gen_texts_path = args.input_gen_texts
    output_path = args.output
    lang = args.lang
    max_sentences = args.max_sentences

    model_name = check_model_name(input_gen_texts_path)

    with open(input_gen_texts_path, "r", encoding="utf-8") as f:
        gen_texts = f.read()

    gen_texts_lines = preprocess_generated_texts(model_name, gen_texts)

    pipe = stanza.Pipeline(lang=lang, processors="tokenize", verbose=False)

    os.makedirs("parsed_texts", exist_ok=True)
    indexes = tokenize_lead_paragraphs(
        gen_texts_lines,
        f"parsed_texts/{model_name}_stanza_tokenized.txt",
        pipe,
        max_sentences,
    )

    modify_original_json(indexes, input_json_path, output_path, model_name)


if __name__ == "__main__":
    main()
