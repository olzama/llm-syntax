import argparse
import json
import logging
import os

from config_loader import load_config
from dotenv import load_dotenv
from openai import OpenAI

from text_generator import TextGenerator

load_dotenv("src/config/article_searcher.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_news_headline_three_words(news_json_path: str) -> list[str]:
    news_headlines = []

    with open(news_json_path, "r", encoding="utf-8") as f:
        news_json = json.load(f)

    for article in news_json:
        news_headlines.append(
            (article.get("headline", ""), article.get("first_three_words", []))
        )

    return news_headlines


def main():

    model_cfg, decoding_cfg, quant_cfg, chat_cfg, news_cfg = load_config(
        path="src/config/config.yaml"
    )

    model = TextGenerator(model_cfg, decoding_cfg, quant_cfg, chat_cfg)
    # TODO meter paths en config?
    path = "data/2025_2-2025_5_modified.json"

    args = argparse.ArgumentParser(description="Generate news leads")
    args.add_argument(
        "--input_json",
        "-ij",
        type=str,
        default="data/2025_2-2025_5_modified.json",
        help="Input JSON file",
    )

    args.add_argument(
        "--output",
        "-o",
        type=str,
        default=f"generated_texts/{model_cfg.name}.txt",
        help="Output JSON file",
    )

    args.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=0,
        help="Number of top headlines to process",
    )
    args = args.parse_args()

    if args.top_k > 0:
        headlines = get_news_headline_three_words(args.input_json)[: args.top_k]
    else:
        headlines = get_news_headline_three_words(args.input_json)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if model_cfg.is_chat_model:
        logger.info(f"Using chat model: {model_cfg.name}")
        prompts = [
            (
                f"""
               You will write a news lead paragraph using the inputs below.
                Inputs
                Headline: {headline}
                LeadThreeWords: {lead_three_words}
                Requirements — Mandatory
                Write one paragraph of several sentences (more than one, e.g. two-three (2-3); no title, no bullets.
                Output format: the paragraph only—no preamble or labels.
                """
            )
            for headline, lead_three_words in headlines
        ]
    else:
        logger.info(f"Using standard model: {model_cfg.name}")
        prompts = [
            f"""
        {headline}
        {lead_three_words}"""
            for headline, lead_three_words in headlines
        ]

    sep = f"\n\n{'-'*30}\n\n"
    with open(args.output, "w", encoding="utf-8") as f:
        name = model_cfg.name.split("/")[-1]
        f.write(f"{name}\n")
        outputs = model.generate_texts(prompts)
        for output in outputs:
            f.write(f"{output}")
            f.write(sep)

    # logging.info(f"{outputs}")


if __name__ == "__main__":

    main()
