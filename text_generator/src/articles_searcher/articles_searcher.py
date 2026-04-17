import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry
from utils import iter_months

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path("src/config/article_searcher.env"))
NYT_API_KEY = os.getenv("NYT_API_KEY")
base_url = "https://api.nytimes.com/svc/archive/v1/"

session = requests.Session()
retries = Retry(
    total=3,
    # backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))


def download_data(start_date: datetime, end_date: datetime) -> list:
    responses = []

    for year, month in iter_months(start_date, end_date):
        url = f"{base_url}{year}/{month}.json?api-key={NYT_API_KEY}"

        logger.info(url)
        logger.info(f"Downloading data from {year}-{month}...")
        try:

            response = session.get(url, timeout=5)
            if response.status_code == 200:
                responses.append(response.json())
        except Exception as e:
            logger.error(f"Error downloading data from {year}-{month}: {e}")

    return responses


def parsing_data(data: json) -> list:
    parsed_data = []

    for response in data:
        docs_info = response.get("response", {}).get("docs", [])

        for doc in docs_info:
            parsed_data.append(
                {
                    "headline": doc.get("headline", {}).get("main", ""),
                    "abstract": doc.get("abstract", ""),
                    "lead_paragraph": doc.get("lead_paragraph", ""),
                    "snippet": doc.get("snippet", ""),
                    "first_three_words": doc.get("abstract", "").split()[
                        :3
                    ],  # TODO mover para variable para cambiar entre abstract,lead y snippet
                    "section": doc.get("section_name", ""),
                    "news_desk": doc.get("news_desk", ""),
                    "keywords": [i.get("value", "") for i in doc.get("keywords", [])],
                    "author": doc.get("byline", {}).get("person", []),
                    "pub_date": doc.get("pub_date", ""),
                    "word_count": doc.get("word_count", 0),
                    "uri": doc.get("uri", ""),
                }
            )

    return parsed_data


def save_data(parsed_data: json, output_dir: str, file_name: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json.dump(
        parsed_data,
        open(os.path.join(output_dir, f"{file_name}"), "w"),
        indent=4,
        ensure_ascii=False,
    )
    logger.info(f"Data saved to {os.path.join(output_dir, file_name)}")


def main():

    parser = argparse.ArgumentParser(description="Download NYT Archive Data.")
    parser.add_argument(
        "start_date",
        type=lambda d: datetime.strptime(d, "%Y-%m"),
        help="Start date in YYYY-MM format",
    )
    parser.add_argument(
        "end_date",
        type=lambda d: datetime.strptime(d, "%Y-%m"),
        help="End date in YYYY-MM format",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for the downloaded data",
        default="data/",
        nargs="?",
    )

    args = parser.parse_args()

    data = download_data(args.start_date, args.end_date)
    parsed_data = parsing_data(data)
    # save_data(
    #     parsed_data,
    #     args.output_dir,
    #     f"{args.start_date.year}_{args.start_date.month}-{args.end_date.year}_{args.end_date.month}.json",
    # )

    logger.info(parsed_data)

    save_data(
        parsed_data,
        args.output_dir,
        f"{args.start_date.year}_{args.start_date.month}-{args.end_date.year}_{args.end_date.month}.json",
    )


if __name__ == "__main__":
    main()
