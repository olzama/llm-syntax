"""
extract_sentences.py — tokenize NYT article paragraphs into sentences,
filter to single-authored articles, and write per-author sentence files.

First step of the author-level diversity analysis pipeline.  Output feeds
sentences2authors.py and the author comparison chain.

Multi-authored bylines (containing ',' or ' and ') and a fixed exclusion list
of institutional bylines (e.g. "The New York Times") are filtered out.

Produces:
  <output_dir>/by-one-author/original-<Author_Name>.txt  — sentences per single author
  <output_dir>/sentences2author.json                      — sentence key -> author(s) + text
  <output_dir>/more_than_100.json                         — authors with >100 sentences
  <output_dir>/num_sentences_per_author.png               — histogram

Usage:
    python scripts/extract_sentences.py <nyt_json> --output-dir <dir>

Arguments:
    nyt_json        Path to raw NYT articles JSON (array of article objects with
                    lead_paragraph, byline, and section_name fields).

Options:
    --output-dir    Directory to write all outputs (default: analysis/sentences).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import stanza

from util import generate_key, load_json_data

EXCLUDE_AUTHORS = {
    'New York Times Games', ' York Times Audio', 'New York Times Audio',
    'The Learning Network', 'Florence Fabricant', 'The New York Times',
    'The New York Times Cooking', 'New York Times Opinion', 'ABC Australia',
    'The New York Times Magazine', 'News Nation', 'NBC News', 'Nbc news',
    'The New York Times Books Staff',
}

_DEFAULT_OUTPUT_DIR = os.path.join('analysis', 'sentences')


def tokenize_paragraph(paragraph, stz):
    doc = stz(paragraph)
    return [sentence.text for sentence in doc.sentences]


def is_single_authored(author):
    """Return True if byline represents a single, non-excluded author."""
    if not author:
        return False
    if ',' in author or ' and ' in author:
        return False
    if author[2:].strip() in EXCLUDE_AUTHORS:
        return False
    return True


def tokenize_articles(articles, stz):
    """Tokenize lead paragraphs; build per-author sentence lists and sentence->author map.

    Returns:
        per_author:      {author: [sentence, ...]}  sorted by sentence count descending
        sentence2authors: {sentence_key: {'authors': {author: count}, 'sentence': str}}
    """
    per_author = {}
    sentence2authors = {}
    for i, article in enumerate(articles):
        print(f"Tokenizing paragraph {i}...")
        author = article['byline']['original']
        if not article['lead_paragraph']:
            continue
        sentences = tokenize_paragraph(article['lead_paragraph'], stz)
        for sen in sentences:
            sen_key = generate_key(sen)
            entry = sentence2authors.setdefault(sen_key, {'authors': {}, 'sentence': sen})
            entry['authors'][author] = entry['authors'].get(author, 0) + 1
        per_author.setdefault(author, []).extend(sentences)
    per_author = dict(sorted(per_author.items(), key=lambda kv: len(kv[1]), reverse=True))
    return per_author, sentence2authors


def filter_single_authored(per_author):
    """Return only entries whose byline represents a single, non-excluded author."""
    return {a: s for a, s in per_author.items() if is_single_authored(a)}


def write_author_files(single_authored, authors_dir):
    """Write one sentence-per-line .txt file per author into authors_dir."""
    for author, sentences in single_authored.items():
        clean = author.replace(' ', '_').replace('&', 'and').replace('.', '')
        if len(clean) > 50:
            clean = 'TooManyAuthors'
        with open(os.path.join(authors_dir, f'original-{clean}.txt'), 'w') as f:
            for sentence in sentences:
                f.write(sentence + '\n')


def report_stats(single_authored):
    """Print sentence-count summary statistics to stdout."""
    vals = list(single_authored.values())
    n = len(vals)
    total = sum(len(s) for s in vals)
    print(f"Number of authors: {n}")
    print(f"Number of sentences: {total}")
    print(f"Maximum sentences per author: {len(vals[0])}")
    print(f"Mean sentences per author: {total / n:.1f}")
    print(f"Median sentences per author: {len(vals[n // 2])}")
    print(f"Authors >100 sentences: {sum(1 for s in vals if len(s) > 100)}")
    print(f"Authors  >50 sentences: {sum(1 for s in vals if len(s) > 50)}")
    print(f"Authors  >25 sentences: {sum(1 for s in vals if len(s) > 25)}")


def save_histogram(single_authored, output_dir):
    """Save a histogram of sentence counts per author as a PNG."""
    counts = [len(s) for s in single_authored.values()]
    plt.hist(counts, bins=np.arange(0, 100, 5))
    plt.xlabel('Number of sentences')
    plt.ylabel('Number of authors')
    plt.title('Number of sentences per author')
    plt.savefig(os.path.join(output_dir, 'num_sentences_per_author.png'))
    plt.close()


def save_outputs(sentence2authors, single_authored, output_dir):
    """Write sentence2author.json and more_than_100.json to output_dir."""
    with open(os.path.join(output_dir, 'sentences2author.json'), 'w', encoding='utf-8') as f:
        json.dump(sentence2authors, f, ensure_ascii=False)
    more_than_100 = [a for a, s in single_authored.items() if len(s) > 100]
    with open(os.path.join(output_dir, 'more_than_100.json'), 'w', encoding='utf-8') as f:
        json.dump(more_than_100, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('nyt_json', help='Path to raw NYT articles JSON.')
    ap.add_argument('--output-dir', default=_DEFAULT_OUTPUT_DIR,
                    help=f'Directory to write all outputs (default: {_DEFAULT_OUTPUT_DIR}).')
    args = ap.parse_args()

    authors_dir = os.path.join(args.output_dir, 'by-one-author')
    os.makedirs(authors_dir, exist_ok=True)

    stz = stanza.Pipeline('en', processors='tokenize')
    articles = load_json_data(args.nyt_json)

    per_author, sentence2authors = tokenize_articles(articles, stz)
    single_authored = filter_single_authored(per_author)
    write_author_files(single_authored, authors_dir)
    report_stats(single_authored)
    save_histogram(single_authored, args.output_dir)
    save_outputs(sentence2authors, single_authored, args.output_dir)
    print('done.')


if __name__ == '__main__':
    main()
