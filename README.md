# llm-syntax

Scripts and data to investigate and illustrate the differences in the distributions of syntactic and lexical types (using HPSG), revealing systematic distinctions between human and LLM-generated writing. 


## Data

The parsed results are provided as JSON files in `analysis/frequencies-json`.

Each file contains a nested dictionary with the structure:

```
'phenomenon' : 'model' : 'type' : 'count'
```

### Keys

* **phenomenon** — category of grammatical phenomenon:

  * `lexrule` — lexical rules (convert a word to a lexeme)
  * `lextype` — lexical types (fine-grained part of speech of a word)
  * `constr` — constructions (form–meaning pairings that combine one or more constituents)
  * `lexentries` - lexical entries (similar to lemmas)

* **model** — text source or model used:

  * `"NYT-2023-human"` — NYT 2023 (human-authored)
  * `"NYT-2025-human"` — NYT 2025 (human-authored)
  * `"WSJ-1987-human"` — Wall Street Journal
  * `"Wikipedia-2008-human"` — Wikipedia
  * `"Llama7B-2023-llm"` — LLaMA 7B (2023)
  * `"Llama13B-2023-llm"` — LLaMA 13B (2023)
  * `"Llama30B-2023-llm"` — LLaMA 30B (2023)
  * `"Llama65B-2023-llm"` — LLaMA 65B (2023)
  * `"Mistral7B-2023-llm"` — Mistral 7B (2023)
  * `"Falcon7B-2023-llm"` — Falcon 7B (2023)
  * `"Llama70B-2025-llm"` — LLaMA 70B (2025)
  * `"Mistral7B_i-2025-llm"` — Mistral 7B Instruct (2025)
  * `"GPT4o-2025-llm"` — GPT-4o (2025)
  * `"Qwen14B-2025-llm"` — Qwen 14B (2025)
  * `"Qwen32B-2025-llm"` — Qwen 32B (2025)
  * `"Qwen72B-2025-llm"` — Qwen 72B (2025)
  * `"TinyLlama-2025-llm"` — TinyLlama (2025)

* **type** — specific linguistic type (examples):

  * `and_or_conj` — the `/'` in *SF/SPCA* (lextype)
  * `n_pl-irreg_olr` — irregular plural, e.g., *child → children* (lexrule)
  * `sb-hd_mc_c` — subject linked to a main clause, e.g., *They arrived* (constr)

* **count** — frequency of each type in the parse. Counts are computed after parsing with the 2025 release of the [English Resource Grammar (ERG)](https://github.com/delph-in/erg/releases/tag/2025).

### Linguistic Diversity Analysis

This script analyzes linguistic diversity in language model outputs using Shannon and Simpson diversity indices.

#### Usage

```bash
python scripts/diversity.py [JSON_FILES...] [OPTIONS]
```

#### Arguments

- `JSON_FILES`: One or more JSON files containing linguistic data in the expected format
- `--phenomena`: Phenomena to analyze (choices: `lexrule`, `lextype`, `constr`, `lexentries`; default: all four)
- `--output-dir`: Directory for output files (default: `out`)
- `--split-punct`: Also produce separate analyses with and without punctuation-related types
- `--explain MODEL_A MODEL_B`: Pairwise JSD explain butterfly plot for two models
- `--group-explain "Model1,Model2,..." "Model3,Model4,..."`: Same as `--explain` but for groups of models
- `--model-registry PATH`: JSON file mapping model names to short integer IDs used in output filenames (default: `analysis/model-ids.json`)
- `--coverage`: Coverage target for Top-K type selection (default: 0.9)
- `--max-top`: Upper cap on number of types shown in explain plots (default: 60)
- `--learning N`: Produce learning curves with N bins per phenomenon

#### Examples

Diversity scatter plots only:
```bash
python scripts/diversity.py analysis/frequencies-json/frequencies-2023.json --output-dir analysis/diversity-repro
```

Pairwise JSD explain plot for two models:
```bash
python scripts/diversity.py analysis/frequencies-json/frequencies-2023.json --explain NYT-2023-human Llama7B-2023-llm --output-dir analysis/diversity-repro
```

Group JSD explain plot (humans vs 2023 LLMs):
```bash
python scripts/diversity.py analysis/frequencies-json/frequencies-2023.json --output-dir analysis/diversity-repro --group-explain "NYT-2023-human,WSJ-1987-human,Wikipedia-2008-human" "Falcon7B-2023-llm,Llama65B-2023-llm,Llama30B-2023-llm,Mistral7B-2023-llm,Llama7B-2023-llm,Llama13B-2023-llm"
```

> **Note:** On headless machines (no display), prefix with `MPLBACKEND=Agg` to avoid a segfault.

#### Output Files

Outputs are written to three subdirectories of `--output-dir`, with a `README.md` explaining the layout:

- `plots/` — scatter PNGs, butterfly plots, cumulative JSD curves
- `mds/` — markdown tables with per-model diversity scores
- `json/` — top-K contributors JSON (includes `"groups"` field with model membership)

Filename pattern for explain outputs: `{phenom}-{gA}--vs--{gB}-{kind}.{ext}`

where `{gA}` and `{gB}` are group tags of the form `g1.3.4` (dot-separated IDs from `analysis/model-ids.json`). `{phenom}` is one of `constr`, `lextype`, `lexrule`, `lexentries`; `{kind}` is `butterfly`, `cumulative`, or `top-contributors`.

#### Dependencies

- numpy
- matplotlib

### Frequency Comparison Plots

`visualize_frequencies.py` generates bar+scatter plots comparing each LLM model's
top-N construction, lexrule, and lextype frequencies against the three human baselines
(NYT, Wikipedia, WSJ), normalized by sentence count.

#### Usage

```bash
python scripts/visualize_frequencies.py <frequencies_json>
```

Run from the repo root. Model names in the JSON must match those defined in `dataset_sizes`
in `scripts/construction_frequencies.py`.

#### Example

```bash
python scripts/visualize_frequencies.py analysis/frequencies-json/frequencies-models-wiki-wsj.json
```

> **Note:** On headless machines (no display), prefix with `MPLBACKEND=Agg`:
> ```bash
> MPLBACKEND=Agg python scripts/visualize_frequencies.py analysis/frequencies-json/frequencies-models-wiki-wsj.json
> ```

#### Output Files

PNG files written to `analysis/plots/frequencies/0-50/`. Two sets are produced:

**Per-model** (one bar per LLM, normalized by sentence count):
```
Top frequencies-llama_30-original-wikipedia-wsj-constr.png
Bottom frequencies-llama_30-original-wikipedia-wsj-constr.png
...
```

**Combined** (all LLMs aggregated as `llm`, normalized by construction count):
```
Top frequencies-llm-original-wikipedia-wsj-constr.png
Bottom frequencies-llm-original-wikipedia-wsj-constr.png
...
```

#### Dependencies

- numpy, pandas, matplotlib

### PCA Plots

`pca.py` performs PCA on pairwise cosine-distance matrices and produces scatter
plots showing how models cluster by syntactic and lexical type distributions.

#### Usage

```bash
python scripts/pca.py [input_dir] [output_dir]
```

Run from the repo root.

#### Example

```bash
python scripts/pca.py \
  analysis/cosine-pairs/models/norm-by-constr-count \
  analysis/plots
```

> **Note:** On headless machines (no display), prefix with `MPLBACKEND=Agg`:
> ```bash
> MPLBACKEND=Agg python scripts/pca.py
> ```

#### Output Files

Three PNG files written to `output_dir` (default: `analysis/plots`):

```
pca_syntax.png   — PCA of syntactic construction distances
pca_lextype.png  — PCA of lexical type distances
pca_lexrule.png  — PCA of lexical rule distances
```

#### Dependencies

- numpy, pandas, matplotlib, scikit-learn

### Author Analysis Preprocessing

`extract_sentences.py` tokenizes NYT article paragraphs into sentences using [Stanza](https://stanfordnlp.github.io/stanza/), filters to single-authored articles, and writes per-author sentence files. This is the first step of the author-level diversity analysis pipeline.

**Input:** Raw NYT articles JSON — an array of article objects with `lead_paragraph`, `byline`, and `section_name` fields. Not included in the repo due to licensing restrictions.

#### Usage

```bash
python scripts/extract_sentences.py <nyt_json> --output-dir analysis/sentences
```

#### Output Files

| File | Description |
| ---- | ----------- |
| `by-one-author/original-<Author>.txt` | One file per single author with their sentences |
| `sentences2author.json` | Maps each sentence (by key) to its author(s) and text |
| `more_than_100.json` | Authors with more than 100 sentences |
| `num_sentences_per_author.png` | Histogram of sentence counts per author |

Multi-authored articles (bylines with `,` or ` and `) and a fixed exclusion list of institutional bylines (e.g. "The New York Times") are filtered out.

#### Dependencies

- stanza, matplotlib, numpy

## Publications

* Olga Zamaraeva, Dan Flickinger, Francis Bond, and Carlos Gómez-Rodríguez. 2025. *[Comparing LLM-generated and human-authored news text using formal syntactic theory](https://aclanthology.org/2025.acl-long.443/)*. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9041–9060, Vienna, Austria. Association for Computational Linguistics.

* Adrián Gude, Roi Santos-Rios, Francis Bond, Dan Flickinger, Carlos Gómez-Rodríguez, and Olga Zamaraeva. To appear. *[More aligned, less diverse? Comparing the grammar and lexicon of two generations of LLMs](https://arxiv.org/abs/2605.06030)*. To appear in *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*, San Diego, CA, USA. Association for Computational Linguistics.

Reusing data from:

* Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, and David Vilares. 2024. Contrasting linguistic patterns in human and LLM-generated news text. Artificial Intelligence Review, 57(10):265.
