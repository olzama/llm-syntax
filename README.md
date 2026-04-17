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
- `--coverage`: Coverage target for Top-K type selection (default: 0.9)
- `--max-top`: Upper cap on number of types shown in explain plots (default: 60)
- `--learning N`: Produce learning curves with N bins per phenomenon

#### Examples

Generate diversity plots for constructions and lexical types (recommended):
```bash
python scripts/diversity.py \
  analysis/frequencies-json/frequencies-2023.json \
  analysis/frequencies-json/frequencies-2025.json \
  --output-dir analysis/plots/plots-diversity \
  --phenomena constr lextype \
  --split-punct
```

> **Note:** On headless machines (no display), prefix with `MPLBACKEND=Agg` to avoid a segfault:
> ```bash
> MPLBACKEND=Agg python scripts/diversity.py ...
> ```

All phenomena, custom output directory:
```bash
python scripts/diversity.py analysis/frequencies-json/*.json --output-dir results
```

Pairwise JSD explain plot for two models:
```bash
python scripts/diversity.py analysis/frequencies-json/*.json \
  --explain NYT-2023-human Llama7B-2023-llm \
  --coverage 0.9 --max-top 40
```

#### Output Files

Outputs are written to two subdirectories of `--output-dir`:

- `png/diversity-{phenom}{suffix}-{index}.png` — scatter plots of diversity scores
- `md/diversity-{phenom}{suffix}-{index}.md` — markdown tables with per-model diversity scores

Where `{phenom}` is one of `constr`, `lextype`, `lexrule`, `lexentries`; `{suffix}` is empty, `_punct`, or `_xpunct` (when using `--split-punct`); and `{index}` is `shannon` or `simpson`.

#### Dependencies

- numpy
- matplotlib

## Publication

* Olga Zamaraeva, Dan Flickinger, Francis Bond, and Carlos Gómez-Rodríguez. 2025. *[Comparing LLM-generated and human-authored news text using formal syntactic theory](https://aclanthology.org/2025.acl-long.443/)*. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9041–9060, Vienna, Austria. Association for Computational Linguistics.

Reusing data from:

* Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, and David Vilares. 2024. Contrasting linguistic patterns in human and LLM-generated news text. Artificial Intelligence Review, 57(10):265.
