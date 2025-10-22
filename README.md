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

  * `"new-original"` — NYT-2025
  * `"original"` — Original NYT
  * `"llama_07"` — LLaMA 7B
  * `"llama_13"` — LLaMA 13B
  * `"llama_30"` — LLaMA 30B
  * `"llama_65"` — LLaMA 65B
  * `"mistral_07"` — Mistral 7B
  * `"falcon_07"` — Falcon 7B
  * `"wsj"` — Wall Street Journal
  * `"wikipedia"` — Wikipedia

* **type** — specific linguistic type (examples):

  * `and_or_conj` — the `/'` in *SF/SPCA* (lextype)
  * `n_pl-irreg_olr` — irregular plural, e.g., *child → children* (lexrule)
  * `sb-hd_mc_c` — subject linked to a main clause, e.g., *They arrived* (constr)

* **count** — frequency of each type in the parse. Counts are computed after parsing with the 2025 release of the [English Resource Grammar (ERG)](https://github.com/delph-in/erg/releases/tag/2025).

### Linguistic Diversity Analysis

This script analyzes linguistic diversity in language model outputs using Shannon and Simpson diversity indices.

#### Usage

```bash
python diversity.py [JSON_FILES...] [OPTIONS]
```

#### Arguments

- `JSON_FILES`: One or more JSON files containing linguistic data in the expected format
- `--phenomena`: Phenomena to analyze (choices: `lexrule`, `lextype`, `constr`; default: all three)
- `--num-bootstrap`: Number of permutation test iterations (default: 10000)
- `--output-dir`: Directory for output files (default: `out`)
- `--explain --coverage --max-top`: Calculate which types contribute the most to the difference in Jessen-Shannon Divergence and produce the butterfly plot for the max-top types.
- `--explain-group "Model1,Model2..." "Model3,Model4..."`: Same as `--explain`, but for groups of models; treats all models equally regardless of the size of the corresponding dataset
- `--split-punct`: Produce separate analyses with and without types related directly to punctuation

#### Examples

Current advised settings:
```
uv run scripts/diversity.py --num-bootstrap 1 analysis/frequencies-json/frequencies-2025.json analysis/frequencies-json/frequencies-2023-repro.json 
```

Basic usage with default settings (using uv for dependencies):
```bash
uv run scripts/diversity.py analysis/frequencies-json/*.json 
OR
python scripts/diversity.py analysis/frequencies-json/*.json 
```

Analyze only constructions and lexical rules, save to custom directory:
```bash
python diversity.py data1.json --phenomena constr lexrule --output-dir results
```

Run with fewer bootstrap iterations for faster execution:
```bash
python diversity.py data1.json --num-bootstrap 1000
```

#### Output Files

The script generates the following files in the output directory:

- `erg-llm-{phenomenon}-{index}.md`: Markdown tables with diversity scores
- `llm-erg-{phenomenon}-{index}.png`: Tufte-style plots showing diversity comparisons

Where `{phenomenon}` is one of `constructions`, `lexical-rules`, or `lexical-types`, and `{index}` is either `shannon` or `simpson`.

#### Dependencies

- numpy
- matplotlib
- json (built-in)
- collections (built-in)

## Publication

* Olga Zamaraeva, Dan Flickinger, Francis Bond, and Carlos Gómez-Rodríguez. 2025. *[Comparing LLM-generated and human-authored news text using formal syntactic theory](https://aclanthology.org/2025.acl-long.443/)*. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9041–9060, Vienna, Austria. Association for Computational Linguistics.

Reusing data from:

* Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, and David Vilares. 2024. Contrasting linguistic patterns in human and LLM-generated news text. Artificial Intelligence Review, 57(10):265.
