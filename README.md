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


## Publication

* Olga Zamaraeva, Dan Flickinger, Francis Bond, and Carlos Gómez-Rodríguez. 2025. *[Comparing LLM-generated and human-authored news text using formal syntactic theory](https://aclanthology.org/2025.acl-long.443/)*. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9041–9060, Vienna, Austria. Association for Computational Linguistics.

Reusing data from:

* Alberto Muñoz-Ortiz, Carlos Gómez-Rodríguez, and David Vilares. 2024. Contrasting linguistic patterns in human and LLM-generated news text. Artificial Intelligence Review, 57(10):265.
