Models
  * `nyt_2025_hum` — NYT-2025
	* NYT times articles from 2025-XX to 2025 YY
  * `nyt_2023_hum` — Original NYT 
	* NYT times articles from 2025-XX to 2025 YY
  * `llama_07` — LLaMA 7B **FIXME**
	* URL: XYZ
  * `llama_13` — LLaMA 13B **FIXME**
  * `llama_30` — LLaMA 30B **FIXME**
  * `llama_65` — LLaMA 65B **FIXME**
  * `mistral_07` — Mistral 7B **FIXME**
  * `falcon_07` — Falcon 7B **FIXME**
  * `wsj_1988_hum` — Wall Street Journal
	* Wall Street Journal from January 1987 to June 1989
	* https://github.com/delph-in/erg/tree/main/tsdb/skeletons/wsj
  * `wikipedia_2008_hum` — Wikipedia
	* Wikipedia dump of 2008 
	* https://github.com/delph-in/erg/blob/main/tsdb/skeletons/wescience/LICENSE
  *  **FIXME** add 2025 models

## Model names are of the form
 * `src_year_type`
    * `src` is of the form
      * `model-version-size-tuning`.  E.g. `llama-v2-13-base` or `mistral-v3-07-inst`
	  * or just `nyt, wsj, wikipedia` for the text written by humans
	* `year` is the year the text was written/generated, e.g. 2008, 2023, …
    * `type` is either `llm` or `hum` 
