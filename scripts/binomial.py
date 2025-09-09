# ------------------------------------------------------------------------
# 0.  Imports  (same as before)  -----------------------------------------
# ------------------------------------------------------------------------
import pandas as pd
import bambi as bmb
import arviz as az
import json

# ------------------------------------------------------------------------
# 1.  Which models are machines?
# ------------------------------------------------------------------------
machines = {
    "falcon_07", "llama_07", "llama_13",
    "llama_30", "llama_65", "mistral_07"
}
def grp_label(model_id: str) -> str:
    """Return 'Machine' or 'Human'."""
    return "Machine" if model_id in machines else "Human"

# ------------------------------------------------------------------------
# 2.  Build a tidy long table *for one category* -------------------------
# ------------------------------------------------------------------------
def make_long_df(one_category_dict: dict, *, category_name: str) -> pd.DataFrame:
    rows = []
    for model_id, constr_dict in one_category_dict.items():
        total = sum(constr_dict.values())                     # n_i
        for construction, count in constr_dict.items():
            rows.append({
                "category"     : category_name,
                "model"        : model_id,
                "grp"          : grp_label(model_id),         # <<–– renamed here
                "construction" : construction,
                "count"        : count,
                "n_tokens"     : total
            })
    df_long = pd.DataFrame(rows)

    # make sure the grouping factor is categorical
    df_long["construction"] = df_long["construction"].astype("category")
    df_long["grp"]          = df_long["grp"].astype("category")

    return df_long

# ------------------------------------------------------------------------
# 3.  Fit the hierarchical binomial GLMM for ONE category ---------------
# ------------------------------------------------------------------------
def fit_binom_glmm(df_long: pd.DataFrame):
    # renamed 'group' → 'grp' on the RHS and inside the random-effect term
    formula = "count|trials(n_tokens) ~ grp + (1 + grp | construction)"
    model   = bmb.Model(formula, data=df_long, family="binomial")
    idata   = model.fit(draws=2000, tune=2000, target_accept=0.9)
    return idata

# ------------------------------------------------------------------------
# 4.  Extract construction-specific contrasts ---------------------------
#    (unchanged except for the new variable name)
# ------------------------------------------------------------------------
def extract_construction_effects(idata: az.InferenceData) -> pd.DataFrame:
    beta_machine = idata.posterior["grp[Machine]"]
    rand_slopes  = idata.posterior["1|construction_grp[Machine]"]
    constr_names = rand_slopes.coords["construction"].values

    records = []
    for i, constr in enumerate(constr_names):
        delta = beta_machine + rand_slopes[..., i]
        mean  = delta.mean().item()
        l95, u95 = az.hdi(delta, hdi_prob=0.95).to_array().values
        records.append({
            "construction": constr,
            "log_odds_diff": mean,
            "l95": l95,
            "u95": u95,
            "sig_95": (l95 > 0) | (u95 < 0)
        })
    return pd.DataFrame(records).sort_values("log_odds_diff", ascending=False)

# ------------------------------------------------------------------------
# 5.  MAIN LOOP  ––  work on each top-level category separately ----------
# ------------------------------------------------------------------------
all_summaries = {}

import sys, formulae, pymc
print("python  :", sys.version)
print("bambi   :", bmb.__version__)
print("formulae:", formulae.__version__)
print("pymc    :", pymc.__version__)

with open('analysis/frequencies-json/frequencies-models-wiki-wsj.json', 'r') as f:
    model_frequencies = json.load(f)

for category_name, per_model_dict in model_frequencies.items():
    print(f"\n=== Category: {category_name} ===")

    # 5·1  Build tidy DF for *this* category only
    df_cat = make_long_df(per_model_dict, category_name=category_name)
    print(f"   → data shape: {df_cat.shape}")

    # 5·2  Fit the GLMM
    idata_cat = fit_binom_glmm(df_cat)

    # 5·3  Summarise per-construction contrasts
    summary_cat = extract_construction_effects(idata_cat)

    # 5·4  Store for later inspection (or write to CSV)
    all_summaries[category_name] = summary_cat
    summary_cat.to_csv(f"{category_name}_machine_vs_human.csv", index=False)

    # quick peek
    print(summary_cat.head(10)[["construction", "log_odds_diff", "l95", "u95", "sig_95"]])
