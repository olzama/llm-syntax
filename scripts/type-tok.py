import json
from collections import defaultdict

# Load the JSON data
with open("analysis/frequencies-json/frequencies-models-wiki-wsj.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# List of construction categories to check
construction_categories = ["constr", "lexrule", "lextype"]

# Store results: {category: {model: {'types': ..., 'tokens': ..., 'ttr': ...}}}
results = defaultdict(dict)

for category in construction_categories:
    if category in data:
        for model, items in data[category].items():
            token_count = sum(items.values())
            type_count = sum(1 for value in items.values() if value > 0)
            ttr = type_count / token_count if token_count > 0 else 0
            results[category][model] = {
                "types": type_count,
                "tokens": token_count,
                "ttr": ttr
            }

# Display the results
for category, model_stats in results.items():
    print(f"\nCategory: {category}")
    for model, stats in model_stats.items():
        print(f"  {model}: Types = {stats['types']}, Tokens = {stats['tokens']}, TTR = {stats['ttr']:.5f}")