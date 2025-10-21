#!/usr/bin/env python3
"""
diversity.py — legacy-format diversity analysis with 'lexentries' support

Input format (OLD):
{
  "constr":    { "<model>": { "<type>": count, ... }, ... },
  "lexrule":   { "<model>": { "<type>": count, ... }, ... },
  "lextype":   { "<model>": { "<type>": count, ... }, ... },
  "lexentries":{ "<model>": { "<type>": count, ... }, ... }  # NEW
}

Features
- Accepts multiple JSON files; merges counts by phenomenon/model/type.
- Phenomena supported: constr, lexrule, lextype, lexentries.
- Computes Shannon (nats) and Simpson diversity per model.
- Per-phenomenon scatter plots with cohort styling:
    * LLMs (2023): blue circle
    * LLMs (2025): teal circle
    * Human (NYT): chestnut star *
    * Human (WSJ/Wiki/other): copper '+'
- Legend matches plotted markers/colors and only shows series that appear.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict as dd, Counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# --------------------------- Metrics ---------------------------

def calculate_shannon_diversity(names):
    """Shannon diversity (natural log)."""
    if not names:
        return float('nan')
    counts = Counter(names)
    N = sum(counts.values())
    if N == 0:
        return float('nan')
    H = 0.0
    for c in counts.values():
        p = c / N
        if p > 0:
            H -= p * math.log(p)
    return H


def simpson_diversity_index(names):
    """Simpson diversity (1 - sum p_i^2)."""
    if not names or len(names) <= 1:
        return float('nan')
    counts = Counter(names)
    N = sum(counts.values())
    if N <= 1:
        return float('nan')
    num = sum(c*c for c in counts.values())
    return 1 - (num / (N*N))


# --------------------------- Cohort styling ---------------------------

COLOR_LLM_2023   = "#3A6DB4"  # cobalt blue
COLOR_LLM_2025   = "#4AA6B4"  # teal-blue
COLOR_HUMAN_NYT  = "#B44E3A"  # chestnut brown
COLOR_HUMAN_OTH  = "#D17F4A"  # copper

def model_year(model: str):
    m = re.search(r"(20\d{2})", model)
    return m.group(1) if m else None

def is_human(model: str) -> bool:
    return "-human" in model.lower()

def human_subtype(model: str) -> str:
    low = model.lower()
    if "nyt" in low:
        return "nyt"
    return "other"

def series_key(model: str) -> str:
    """Return one of: 'llm2023', 'llm2025', 'human_nyt', 'human_other'."""
    if is_human(model):
        return "human_nyt" if human_subtype(model) == "nyt" else "human_other"
    y = model_year(model) or ""
    if "2023" in y:
        return "llm2023"
    if "2025" in y:
        return "llm2025"
    # default: treat as newer LLMs
    return "llm2025"

SERIES_STYLE = {
    "llm2023":     dict(color=COLOR_LLM_2023, marker="o", size=38, label="LLMs (2023)"),
    "llm2025":     dict(color=COLOR_LLM_2025, marker="o", size=38, label="LLMs (2025)"),
    "human_nyt":   dict(color=COLOR_HUMAN_NYT, marker="*", size=60, label="Human (NYT)"),
    "human_other": dict(color=COLOR_HUMAN_OTH, marker="+", size=52, label="Human (WSJ/Wiki)"),
}


# --------------------------- Load & Merge ---------------------------

SUPPORTED_PHENOMENA = ["constr", "lexrule", "lextype", "lexentries"]

def load_and_merge(files, phenomena):
    """
    Return: dict[phenomenon][model] -> Counter(type -> count)
    Only keeps requested phenomena that exist in any file.
    """
    data = {p: {} for p in SUPPORTED_PHENOMENA}
    seen_any = False

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for p in phenomena:
            if p not in obj or not isinstance(obj[p], dict):
                continue
            seen_any = True
            for model, type_counts in obj[p].items():
                if not isinstance(type_counts, dict):
                    continue
                tgt = data[p].setdefault(model, Counter())
                for tname, c in type_counts.items():
                    try:
                        c = int(c)
                    except Exception:
                        continue
                    if c > 0:
                        tgt[tname] += c

    # drop empty phenomena
    data = {p: md for p, md in data.items() if md}
    return data, seen_any


# --------------------------- Plotting ---------------------------

def plot_scatter_for_phenomenon(phenom, model_counters, outdir):
    """
    model_counters: dict[model] -> Counter(type -> count)
    Writes: PNGs and MD tables for Shannon & Simpson.
    """
    # Expand to repeated list once per model
    model_names_list = {}
    for model, ctr in model_counters.items():
        expanded = []
        for t, c in ctr.items():
            if c > 0:
                expanded.extend([t]*c)
        if expanded:
            model_names_list[model] = expanded

    # Compute indices
    sh_scores = []
    si_scores = []
    for model, names in model_names_list.items():
        if len(names) <= 1:
            continue
        sh = calculate_shannon_diversity(names)
        si = simpson_diversity_index(names)
        sh_scores.append((model, sh))
        si_scores.append((model, si))

    # Sort by score for stable Y order
    sh_scores.sort(key=lambda x: (x[1], x[0]))
    si_scores.sort(key=lambda x: (x[1], x[0]))

    # Emit tables
    def save_md(scores, stem):
        if not scores: return
        p = os.path.join(outdir, f"{stem}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("| Model | Diversity |\n| --- | --- |\n")
            for m, v in scores:
                f.write(f"| {m} | {v:.3f} |\n")

    os.makedirs(outdir, exist_ok=True)
    save_md(sh_scores, f"diversity-{phenom}-shannon")
    save_md(si_scores, f"diversity-{phenom}-simpson")

    # Common plotting helper
    def scatter(scores, idx_label, fname):
        if not scores: return
        models = [m for m, _ in scores]
        vals   = [v for _, v in scores]
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        y = np.arange(len(models))

        # draw points with cohort styling
        present_series = set()
        for i, (m, v) in enumerate(scores):
            sk = series_key(m)
            st = SERIES_STYLE[sk]
            present_series.add(sk)
            if st["marker"] == "+":
                # plus is edge-only for readability over grid
                ax.scatter(v, i, s=st["size"], marker=st["marker"], c=st["color"])
            else:
                ax.scatter(v, i, s=st["size"], marker=st["marker"], c=st["color"])

        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel(idx_label, fontsize=10)
        ax.set_title(f"Diversity — {phenom} ({idx_label})", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # legend that matches plotted markers/colors; only present cohorts
        handles = []
        for sk in ["llm2023","llm2025","human_nyt","human_other"]:
            if sk not in present_series:
                continue
            st = SERIES_STYLE[sk]
            handles.append(Line2D([0],[0], marker=st["marker"], linestyle="None",
                                  markerfacecolor=st["color"] if st["marker"] != "+" else "none",
                                  markeredgecolor=st["color"], markeredgewidth=1.2,
                                  color=st["color"], label=st["label"], markersize=8))
        # Place outside right to avoid occluding points
        if handles:
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=8, title="Model Type", frameon=True)

        plt.tight_layout()
        outp = os.path.join(outdir, fname)
        plt.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close()

    scatter(sh_scores, "Shannon Index", f"diversity-{phenom}-shannon.png")
    scatter(si_scores, "Simpson Index", f"diversity-{phenom}-simpson.png")


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Diversity analysis for legacy-format JSON (with 'lexentries').")
    ap.add_argument("json_files", nargs="+", help="Input JSON files in the OLD format.")
    ap.add_argument("--output-dir", default="out", help="Directory to write outputs.")
    ap.add_argument("--phenomena", nargs="+",
                    choices=SUPPORTED_PHENOMENA,
                    default=["constr", "lextype", "lexrule", "lexentries"],
                    help="Which phenomena to analyze.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data, seen_any = load_and_merge(args.json_files, args.phenomena)
    if not seen_any:
        print("No recognized phenomena found in the provided files.")
        return
    if not data:
        print("No non-empty phenomena after merging (all were empty).")
        return

    for phenom, model_counters in data.items():
        if not model_counters:
            continue
        plot_scatter_for_phenomenon(phenom, model_counters, args.output_dir)


if __name__ == "__main__":
    main()
