#!/usr/bin/env python3
"""
diversity.py — legacy-format diversity analysis with pairwise JSD explain plots

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
- Baseline per-phenomenon scatter plots (Shannon/Simpson) with cohort-aware colors/markers.
- Pairwise per-type Jensen–Shannon divergence (JSD) **explain visuals**:
    * Butterfly bar chart with coverage-based Top-K contributors (direction = which model has higher p).
    * Cumulative contribution curve (rank vs fraction of total JSD).
    * Optional group-level butterfly via --group-map JSON (type -> group).
- Top-K selection is by **coverage** (e.g., 90% of total JSD), not by frequency.

Usage examples:
  python diversity.py results_2023.json results_2025.json --output-dir out --phenomena lextype lexentries
  python diversity.py results_2023.json results_2025.json --output-dir out \
      --explain NYT-2023-human NYT-2025-human --coverage 0.9 --max-top 60 --phenomena lextype
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict as dd, Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# --------------------------- Metrics ---------------------------

def calculate_shannon_diversity(names: List[str]) -> float:
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


def simpson_diversity_index(names: List[str]) -> float:
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
    m = re.search(r"(20\\d{2})", model)
    return m.group(1) if m else None

def is_human(model: str) -> bool:
    return "-human" in model.lower()

def human_subtype(model: str) -> str:
    low = model.lower()
    return "nyt" if "nyt" in low else "other"

def series_key(model: str) -> str:
    """Return one of: 'llm2023', 'llm2025', 'human_nyt', 'human_other'."""
    if is_human(model):
        return "human_nyt" if human_subtype(model) == "nyt" else "human_other"
    y = model_year(model) or ""
    if "2023" in y:
        return "llm2023"
    if "2025" in y:
        return "llm2025"
    # default to newer LLM color if no year is present
    return "llm2025"

SERIES_STYLE = {
    "llm2023":     dict(color=COLOR_LLM_2023, marker="o", size=38, label="LLMs (2023)"),
    "llm2025":     dict(color=COLOR_LLM_2025, marker="o", size=38, label="LLMs (2025)"),
    "human_nyt":   dict(color=COLOR_HUMAN_NYT, marker="*", size=60, label="Human (NYT)"),
    "human_other": dict(color=COLOR_HUMAN_OTH, marker="+", size=52, label="Human (WSJ/Wiki)"),
}


# --------------------------- Load & Merge ---------------------------

SUPPORTED_PHENOMENA = ["constr", "lexrule", "lextype", "lexentries"]

def load_and_merge(files: List[str], phenomena: List[str]):
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


# --------------------------- Helpers ---------------------------

def to_prob(counter: Counter, eps: float = 0.0) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    # No need to add eps to each cell before norm here; do smoothing after alignment
    return {k: (v / total) for k, v in counter.items()}

def align_probs(p: Dict[str, float], q: Dict[str, float], eps: float = 0.0) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return union vocabulary and aligned probability arrays; optional tiny smoothing then renorm."""
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_arr = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_arr = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    if eps > 0.0:
        p_arr = p_arr + eps
        q_arr = q_arr + eps
    # Renormalize to sum to 1 (avoid numerical drift)
    p_sum = p_arr.sum()
    q_sum = q_arr.sum()
    if p_sum > 0:
        p_arr = p_arr / p_sum
    if q_sum > 0:
        q_arr = q_arr / q_sum
    return keys, p_arr, q_arr

def jsd_contributions(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-dimension JSD contributions; handles zeros by the convention 0*log(0/·)=0."""
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        term_p = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0)
        term_q = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0)
    return 0.5 * (term_p + term_q)  # nonnegative, sum = JSD

def coverage_top(contribs: np.ndarray, coverage: float, max_top: int) -> int:
    """Return minimal K such that first K sorted contribs cover >= coverage of total, capped by max_top."""
    if contribs.size == 0:
        return 0
    order = np.argsort(contribs)[::-1]
    sorted_vals = contribs[order]
    total = sorted_vals.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(sorted_vals) / total
    K = int(np.searchsorted(cum, coverage) + 1)
    return int(min(K, len(sorted_vals), max_top))


# --------------------------- Plotting: core explain visuals ---------------------------

def butterfly_explain(keys: List[str], p: np.ndarray, q: np.ndarray, contribs: np.ndarray,
                      model_a: str, model_b: str, outpath: str,
                      coverage: float = 0.9, max_top: int = 60):
    """Mirrored bar chart of signed top contributors (coverage-based)."""
    delta = p - q
    total_jsd = contribs.sum()
    order = np.argsort(contribs)[::-1]
    K = coverage_top(contribs, coverage, max_top)
    top_idx = order[:K]

    left = [(keys[i], contribs[i]) for i in top_idx if delta[i] > 0]
    right = [(keys[i], contribs[i]) for i in top_idx if delta[i] <= 0]
    left.sort(key=lambda x: x[1])   # small to large for nice stacking on y
    right.sort(key=lambda x: x[1])

    labels = [k for k, _ in left] + [k for k, _ in right]
    y_pos = np.arange(len(labels))
    vals_left = [-v for _, v in left]  # negative for left side
    vals_right = [v for _, v in right]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(labels) + 1.5)))
    yl = np.arange(len(left))
    yr = np.arange(len(left), len(labels))
    if left:
        ax.barh(yl, vals_left, color="#3A6DB4", alpha=0.9, edgecolor='white', linewidth=0.5, label=model_a)
    if right:
        ax.barh(yr, vals_right, color="#D17F4A", alpha=0.9, edgecolor='white', linewidth=0.5, label=model_b)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Per-type JSD contribution")
    cov = contribs[order[:K]].sum()/total_jsd if total_jsd > 0 else 0.0
    ax.set_title(f"Top contributors by coverage — {model_a} vs {model_b}\n"
                 f"K={K} explain {cov:.1%} of JSD; tail={1-cov:.1%} across {len(contribs)-K} types")
    ax.axvline(0, color='gray', linewidth=1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def cumulative_curve(contribs: np.ndarray, outpath: str, coverage: float = 0.9, max_top: int = 60):
    """Cumulative fraction of JSD vs rank; marks K chosen by coverage."""
    order = np.argsort(contribs)[::-1]
    sorted_vals = contribs[order]
    total = sorted_vals.sum()
    if total <= 0:
        return
    cum = np.cumsum(sorted_vals) / total
    K = coverage_top(contribs, coverage, max_top)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(np.arange(1, len(sorted_vals) + 1), cum, linewidth=2.0)
    ax.axhline(coverage, color='gray', linestyle='--', linewidth=1.0)
    ax.axvline(K, color='gray', linestyle='--', linewidth=1.0)
    ax.set_xlabel("Ranked types")
    ax.set_ylabel("Cumulative fraction of JSD")
    ax.set_title(f"Cumulative JSD contributions\nK={K} reaches {coverage:.0%} coverage; total types={len(sorted_vals)}")
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def group_butterfly(keys: List[str], p: np.ndarray, q: np.ndarray, contribs: np.ndarray,
                    model_a: str, model_b: str, outpath: str, group_map: Dict[str, str],
                    coverage: float = 0.9, max_top: int = 60):
    """Butterfly for group contributions using provided mapping: type -> group."""
    groups = dd(lambda: {'abs': 0.0, 'signed': 0.0})
    for i, k in enumerate(keys):
        g = group_map.get(k, None)
        if g is None:
            continue
        c = contribs[i]
        if c <= 0:
            continue
        groups[g]['abs'] += c
        # signed direction by delta p
        sgn = 1.0 if (p[i] - q[i]) > 0 else -1.0
        groups[g]['signed'] += sgn * c

    if not groups:
        return

    # Prepare contributions sorted by absolute contribution
    items = sorted(groups.items(), key=lambda kv: kv[1]['abs'], reverse=True)
    abs_vals = np.array([kv[1]['abs'] for kv in items], dtype=float)
    K = coverage_top(abs_vals, coverage, max_top)
    top_items = items[:K]

    left = [(g, d['abs']) for g, d in top_items if d['signed'] > 0]
    right = [(g, d['abs']) for g, d in top_items if d['signed'] <= 0]
    left.sort(key=lambda x: x[1])
    right.sort(key=lambda x: x[1])

    labels = [g for g, _ in left] + [g for g, _ in right]
    y_pos = np.arange(len(labels))
    vals_left = [-v for _, v in left]
    vals_right = [v for _, v in right]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(labels) + 1.5)))
    yl = np.arange(len(left))
    yr = np.arange(len(left), len(labels))
    if left:
        ax.barh(yl, vals_left, color="#3A6DB4", alpha=0.9, edgecolor='white', linewidth=0.5, label=model_a)
    if right:
        ax.barh(yr, vals_right, color="#D17F4A", alpha=0.9, edgecolor='white', linewidth=0.5, label=model_b)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Group JSD contribution")
    covered = abs_vals[:K].sum() / abs_vals.sum() if abs_vals.sum() > 0 else 0.0
    ax.set_title(f"Group contributors by coverage — {model_a} vs {model_b}\n"
                 f"K={K} groups explain {covered:.1%} of grouped JSD; tail={1 - covered:.1%}")
    ax.axvline(0, color='gray', linewidth=1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


# --------------------------- Plotting: baseline scatter ---------------------------

def plot_scatter_for_phenomenon(phenom: str, model_counters: Dict[str, Counter], outdir: str):
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
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        y = np.arange(len(models))

        # draw points with cohort styling
        present_series = set()
        for i, (m, v) in enumerate(scores):
            sk = series_key(m)
            st = SERIES_STYLE[sk]
            present_series.add(sk)
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
        if handles:
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=8, title="Model Type", frameon=True)

        plt.tight_layout()
        outp = os.path.join(outdir, fname)
        plt.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close()

    scatter(sh_scores, "Shannon Index", f"diversity-{phenom}-shannon.png")
    scatter(si_scores, "Simpson Index", f"diversity-{phenom}-simpson.png")


# --------------------------- Explain pipeline ---------------------------

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)

def run_explain_for_phenom(phenom: str, model_counters: Dict[str, Counter], outdir: str,
                           model_a: str, model_b: str, coverage: float, max_top: int,
                           group_map: Dict[str, str] = None):
    """Compute per-type JSD contributions for A vs B and plot visuals."""
    if model_a not in model_counters or model_b not in model_counters:
        return

    p_dict = to_prob(model_counters[model_a])
    q_dict = to_prob(model_counters[model_b])
    keys, p, q = align_probs(p_dict, q_dict, eps=1e-12)  # tiny smoothing
    contribs = jsd_contributions(p, q)

    base = f"explain-jsd-{sanitize(phenom)}-{sanitize(model_a)}--vs--{sanitize(model_b)}"
    butterfly_explain(keys, p, q, contribs, model_a, model_b,
                      os.path.join(outdir, base + "-butterfly.png"),
                      coverage=coverage, max_top=max_top)
    cumulative_curve(contribs, os.path.join(outdir, base + "-cumulative.png"),
                     coverage=coverage, max_top=max_top)

    if group_map:
        group_butterfly(keys, p, q, contribs, model_a, model_b,
                        os.path.join(outdir, base + "-group-butterfly.png"),
                        group_map=group_map, coverage=coverage, max_top=max_top)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Diversity analysis (legacy format) with pairwise JSD explain plots.")
    ap.add_argument("json_files", nargs="+", help="Input JSON files in the OLD format.")
    ap.add_argument("--output-dir", default="out", help="Directory to write outputs.")
    ap.add_argument("--phenomena", nargs="+",
                    choices=SUPPORTED_PHENOMENA,
                    default=["constr", "lextype", "lexrule", "lexentries"],
                    help="Which phenomena to analyze.")
    # Pairwise explain options
    ap.add_argument("--explain", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                    help="Produce pairwise JSD explain plots for the two named models.")
    ap.add_argument("--coverage", type=float, default=0.9,
                    help="Coverage target (fraction of total JSD) for Top-K selection (default 0.9).")
    ap.add_argument("--max-top", type=int, default=60,
                    help="Upper cap on Top-K items/groups (default 60).")
    ap.add_argument("--group-map", type=str,
                    help="Optional JSON file mapping type -> group (for group-level butterfly).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data, seen_any = load_and_merge(args.json_files, args.phenomena)
    if not seen_any:
        print("No recognized phenomena found in the provided files.")
        return
    if not data:
        print("No non-empty phenomena after merging (all were empty).")
        return

    # Baseline scatter plots
    for phenom, model_counters in data.items():
        if not model_counters:
            continue
        plot_scatter_for_phenomenon(phenom, model_counters, args.output_dir)

    # Pairwise explain
    if args.explain:
        model_a, model_b = args.explain[0], args.explain[1]
        group_map = None
        if args.group_map:
            with open(args.group_map, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                group_map = loaded
            else:
                print("Warning: --group-map JSON must be a dict of {type: group}; ignoring.")
        for phenom, model_counters in data.items():
            run_explain_for_phenom(phenom, model_counters, args.output_dir,
                                   model_a, model_b, coverage=args.coverage,
                                   max_top=args.max_top, group_map=group_map)


if __name__ == "__main__":
    main()
