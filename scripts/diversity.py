#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Diversity analysis (legacy format) with pairwise & group JSD explain plots."""

import argparse, json, math, os, re, random
from collections import defaultdict as dd, Counter
from typing import Dict, List, Tuple, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# --------------------------- Metrics ---------------------------
def calculate_shannon_diversity(names: List[str]) -> float:
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
    if not names or len(names) <= 1:
        return float('nan')
    counts = Counter(names)
    N = sum(counts.values())
    if N <= 1:
        return float('nan')
    num = sum(c*c for c in counts.values())
    return 1 - (num / (N*N))

# --------------------------- Cohort styling ---------------------------
COLOR_LLM_2023   = "#3A6DB4"
COLOR_LLM_2025   = "#4AA6B4"
COLOR_HUMAN_NYT  = "#B44E3A"
COLOR_HUMAN_OTH  = "#D17F4A"

def model_year(model: str):
    m = re.search(r"(20\d{2})", model)
    return m.group(1) if m else None

def is_human(model: str) -> bool:
    return "-human" in model.lower()

def human_subtype(model: str) -> str:
    return "nyt" if "nyt" in model.lower() else "other"

def series_key(model: str) -> str:
    if is_human(model):
        return "human_nyt" if human_subtype(model) == "nyt" else "human_other"
    y = model_year(model) or ""
    if "2023" in y:
        return "llm2023"
    if "2025" in y:
        return "llm2025"
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
    data = {p: md for p, md in data.items() if md}
    return data, seen_any

# --------------------------- Helpers ---------------------------
def to_prob(counter: Counter, eps: float = 0.0) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: (v / total) for k, v in counter.items()}

def align_probs(p: Dict[str, float], q: Dict[str, float], eps: float = 0.0) -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_arr = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_arr = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    if eps > 0.0:
        p_arr = p_arr + eps
        q_arr = q_arr + eps
    p_sum = p_arr.sum()
    q_sum = q_arr.sum()
    if p_sum > 0:
        p_arr = p_arr / p_sum
    if q_sum > 0:
        q_arr = q_arr / q_sum
    return keys, p_arr, q_arr

def jsd_contributions(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        term_p = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0)
        term_q = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0)
    return 0.5 * (term_p + term_q)

def coverage_top(contribs: np.ndarray, coverage: float, max_top: int) -> int:
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

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)

# --------------------------- Plotting: core explain visuals ---------------------------
def butterfly_explain(keys, p, q, contribs, model_a, model_b, outpath, coverage=0.9, max_top=60):
    delta = p - q
    total_jsd = contribs.sum()
    order = np.argsort(contribs)[::-1]
    K = coverage_top(contribs, coverage, max_top)
    top_idx = order[:K]

    left = [(keys[i], contribs[i]) for i in top_idx if delta[i] > 0]
    right = [(keys[i], contribs[i]) for i in top_idx if delta[i] <= 0]
    left.sort(key=lambda x: x[1])
    right.sort(key=lambda x: x[1])

    labels = [k for k, _ in left] + [k for k, _ in right]
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

def cumulative_curve(contribs, outpath, coverage=0.9, max_top=60):
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

def group_butterfly(keys, p, q, contribs, model_a, model_b, outpath, group_map, coverage=0.9, max_top=60):
    groups = dd(lambda: {'abs': 0.0, 'signed': 0.0})
    for i, k in enumerate(keys):
        g = group_map.get(k, None)
        if g is None:
            continue
        c = contribs[i]
        if c <= 0:
            continue
        groups[g]['abs'] += c
        sgn = 1.0 if (p[i] - q[i]) > 0 else -1.0
        groups[g]['signed'] += sgn * c

    if not groups:
        return

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
    model_names_list = {}
    for model, ctr in model_counters.items():
        expanded = []
        for t, c in ctr.items():
            if c > 0:
                expanded.extend([t]*c)
        if expanded:
            model_names_list[model] = expanded

    sh_scores = []
    si_scores = []
    for model, names in model_names_list.items():
        if len(names) <= 1:
            continue
        sh = calculate_shannon_diversity(names)
        si = simpson_diversity_index(names)
        sh_scores.append((model, sh))
        si_scores.append((model, si))

    sh_scores.sort(key=lambda x: (x[1], x[0]))
    si_scores.sort(key=lambda x: (x[1], x[0]))

    def scatter(scores, idx_label, fname):
        if not scores: return
        models = [m for m, _ in scores]
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        y = np.arange(len(models))

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

    os.makedirs(outdir, exist_ok=True)
    def save_md(scores, stem):
        if not scores: return
        p = os.path.join(outdir, f"{stem}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("| Model | Diversity |\n| --- | --- |\n")
            for m, v in scores:
                f.write(f"| {m} | {v:.3f} |\n")

    save_md(sh_scores, f"diversity-{phenom}-shannon")
    save_md(si_scores, f"diversity-{phenom}-simpson")
    scatter(sh_scores, "Shannon Index", f"diversity-{phenom}-shannon.png")
    scatter(si_scores, "Simpson Index", f"diversity-{phenom}-simpson.png")

# --------------------------- Explain pipeline (pairwise & group) ---------------------------
def group_mean_prob(model_counters: Dict[str, Counter], models: Sequence[str]) -> Dict[str, float]:
    probs = []
    for m in models:
        if m not in model_counters:
            continue
        p = to_prob(model_counters[m])
        if p:
            probs.append(p)
    if not probs:
        return {}
    keys = set()
    for p in probs:
        keys |= set(p.keys())
    keys = sorted(keys)
    arrs = []
    for p in probs:
        arrs.append(np.array([p.get(k, 0.0) for k in keys], dtype=float))
    M = np.vstack(arrs)
    mean = M.mean(axis=0)
    return {k: float(mean[i]) for i, k in enumerate(keys)}

def run_explain_for_phenom(phenom, model_counters, outdir, model_a, model_b, coverage, max_top, group_map=None):
    if model_a not in model_counters or model_b not in model_counters:
        return
    p_dict = to_prob(model_counters[model_a])
    q_dict = to_prob(model_counters[model_b])
    keys, p, q = align_probs(p_dict, q_dict, eps=1e-12)
    contribs = jsd_contributions(p, q)
    base = f"explain-jsd-{sanitize(phenom)}-{sanitize(model_a)}--vs--{sanitize(model_b)}"
    butterfly_explain(keys, p, q, contribs, model_a, model_b, os.path.join(outdir, base + "-butterfly.png"), coverage, max_top)
    cumulative_curve(contribs, os.path.join(outdir, base + "-cumulative.png"), coverage, max_top)
    if group_map:
        group_butterfly(keys, p, q, contribs, model_a, model_b, os.path.join(outdir, base + "-group-butterfly.png"), group_map, coverage, max_top)

def run_group_explain_for_phenom(phenom, model_counters, outdir, groupA, groupB, coverage, max_top, group_map=None, permute_models=0, rng_seed=1337):
    p_mean = group_mean_prob(model_counters, groupA)
    q_mean = group_mean_prob(model_counters, groupB)
    if not p_mean or not q_mean:
        return
    keys, p, q = align_probs(p_mean, q_mean, eps=1e-12)
    contribs = jsd_contributions(p, q)
    jsd_total = float(contribs.sum())

    labelA = "GRP{" + ",".join(groupA) + "}"
    labelB = "GRP{" + ",".join(groupB) + "}"
    base = f"explain-jsd-{sanitize(phenom)}-{sanitize(labelA)}--vs--{sanitize(labelB)}"
    butterfly_explain(keys, p, q, contribs, labelA, labelB, os.path.join(outdir, base + "-butterfly.png"), coverage, max_top)
    cumulative_curve(contribs, os.path.join(outdir, base + "-cumulative.png"), coverage, max_top)
    if group_map:
        group_butterfly(keys, p, q, contribs, labelA, labelB, os.path.join(outdir, base + "-group-butterfly.png"), group_map, coverage, max_top)

    if permute_models and permute_models > 0:
        rng = random.Random(rng_seed)
        present = [m for m in model_counters.keys() if m in set(groupA) | set(groupB)]
        A_size = len([m for m in groupA if m in model_counters])
        B_size = len([m for m in groupB if m in model_counters])
        if A_size == 0 or B_size == 0 or A_size + B_size < 2:
            return
        jsd_null = []
        for _ in range(int(permute_models)):
            rng.shuffle(present)
            A_perm = present[:A_size]
            B_perm = present[A_size:A_size+B_size]
            p_perm = group_mean_prob(model_counters, A_perm)
            q_perm = group_mean_prob(model_counters, B_perm)
            if not p_perm or not q_perm:
                continue
            _, pp, qq = align_probs(p_perm, q_perm, eps=1e-12)
            jsd_perm = float(jsd_contributions(pp, qq).sum())
            jsd_null.append(jsd_perm)
        if jsd_null:
            ge = sum(1 for v in jsd_null if v >= jsd_total)
            pval = (ge + 1) / (len(jsd_null) + 1)
            txt = os.path.join(outdir, base + "-perm-summary.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(f"Observed group JSD: {jsd_total:.6f}\n")
                f.write(f"Permutations: {len(jsd_null)}\n")
                f.write(f"p-value (>= observed): {pval:.6g}\n")
                f.write(f"Null mean: {np.mean(jsd_null):.6f}, std: {np.std(jsd_null):.6f}\n")
                q05, q50, q95 = np.quantile(jsd_null, [0.05, 0.5, 0.95])
                f.write(f"Null quantiles: 5%={q05:.6f}, 50%={q50:.6f}, 95%={q95:.6f}\n")

# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Diversity (legacy) with pairwise & group JSD explain plots.")
    ap.add_argument("json_files", nargs="+", help="Input JSON files in the OLD format.")
    ap.add_argument("--output-dir", default="out", help="Directory to write outputs.")
    ap.add_argument("--phenomena", nargs="+", choices=SUPPORTED_PHENOMENA, default=["constr", "lextype", "lexrule", "lexentries"], help="Phenomena to analyze.")
    ap.add_argument("--explain", nargs=2, metavar=("MODEL_A", "MODEL_B"), help="Pairwise JSD explain plots for two models.")
    ap.add_argument("--group-explain", nargs=2, metavar=("GROUP_A", "GROUP_B"), help="Compare two comma-separated model lists, e.g., \"A,B,C\" \"D,E,F\".")
    ap.add_argument("--coverage", type=float, default=0.9, help="Coverage target for Top-K selection (default 0.9).")
    ap.add_argument("--max-top", type=int, default=60, help="Upper cap on Top-K items/groups (default 60).")
    ap.add_argument("--group-map", type=str, help="Optional JSON {type: group} for group-level butterfly.")
    ap.add_argument("--permute-models", type=int, default=0, help="If >0, run that many model-level permutations for group JSD.")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for permutations.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data, seen_any = load_and_merge(args.json_files, args.phenomena)
    if not seen_any:
        print("No recognized phenomena found in the provided files."); return
    if not data:
        print("No non-empty phenomena after merging (all were empty).\n"); return

    for phenom, model_counters in data.items():
        if not model_counters:
            continue
        plot_scatter_for_phenomenon(phenom, model_counters, args.output_dir)

    group_map = None
    if args.group_map:
        with open(args.group_map, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            group_map = loaded
        else:
            print("Warning: --group-map JSON must be a dict of {type: group}; ignoring.")

    if args.explain:
        model_a, model_b = args.explain[0], args.explain[1]
        for phenom, model_counters in data.items():
            run_explain_for_phenom(phenom, model_counters, args.output_dir, model_a, model_b, args.coverage, args.max_top, group_map)

    if args.group_explain:
        grpA = [s for s in args.group_explain[0].split(",") if s.strip()]
        grpB = [s for s in args.group_explain[1].split(",") if s.strip()]
        for phenom, model_counters in data.items():
            run_group_explain_for_phenom(phenom, model_counters, args.output_dir, grpA, grpB, args.coverage, args.max_top, group_map, args.permute_models, args.seed)

if __name__ == "__main__":
    main()
