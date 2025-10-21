#!/usr/bin/env python3
"""
diversity_depth.py  —  depth-based diversity analysis (clean version)

- Supports ONLY the new JSON format: top-level keys 1..5 (depth) → model → {lextype: count}.
- Accepts multiple input files; merges by depth/model (adds counts).
- Imports unchanged metrics from your original `diversity.py` (Shannon, Simpson, permutation_test).
- Per-depth scatter: cohort colors (2023/2025/humans) and markers (LLM vs human);
  legend matches styling.
- Cross-depth plots: unique color PER MODEL (no reuse), line style by cohort
  (dotted=2023, solid=2025, dash-dot=human). Legends are placed OUTSIDE the axes
  (styles on top, models on the right). Optional numeric markers show depth numbers.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict as dd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --------------------------- Imports from diversity.py ---------------------------

try:
    from diversity import calculate_shannon_diversity, simpson_diversity_index, permutation_test
except Exception:
    # Fallbacks (only used if diversity.py is not available)
    def calculate_shannon_diversity(names):
        if not names:
            return float('nan')
        counts = dd(int)
        for n in names:
            counts[n] += 1
        total = len(names)
        if total == 0:
            return float('nan')
        H = 0.0
        for c in counts.values():
            p = c / total
            H -= p * math.log(p)
        return H

    def simpson_diversity_index(names):
        if not names or len(names) <= 1:
            return float('nan')
        counts = Counter(names)
        N = sum(counts.values())
        if N <= 1:
            return float('nan')
        num = sum(c*c for c in counts.values())
        return 1 - (num / (N*N))

    def permutation_test(*args, **kwargs):
        raise NotImplementedError("permutation_test requires diversity.py")


# ------------------------------- Styling -----------------------------------

# Cohort colors (match spirit of old script)
COLOR_LLM_2023   = "#3A6DB4"  # cobalt blue
COLOR_LLM_2025   = "#4AA6B4"  # teal-blue
COLOR_HUMAN_NYT  = "#B44E3A"  # chestnut brown
COLOR_HUMAN_OTH  = "#D17F4A"  # copper

def is_human_model_name(model_name: str) -> bool:
    return "-human" in model_name.lower()

def model_year(model_name: str):
    m = re.search(r"(20\d{2})", model_name)
    return m.group(1) if m else None

def human_subtype(model_name: str):
    return "nyt" if "nyt" in model_name.lower() else "other"

def scatter_style_for_model(model_name: str):
    """Colors/markers for per-depth scatter plots."""
    if is_human_model_name(model_name):
        if human_subtype(model_name) == "nyt":
            return dict(color=COLOR_HUMAN_NYT, marker="*", size=60)
        else:
            return dict(color=COLOR_HUMAN_OTH, marker="+", size=55)
    y = model_year(model_name)
    if y == "2023":
        return dict(color=COLOR_LLM_2023, marker="o", size=40)
    elif y == "2025":
        return dict(color=COLOR_LLM_2025, marker="o", size=40)
    return dict(color=COLOR_LLM_2025, marker="o", size=40)

def line_style_for_model(model_name: str):
    """Dotted for 2023 (older), solid for 2025 (newer), dash-dot for humans."""
    if is_human_model_name(model_name):
        return "-."
    y = model_year(model_name)
    if y == "2023":
        return ":"
    elif y == "2025":
        return "-"
    return "--"



def marker_for_model(model_name: str):
    """Marker choice for depth-comparison lines."""
    if is_human_model_name(model_name):
        return "*" if human_subtype(model_name) == "nyt" else "+"
    return "o"
def distinct_color_map(models):
    """
    Pleasant, non-repeating qualitative colors.
    Start with tab20 (de-paired), extend with Set2 and Set3;
    if still short, add softly saturated HSV tones.
    """
    models_sorted = sorted(models)
    n = len(models_sorted)
    # tab20 de-paired
    tab20 = list(plt.get_cmap('tab20').colors)
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    palette = [tab20[i] for i in order]
    # extend with pleasant qualitative sets
    palette += list(plt.get_cmap('Set2').colors)
    palette += list(plt.get_cmap('Set3').colors)
    # fallback gentle HSV if needed
    import matplotlib
    while len(palette) < n:
        h = (len(palette) * 0.61803398875) % 1.0
        palette.append(matplotlib.colors.hsv_to_rgb((h, 0.55, 0.85)))
    return {m: palette[i] for i, m in enumerate(models_sorted[:n])}


# ------------------------------- Heuristics ---------------------------------

def is_punct_type(tname: str) -> bool:
    t = tname.lower()
    return t.startswith('pt') or ('pct' in t) or t.endswith('plr')


# ------------------------------- IO ----------------------------------------

def load_depth_data(json_paths, selected_depths=None, min_items=2):
    """
    Return dict[int depth][str model] -> list[str lextype repeated by count]
    Only keeps depths where at least one model has >= min_items items.
    """
    data = dd(lambda: dd(list))
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        for k, models in obj.items():
            try:
                d = int(k)
            except Exception:
                continue
            if d < 1 or d > 5:
                continue
            if selected_depths and d not in selected_depths:
                continue
            if not isinstance(models, dict):
                continue
            for model, counts in models.items():
                if not isinstance(counts, dict):
                    continue
                expanded = []
                for tname, c in counts.items():
                    try:
                        c = int(c)
                    except Exception:
                        continue
                    if c <= 0:
                        continue
                    expanded.extend([tname] * c)
                if expanded:
                    data[d][model].extend(expanded)

    # keep depths with at least one model meeting the threshold
    data = {d: md for d, md in data.items() if any(len(lst) >= min_items for lst in md.values())}
    return data

def debug_summary(json_paths, selected_depths, raw_data, min_items):
    print("\n=== DEBUG SUMMARY ===")
    print(f"Files: {', '.join(json_paths)}")
    if selected_depths:
        print(f"Selected depths: {sorted(selected_depths)}")
    else:
        print("Selected depths: ALL present in files")
    if not raw_data:
        print("No depths found at all. Expect top-level keys 1..5 → model → {type: count}.")
        return
    for d in sorted(raw_data.keys()):
        models = raw_data[d]
        num_models = len(models)
        num_meet = sum(1 for m, names in models.items() if len(names) >= min_items)
        print(f"- Depth {d}: {num_models} models; {num_meet} with >= {min_items} items.")
        for m, names in list(models.items())[:5]:
            print(f"    · {m}: {len(names)} items")
    print("A depth is kept only if ≥1 model has ≥ min-items-per-model items (default 2).")


# ------------------------------- Plotting ----------------------------------

def plot_scatter_diversity(scores, title, xlabel, outpath):
    """scores: list[(model, value)]"""
    if not scores:
        return
    models = [m for m, _ in scores]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(models))
    for i, (m, v) in enumerate(scores):
        st = scatter_style_for_model(m)
        ax.scatter(v, i, s=st.get('size', 40), marker=st['marker'], color=st['color'])

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Cohort legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor=COLOR_LLM_2023, markersize=8, label='LLMs (2023)'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=COLOR_LLM_2025, markersize=8, label='LLMs (2025)'),
        Line2D([0],[0], marker='*', color='none', markerfacecolor=COLOR_HUMAN_NYT, markersize=10, label='Human (NYT)'),
        Line2D([0],[0], marker='+', color=COLOR_HUMAN_OTH, markersize=8, label='Human (WSJ/Wiki)'),
    ]
    # Place legend outside to avoid occlusion
    ax.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8, title='Model Type', frameon=True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def save_markdown(scores, outpath):
    if not scores:
        return
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write("| Model | Diversity |\n| --- | --- |\n")
        for m, v in scores:
            f.write(f"| {m} | {v:.3f} |\n")

def learning_curve(names, n_bins):
    """Return two lists (Shannon, Simpson) over n_bins cumulative steps."""
    if not names or len(names) <= n_bins:
        return [], []
    shuffled = names.copy()
    random.shuffle(shuffled)
    total = len(shuffled)
    s_curve, sim_curve = [], []
    running = dd(int)
    for i in range(1, n_bins + 1):
        end = int((i / n_bins) * total)
        start = int(((i - 1) / n_bins) * total) if i > 1 else 0
        for name in shuffled[start:end]:
            running[name] += 1
        expanded = []
        for name, c in running.items():
            expanded.extend([name] * c)
        s_curve.append(calculate_shannon_diversity(expanded) if len(expanded) > 0 else float('nan'))
        sim_curve.append(simpson_diversity_index(expanded) if len(expanded) > 1 else float('nan'))
    return s_curve, sim_curve

def plot_learning_curves(depth, model_to_names, outdir, n_bins):
    xs = [100 * (i + 1) / n_bins for i in range(n_bins)]
    for idx_name, idx_extractor in [('Shannon', lambda sc: sc[0]), ('Simpson', lambda sc: sc[1])]:
        fig, ax = plt.subplots(figsize=(10, 5))
        any_plotted = False
        for model, names in model_to_names.items():
            s_curve, sim_curve = learning_curve(names, n_bins)
            if not s_curve or not sim_curve:
                continue
            ys = idx_extractor((s_curve, sim_curve))
            if len(ys) != len(xs):
                continue
            ax.plot(xs, ys, marker='o', linewidth=1.5, alpha=0.85, label=model)
            any_plotted = True
        if not any_plotted:
            plt.close()
            continue
        ax.set_xlabel('Percentage of Data (%)')
        ax.set_ylabel(f'{idx_name} Diversity')
        ax.set_title(f'Learning Curve — Depth {depth} ({idx_name})')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        out = os.path.join(outdir, f"learning-depth-{depth}-{idx_name.lower()}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()

def plot_cross_depth(models_by_depth, outdir, suffix="", number_markers=False):
    """
    One plot per index (Shannon, Simpson), x-axis = depth, line per model.
    number_markers=True overlays the depth number at each point.
    """
    depths = sorted(models_by_depth.keys())
    all_models = set()
    for d in depths:
        all_models.update(models_by_depth[d].keys())
    if not all_models:
        return

    color_map = distinct_color_map(all_models)
    indices = {'Shannon': calculate_shannon_diversity, 'Simpson': simpson_diversity_index}
    title_suffix = {'': '', '_punct': ' — punctuation only', '_xpunct': ' — no punctuation'}[suffix]

    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0],[0], color='k', linestyle=':',  label='2023 (older)'),
        Line2D([0],[0], color='k', linestyle='-',  label='2025 (newer)'),
        Line2D([0],[0], color='k', linestyle='-.', label='Human'),
    ]

    for idx_name, idx_fn in indices.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        any_plotted = False
        for model in sorted(all_models):
            xs, ys = [], []
            for d in depths:
                names = models_by_depth[d].get(model, [])
                val = idx_fn(names) if len(names) > 1 else float('nan')
                if not np.isnan(val):
                    xs.append(d)
                    ys.append(val)
            if xs:
                ls = line_style_for_model(model)
                color = color_map[model]
                ax.plot(xs, ys, linewidth=2.2, alpha=0.95, label=model, color=color, linestyle=ls)
                mk = marker_for_model(model)
                if number_markers and not is_human_model_name(model):
                    for x, y in zip(xs, ys):
                        ax.plot([x], [y], linestyle='None', marker=f"${int(x)}$", markersize=10, color=color)
                else:
                    ax.plot(xs, ys, linestyle='None', marker=mk, markersize=5.5,
                            markerfacecolor=color if mk!='+' else 'none', markeredgecolor='white', markeredgewidth=0.7)
                any_plotted = True
        if not any_plotted:
            plt.close()
            continue
        ax.set_xlabel('Lexical Type Depth')
        ax.set_ylabel(f'{idx_name} Diversity')
        ax.set_title(f'Diversity across depths{title_suffix} ({idx_name})')
        ax.set_xticks(depths)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Style legend ABOVE the plot
        leg1 = ax.legend(handles=style_legend, loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3,
                         fontsize=8, title='Line style', frameon=True)
        ax.add_artist(leg1)

        # Model legend on the RIGHT with exact color+linestyle
        end_values = {}
        max_depth = max(depths) if depths else None
        # Compute rightmost y for each model
        for m in all_models:
            ys_m = []
            for d in depths:
                names = models_by_depth[d].get(m, [])
                val = idx_fn(names) if len(names) > 1 else float('nan')
                ys_m.append((d, val))
            # pick the value at the largest depth with a finite value
            y_right = None
            for d, v in sorted(ys_m, key=lambda t: t[0], reverse=True):
                if not (isinstance(v, float) and np.isnan(v)):
                    y_right = v
                    break
            end_values[m] = y_right if y_right is not None else float('nan')
        legend_models = sorted(all_models, key=lambda m: end_values[m] if end_values[m]==end_values[m] else -1, reverse=True)
        from matplotlib.lines import Line2D
        model_handles = []
        for m in legend_models:
            ls = line_style_for_model(m)
            mk = marker_for_model(m)
            model_handles.append(Line2D([0],[0], color=color_map[m], linestyle=ls, marker=mk,
                                        markerfacecolor=color_map[m] if mk!='+' else 'none', markeredgecolor='white',
                                        markeredgewidth=0.7, linewidth=2.2, label=m))
        leg2 = ax.legend(handles=model_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                         fontsize=8, title='Model', frameon=True)

        plt.tight_layout()
        out = os.path.join(outdir, f"depth-comparison{suffix}-{idx_name.lower()}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()


# ------------------------------- Pipeline ----------------------------------

def analyze_depth(depth, model_to_names, outdir, tag=""):
    """Compute diversity per model for one depth; save tables/plots."""
    sh_scores, sim_scores = [], []
    for model, names in model_to_names.items():
        if len(names) <= 1:
            continue
        sh = calculate_shannon_diversity(names)
        si = simpson_diversity_index(names)
        sh_scores.append((model, sh))
        sim_scores.append((model, si))
    sh_scores.sort(key=lambda x: (x[1], x[0]))
    sim_scores.sort(key=lambda x: (x[1], x[0]))
    save_markdown(sh_scores, os.path.join(outdir, f"diversity-depth-{depth}{tag}-shannon.md"))
    save_markdown(sim_scores, os.path.join(outdir, f"diversity-depth-{depth}{tag}-simpson.md"))
    title_base = f"Diversity — Depth {depth}"
    plot_scatter_diversity(sh_scores, f"{title_base}{tag} (Shannon)", "Shannon Index",
                           os.path.join(outdir, f"diversity-depth-{depth}{tag}-shannon.png"))
    plot_scatter_diversity(sim_scores, f"{title_base}{tag} (Simpson)", "Simpson Index",
                           os.path.join(outdir, f"diversity-depth-{depth}{tag}-simpson.png"))

def split_punct_views(models_by_depth):
    """Return base, punct-only, no-punct dicts with shape depth->model->list."""
    base, punct, xpunct = {}, {}, {}
    for d, md in models_by_depth.items():
        base[d], punct[d], xpunct[d] = {}, {}, {}
        for model, names in md.items():
            base[d][model] = list(names)
            punct[d][model] = [t for t in names if is_punct_type(t)]
            xpunct[d][model] = [t for t in names if not is_punct_type(t)]
    return base, punct, xpunct

def main():
    parser = argparse.ArgumentParser(description="Compute & visualize diversity for depth-based lextype JSON.")
    parser.add_argument("json_files", nargs="+", help="Input JSON file(s) in the new depth format.")
    parser.add_argument("--output-dir", default="out", help="Directory to write outputs.")
    parser.add_argument("--depths", nargs="+", type=int, choices=[1,2,3,4,5], help="Only analyze these depths.")
    parser.add_argument("--split-punct", action="store_true", help="Also produce punctuation-only and no-punct variants.")
    parser.add_argument("--learning", type=int, metavar="N", help="Generate learning curves with N bins (per depth).")
    parser.add_argument("--compare-depths", action="store_true", help="Plot diversity across depths (per model).")
    parser.add_argument("--number-markers", action="store_true", help="Use numeric markers ($1$..$5$) on depth plots.")
    parser.add_argument("--min-items-per-model", type=int, default=2,
                        help="Minimum items per model to keep a depth (default: 2).")
    parser.add_argument("--debug", action="store_true",
                        help="Print diagnostics about what was loaded and why depths/models may be skipped.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load + optional debug
    selected = set(args.depths) if args.depths else None
    raw_loaded = load_depth_data(args.json_files, selected_depths=selected, min_items=1)
    if args.debug:
        debug_summary(args.json_files, selected, raw_loaded, args.min_items_per_model)
    models_by_depth = load_depth_data(args.json_files, selected_depths=selected,
                                      min_items=args.min_items_per_model)
    if not models_by_depth:
        print("No depth data found with observations meeting the threshold. Try --debug or lower --min-items-per-model 1")
        return

    # Views
    base = models_by_depth
    if args.split_punct:
        base, punct, xpunct = split_punct_views(models_by_depth)
        views = [("", base), ("_punct", punct), ("_xpunct", xpunct)]
    else:
        views = [("", base)]

    # Per-depth analysis
    for tag, view in views:
        for depth in sorted(view.keys()):
            analyze_depth(depth, view[depth], args.output_dir, tag=tag)
            if args.learning:
                plot_learning_curves(depth, view[depth], args.output_dir, n_bins=args.learning)

    # Cross-depth comparison
    if args.compare_depths:
        for tag, view in views:
            plot_cross_depth(view, args.output_dir, suffix=tag, number_markers=args.number_markers)

if __name__ == "__main__":
    main()
