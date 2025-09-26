#!/usr/bin/env python3
import json
import math
import os
from typing import Dict, List

import matplotlib

# Set global font sizes
matplotlib.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

def read_json_array(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


DIMENSIONS = ["IIA", "OPC", "AU", "IE", "Overall"]
MODEL_KEYS = {
    "evaluation_gpt-4o": "GPT-4o",
    "evaluation_deepseek-r1": "Deepseek-R1",
    "evaluation_claude-3-7-sonnet": "Claude-3.7-Sonnet",
}

def compute_model_dimension_averages(records: List[dict]) -> Dict[str, Dict[str, float]]:
    # Initialize sums and counts (excluding Overall for now)
    individual_dims = ["IIA", "OPC", "AU", "IE"]
    sums: Dict[str, Dict[str, float]] = {model: {d: 0.0 for d in individual_dims} for model in MODEL_KEYS.values()}
    counts: Dict[str, Dict[str, int]] = {model: {d: 0 for d in individual_dims} for model in MODEL_KEYS.values()}

    for rec in records:
        for raw_key, model in MODEL_KEYS.items():
            eval_block = rec.get(raw_key) or {}
            for dim in individual_dims:
                entry = eval_block.get(dim) or {}
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    sums[model][dim] += float(score)
                    counts[model][dim] += 1

    # Calculate averages for individual dimensions
    result = {
        model: {
            dim: (sums[model][dim] / counts[model][dim] if counts[model][dim] > 0 else 0.0)
            for dim in individual_dims
        }
        for model in MODEL_KEYS.values()
    }
    
    # Calculate overall average for each model
    for model in MODEL_KEYS.values():
        individual_scores = [result[model][dim] for dim in individual_dims]
        result[model]["Overall"] = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
    
    return result

def make_grouped_bar_single_panel(models: Dict[str, Dict[str, float]], labels: List[str], out_path: str, title: str = "Pxplore.json") -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["GPT-4o", "Claude-3.7-Sonnet", "Deepseek-R1"]
    # Darker solid palette (Pxplore only)
    colors = {
        "GPT-4o": "#E05A6E",            # red
        "Claude-3.7-Sonnet": "#3AA374",  # green
        "Deepseek-R1": "#4A90E2",       # blue
    }

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = np.linspace(-width, width, len(model_order))
    for idx, model in enumerate(model_order):
        vals = [models.get(model, {}).get(dim, 0.0) for dim in labels]
        ax.bar(x + offsets[idx], vals, width, label=model, color=colors.get(model, None), alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel('Average Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(2.5, 5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)

def make_grouped_bar_single_panel_with_baseline(models_eval: Dict[str, Dict[str, float]], models_base: Dict[str, Dict[str, float]], labels: List[str], out_path: str) -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["GPT-4o", "Claude-3.7-Sonnet", "Deepseek-R1"]
    # Solid (Pxplore): lighter fill with darker outline
    eval_fill_colors = {
        "GPT-4o": "#E88993",            # darker red fill
        "Claude-3.7-Sonnet": "#97D3A8",  # darker green fill
        "Deepseek-R1": "#7EB6FF",       # darker blue fill
    }
    eval_edge_colors = {
        "GPT-4o": "#E05A6E",            # red edge
        "Claude-3.7-Sonnet": "#3AA374",  # green edge
        "Deepseek-R1": "#4A90E2",       # blue edge
    }
    base_colors = {
        "GPT-4o": "#F7BFC4",            # light red
        "Claude-3.7-Sonnet": "#CDECCF",  # light green
        "Deepseek-R1": "#CFE8FF",       # light blue
    }
    # Subtle texture (hatch) for baseline bars, one per model
    base_hatches = {
        "GPT-4o": "//",
        "Deepseek-R1": "//",
        "Claude-3.7-Sonnet": "//",
    }

    x = np.arange(len(labels))
    group_width = 0.82  # slightly narrower clusters for a bit more spacing
    pair_width = group_width / 3.0  # width allotted per model pair (Pxplore + Baseline)
    bar_width = pair_width / 2.0  # slightly narrower bars to avoid crowding

    fig, ax = plt.subplots(figsize=(14, 6))

    # For each model, place a pair (Pxplore solid, Baseline dashed) side by side within the dimension
    # Offsets center the three pairs around each x tick
    pair_offsets = np.linspace(-group_width / 2 + pair_width / 2, group_width / 2 - pair_width / 2, len(model_order))

    for idx, model in enumerate(model_order):
        eval_vals = [models_eval.get(model, {}).get(dim, 0.0) for dim in labels]
        base_vals = [models_base.get(model, {}).get(dim, 0.0) for dim in labels]

        centers = x + pair_offsets[idx]
        # Pxplore (solid) to the left within the pair
        ax.bar(centers - bar_width / 2, eval_vals, bar_width,
               color=eval_fill_colors.get(model, None), alpha=0.8,
               edgecolor=eval_edge_colors.get(model, None), linewidth=1.2)
        # Baseline (dashed) to the right within the pair
        ax.bar(centers + bar_width / 2, base_vals, bar_width,
               color=base_colors.get(model, None), alpha=0.35,
               edgecolor=eval_edge_colors.get(model, None), linewidth=1,
               hatch=base_hatches.get(model, None))

    ax.set_ylabel('Average Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(2.5, 5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.margins(x=0.02)

    # Legend: model colors + pattern meaning
    from matplotlib.patches import Patch
    model_handles = [Patch(facecolor=eval_fill_colors[m], edgecolor=eval_edge_colors[m], label=m) for m in model_order]
    pattern_handles = [
        Patch(facecolor='gray', edgecolor='gray', label='Pxplore', alpha=0.3),
        Patch(facecolor='none', edgecolor='gray', label='Baseline', hatch='//')
    ]
    ax.legend(handles=(model_handles + pattern_handles), loc='upper left', ncol=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    eval_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_profile.json")
    baseline_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_profile_baseline.json")

    eval_records = read_json_array(eval_path)
    baseline_records = read_json_array(baseline_path)

    # Compute per-model per-dimension averages
    models_eval = compute_model_dimension_averages(eval_records)
    models_base = compute_model_dimension_averages(baseline_records)

    out_pdf = os.path.join(repo_root, "service", "scripts", "evaluation", "result_profile.pdf")
    make_grouped_bar_single_panel_with_baseline(models_eval, models_base, DIMENSIONS, out_pdf)
    print(f"Saved bar chart to: {out_pdf}")


if __name__ == "__main__":
    main()


