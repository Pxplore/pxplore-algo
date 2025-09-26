#!/usr/bin/env python3
import json
import os
from typing import Dict, List

import matplotlib

# Set global font sizes to match draw_profile.py
matplotlib.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})


def read_json_array(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Seven adaptation dimensions + Overall
DIMENSIONS = ["CC", "PE", "PS", "CE", "MA", "Overall"]

# Keys present in the JSON mapped to pretty model names
MODEL_KEYS = {
    "evaluation_gpt-4o": "GPT-4o",
    "evaluation_deepseek-r1": "Deepseek-R1",
    "evaluation_claude-3-7-sonnet": "Claude-3.7-Sonnet",
}


def compute_model_dimension_averages(records: List[dict], setting: str) -> Dict[str, Dict[str, float]]:
    """
    Compute per-model averages for the given setting ("original" or "adaptation").

    The input records follow this structure under each model key:
      records[*][model_key][setting][dimension] = { "score": number, ... }
    """
    individual_dims = ["CC", "PC", "PE", "PS", "CE", "MA", "RS"]

    sums: Dict[str, Dict[str, float]] = {model: {d: 0.0 for d in individual_dims} for model in MODEL_KEYS.values()}
    counts: Dict[str, Dict[str, int]] = {model: {d: 0 for d in individual_dims} for model in MODEL_KEYS.values()}

    for rec in records:
        for raw_key, model in MODEL_KEYS.items():
            eval_block = rec.get(raw_key) or {}
            data = (eval_block.get(setting) or {}) if isinstance(eval_block, dict) else {}
            for dim in individual_dims:
                entry = data.get(dim) or {}
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    sums[model][dim] += float(score)
                    counts[model][dim] += 1

    # Calculate averages for individual dimensions
    result: Dict[str, Dict[str, float]] = {
        model: {
            dim: (sums[model][dim] / counts[model][dim] if counts[model][dim] > 0 else 0.0)
            for dim in individual_dims
        }
        for model in MODEL_KEYS.values()
    }

    # Calculate overall as the mean across the seven dimensions
    for model in MODEL_KEYS.values():
        individual_scores = [result[model][dim] for dim in individual_dims]
        result[model]["Overall"] = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0

    return result


def make_grouped_bar_original_vs_adaptation(models_adapt: Dict[str, Dict[str, float]],
                                            models_orig: Dict[str, Dict[str, float]],
                                            labels: List[str],
                                            out_path: str) -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["GPT-4o", "Claude-3.7-Sonnet", "Deepseek-R1"]

    # Colors styled similar to draw_profile.py
    adapt_fill_colors = {
        "GPT-4o": "#E88993",
        "Deepseek-R1": "#7EB6FF",
        "Claude-3.7-Sonnet": "#97D3A8",
    }
    edge_colors = {
        "GPT-4o": "#E05A6E",
        "Deepseek-R1": "#4A90E2",
        "Claude-3.7-Sonnet": "#3AA374",
    }
    orig_colors = {
        "GPT-4o": "#F7BFC4",
        "Deepseek-R1": "#CFE8FF",
        "Claude-3.7-Sonnet": "#CDECCF",
    }
    hatch_pattern = "//"

    x = np.arange(len(labels))
    group_width = 0.82
    pair_width = group_width / 3.0
    bar_width = pair_width / 2.0

    fig, ax = plt.subplots(figsize=(14, 5))

    pair_offsets = np.linspace(-group_width / 2 + pair_width / 2, group_width / 2 - pair_width / 2, len(model_order))

    for idx, model in enumerate(model_order):
        adapt_vals = [models_adapt.get(model, {}).get(dim, 0.0) for dim in labels]
        orig_vals = [models_orig.get(model, {}).get(dim, 0.0) for dim in labels]

        centers = x + pair_offsets[idx]
        # Adaptation on the left of the pair
        ax.bar(centers - bar_width / 2, adapt_vals, bar_width,
               color=adapt_fill_colors.get(model), alpha=0.8,
               edgecolor=edge_colors.get(model), linewidth=1.2)
        # Original on the right of the pair
        ax.bar(centers + bar_width / 2, orig_vals, bar_width,
               color=orig_colors.get(model), alpha=0.35,
               edgecolor=edge_colors.get(model), linewidth=1,
               hatch=hatch_pattern)

    ax.set_ylabel('Average Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(2, 5.2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.margins(x=0.02)

    # Legend
    from matplotlib.patches import Patch
    model_handles = [Patch(facecolor=adapt_fill_colors[m], edgecolor=edge_colors[m], label=m) for m in model_order]
    pattern_handles = [
        Patch(facecolor='gray', edgecolor='gray', label='Pxplore', alpha=0.3),
        Patch(facecolor='none', edgecolor='gray', label='Original', hatch=hatch_pattern)
    ]
    ax.legend(handles=(model_handles + pattern_handles), loc='upper right', ncol=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_adaptation.json")

    records = read_json_array(data_path)

    models_original = compute_model_dimension_averages(records, setting="original")
    models_adaptation = compute_model_dimension_averages(records, setting="adaptation")

    out_pdf = os.path.join(repo_root, "service", "scripts", "evaluation", "result_adaptation.pdf")
    make_grouped_bar_original_vs_adaptation(models_adaptation, models_original, DIMENSIONS, out_pdf)
    print(f"Saved bar chart to: {out_pdf}")


if __name__ == "__main__":
    main()


