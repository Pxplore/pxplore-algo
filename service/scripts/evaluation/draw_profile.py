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


DIMENSIONS = ["IIA", "CSA", "LNI", "OPC", "TCG", "AU", "IE"]
MODEL_KEYS = {
    "evaluation_gpt-4o": "GPT-4o",
    "evaluation_deepseek-r1": "Deepseek-R1",
    "evaluation_claude-3-7-sonnet": "Claude-3.7-Sonnet",
}


def compute_dimension_averages(records: List[dict]) -> Dict[str, float]:
    sums = {k: 0.0 for k in DIMENSIONS}
    counts = {k: 0 for k in DIMENSIONS}

    for rec in records:
        avg = rec.get("average_score") or {}
        for dim in DIMENSIONS:
            val = avg.get(dim)
            if isinstance(val, (int, float)):
                sums[dim] += float(val)
                counts[dim] += 1

    return {dim: (sums[dim] / counts[dim] if counts[dim] > 0 else 0.0) for dim in DIMENSIONS}


def compute_model_dimension_averages(records: List[dict]) -> Dict[str, Dict[str, float]]:
    # Initialize sums and counts
    sums: Dict[str, Dict[str, float]] = {model: {d: 0.0 for d in DIMENSIONS} for model in MODEL_KEYS.values()}
    counts: Dict[str, Dict[str, int]] = {model: {d: 0 for d in DIMENSIONS} for model in MODEL_KEYS.values()}

    for rec in records:
        for raw_key, model in MODEL_KEYS.items():
            eval_block = rec.get(raw_key) or {}
            for dim in DIMENSIONS:
                entry = eval_block.get(dim) or {}
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    sums[model][dim] += float(score)
                    counts[model][dim] += 1

    return {
        model: {
            dim: (sums[model][dim] / counts[model][dim] if counts[model][dim] > 0 else 0.0)
            for dim in DIMENSIONS
        }
        for model in MODEL_KEYS.values()
    }


def make_radar_pdf(values_a: Dict[str, float], values_b: Dict[str, float], labels: List[str], out_path: str, title: str = "Evaluation vs Baseline") -> None:
    # Lazy import matplotlib backends after setting non-interactive backend
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    # close the loop
    angles += angles[:1]

    vals_a = [values_a.get(l, 0.0) for l in labels]
    vals_b = [values_b.get(l, 0.0) for l in labels]
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Determine max scale slightly above observed max, default to 5
    max_val = max(5.0, max(vals_a + vals_b))
    upper = math.ceil(max_val * 2) / 2.0  # round to .5
    ax.set_ylim(0, upper)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([upper * x / 5 for x in range(1, 5)])
    ax.set_yticklabels([f"{upper * x / 5:.1f}" for x in range(1, 5)])
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.plot(angles, vals_a, color='#1f77b4', linewidth=2, label='Pxplore.json')
    ax.fill(angles, vals_a, color='#1f77b4', alpha=0.15)

    ax.plot(angles, vals_b, color='#d62728', linewidth=2, label='GPT-4o.json')
    ax.fill(angles, vals_b, color='#d62728', alpha=0.15)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


def make_grouped_bar_pdf(values_a: Dict[str, float], values_b: Dict[str, float], labels: List[str], out_path: str, title: str = "Evaluation vs Baseline") -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    width = 0.35

    vals_a = [values_a.get(l, 0.0) for l in labels]
    vals_b = [values_b.get(l, 0.0) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, vals_a, width, label='Pxplore.json', color='#1f77b4', alpha=0.8)
    rects2 = ax.bar(x + width / 2, vals_b, width, label='GPT-4o.json', color='#d62728', alpha=0.8)

    ax.set_ylabel('Average Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(2, 5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()

    # Removed numeric labels for a cleaner look

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


def make_grouped_bar_per_model_pdf(models_a: Dict[str, Dict[str, float]], models_b: Dict[str, Dict[str, float]], labels: List[str], out_path: str, title_a: str = "Pxplore.json", title_b: str = "GPT-4o.json") -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["GPT-4o", "Deepseek-R1", "Claude-3.7-Sonnet"]
    # Solid palette for single datasets (kept for completeness)
    colors = {
        "GPT-4o": "#E05A6E",            # red (outline tone)
        "Deepseek-R1": "#3AA374",       # green (outline tone)
        "Claude-3.7-Sonnet": "#4A90E2",  # blue (outline tone)
    }

    x = np.arange(len(labels))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, models_vals, title in [(axes[0], models_a, title_a), (axes[1], models_b, title_b)]:
        offsets = np.linspace(-width, width, len(model_order))
        for idx, model in enumerate(model_order):
            vals = [models_vals.get(model, {}).get(dim, 0.0) for dim in labels]
            ax.bar(x + offsets[idx], vals, width, label=model, color=colors.get(model, None), alpha=0.85)

        ax.set_title(title)
        ax.set_ylabel('Average Score')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(2.5, 5)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', ncol=3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


def make_grouped_bar_single_panel(models: Dict[str, Dict[str, float]], labels: List[str], out_path: str, title: str = "Pxplore.json") -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["GPT-4o", "Deepseek-R1", "Claude-3.7-Sonnet"]
    # Darker solid palette (Pxplore only)
    colors = {
        "GPT-4o": "#E05A6E",            # red
        "Deepseek-R1": "#3AA374",       # green
        "Claude-3.7-Sonnet": "#4A90E2",  # blue
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


def make_grouped_bar_single_panel_with_baseline(models_eval: Dict[str, Dict[str, float]], models_base: Dict[str, Dict[str, float]], labels: List[str], out_path: str, title: str = "Pxplore vs Baseline") -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["GPT-4o", "Deepseek-R1", "Claude-3.7-Sonnet"]
    # Solid (Pxplore): lighter fill with darker outline
    eval_fill_colors = {
        "GPT-4o": "#E88993",            # darker red fill
        "Deepseek-R1": "#97D3A8",       # darker green fill
        "Claude-3.7-Sonnet": "#7EB6FF",  # darker blue fill
    }
    eval_edge_colors = {
        "GPT-4o": "#E05A6E",            # red edge
        "Deepseek-R1": "#3AA374",       # green edge
        "Claude-3.7-Sonnet": "#4A90E2",  # blue edge
    }
    base_colors = {
        "GPT-4o": "#F7BFC4",            # light red
        "Deepseek-R1": "#CDECCF",       # light green
        "Claude-3.7-Sonnet": "#CFE8FF",  # light blue
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
    make_grouped_bar_single_panel_with_baseline(models_eval, models_base, DIMENSIONS, out_pdf, title='Pxplore vs Baseline')
    print(f"Saved bar chart to: {out_pdf}")


if __name__ == "__main__":
    main()


