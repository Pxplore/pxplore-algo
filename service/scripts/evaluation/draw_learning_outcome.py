#!/usr/bin/env python3
import os

import matplotlib

# Match plotting style used in other evaluation scripts
matplotlib.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

# Results: values are percentages
result = {
    "Pxplore": {
        "pre-test": 61.81,
        "post-test": 90.09,
        "knowledge-gain": 28.28,
    },
    "Retrieval": {
        "pre-test": 70.00,
        "post-test": 84.18,
        "knowledge-gain": 14.18,
    }
}


def draw_learning_outcome_chart() -> str:
    """Draw two stacked bars (one per system) with segments for pre-test and knowledge gain.

    The total bar height equals post-test. The top segment equals knowledge gain.
    The gain value is annotated on each bar. Saves a PDF next to this script.
    Returns the output file path.
    """
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Display Retrieval first, then Pxplore
    systems = ["Retrieval", "Pxplore"]
    pre_vals = [result[s]["pre-test"] for s in systems]
    gain_vals = [result[s]["knowledge-gain"] for s in systems]
    post_vals = [result[s]["post-test"] for s in systems]

    # Colors aligned with draw_userstudy.py
    # Use fill for pre-test and a slightly darker edge color as gain fill
    pxplore_fill = "#b0e3e6"
    pxplore_edge = "#8dd3d8"
    retrieval_fill = "#d1cee3"
    retrieval_edge = "#b8b3d1"

    fig, ax = plt.subplots(figsize=(10, 2.6))
    y = np.arange(len(systems))
    height = 0.7

    # Determine per-system colors from userstudy palette
    pre_colors = [pxplore_fill if s == "Pxplore" else retrieval_fill for s in systems]
    gain_colors = [pxplore_edge if s == "Pxplore" else retrieval_edge for s in systems]
    edge_colors = [pxplore_edge if s == "Pxplore" else retrieval_edge for s in systems]


    # Pre segment (left part)
    ax.barh(
        y,
        pre_vals,
        height,
        color=pre_colors,
        edgecolor=edge_colors,
        linewidth=1.2,
        label='Pre-test',
    )

    # Gain segment stacked to the right
    ax.barh(
        y,
        gain_vals,
        height,
        left=pre_vals,
        color=gain_colors,
        edgecolor=edge_colors,
        linewidth=1.2,
        label='Knowledge Gain',
    )

    # Annotate knowledge gain centered in its segment and add direct Pre/Post labels
    for i, (pre, gain, post) in enumerate(zip(pre_vals, gain_vals, post_vals)):
        x_center = pre + gain / 2.0
        ax.text(
            x_center,
            y[i],
            f"Post-test (+{gain:.2f})",
            ha='center', va='center',
            fontsize=16,
        )
        # Pre-test label centered within the visible pre segment (x from 50 to pre)
        pre_label_x = (50 + pre) / 2.0
        ax.text(
            pre_label_x,
            y[i],
            f"Pre-test",
            ha='center', va='center',
            fontsize=16,
            color='black',
        )
    ax.set_xlabel('Average Score (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(systems)
    ax.set_xlim(50, 92)
    ax.set_ylim(-0.6, len(systems) - 1 + 0.6)  # Add vertical margins
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'result_learning_outcome.pdf')
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    output = draw_learning_outcome_chart()
    print(f"Saved learning outcome chart to: {output}")