#!/usr/bin/env python3
import json
import math
import os
from typing import Dict, List

import matplotlib
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Set global font sizes
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 10,
    'axes.labelsize': 14,
    'xtick.labelsize': 20,
    'ytick.labelsize': 12,
    'legend.fontsize': 18,
})

def read_json_array(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


DIMENSIONS = ["IIA", "OPC", "AU", "IE", "Overall"]
MODEL_KEYS = {
    "evaluation_gpt-4o": "GPT-4o",
    "evaluation_deepseek-r1": "Deepseek-R1",
    "evaluation_claude-3-7-sonnet": "Claude-3.7-Sonnet",
    "human_evaluation": "Human",
}

def compute_model_dimension_averages(records: List[dict]) -> Dict[str, Dict[str, float]]:
    # Initialize sums and counts (excluding Overall for now)
    individual_dims = ["IIA", "OPC", "AU", "IE"]
    sums: Dict[str, Dict[str, float]] = {model: {d: 0.0 for d in individual_dims} for model in MODEL_KEYS.keys()}
    counts: Dict[str, Dict[str, int]] = {model: {d: 0 for d in individual_dims} for model in MODEL_KEYS.keys()}

    for rec in records:
        for raw_key, model in MODEL_KEYS.items():
            if raw_key == "human_evaluation":
                continue  # Skip human evaluation in this function
            eval_block = rec.get(raw_key) or {}
            for dim in individual_dims:
                entry = eval_block.get(dim) or {}
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    sums[raw_key][dim] += float(score)
                    counts[raw_key][dim] += 1

    # Calculate averages for individual dimensions
    result = {}
    for raw_key, model in MODEL_KEYS.items():
        if raw_key == "human_evaluation":
            continue  # Skip human evaluation in this function
        result[model] = {
            dim: (sums[raw_key][dim] / counts[raw_key][dim] if counts[raw_key][dim] > 0 else 0.0)
            for dim in individual_dims
        }
    
    # Calculate overall average for each model (excluding Human)
    for model in result.keys():
        individual_scores = [result[model][dim] for dim in individual_dims]
        result[model]["Overall"] = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
    
    return result


def compute_human_dimension_averages(annotation_records: List[dict], system_type: str) -> Dict[str, float]:
    """Compute average scores for human evaluation of a specific system type (pxplore or baseline)"""
    individual_dims = ["IIA", "OPC", "AU", "IE"]
    sums: Dict[str, float] = {dim: 0.0 for dim in individual_dims}
    counts: Dict[str, int] = {dim: 0 for dim in individual_dims}
    
    for rec in annotation_records:
        for dim in individual_dims:
            field_name = f"part_a_{system_type}_{dim}"
            score = rec.get(field_name)
            if isinstance(score, (int, float)):
                sums[dim] += float(score)
                counts[dim] += 1
    
    # Calculate averages for individual dimensions
    result = {}
    for dim in individual_dims:
        result[dim] = sums[dim] / counts[dim] if counts[dim] > 0 else 0.0
    
    # Calculate overall average
    individual_scores = [result[dim] for dim in individual_dims]
    result["Overall"] = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
    
    return result


def calculate_pearson_correlations(models_eval: Dict[str, Dict[str, float]], models_base: Dict[str, Dict[str, float]]):
    """Calculate Pearson correlations between human and automated model scores"""
    print("\nPEARSON CORRELATION ANALYSIS")
    print("-"*50)
    
    # Extract scores for correlation analysis
    dimensions = ['IIA', 'OPC', 'AU', 'IE', 'Overall']
    
    # Human scores (as reference)
    human_pxplore_scores = [models_eval["Human"][dim] for dim in dimensions]
    human_baseline_scores = [models_base["Human"][dim] for dim in dimensions]
    
    # Calculate average automated model scores
    automated_models = [model for model in models_eval.keys() if model != "Human"]
    avg_auto_pxplore_scores = []
    avg_auto_baseline_scores = [] 
    
    for dim in dimensions:
        pxplore_vals = [models_eval[model][dim] for model in automated_models]
        baseline_vals = [models_base[model][dim] for model in automated_models]
        avg_auto_pxplore_scores.append(np.mean(pxplore_vals))
        avg_auto_baseline_scores.append(np.mean(baseline_vals))
    
    print("Correlation between Human and Automated Models:")
    print(f"{'Model':<20} {'Pxplore r':>10} {'Pxplore p':>10} {'Baseline r':>12} {'Baseline p':>12}")
    print("-"*70)
    
    # Calculate correlations for each automated model
    for model in automated_models:
        model_pxplore_scores = [models_eval[model][dim] for dim in dimensions]
        model_baseline_scores = [models_base[model][dim] for dim in dimensions]
        
        # Pxplore correlation
        pxplore_r, pxplore_p = pearsonr(human_pxplore_scores, model_pxplore_scores)
        # Baseline correlation  
        baseline_r, baseline_p = pearsonr(human_baseline_scores, model_baseline_scores)
        
        print(f"{model:<20} {pxplore_r:+8.4f} {pxplore_p:>8.4f} {baseline_r:+10.4f} {baseline_p:>10.4f}")
    
    # Calculate correlations with average automated model
    avg_pxplore_r, avg_pxplore_p = pearsonr(human_pxplore_scores, avg_auto_pxplore_scores)
    avg_baseline_r, avg_baseline_p = pearsonr(human_baseline_scores, avg_auto_baseline_scores)
    
    print("-"*70)
    print(f"{'Average (Auto)':<20} {avg_pxplore_r:+8.4f} {avg_pxplore_p:>8.4f} {avg_baseline_r:+10.4f} {avg_baseline_p:>10.4f}")
    
    print("\nInterpretation:")
    print("- r: Correlation coefficient (-1 to +1, closer to ±1 = stronger correlation)")
    print("- p: p-value (< 0.05 typically considered statistically significant)")
    print("- Higher r values indicate automated models better predict human judgment")
    print("\n" + "="*80)


def print_detailed_scores_table(models_eval: Dict[str, Dict[str, float]], models_base: Dict[str, Dict[str, float]]):
    """Print a detailed table of all scores"""
    print("\n" + "="*80)
    print("DETAILED SCORES TABLE")
    print("="*80)
    
    # Create DataFrames for better formatting
    eval_df = pd.DataFrame(models_eval).T
    base_df = pd.DataFrame(models_base).T
    
    eval_df.columns = [f"{col} (Pxplore)" for col in eval_df.columns]
    base_df.columns = [f"{col} (Baseline)" for col in base_df.columns]
    
    # Combine dataframes side by side
    combined_df = pd.concat([eval_df, base_df], axis=1)
    
    # Round to 2 decimal places for better readability
    combined_df = combined_df.round(2)
    
    print("\nFormat: Model | Dimension (System) = Score")
    print("-"*80)
    
    # Print each model with detailed breakdown
    for model in combined_df.index:
        print(f"\n{model}:")
        for dimension in ['IIA', 'OPC', 'AU', 'IE', 'Overall']:
            pxplore_col = f"{dimension} (Pxplore)"
            baseline_col = f"{dimension} (Baseline)"
            
            pxplore_score = combined_df.loc[model, pxplore_col]
            baseline_score = combined_df.loc[model, baseline_col]
            
            print(f"  {dimension:7s}: Pxplore={pxplore_score:5.2f}, Baseline={baseline_score:5.2f}")
    
    print("\n" + "="*80)
    
    # Calculate Pearson correlations
    calculate_pearson_correlations(models_eval, models_base)
    
    # Add improvement summary table
    print("\nIMPROVEMENT SUMMARY (Pxplore - Baseline)")
    print("-"*58)
    print(f"{'Model':<20} {'IIA':>7} {'OPC':>7} {'AU':>7} {'IE':>7} {'Overall':>9}")
    print("-"*58)
    
    # Calculate averages for automated models
    automated_models = []
    automated_dim_improvements = {'IIA': [], 'OPC': [], 'AU': [], 'IE': [], 'Overall': []}
    
    for model in combined_df.index:
        improvements = []
        pxplore_overall = combined_df.loc[model, 'Overall (Pxplore)']
        baseline_overall = combined_df.loc[model, 'Overall (Baseline)']
        
        for dimension in ['IIA', 'OPC', 'AU', 'IE']:
            pxplore_score = combined_df.loc[model, f'{dimension} (Pxplore)']
            baseline_score = combined_df.loc[model, f'{dimension} (Baseline)']
            improvement = pxplore_score - baseline_score
            improvement_pct = (improvement / baseline_score) * 100 if baseline_score != 0 else 0
            improvements.append(f"{improvement_pct:+6.2f}%")
            
            # Collect improvements for automated models average
            if model != "Human":
                automated_dim_improvements[dimension].append(improvement_pct)
        
        overall_improvement = pxplore_overall - baseline_overall
        overall_improvement_pct = (overall_improvement / baseline_overall) * 100 if baseline_overall != 0 else 0
        improvements.append(f"{overall_improvement_pct:+7.2f}%")
        
        print(f"{model:<20} {improvements[0]:>7} {improvements[1]:>7} {improvements[2]:>7} {improvements[3]:>7} {improvements[4]:>9}")
        
        # Collect overall improvements for automated models average
        if model != "Human":
            automated_dim_improvements['Overall'].append(overall_improvement_pct)
    
    # Calculate and print average of automated models
    avg_improvements = []
    for dimension in ['IIA', 'OPC', 'AU', 'IE']:
        avg_val = sum(automated_dim_improvements[dimension]) / len(automated_dim_improvements[dimension])
        avg_improvements.append(f"{avg_val:+6.2f}%")
    
    avg_overall = sum(automated_dim_improvements['Overall']) / len(automated_dim_improvements['Overall'])
    avg_improvements.append(f"{avg_overall:+7.2f}%")
    
    print("-"*58)
    print(f"{'Average (Auto)':<20} {avg_improvements[0]:>7} {avg_improvements[1]:>7} {avg_improvements[2]:>7} {avg_improvements[3]:>7} {avg_improvements[4]:>9}")
    
    print("-"*58)
    print("Note: Positive values indicate Pxplore performs better")
    
    # Add model comparison summary
    print("\nMODEL+COMPARISON SUMMARY")
    print("-"*40)
    
    # Extract overall improvements for comparison
    human_improvement = None
    model_improvements = {}
    
    for model in combined_df.index:
        pxplore_overall = combined_df.loc[model, 'Overall (Pxplore)']
        baseline_overall = combined_df.loc[model, 'Overall (Baseline)']
        improvement = pxplore_overall - baseline_overall
        
        if model == "Human":
            human_improvement = improvement
        else:
            model_improvements[model] = improvement
    
    print(f"Human overall improvement: {human_improvement:+.3f}")
    
    # Calculate average improvement of automated models
    avg_automated_improvement = sum(model_improvements.values()) / len(model_improvements)
    print(f"Average automated model improvement: {avg_automated_improvement:+.3f}")
    print(f"Human vs Automated average: {human_improvement - avg_automated_improvement:+.3f}")
    
    print("\nModel vs Human improvement comparison:")
    print(f"{'Model':<20} {'Improvement':>10} {'vs Human':>10}")
    print("-"*40)
    
    for model, improvement in model_improvements.items():
        difference = human_improvement - improvement
        print(f"{model:<20} {improvement:+8.3f} {difference:+8.3f}")
    
    print(f"{'Average (Auto)':<20} {avg_automated_improvement:+8.3f} {human_improvement - avg_automated_improvement:+8.3f}")
    
    print("\nLegend:")
    print("- Overall Improvement: Pxplore - Baseline performance")
    print("- vs Human: How much better/worse than human improvement")
    print("- Average (Auto): Average of GPT-4o, Deepseek-R1, Claude-3.7-Sonnet")
    print("\n" + "="*80)

def make_grouped_bar_single_panel_with_baseline(models_eval: Dict[str, Dict[str, float]], models_base: Dict[str, Dict[str, float]], labels: List[str], out_path: str) -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["Human", "GPT-4o", "Claude-3.7-Sonnet", "Deepseek-R1"]
    # Colors
    eval_fill_colors = {
        "GPT-4o": "#fcd7ac",            # light peach fill
        "Claude-3.7-Sonnet": "#d1cee3",  # light purple fill
        "Deepseek-R1": "#b0e3e6",       # light blue-green fill
        "Human": "#fadad5",             # light orange fill
    }
    eval_edge_colors = {
        "GPT-4o": "#f0b87a",            # darker peach edge
        "Claude-3.7-Sonnet": "#b8b3d1",  # darker purple edge
        "Deepseek-R1": "#8dd3d8",       # darker blue-green edge
        "Human": "#e6c4bc",             # darker orange edge
    }
    base_colors = {
        "GPT-4o": "#fef8f2",            # very light peach
        "Claude-3.7-Sonnet": "#faf9fd",  # very light purple
        "Deepseek-R1": "#f0fdfe",       # very light blue-green
        "Human": "#fef7f5",             # very light orange
    }
    # Subtle texture (hatch) for baseline bars, one per model
    base_hatches = {
        "GPT-4o": "//",
        "Deepseek-R1": "//",
        "Claude-3.7-Sonnet": "//",
        "Human": "//",
    }

    x = np.arange(len(labels))
    group_width = 0.82  # slightly narrower clusters for a bit more spacing
    pair_width = group_width / 4.0  # width allotted per model pair (Pxplore + Baseline)
    bar_width = pair_width / 2.0  # slightly narrower bars to avoid crowding

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # For each model, place a pair (Pxplore solid, Baseline dashed) side by side within the dimension
    # Offsets center the three pairs around each x tick
    pair_offsets = np.linspace(-group_width / 2 + pair_width / 2, group_width / 2 - pair_width / 2, len(model_order))

    for idx, model in enumerate(model_order):
        eval_vals = [models_eval.get(model, {}).get(dim, 0.0) for dim in labels]
        base_vals = [models_base.get(model, {}).get(dim, 0.0) for dim in labels]

        centers = x + pair_offsets[idx]
        # Pxplore (solid) to the left within the pair
        ax.bar(centers - bar_width / 2, eval_vals, bar_width,
               color=eval_fill_colors.get(model, None), alpha=1,
               edgecolor=eval_edge_colors.get(model, None), linewidth=1.2)
        # Baseline (dashed) to the right within the pair
        ax.bar(centers + bar_width / 2, base_vals, bar_width,
               color=base_colors.get(model, None), alpha=1,
               edgecolor=eval_edge_colors.get(model, None), linewidth=1,
               hatch=base_hatches.get(model, None))

    ax.set_ylabel('Average Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(2.8, 5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.margins(x=0.02)

    # Legend: model colors + pattern meaning
    from matplotlib.patches import Patch
    
    # First legend line: models only
    model_handles = [Patch(facecolor=eval_fill_colors[m], edgecolor=eval_edge_colors[m], label=m) for m in model_order]
    legend1 = ax.legend(handles=model_handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    ax.add_artist(legend1)
    
    # Second legend line: patterns
    pattern_handles = [
        Patch(facecolor='white', edgecolor='gray', label='Pxplore'),
        Patch(facecolor='white', edgecolor='gray', label='Baseline', hatch='//')
    ]
    legend2 = ax.legend(handles=pattern_handles, loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)

def make_heatmap_comparison(models_eval: Dict[str, Dict[str, float]], 
                            models_base: Dict[str, Dict[str, float]], 
                            out_path: str) -> None:
    """Create heatmap showing score differences (Pxplore - Baseline)
    
    热力
  
    """
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    
    dimensions = ["IIA", "OPC", "AU", "IE", "Avg"]
    model_order = ["Human", "GPT-4o", "Claude-3.7-Sonnet", "Deepseek-R1"]
    
    # Calculate differences
    diff_matrix = []
    for model in model_order:
        row = []
        for dim in dimensions:
            # 处理Overall到Avg的映射
            actual_dim = "Overall" if dim == "Avg" else dim
            diff = models_eval[model][actual_dim] - models_base[model][actual_dim]
            row.append(diff)
        diff_matrix.append(row)
    
    diff_matrix = np.array(diff_matrix)
    
   
    # 红色(负值) -> 浅色(接近0) -> 蓝色(正值)
    colors_list = [
        '#C72324',  # 深红（负值）
        '#FEA983',  # 浅橙（接近0负）
        '#FFFFFF',  # 白色（0）
        '#9AC9DB',  # 浅蓝（接近0正）
        '#2978B5',  # 深蓝（正值）
    ]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    
    # Create heatmap
    im = ax.imshow(diff_matrix, cmap=cmap, aspect='auto', vmin=-0.5, vmax=0.5)
    
    # Set ticks
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_xticklabels(dimensions, fontsize=14)
    # Use wrapped label for Claude to move "-Sonnet" to next line
    display_model_labels = [
        (m.replace("-Sonnet", "\n-Sonnet") if m == "Claude-3.7-Sonnet" else (m.replace("-R1", "\n-R1") if m == "Deepseek-R1" else m))
        for m in model_order
    ]
    ax.set_yticklabels(display_model_labels, fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label('Score Improvement (Pxplore - Baseline)', rotation=270, labelpad=25, fontsize=14)
    
    # Add text annotations with adaptive color
    for i in range(len(model_order)):
        for j in range(len(dimensions)):
            value = diff_matrix[i, j]
            # 根据背景颜色选择文字颜色
            text_color = 'white' if abs(value) > 0.25 else 'black'
            text = ax.text(j, i, f'{value:.2f}',
                          ha="center", va="center", color=text_color, 
                          fontsize=13, fontweight='bold')
    
    ax.set_title('Score Improvement (Pxplore - Baseline)', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    eval_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_profile.json")
    baseline_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_profile_baseline.json")
    annotation_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_annotation.json")

    eval_records = read_json_array(eval_path)
    baseline_records = read_json_array(baseline_path)
    annotation_records = read_json_array(annotation_path)

    # Compute per-model per-dimension averages
    models_eval = compute_model_dimension_averages(eval_records)
    models_base = compute_model_dimension_averages(baseline_records)
    
    # Add human evaluation data
    human_eval_scores = compute_human_dimension_averages(annotation_records, "pxplore")
    human_base_scores = compute_human_dimension_averages(annotation_records, "baseline")
    
    # Add human scores to the dictionaries
    models_eval["Human"] = human_eval_scores
    models_base["Human"] = human_base_scores

    # Print detailed scores table
    print_detailed_scores_table(models_eval, models_base)
    
    # out_pdf = os.path.join(repo_root, "service", "scripts", "evaluation", "result_profile.pdf")
    # make_grouped_bar_single_panel_with_baseline(models_eval, models_base, DIMENSIONS, out_pdf)
    # print(f"Saved bar chart to: {out_pdf}")
    
    # Heatmap (热力图)
    out_heatmap = os.path.join(repo_root, "service", "scripts", "evaluation", "result_profile.pdf")
    make_heatmap_comparison(models_eval, models_base, out_heatmap)
    print(f"Saved heatmap to: {out_heatmap}")


if __name__ == "__main__":
    main()


