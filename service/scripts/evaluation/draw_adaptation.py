#!/usr/bin/env python3
import json
import os
from typing import Dict, List

import matplotlib
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Set global font sizes to match draw_profile.py
matplotlib.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})


def read_json_array(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Seven adaptation dimensions + Overall
DIMENSIONS = ["CC", "PE", "PS", "CE", "MA", "Overall"]

# Mapping of short dimension codes to full display names
DIMENSION_NAME_MAP = {
    "CC": "Contextual Coherence (CC)",
    "PE": "Personalization & Empathy (PE)",
    "PS": "Pedagogical Scaffolding (PS)",
    "CE": "Conciseness & Efficiency (CE)",
    "MA": "Motivational Appeal (MA)",
    "Avg": "Overall",
    "Overall": "Overall",
}

# Keys present in the JSON mapped to pretty model names
MODEL_KEYS = {
    "evaluation_gpt-4o": "GPT-4o",
    "evaluation_deepseek-r1": "DS-R1",
    "evaluation_claude-3-7-sonnet": "Claude-3.7",
    "human_evaluation": "Human",
}


def compute_model_dimension_averages(records: List[dict], setting: str) -> Dict[str, Dict[str, float]]:
    """
    Compute per-model averages for the given setting ("original" or "adaptation").

    The input records follow this structure under each model key:
      records[*][model_key][setting][dimension] = { "score": number, ... }
    """
    individual_dims = ["CC", "PE", "PS", "CE", "MA"]

    sums: Dict[str, Dict[str, float]] = {model: {d: 0.0 for d in individual_dims} for model in MODEL_KEYS.keys()}
    counts: Dict[str, Dict[str, int]] = {model: {d: 0 for d in individual_dims} for model in MODEL_KEYS.keys()}

    for rec in records:
        for raw_key, model in MODEL_KEYS.items():
            if raw_key == "human_evaluation":
                continue  # Skip human evaluation in this function
            eval_block = rec.get(raw_key) or {}
            data = (eval_block.get(setting) or {}) if isinstance(eval_block, dict) else {}
            for dim in individual_dims:
                entry = data.get(dim) or {}
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    sums[raw_key][dim] += float(score)
                    counts[raw_key][dim] += 1

    # Calculate averages for individual dimensions
    result: Dict[str, Dict[str, float]] = {}
    for raw_key, model in MODEL_KEYS.items():
        if raw_key == "human_evaluation":
            continue  # Skip human evaluation in this function
        result[model] = {
            dim: (sums[raw_key][dim] / counts[raw_key][dim] if counts[raw_key][dim] > 0 else 0.0)
            for dim in individual_dims
        }

    # Calculate overall as the mean across the dimensions
    for model in result.keys():
        individual_scores = [result[model][dim] for dim in individual_dims]
        result[model]["Overall"] = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0

    return result


def compute_human_adaptation_averages(annotation_records: List[dict], setting: str) -> Dict[str, float]:
    """Compute average scores for human evaluation of adaptation dimensions"""
    individual_dims = ["CC", "PE", "PS", "CE", "MA"]
    sums: Dict[str, float] = {dim: 0.0 for dim in individual_dims}
    counts: Dict[str, int] = {dim: 0 for dim in individual_dims}
    
    for rec in annotation_records:
        for dim in individual_dims:
            field_name = f"part_c_{setting}_{dim}"
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
    dimensions = ['CC', 'PE', 'PS', 'CE', 'MA', 'Overall']
    
    # Human scores (as reference)
    human_eval_scores = [models_eval["Human"][dim] for dim in dimensions]
    human_base_scores = [models_base["Human"][dim] for dim in dimensions]
    
    # Calculate average automated model scores
    automated_models = [model for model in models_eval.keys() if model != "Human"]
    avg_auto_eval_scores = []
    avg_auto_base_scores = [] 
    
    for dim in dimensions:
        eval_vals = [models_eval[model][dim] for model in automated_models]
        base_vals = [models_base[model][dim] for model in automated_models]
        avg_auto_eval_scores.append(np.mean(eval_vals))
        avg_auto_base_scores.append(np.mean(base_vals))
    
    print("Correlation between Human and Automated Models:")
    print(f"{'Model':<20} {'Adaptation r':>12} {'Adaptation p':>12} {'Original r':>12} {'Original p':>12}")
    print("-"*75)
    
    # Calculate correlations for each automated model
    for model in automated_models:
        model_eval_scores = [models_eval[model][dim] for dim in dimensions]
        model_base_scores = [models_base[model][dim] for dim in dimensions]
        
        # Adaptation correlation
        eval_r, eval_p = pearsonr(human_eval_scores, model_eval_scores)
        # Original correlation  
        base_r, base_p = pearsonr(human_base_scores, model_base_scores)
        
        print(f"{model:<20} {eval_r:+10.4f} {eval_p:>10.4f} {base_r:+10.4f} {base_p:>10.4f}")
    
    # Calculate correlations with average automated model
    avg_eval_r, avg_eval_p = pearsonr(human_eval_scores, avg_auto_eval_scores)
    avg_base_r, avg_base_p = pearsonr(human_base_scores, avg_auto_base_scores)
    
    print("-"*75)
    print(f"{'Average (Auto)':<20} {avg_eval_r:+10.4f} {avg_eval_p:>10.4f} {avg_base_r:+10.4f} {avg_base_p:>10.4f}")
    
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
    
    eval_df.columns = [f"{col} (Adaptation)" for col in eval_df.columns]
    base_df.columns = [f"{col} (Original)" for col in base_df.columns]
    
    # Combine dataframes side by side
    combined_df = pd.concat([eval_df, base_df], axis=1)
    
    # Round to 2 decimal places for better readability
    combined_df = combined_df.round(2)
    
    print("\nFormat: Model | Dimension (System) = Score")
    print("-"*80)
    
    # Print each model with detailed breakdown
    for model in combined_df.index:
        print(f"\n{model}:")
        for dimension in ['CC', 'PE', 'PS', 'CE', 'MA', 'Overall']:
            eval_col = f"{dimension} (Adaptation)"
            base_col = f"{dimension} (Original)"
            
            eval_score = combined_df.loc[model, eval_col]
            base_score = combined_df.loc[model, base_col]
            
            print(f"  {dimension:7s}: Adaptation={eval_score:5.2f}, Original={base_score:5.2f}")
    
    print("\n" + "="*80)
    
    # Calculate Pearson correlations
    calculate_pearson_correlations(models_eval, models_base)
    
    # Add improvement summary table
    print("\nIMPROVEMENT SUMMARY (Adaptation - Original)")
    print("-"*58)
    print(f"{'Model':<20} {'CC':>7} {'PE':>7} {'PS':>7} {'CE':>7} {'MA':>7} {'Overall':>9}")
    print("-"*58)
    
    # Calculate averages for automated models
    automated_models = []
    automated_dim_improvements = {'CC': [], 'PE': [], 'PS': [], 'CE': [], 'MA': [], 'Overall': []}
    
    for model in combined_df.index:
        improvements = []
        
        for dimension in ['CC', 'PE', 'PS', 'CE', 'MA']:
            evaluation_col = f"{dimension} (Adaptation)"
            original_col = f"{dimension} (Original)"
            
            evaluation_score = combined_df.loc[model, evaluation_col]
            original_score = combined_df.loc[model, original_col]
            improvement = evaluation_score - original_score
            improvement_pct = (improvement / original_score) * 100 if original_score != 0 else 0
            improvements.append(f"{improvement_pct:+6.2f}%")
            
            # Collect improvements for automated models average
            if model != "Human":
                automated_dim_improvements[dimension].append(improvement_pct)
        
        # Overall improvement
        evaluation_col = f"Overall (Adaptation)"
        original_col = f"Overall (Original)"
        eval_score = combined_df.loc[model, evaluation_col]
        orig_score = combined_df.loc[model, original_col]
        overall_improvement = eval_score - orig_score
        overall_improvement_pct = (overall_improvement / orig_score) * 100 if orig_score != 0 else 0
        improvements.append(f"{overall_improvement_pct:+7.2f}%")
        
        print(f"{model:<20} {improvements[0]:>7} {improvements[1]:>7} {improvements[2]:>7} {improvements[3]:>7} {improvements[4]:>7} {improvements[5]:>9}")
        
        # Collect overall improvements for automated models average
        if model != "Human":
            automated_dim_improvements['Overall'].append(overall_improvement_pct)
    
    # Calculate and print average of automated models
    avg_improvements = []
    for dimension in ['CC', 'PE', 'PS', 'CE', 'MA']:
        avg_val = sum(automated_dim_improvements[dimension]) / len(automated_dim_improvements[dimension])
        avg_improvements.append(f"{avg_val:+6.2f}%")
    
    avg_overall = sum(automated_dim_improvements['Overall']) / len(automated_dim_improvements['Overall'])
    avg_improvements.append(f"{avg_overall:+7.2f}%")
    
    print("-"*58)
    print(f"{'Average (Auto)':<20} {avg_improvements[0]:>7} {avg_improvements[1]:>7} {avg_improvements[2]:>7} {avg_improvements[3]:>7} {avg_improvements[4]:>7} {avg_improvements[5]:>9}")
    
    print("-"*58)
    print("Note: Positive values indicate Adaptation performs better")
    
    # Add model comparison summary
    print("\nMODEL COMPARISON SUMMARY")
    print("-"*40)
    
    # Extract overall improvements for comparison
    human_improvement = None
    model_improvements = {}
    
    for model in combined_df.index:
        eval_overall = combined_df.loc[model, 'Overall (Adaptation)']
        base_overall = combined_df.loc[model, 'Overall (Original)']
        improvement = eval_overall - base_overall
        
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
    print("- Overall Improvement: Adaptation - Original performance")
    print("- vs Human: How much better/worse than human improvement")
    print("- Average (Auto): Average of GPT-4o, DS-R1, Claude-3.7")
    print("\n" + "="*80)


def make_grouped_bar_original_vs_adaptation(models_adapt: Dict[str, Dict[str, float]],
                                            models_orig: Dict[str, Dict[str, float]],
                                            labels: List[str],
                                            out_path: str) -> None:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["Human", "GPT-4o", "Claude-3.7", "DS-R1"]

    # Colors
    adapt_fill_colors = {
        "Human": "#fadad5",
        "GPT-4o": "#fcd7ac",
        "DS-R1": "#b0e3e6",
        "Claude-3.7": "#d1cee3",
    }
    edge_colors = {
        "Human": "#e6c4bc",
        "GPT-4o": "#eea764",
        "DS-R1": "#8dd3d8",
        "Claude-3.7": "#b8b3d1",
    }
    orig_colors = {
        "Human": "#fef7f5",
        "GPT-4o": "#fef3e6",
        "DS-R1": "#f0fdfe",
        "Claude-3.7": "#faf9fd",
    }
    hatch_pattern = "//"

    x = np.arange(len(labels))
    group_width = 0.82
    pair_width = group_width / 4.0
    bar_width = pair_width / 2.0

    fig, ax = plt.subplots(figsize=(14, 5))

    pair_offsets = np.linspace(-group_width / 2 + pair_width / 2, group_width / 2 - pair_width / 2, len(model_order))

    for idx, model in enumerate(model_order):
        adapt_vals = [models_adapt.get(model, {}).get(dim, 0.0) for dim in labels]
        orig_vals = [models_orig.get(model, {}).get(dim, 0.0) for dim in labels]

        centers = x + pair_offsets[idx]
        # Adaptation on the left of the pair
        ax.bar(centers - bar_width / 2, adapt_vals, bar_width,
               color=adapt_fill_colors.get(model), alpha=1,
               edgecolor=edge_colors.get(model), linewidth=1.2)
        # Original on the right of the pair
        ax.bar(centers + bar_width / 2, orig_vals, bar_width,
               color=orig_colors.get(model), alpha=1,
               edgecolor=edge_colors.get(model), linewidth=1,
               hatch=hatch_pattern)

    ax.set_ylabel('Average Score')
    ax.set_xticks(x)
    # Display full dimension names on x-axis
    display_labels = [DIMENSION_NAME_MAP.get(label, label) for label in labels]
    ax.set_xticklabels(display_labels)
    ax.set_ylim(2.5, 5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.margins(x=0.02)

    # Legend - two separate legends like in draw_profile.py
    from matplotlib.patches import Patch
    model_handles = [Patch(facecolor=adapt_fill_colors[m], edgecolor=edge_colors[m], label=m) for m in model_order]
    legend1 = ax.legend(handles=model_handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    ax.add_artist(legend1)
    
    pattern_handles = [
        Patch(facecolor='gray', edgecolor='gray', label='Adaptation', alpha=0.7),
        Patch(facecolor='none', edgecolor='gray', label='Original', hatch=hatch_pattern)
    ]
    legend2 = ax.legend(handles=pattern_handles, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


def collect_raw_scores_with_human(records: List[dict], annotation_records: List[dict], 
                                  setting: str, annotation_setting: str) -> Dict[str, Dict[str, List[float]]]:
    """
    收集原始分数（包括Human数据），用于绘制小提琴图
    小提琴图
   
    """
    individual_dims = ["CC", "PE", "PS", "CE", "MA"]
    raw_data: Dict[str, Dict[str, List[float]]] = {}
    
    # 收集自动化模型的数据
    for raw_key, model in MODEL_KEYS.items():
        if raw_key == "human_evaluation":
            continue
        raw_data[model] = {dim: [] for dim in individual_dims}
    
    for rec in records:
        for raw_key, model in MODEL_KEYS.items():
            if raw_key == "human_evaluation":
                continue
            eval_block = rec.get(raw_key) or {}
            data = (eval_block.get(setting) or {}) if isinstance(eval_block, dict) else {}
            for dim in individual_dims:
                entry = data.get(dim) or {}
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    raw_data[model][dim].append(float(score))
    
    # 收集Human数据
    raw_data["Human"] = {dim: [] for dim in individual_dims}
    for rec in annotation_records:
        for dim in individual_dims:
            field_name = f"part_c_{annotation_setting}_{dim}"
            score = rec.get(field_name)
            if isinstance(score, (int, float)):
                raw_data["Human"][dim].append(float(score))
    
    # 计算Overall/Avg
    for model in raw_data.keys():
        raw_data[model]["Avg"] = []
        # 找到最小记录数
        dim_counts = [len(raw_data[model][dim]) for dim in individual_dims if raw_data[model][dim]]
        if dim_counts:
            num_records = min(dim_counts)
            for i in range(num_records):
                scores = [raw_data[model][dim][i] for dim in individual_dims if i < len(raw_data[model][dim])]
                if scores:
                    raw_data[model]["Avg"].append(sum(scores) / len(scores))
    
    return raw_data


def make_violin_plot_adaptation_vs_original(raw_adapt: Dict[str, Dict[str, List[float]]], 
                                             raw_orig: Dict[str, Dict[str, List[float]]],
                                             out_path: str) -> None:
    """创建小提琴图对比Adaptation vs Original

    """
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    dimensions = ["CC", "PE", "PS", "CE", "MA", "Avg"]
    model_order = ["Human", "GPT-4o", "Claude-3.7", "DS-R1"]
    
  
    colors = {
        "Human": "#C72324",           # 深红
        "GPT-4o": "#FEA983",          # 浅橙
        "Claude-3.7": "#9AC9DB",  # 浅蓝
        "DS-R1": "#2978B5",     # 深蓝
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 6.5))
    axes = axes.flatten()
    
    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        
        # 为每个模型准备数据
        positions = []
        all_data = []
        all_colors = []
        # 组标签与中心（每个模型一组：Adaptation vs Original）
        group_centers = []
        group_labels = []
        
        pos = 0
        for model in model_order:
            model_positions = []
            # Adaptation数据
            adapt_data = raw_adapt[model].get(dim, [])
            if adapt_data:
                all_data.append(adapt_data)
                positions.append(pos)
                all_colors.append(colors[model])
                model_positions.append(pos)
                pos += 1
            
            # Original数据
            orig_data = raw_orig[model].get(dim, [])
            if orig_data:
                all_data.append(orig_data)
                positions.append(pos)
                all_colors.append(colors[model])
                model_positions.append(pos)
                pos += 1
            
            # 本模型组的中心点与标签
            if model_positions:
                center = sum(model_positions) / len(model_positions)
                group_centers.append(center)
                group_labels.append(f"{model}")

            pos += 0.3  # 模型之间的间距
        
        # 绘制小提琴图
        if all_data:
            parts = ax.violinplot(all_data, positions=positions, widths=0.7,
                                 showmeans=True, showmedians=True)
            
            # 为每个小提琴着色
            for pc, color in zip(parts['bodies'], all_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
        
        ax.set_title(DIMENSION_NAME_MAP.get(dim, dim), fontsize=20, fontweight='bold')
        # 使用每对小提琴的中心作为一个刻度，仅显示一条合并标签
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontsize=18, rotation=0, ha='center')
        ax.set_ylim(2.5, 5)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylabel('Score', fontsize=18)
    
    fig.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def make_violin_plot_single_setting(raw_data: Dict[str, Dict[str, List[float]]], 
                                     out_path: str, 
                                     title: str) -> None:
    """创建单一设置的小提琴图（只有Adaptation或只有Original）
    
    每个维度展示4个模型的数据分布
    """
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    dimensions = ["CC", "PE", "PS", "CE", "MA", "Avg"]
    model_order = ["Human", "GPT-4o", "Claude-3.7", "DS-R1"]
    
    
    colors = {
        "Human": "#C72324",           # 深红
        "GPT-4o": "#FEA983",          # 浅橙
        "Claude-3.7": "#9AC9DB",  # 浅蓝
        "DS-R1": "#2978B5",     # 深蓝
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        
        # 准备数据
        positions = []
        all_data = []
        all_colors = []
        labels = []
        
        for i, model in enumerate(model_order):
            data = raw_data[model].get(dim, [])
            if data:
                all_data.append(data)
                positions.append(i)
                all_colors.append(colors[model])
                # 截断模型名称以节省空间
                label = model if len(model) <= 10 else model[:8] + ".."
                labels.append(label)
        
        # 绘制小提琴图
        if all_data:
            parts = ax.violinplot(all_data, positions=positions, widths=0.7,
                                 showmeans=True, showmedians=True)
            
            # 为每个小提琴着色
            for pc, color in zip(parts['bodies'], all_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.8)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.2)
        
        ax.set_title(DIMENSION_NAME_MAP.get(dim, dim), fontsize=20, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=14, rotation=30, ha='right')
        ax.set_ylim(2.5, 5)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylabel('Score', fontsize=16)
    
    fig.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_adaptation.json")
    annotation_path = os.path.join(repo_root, "service", "scripts", "buffer", "eval_annotation.json")

    records = read_json_array(data_path)
    annotation_records = read_json_array(annotation_path)

    models_original = compute_model_dimension_averages(records, setting="original")
    models_adaptation = compute_model_dimension_averages(records, setting="adaptation")
    
    # Compute human evaluation for both settings
    human_original = compute_human_adaptation_averages(annotation_records, "initial")
    human_adaptation = compute_human_adaptation_averages(annotation_records, "adapted")
    
    # Add human evaluation to models dictionaries
    models_original["Human"] = human_original
    models_adaptation["Human"] = human_adaptation
    
    # Print detailed analysis tables
    print_detailed_scores_table(models_adaptation, models_original)

    # out_pdf = os.path.join(repo_root, "service", "scripts", "evaluation", "result_adaptation.pdf")
    # make_grouped_bar_original_vs_adaptation(models_adaptation, models_original, DIMENSIONS, out_pdf)
    # print(f"\nSaved bar chart to: {out_pdf}")
    
    # 收集原始分数数据（包括Human）
    raw_adapt = collect_raw_scores_with_human(records, annotation_records, "adaptation", "adapted")
    raw_orig = collect_raw_scores_with_human(records, annotation_records, "original", "initial")
    
    # 小提琴图1 (Violin Plot) - 展示Adaptation vs Original的对比
    out_violin = os.path.join(repo_root, "service", "scripts", "evaluation", "result_adaptation.pdf")
    make_violin_plot_adaptation_vs_original(raw_adapt, raw_orig, out_violin)
    print(f"Saved violin plot (Adaptation vs Original) to: {out_violin}")
    
    # 小提琴图2 - 只展示Adaptation
    # out_violin_adapt = os.path.join(repo_root, "service", "scripts", "evaluation", "result_adaptation_violin_adapt_only.pdf")
    # make_violin_plot_single_setting(raw_adapt, out_violin_adapt, "Adaptation Score Distribution")
    # print(f"Saved violin plot (Adaptation only) to: {out_violin_adapt}")
    
    # 小提琴图3 - 只展示Original
    # out_violin_orig = os.path.join(repo_root, "service", "scripts", "evaluation", "result_adaptation_violin_original_only.pdf")
    # make_violin_plot_single_setting(raw_orig, out_violin_orig, "Original Score Distribution")
    # print(f"Saved violin plot (Original only) to: {out_violin_orig}")


if __name__ == "__main__":
    main()


