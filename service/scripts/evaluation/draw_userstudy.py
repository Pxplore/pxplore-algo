
#!/usr/bin/env python3
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Set global font sizes to match draw_adaptation.py
matplotlib.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 22,
    'ytick.labelsize': 18,
    'legend.fontsize': 22,
})

result_pxplore = {
    "Help.": 4.36,
    "Div.": 4.27,
    "Rel.": 4.36,
    "Pers.": 4.18,
    "Clar.": 4.55,
    "Mot.": 4.73,
    "Und.": 4.64,
    "Sat.": 4.64
}

result_baseline = {
    "Help.": 4.36,
    "Div.": 4.18,
    "Rel.": 4,
    "Pers.": 4,
    "Clar.": 4.45,
    "Mot.": 4.64,
    "Und.": 4.64,
    "Sat.": 4.55
}

def calculate_overall_score(results):
    """Calculate overall score as average of all dimensions"""
    return sum(results.values()) / len(results)

def create_userstudy_bar_chart():
    # Calculate overall scores
    pxplore_overall = calculate_overall_score(result_pxplore)
    baseline_overall = calculate_overall_score(result_baseline)
    
    # Prepare data with overall scores
    dimensions = list(result_pxplore.keys()) + ["Overall"]
    pxplore_scores = list(result_pxplore.values()) + [pxplore_overall]
    baseline_scores = list(result_baseline.values()) + [baseline_overall]
    
    # Set up the plot
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(14, 4))
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    # Colors
    pxplore_color = "#C72324"  
    baseline_color = "#FEA983"  
    pxplore_edge = "#000" 
    baseline_edge = "#000"
    
    # Create bars
    bars1 = ax.bar(x - width/2, pxplore_scores, width, 
                   color=pxplore_color, alpha=1, 
                   edgecolor=pxplore_edge, linewidth=1.2,
                   label='Pxplore')
    bars2 = ax.bar(x + width/2, baseline_scores, width,
                   color=baseline_color, alpha=1,
                   edgecolor=baseline_edge, linewidth=1.2,
                   label='Retrieval')
    
    # Customize the plot
    ax.set_ylabel('Average Score')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.set_ylim(3.8, 4.8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')
    
    # Adjust layout and save
    fig.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "result_user_study.pdf")
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved user study bar chart to: {output_path}")
    print(f"Pxplore Overall Score: {pxplore_overall:.3f}")
    print(f"Baseline Overall Score: {baseline_overall:.3f}")

if __name__ == "__main__":
    create_userstudy_bar_chart()
