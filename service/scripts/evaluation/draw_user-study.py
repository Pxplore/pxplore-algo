#!/usr/bin/env python3
import argparse
import os
from typing import Optional, Sequence

import matplotlib

# Match plotting style used in other evaluation scripts
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# User study results (from draw_userstudy.py)
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


def _read_excel(
    excel_path: str,
    sheet: Optional[str],
):
    import pandas as pd

    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet)

    # If a specific sheet was not provided and the file returns a dict-like (multiple sheets),
    # prefer the first sheet.
    if isinstance(df, dict):
        # Return the dict so we can use sheet names as settings later
        return df

    return df


def calculate_overall_score(results):
    """Calculate overall score as average of all dimensions"""
    return sum(results.values()) / len(results)


def draw_userstudy_bar_chart(ax):
    """Draw user study bar chart comparing Pxplore vs Retrieval across dimensions.
    
    Modified to work with a subplot axis instead of creating its own figure.
    """
    import numpy as np
    
    # Calculate overall scores
    pxplore_overall = calculate_overall_score(result_pxplore)
    baseline_overall = calculate_overall_score(result_baseline)
    
    # Prepare data with overall scores
    dimensions = list(result_pxplore.keys()) + ["Overall"]
    pxplore_scores = list(result_pxplore.values()) + [pxplore_overall]
    baseline_scores = list(result_baseline.values()) + [baseline_overall]
    
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
    ax.set_title('Learner Experience (1-5)', fontsize=12, fontweight='bold')


def draw_raincloud(
    df,
    system_col: str,
    phase_col: str,
    score_col: str,
    systems: Optional[Sequence[str]] = None,
    phases: Optional[Sequence[str]] = None,
    ax=None,
):
    """Draw a raincloud plot comparing score distributions across systems and phases.

    Modified to work with a subplot axis instead of creating its own figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Try ptitprince if available; otherwise fall back to violin + strip
    try:
        import ptitprince as pt
        use_ptit = True
    except Exception:
        use_ptit = False

    # Palette by phase (Pre-test vs Post-test)
    unique_systems = df[system_col].dropna().astype(str).unique().tolist()
    # Use project palette: Pre-test #9AC9DB, Post-test #2978B5
    phase_palette = {
        'Pre-test': '#9AC9DB',
        'Post-test': '#2978B5',
    }
    import seaborn as sns
    # Ensure palette covers any unexpected phase labels
    unique_phases = df[phase_col].dropna().astype(str).unique().tolist()
    if any(p not in phase_palette for p in unique_phases):
        extra = [p for p in unique_phases if p not in phase_palette]
        extra_colors = sns.color_palette("Set2", n_colors=len(extra))
        for p, c in zip(extra, extra_colors):
            phase_palette[p] = c

    if systems is not None:
        df = df[df[system_col].isin(list(systems))]
    if phases is not None:
        df = df[df[phase_col].isin(list(phases))]

    # Ensure categorical order if provided
    if systems is not None:
        df[system_col] = df[system_col].astype('category')
        df[system_col] = df[system_col].cat.set_categories(list(systems), ordered=True)
    if phases is not None:
        df[phase_col] = df[phase_col].astype('category')
        df[phase_col] = df[phase_col].cat.set_categories(list(phases), ordered=True)

    if use_ptit:
        # ptitprince RainCloud: combines half-violin, box, and strip
        # Orient with systems on x, scores on y, hue by phase
        hue_order = ['Pre-test', 'Post-test'] if set(['Pre-test','Post-test']).issubset(set(unique_phases)) else unique_phases
        pt.RainCloud(
            x=system_col,
            y=score_col,
            hue=phase_col,
            data=df,
            palette=[phase_palette.get(p) for p in hue_order],
            width_viol=.4,
            width_box=.15,
            move=.12,
            alpha=.8,
            ax=ax,
            orient='h' if False else 'v',
            dodge=True,
        )
    else:
        # Fallback: violinplot + stripplot
        sns.violinplot(
            data=df,
            x=system_col,
            y=score_col,
            hue=phase_col,
            palette=phase_palette,
            cut=0,
            inner=None,
            linewidth=1,
            width=0.5,
            ax=ax,
        )
        # Reduce spacing between groups
        ax.set_xlim(-0.3, len(df[system_col].unique()) - 1 + 0.3)
        sns.stripplot(
            data=df,
            x=system_col,
            y=score_col,
            hue=phase_col if not use_ptit else None,
            dodge=True,
            palette=phase_palette,
            size=2.5,
            jitter=0.15,
            alpha=0.6,
            ax=ax,
        )

    ax.set_xlabel('')
    ax.set_ylabel('Score', labelpad=1)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.margins(x=0)
    ax.set_title('Learning Outcome (%)', fontsize=12, fontweight='bold')

    # Manage legend: in fallback we may have duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        # Keep unique system labels
        unique = []
        seen = set()
        for h, l in zip(handles, labels):
            # We want legend for phases, not systems
            if l not in seen and l in df[phase_col].astype(str).unique().tolist():
                unique.append((h, l))
                seen.add(l)
        if unique:
            ax.legend(
                [u[0] for u in unique], [u[1] for u in unique], ncol=1, 
                loc='lower center'
            )
        else:
            ax.legend().remove()

    # Overlay knowledge gain (difference of means) as arrows from Pre to Post for each system
    try:
        import numpy as np
        # Only annotate if both phases are present
        if phases and ('Pre-test' in phases and 'Post-test' in phases):
            grouped = df.groupby([system_col, phase_col])[score_col].mean().unstack(fill_value=np.nan)
            # Map system label to actual x-position on the axis to avoid mismatches
            try:
                tick_positions = ax.get_xticks()
                tick_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
                x_pos_by_system = {lab: tick_positions[i] for i, lab in enumerate(tick_labels)}
            except Exception:
                x_pos_by_system = {}

            system_list = list(df[system_col].dropna().astype(str).unique())
            for i, system in enumerate(system_list):
                if system in grouped.index:
                    pre_val = float(grouped.loc[system].get('Pre-test', np.nan))
                    post_val = float(grouped.loc[system].get('Post-test', np.nan))
                    if not np.isnan(pre_val) and not np.isnan(post_val):
                        gain = post_val - pre_val
                        x_pos = x_pos_by_system.get(system, i)
                        gain_color = '#C72324'  # project red to emphasize knowledge gain
                        # Draw vertical arrow
                        ax.annotate(
                            '',
                            xy=(x_pos, post_val),
                            xytext=(x_pos, pre_val),
                            arrowprops=dict(arrowstyle='->', color=gain_color, lw=2, alpha=0.9),
                        )
                        # Endpoint markers
                        ax.plot([x_pos], [pre_val], marker='o', color=phase_palette.get('Pre-test', '#9AC9DB'), markersize=4)
                        ax.plot([x_pos], [post_val], marker='o', color=phase_palette.get('Post-test', '#2978B5'), markersize=4)
                        # Label slightly below the midpoint with percentage sign
                        mid_y = (pre_val + post_val) / 2.0
                        ax.text(x_pos + 0.05, mid_y - 3, f"+{gain:.2f}%", color=gain_color, ha='left', va='center')
    except Exception:
        pass


def draw_combined_figure(
    df,
    system_col: str,
    phase_col: str,
    score_col: str,
    systems: Optional[Sequence[str]] = None,
    phases: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
):
    """Draw both the user study bar chart and raincloud plot side by side."""
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Create figure with two subplots side by side, giving more space to the bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2), gridspec_kw={'width_ratios': [1, 2]})
    
    # Draw raincloud plot on the left
    draw_raincloud(
        df=df,
        system_col=system_col,
        phase_col=phase_col,
        score_col=score_col,
        systems=systems,
        phases=phases,
        ax=ax1,
    )
    
    # Draw user study bar chart on the right
    draw_userstudy_bar_chart(ax2)

    # Adjust layout and save
    plt.tight_layout(pad=-1)
    out_path = output_path or os.path.join(os.path.dirname(__file__), 'result_user_study.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format=os.path.splitext(out_path)[1][1:] or 'pdf', bbox_inches='tight')
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Draw combined user study and raincloud plots')
    parser.add_argument('--excel', type=str, default=None, help='Path to Excel file (default: learning_outcome.xlsx next to this script)')
    parser.add_argument('--sheet', type=str, default=None, help='Sheet name (default: first)')
    parser.add_argument('--system-col', type=str, default='System', help='Column for system name')
    parser.add_argument('--phase-col', type=str, default='Phase', help='Column for phase name (e.g., pre/post)')
    parser.add_argument('--score-col', type=str, default='Score', help='Column for numeric score')
    parser.add_argument('--systems', type=str, nargs='*', default=None, help='Filter to these systems (order preserved)')
    parser.add_argument('--phases', type=str, nargs='*', default=None, help='Filter to these phases (order preserved)')
    parser.add_argument('--out', type=str, default=None, help='Output path (default: result_user_study.pdf next to this script)')

    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    excel_path = args.excel or os.path.join(script_dir, 'learning_outcome.xlsx')

    import pandas as pd

    raw_df = _read_excel(
        excel_path=excel_path,
        sheet=args.sheet,
    )

    # Try to normalize to a long-form DataFrame with columns: System, Phase, Score
    import pandas as pd
    if isinstance(raw_df, dict):
        # Multiple sheets: use sheet names as systems if there's no explicit system column
        long_frames = []
        for sheet_name, sdf in raw_df.items():
            sdf = sdf.copy()
            # If explicit system column is present, keep it; else add one from sheet name
            if args.system_col not in sdf.columns:
                sdf[args.system_col] = sheet_name
            long_frames.append(sdf)
        df = pd.concat(long_frames, ignore_index=True)
    else:
        df = raw_df.copy()

    # If the expected columns already exist, keep them
    has_system = args.system_col in df.columns
    has_phase = args.phase_col in df.columns
    has_score = args.score_col in df.columns

    # Prefer 'Setting' as the system column if present
    if args.system_col not in df.columns and 'Setting' in df.columns:
        args.system_col = 'Setting'

    if not (has_phase and has_score):
        # Attempt to detect phase columns in wide format
        candidate_phase_cols = []
        for c in df.columns:
            cname = str(c).lower()
            if any(k in cname for k in ["pre", "post", "knowledge"]):
                candidate_phase_cols.append(c)
        if candidate_phase_cols:
            # Infer system column: prefer a text column with small unique cardinality, or index
            inferred_system_col = None
            for c in df.columns:
                if c in candidate_phase_cols:
                    continue
                series = df[c]
                if series.dtype == object and 1 <= series.nunique() <= 20:
                    inferred_system_col = c
                    break
            if inferred_system_col is None:
                # Use index as System if it looks categorical
                if df.index.dtype == object or df.index.nunique() <= 20:
                    df = df.reset_index().rename(columns={"index": "System"})
                    inferred_system_col = "System"
                else:
                    # Create a single system label
                    df["System"] = "All"
                    inferred_system_col = "System"

            # Prefer explicit Pre-test/Post-test columns if both exist
            explicit_phases = [c for c in df.columns if str(c).strip().lower() in ["pre-test", "post-test"]]
            value_vars = explicit_phases if len(explicit_phases) >= 2 else candidate_phase_cols

            long_df = df.melt(
                id_vars=[inferred_system_col],
                value_vars=value_vars,
                var_name=args.phase_col,
                value_name=args.score_col,
            )
            long_df = long_df.dropna(subset=[args.score_col])
            # Rename inferred system column to desired name if needed
            if inferred_system_col != args.system_col:
                long_df = long_df.rename(columns={inferred_system_col: args.system_col})
            df = long_df
            has_system, has_phase, has_score = True, True, True

    # If still missing required columns, raise informative error
    for col, present in [
        (args.system_col, args.system_col in df.columns),
        (args.phase_col, args.phase_col in df.columns),
        (args.score_col, args.score_col in df.columns),
    ]:
        if not present:
            raise KeyError(
                f"Column '{col}' not found and could not be inferred. Available: {list(df.columns)}"
            )

    # Standardize phases: collapse 'Post-test Part1/Part2' into 'Post-test'
    def _standardize_phase(name: str) -> str:
        n = str(name)
        low = n.lower()
        if 'pre' in low:
            return 'Pre-test'
        if 'post' in low:
            return 'Post-test'
        if 'gain' in low or 'knowledge' in low:
            return 'Knowledge gains'
        return n

    df[args.phase_col] = df[args.phase_col].apply(_standardize_phase)

    # Only compare two settings: Pre-test vs Post-test
    desired_phases = ['Pre-test', 'Post-test']
    df_plot = df[df[args.phase_col].isin(desired_phases)].copy()
    if df_plot.empty:
        raise ValueError("No data found for phases Pre-test/Post-test after standardization.")

    # Default ordering consistent with other scripts
    systems = args.systems or [s for s in pd.Series(df_plot[args.system_col]).dropna().unique()]
    # Prefer specific order if typical
    if set(["Retrieval", "Pxplore"]).issubset(set(systems)):
        systems = ["Retrieval", "Pxplore"]

    phases = args.phases or desired_phases

    out_path = draw_combined_figure(
        df=df_plot,
        system_col=args.system_col,
        phase_col=args.phase_col,
        score_col=args.score_col,
        systems=systems,
        phases=phases,
        output_path=args.out,
    )
    print(f"Saved combined user study plot to: {out_path}")

    # Print user study summary
    pxplore_overall = calculate_overall_score(result_pxplore)
    baseline_overall = calculate_overall_score(result_baseline)
    print(f"Pxplore Overall Score: {pxplore_overall:.3f}")
    print(f"Retrieval Overall Score: {baseline_overall:.3f}")

    # Print raincloud summary
    import pandas as pd
    summary_rows = []

    # Identify canonical phase names from available data
    canonical_pre = None
    canonical_post = None
    canonical_gain = None
    for phase in pd.Series(df_plot[args.phase_col].astype(str).unique()).tolist():
        low = phase.lower()
        if canonical_pre is None and 'pre' in low:
            canonical_pre = phase
        if canonical_post is None and 'post' in low and 'part' not in low:
            canonical_post = phase
        if canonical_gain is None and ('gain' in low or 'knowledge' in low):
            canonical_gain = phase

    def mean_for(system: str, phase_name: str) -> float:
        subset = df_plot[(df_plot[args.system_col] == system) & (df_plot[args.phase_col] == phase_name)]
        if subset.empty:
            return float('nan')
        return float(pd.to_numeric(subset[args.score_col], errors='coerce').mean())

    ordered_systems = systems
    # Ensure Retrieval then Pxplore if both present
    if set(["Retrieval", "Pxplore"]).issubset(set(ordered_systems)):
        ordered_systems = ["Retrieval", "Pxplore"]

    print("\nLearning Outcome Results:")
    for idx, system in enumerate(ordered_systems):
        pre_v = mean_for(system, canonical_pre) if canonical_pre else float('nan')
        post_v = mean_for(system, canonical_post) if canonical_post else float('nan')
        if canonical_gain:
            gain_v = mean_for(system, canonical_gain)
        else:
            # Derive gain as post - pre if possible
            gain_v = (post_v - pre_v) if (pre_v == pre_v and post_v == post_v) else float('nan')

        print(system)
        if pre_v == pre_v:
            print(f"Pre-test: {pre_v:.2f}")
        if post_v == post_v:
            print(f"Post-test: {post_v:.2f}")
        if gain_v == gain_v:
            print(f"Knowledge gains: {gain_v:.2f}")
        if idx != len(ordered_systems) - 1:
            print("")


if __name__ == '__main__':
    main()