import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

def krippendorff_alpha(data, level_of_measurement='interval'):
    """
    Calculate Krippendorff's alpha for inter-annotator agreement.
    Adapted from implementation for interval/ratio data.
    """
    # Convert data to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Remove missing values (represented as NaN)
    mask = ~np.isnan(data)
    clean_data = data.copy()
    clean_data[~mask] = 0
    
    n_items, n_raters = clean_data.shape
    
    if n_items < 2 or n_raters < 2:
        return np.nan
    
    # Calculate observed disagreement
    observed_disagreement = 0
    total_pairs = 0
    
    for i in range(n_items):
        for j in range(n_raters):
            if mask[i, j]:  # Only count valid ratings
                for k in range(n_items):
                    for l in range(n_raters):
                        if mask[k, l] and (i != k or j != l):  # Different items or raters
                            observed_disagreement += (clean_data[i, j] - clean_data[k, l]) ** 2
                            total_pairs += 1
    
    if total_pairs == 0:
        return np.nan
    
    observed_disagreement /= total_pairs
    
    # Calculate expected disagreement
    expected_disagreement = np.var(clean_data[mask])
    
    if expected_disagreement == 0:
        return 1.0  # Perfect agreement
    
    alpha = 1 - (observed_disagreement / expected_disagreement)
    return alpha

class AnnotationEvaluator:
    """
    A class to analyze annotation evaluation data from multiple annotators.
    Includes functionality for calculating averages, correlations, and generating reports.
    """
    
    def __init__(self, json_path):
        """Initialize the evaluator with annotation data."""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.annotators = self._get_annotators()
        self.df = self._flatten_data()
    
    def _get_annotators(self):
        """Extract unique annotators from the data."""
        annotators = set()
        for record in self.data:
            annotators.add(record['annotator'])
        return sorted(list(annotators))
    
    def _flatten_data(self):
        """Convert JSON array data to a pandas DataFrame for analysis."""
        rows = []
        
        for record in self.data:
            annotator = record['annotator']
            record_id = record['record_id']
            student_name = record['student_name']
            
            # Process Part A data (part_a_pxplore and part_a_baseline)
            for module_type in ['part_a_pxplore', 'part_a_baseline']:
                if module_type == 'part_a_pxplore':
                    module_name = "pxplore画像模块评估"
                    dimensions = ['IIA', 'OPC', 'AU', 'IE']
                else:
                    module_name = "baseline画像模块评估"
                    dimensions = ['IIA', 'OPC', 'AU', 'IE']
                
                # Extract dimension ratings for this module
                for dimension_id in dimensions:
                    column_name = f"{module_type}_{dimension_id}"
                    if column_name in record:
                        raw_score = record[column_name]
                        # Convert raw score (1-5) to percentage (20%-100%)
                        percentage_score = raw_score * 20
                        
                        rows.append({
                            'record_id': record_id,
                            'student_name': student_name,
                            'annotator': annotator,
                            'module_type': module_type,
                            'module_name': module_name,
                            'dimension_id': dimension_id,
                            'dimension_name': self._get_dimension_name(dimension_id),
                            'raw_score': raw_score,
                            'percentage_score': percentage_score
                        })
            
            # Process Part C data (part_c_initial and part_c_adapted)
            for module_type in ['part_c_initial', 'part_c_adapted']:
                if module_type == 'part_c_initial':
                    module_name = "initial画像模块评估"
                    dimensions = ['CC', 'PE', 'PS', 'CE', 'MA']
                else:
                    module_name = "adapted画像模块评估"
                    dimensions = ['CC', 'PE', 'PS', 'CE', 'MA']
                
                # Extract dimension ratings for this module
                for dimension_id in dimensions:
                    column_name = f"{module_type}_{dimension_id}"
                    if column_name in record:
                        raw_score = record[column_name]
                        # Convert raw score (1-5) to percentage (20%-100%)
                        percentage_score = raw_score * 20
                        
                        rows.append({
                            'record_id': record_id,
                            'student_name': student_name,
                            'annotator': annotator,
                            'module_type': module_type,
                            'module_name': module_name,
                            'dimension_id': dimension_id,
                            'dimension_name': self._get_part_c_dimension_name(dimension_id),
                            'raw_score': raw_score,
                            'percentage_score': percentage_score
                        })
        
        return pd.DataFrame(rows)
    
    def _get_dimension_name(self, dimension_id):
        """Get full dimension name for Part A."""
        dimension_names = {
            'IIA': 'Interest Identification Accuracy (IIA) - 兴趣识别准确性',
            'OPC': 'Overall Profile Completeness (OPC) - 整体画像完整性', 
            'AU': 'Actionability / Utility (AU) - 可操作性 / 实用性',
            'IE': 'Interpretability & Explainability (IE) - 可解释性与说明性'
        }
        return dimension_names.get(dimension_id, dimension_id)
    
    def _get_part_c_dimension_name(self, dimension_id):
        """Get full dimension name for Part C."""
        dimension_names = {
            'CC': 'Content Coverage (CC) - 内容覆盖',
            'PE': 'Pedagogical Effectiveness (PE) - 教学效果',
            'PS': 'Presentation Style (PS) - 展示风格',
            'CE': 'Cultural Engagement (CE) - 文化参与',
            'MA': 'Motivational Appeal (MA) - 激励吸引'
        }
        return dimension_names.get(dimension_id, dimension_id)
    
    def calculate_average_scores(self):
        """Calculate comprehensive average scores across metrics and annotators."""
        print("=" * 80)
        print("PART A PROFILE MODULE ANALYSIS")
        print("=" * 80)
        
        # Overall averages across all annotators
        overall_stats = self.df.groupby(['module_type', 'dimension_id']).agg({
            'raw_score': ['mean', 'std', 'min', 'max'],
            'percentage_score': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        print("\n1. Overall Averages Across All Annotators:")
        
        # Flatten column names for easier reading
        overall_stats.columns = ['_'.join(col).strip() for col in overall_stats.columns]
        
        # Sort by module and dimension (only part_a)
        module_order = ['part_a_pxplore', 'part_a_baseline']
        dimension_order = ['IIA', 'OPC', 'AU', 'IE']
        
        for module_type in module_order:
            if module_type in overall_stats.index.get_level_values('module_type'):
                print(f"\n{module_type.upper()}:")
                module_data = overall_stats.loc[module_type]
                for dim_id in dimension_order:
                    if dim_id in module_data.index:
                        row = module_data.loc[dim_id]
                        dim_name = self.df[self.df['dimension_id'] == dim_id]['dimension_name'].iloc[0]
                        print(f"  {dim_id} ({dim_name}):")
                        print(f"    Raw Score: {row['raw_score_mean']:.2f} ± {row['raw_score_std']:.2f} (Range: {row['raw_score_min']}-{row['raw_score_max']})")
        
        return overall_stats
    
    def calculate_correlations(self):
        """
        Calculate Pearson correlation, Spearman correlation, and Krippendorf's alpha 
        between ratings from different annotators.
        """
        
        # Create pivot table for correlation analysis
        pivot_raw = self.df.pivot_table(
            index=['module_type', 'dimension_id'], 
            columns='annotator', 
            values='raw_score', 
            aggfunc='mean'
        ).dropna()
        
        pivot_percentage = self.df.pivot_table(
            index=['module_type', 'dimension_id'], 
            columns='annotator', 
            values='percentage_score', 
            aggfunc='mean'
        ).dropna()
        
        if len(self.annotators) >= 2:
            annotator_combinations = list(combinations(self.annotators, 2))
            
            # Raw scores
            corr_matrix_raw = pivot_raw.corr()
            
            # Percentage scores
            corr_matrix_pct = pivot_percentage.corr()
            
            for ann1, ann2 in annotator_combinations:
                if ann1 in corr_matrix_pct.index and ann2 in corr_matrix_pct.columns:
                    p_raw = corr_matrix_raw.loc[ann1, ann2]
                    p_pct = corr_matrix_pct.loc[ann1, ann2]
            
            # Calculate Spearman correlations
            spearman_raw_matrix = pivot_raw.corr(method='spearman')
            spearman_pct_matrix = pivot_percentage.corr(method='spearman')

            for ann1, ann2 in annotator_combinations:
                if ann1 in spearman_pct_matrix.index and ann2 in spearman_pct_matrix.columns:
                    s_raw = spearman_raw_matrix.loc[ann1, ann2]
                    s_pct = spearman_pct_matrix.loc[ann1, ann2]
            
            # Calculate Krippendorff's alpha
            alpha_raw = krippendorff_alpha(pivot_raw)
            alpha_pct = krippendorff_alpha(pivot_percentage)

            # Module-specific alphas
            for module_type in self.df['module_type'].unique():
                module_data = self.df[self.df['module_type'] == module_type]
                module_pivot = module_data.pivot_table(
                    index='dimension_id', 
                    columns='annotator', 
                    values='percentage_score', 
                    aggfunc='mean'
                ).dropna()
                
                if module_pivot.shape[0] > 1 and module_pivot.shape[1] > 1:
                    module_alpha = krippendorff_alpha(module_pivot)
                    module_name = module_data['module_name'].iloc[0]
        
        # Module-specific correlations
        
        for module_type in self.df['module_type'].unique():
            module_data = self.df[self.df['module_type'] == module_type]
            module_pivot = module_data.pivot_table(
                index='dimension_id', 
                columns='annotator', 
                values='percentage_score', 
                aggfunc='mean'
            ).dropna()
            
            if len(self.annotators) >= 2 and module_pivot.shape[0] > 1:
                module_corr_pearson = module_pivot.corr()
                module_corr_spearman = module_pivot.corr(method='spearman')
                module_name = module_data['module_name'].iloc[0]
                
                for ann1, ann2 in annotator_combinations:
                    if ann1 in module_corr_pearson.index and ann2 in module_corr_pearson.columns:
                        pearson_val = module_corr_pearson.loc[ann1, ann2]
                        spearman_val = module_corr_spearman.loc[ann1, ann2]
        
        return corr_matrix_pct if len(self.annotators) >= 2 else None

    def create_detailed_comparison_table(self):
        """Create detailed comparison tables showing annotator results."""
        print("\n" + "=" * 100)
        print("DETAILED ANNOTATOR COMPARISON TABLES")
        print("=" * 100)
        
        # Create comprehensive comparison tables
        dimensions = ['IIA', 'OPC', 'AU', 'IE']
        modules = ['part_a_pxplore', 'part_a_baseline']
        
        # Overall averages table
        print("\n1. OVERALL AVERAGES BY MODULE AND ANNOTATOR")
        print("-" * 80)
        
        comparison_data = []
        for module_type in modules:
            module_name = "Pxplore Module" if module_type == 'part_a_pxplore' else "Baseline Module"
            for dimension in dimensions:
                dim_data = self.df[(self.df['dimension_id'] == dimension) & (self.df['module_type'] == module_type)]
                
                # Calculate by annotator
                annotator_stats = {}
                for annotator in ['yaoxd25', '郑金山']:
                    ann_data = dim_data[dim_data['annotator'] == annotator]
                    if len(ann_data) > 0:
                        annotator_stats[annotator] = {
                            'mean_raw': ann_data['raw_score'].mean(),
                            'std_raw': ann_data['raw_score'].std(),
                            'mean_%': ann_data['percentage_score'].mean(),
                            'std_%': ann_data['percentage_score'].std(),
                            'min': ann_data['raw_score'].min(),
                            'max': ann_data['raw_score'].max(),
                            'count': len(ann_data)
                        }
                    else:
                        annotator_stats[annotator] = {
                            'mean_raw': 0, 'std_raw': 0, 'mean_%': 0, 'std_%': 0,
                            'min': 0, 'max': 0, 'count': 0
                        }
                
                comparison_data.append({
                    'module': module_name,
                    'dimension': dimension,
                    'yaoxd25_raw': f"{annotator_stats['yaoxd25']['mean_raw']:.2f} ± {annotator_stats['yaoxd25']['std_raw']:.2f}",
                    'yaoxd25_pct': f"{annotator_stats['yaoxd25']['mean_%']:.1f}%",
                    'jin_raw': f"{annotator_stats['郑金山']['mean_raw']:.2f} ± {annotator_stats['郑金山']['std_raw']:.2f}",
                    'jin_pct': f"{annotator_stats['郑金山']['mean_%']:.1f}%",
                    'diff_raw': f"{annotator_stats['郑金山']['mean_raw'] - annotator_stats['yaoxd25']['mean_raw']:+.2f}",
                    'yaoxd25_n': annotator_stats['yaoxd25']['count'],
                    'jin_n': annotator_stats['郑金山']['count']
                })
        
        # Print formatted table
        print(f"{'Module':<15} {'Dimension':<4} {'Ann1_Raw':<12} {'Ann1_%':<8} {'Ann2_%':<8} {'Diff':<6} {'N1':<3} {'N2':<3}")
        print("-" * 100)
        
        for row in comparison_data:
            print(f"{row['module']:<15} {row['dimension']:<4} "
                  f"{row['yaoxd25_raw']:<12} {row['yaoxd25_pct']:<8} "
                  f"{row['jin_pct']:<8} "
                  f"{row['diff_raw']:<6} {row['yaoxd25_n']:<3} {row['jin_n']:<3}")
        
        # Detailed breakdown by dimension
        print("\n\n2. DETAILED BREAKDOWN BY DIMENSION")
        print("=" * 80)
        
        for dimension in dimensions:
            dim_name = self._get_dimension_name(dimension)
            print(f"\n{dimension} ({dim_name}):")
            print("-" * 60)
            
            dim_data = self.df[self.df['dimension_id'] == dimension]
            
            for module_type in modules:
                module_name = "Pxplore" if module_type == 'part_a_pxplore' else "Baseline"
                module_data = dim_data[dim_data['module_type'] == module_type]
                
                print(f"\n{module_name} Module:")
                
                for annotator in ['yaoxd25', '郑金山']:
                    ann_data = module_data[module_data['annotator'] == annotator]
                    if len(ann_data) > 0:
                        ann_raw_mean = ann_data['raw_score'].mean()
                        ann_pct_mean = ann_data['percentage_score'].mean()
                        ann_std = ann_data['raw_score'].std()
                        ann_range = f"{ann_data['raw_score'].min()}-{ann_data['raw_score'].max()}"
                        ann_count = len(ann_data)
                        
                        print(f"  {annotator:<8}: {ann_raw_mean:.2f} ± {ann_std:.2f} "
                              f"({ann_range}) = {ann_pct_mean:.1f}% (n={ann_count})")
                    else:
                        print(f"  {annotator:<8}: No data")
        
        # Agreement analysis
        print("\n\n3. ANNOTATOR AGREEMENT ANALYSIS")
        print("=" * 60)
        
        print(f"{'Dimension':<4} {'n_Pairs':<7} {'Pearson':<8} {'p-val':<8} {'Spearman':<9} {'p-val':<8} {'Agreement':<15}")
        print("-" * 65)
        
        # Recalculate correlations for summary
        for dimension in dimensions:
            dim_data = self.df[self.df['dimension_id'] == dimension]
            dim_pivot = dim_data.pivot_table(
                index='record_id',
                columns='annotator',
                values='percentage_score',
                aggfunc='mean'
            )
            
            valid_annotators = [col for col in dim_pivot.columns if col.strip()]
            
            if len(valid_annotators) == 2:
                annotator1, annotator2 = valid_annotators[0], valid_annotators[1]
                valid_pivot = dim_pivot[valid_annotators]
                valid_pairs = valid_pivot.dropna()
                
                if len(valid_pairs) >= 3:
                    values1 = valid_pairs[annotator1].values
                    values2 = valid_pairs[annotator2].values
                    
                    pearson_corr, pearson_p = pearsonr(values1, values2)
                    spearman_corr, spearman_p = spearmanr(values1, values2)
                    
                    # Determine agreement level
                    if abs(pearson_corr) >= 0.7:
                        agreement = "Excellent"
                    elif abs(pearson_corr) >= 0.5:
                        agreement = "Good"
                    elif abs(pearson_corr) >= 0.3:
                        agreement = "Moderate"
                    elif abs(pearson_corr) >= 0.1:
                        agreement = "Poor"
                    else:
                        agreement = "Very Poor"
                    
                    sig_p = "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
                    
                    print(f"{dimension:<4} {len(valid_pairs):<7} {pearson_corr:<7.3f} "
                          f"{pearson_p:<7.3f}{sig_p:<1} {spearman_corr:<8.3f} "
                          f"{spearman_p:<7.3f}{sig_p:<1} {agreement:<15}")
        
        print("\nLegend: ** p<0.01, * p<0.05")
        
        return comparison_data

    def calculate_dimension_annotator_correlations(self):
        """Calculate correlations between annotators for each individual dimension."""
        
        # Calculate correlations for each dimension across annotators
        dimensions = ['AU', 'IE', 'IIA', 'OPC']
        
        correlation_summary = []
        
        for dimension in dimensions:
            # For each dimension, compare annotator ratings across all samples
            dim_data = self.df[self.df['dimension_id'] == dimension]
            
            # Create pivot table with record_id and annotator  
            dim_pivot = dim_data.pivot_table(
                index='record_id',
                columns='annotator',
                values='percentage_score',
                aggfunc='mean'
            )
            
            # Filter out annotators with empty names (just spaces)
            valid_annotators = [col for col in dim_pivot.columns if col.strip()]
            
            if len(valid_annotators) == 2:  # Two valid annotators
                annotator1 = valid_annotators[0]
                annotator2 = valid_annotators[1]
                
                # Get values for correlation calculation (drop NaN pairs)
                # Only use the valid annotators
                valid_pivot = dim_pivot[valid_annotators]
                valid_pairs = valid_pivot.dropna()
                
                if len(valid_pairs) >= 3:  # Need at least 3 pairs for meaningful correlation
                    values1 = valid_pairs[annotator1].values
                    values2 = valid_pairs[annotator2].values
                    
                    # Calculate Pearson and Spearman correlations
                    pearson_corr, pearson_p = pearsonr(values1, values2)
                    spearman_corr, spearman_p = spearmanr(values1, values2)
                    
                    # Get dimension name
                    dim_name = dim_data['dimension_name'].iloc[0]
                    
                    for module_type in ['part_a_pxplore', 'part_a_baseline']:
                        module_pivot = dim_data[dim_data['module_type'] == module_type].pivot_table(
                            index='record_id',
                            columns='annotator', 
                            values='percentage_score',
                            aggfunc='mean'
                        )
                        module_valid = module_pivot.dropna()
                        if len(module_valid) >= 2:
                            mod_values1 = module_valid.iloc[:, 0].values
                            mod_values2 = module_valid.iloc[:, 1].values
                            mod_pearlson = pearsonr(mod_values1, mod_values2)[0]
                    
                    correlation_summary.append({
                        'dimension': dimension,
                        'pearson': pearson_corr,
                        'spearman': spearman_corr,
                        'p_pearson': pearson_p,
                        'p_spearman': spearman_p,
                        'n_samples': len(valid_pairs)
                    })
                else:
                    print(f"\n{dimension}: Insufficient data pairs ({len(valid_pairs)} < 3)")
            else:
                print(f"\n{dimension}: Cannot calculate correlation - need exactly 2 valid annotators, found {len(valid_annotators)}")
        
        # Summary table
        print(f"\n" + "=" * 70)
        print(f"DIMENSION CORRELATION SUMMARY")
        print(f"=" * 70)
        
        print(f"{'Dimension':<10} {'Pearson r':<10} {'p-value':<10} {'Spearman ρ':<10} {'p-value':<10} {'Samples':<8}")
        print(f"-" * 70)
        for item in correlation_summary:
            print(f"{item['dimension']:<10} {item['pearson']:<10.3f} {item['p_pearson']:<10.3f} {item['spearman']:<10.3f} {item['p_spearman']:<10.3f} {item['n_samples']:<8}")
        
        # Overall summary (average of all dimensions)
        if correlation_summary:
            overall_pearson = sum(item['pearson'] for item in correlation_summary) / len(correlation_summary)
            overall_spearman = sum(item['spearman'] for item in correlation_summary) / len(correlation_summary)
            avg_samples = sum(item['n_samples'] for item in correlation_summary) / len(correlation_summary)
            
            print(f"-" * 70)
            print(f"{'OVERALL':<10} {overall_pearson:<10.3f} {'':<10} {overall_spearman:<10.3f} {'':<10} {avg_samples:<8.0f}")
        
        # PXPLORE vs BASELINE COMPARISON TABLE
        print(f"\n" + "=" * 90)
        print(f"PXPLORE vs BASELINE MODULE COMPARISON TABLE")
        print(f"=" * 90)
        
        print(f"{'Dimension':<25} {'Pxplore (Raw±SD)':<20} {'Baseline (Raw±SD)':<20} {'Difference':<12} {'Winner':<8}")
        print(f"-" * 90)
        
        # Calculate comparison data
        dimensions = ['IIA', 'OPC', 'AU', 'IE']
        comparison_results = []
        
        for dimension in dimensions:
            dimension_stats = self.df.groupby(['module_type', 'dimension_id']).agg({
                'raw_score': ['mean', 'std']
            }).round(2)
            
            # Extract pxplore data
            pxplore_data = dimension_stats.loc[('part_a_pxplore', dimension), ('raw_score', 'mean')]
            pxplore_std = dimension_stats.loc[('part_a_pxplore', dimension), ('raw_score', 'std')]
            
            # Extract baseline data  
            baseline_data = dimension_stats.loc[('part_a_baseline', dimension), ('raw_score', 'mean')]
            baseline_std = dimension_stats.loc[('part_a_baseline', dimension), ('raw_score', 'std')]
            
            # Calculate difference
            difference = pxplore_data - baseline_data
            
            # Determine winner
            if difference > 0:
                winner = "Pxplore"
            elif difference < 0:
                winner = "Baseline"
            else:
                winner = "Tie"
            
            comparison_results.append({
                'dimension': dimension,
                'pxplore': f"{pxplore_data:.2f} ± {pxplore_std:.2f}",
                'baseline': f"{baseline_data:.2f} ± {baseline_std:.2f}",
                'difference': f"{difference:+.2f}",
                'winner': winner
            })
        
        # Print the table rows
        for result in comparison_results:
            print(f"{result['dimension']:<25} {result['pxplore']:<20} {result['baseline']:<20} "
                  f"{result['difference']:<12} {result['winner']:<8}")
        
        # Summary statistics
        pxplore_total = dimension_stats.xs('part_a_pxplore', level='module_type')[('raw_score', 'mean')].mean()
        baseline_total = dimension_stats.xs('part_a_baseline', level='module_type')[('raw_score', 'mean')].mean()
        overall_diff = pxplore_total - baseline_total
        
        print(f"-" * 90)
        print(f"{'OVERALL AVERAGE':<25} {pxplore_total:.2f} ± {'TBD':<17} {baseline_total:.2f} ± {'TBD':<17} "
                f"{overall_diff:+.2f} {'Pxplore' if overall_diff > 0 else 'Baseline' if overall_diff < 0 else 'Tie':<8}")
        
        # PART C COMPARISON TABLE (Initial vs Adapted)
        print(f"\n" + "=" * 90)
        print(f"PART C: INITIAL vs ADAPTED MODULE COMPARISON TABLE")
        print(f"=" * 90)
        
        print(f"{'Dimension':<30} {'Initial (Raw±SD)':<18} {'Adapted (Raw±SD)':<18} {'Difference':<12} {'Winner':<8}")
        print(f"-" * 90)
        
        # Calculate Part C comparison data
        part_c_dimensions = ['CC', 'PE', 'PS', 'CE', 'MA']
        part_c_results = []
        
        for dimension in part_c_dimensions:
            part_c_data = self.df[self.df['dimension_id'] == dimension]
            
            if len(part_c_data) > 0:
                part_c_stats = part_c_data.groupby(['module_type']).agg({
                    'raw_score': ['mean', 'std', 'count']
                }).round(2)
                
                # Extract Initial data
                try:
                    initial_data = part_c_stats.loc['part_c_initial', ('raw_score', 'mean')]
                    initial_std = part_c_stats.loc['part_c_initial', ('raw_score', 'std')]
                    initial_count = part_c_stats.loc['part_c_initial', ('raw_score', 'count')]
                except KeyError:
                    initial_data = 0
                    initial_std = 0
                    initial_count = 0
                
                # Extract Adapted data  
                try:
                    adapted_data = part_c_stats.loc['part_c_adapted', ('raw_score', 'mean')]
                    adapted_std = part_c_stats.loc['part_c_adapted', ('raw_score', 'std')]
                    adapted_count = part_c_stats.loc['part_c_adapted', ('raw_score', 'count')]
                except KeyError:
                    adapted_data = 0
                    adapted_std = 0
                    adapted_count = 0
                
                # Calculate difference
                difference = adapted_data - initial_data
                
                # Determine winner
                if difference > 0:
                    winner = "Adapted"
                elif difference < 0:
                    winner = "Initial"
                else:
                    winner = "Tie"
                
                part_c_results.append({
                    'dimension': dimension,
                    'dimension_name': self._get_part_c_dimension_name(dimension),
                    'initial': f"{initial_data:.2f} ± {initial_std:.2f}",
                    'adapted': f"{adapted_data:.2f} ± {adapted_std:.2f}",
                    'difference': f"{difference:+.2f}",
                    'winner': winner
                })
                
                # Print dimension row
                dim_name = self._get_part_c_dimension_name(dimension)[:26]  # Truncate long names
                print(f"{dim_name + ' (' + dimension + ')':<30} {initial_data:.2f} ± {initial_std:.2f} {'':11} "
                      f"{adapted_data:.2f} ± {adapted_std:.2f} {'':10} "
                      f"{difference:+.2f} {'':9} {winner:<8}")
        
        # Summary statistics for Part C
        if part_c_results:
            initial_values = []
            adapted_values = []
            for item in part_c_results:
                initial_values.append(float(item['initial'].split(' ')[0]))
                adapted_values.append(float(item['adapted'].split(' ')[0]))
            
            initial_total = sum(initial_values) / len(initial_values)
            adapted_total = sum(adapted_values) / len(adapted_values)
            overall_diff_c = adapted_total - initial_total
            
            print(f"-" * 90)
            print(f"{'OVERALL AVERAGE':<30} {initial_total:.2f} {'':14} {adapted_total:.2f} {'':14} "
                  f"{overall_diff_c:+.2f} {'':9} {'Adapted' if overall_diff_c > 0 else 'Initial' if overall_diff_c < 0 else 'Tie':<8}")
        
        return comparison_results
    
    def calculate_dimension_correlations(self):
        """Calculate correlations across different dimensions within Part A."""
        print("\n" + "=" * 80)
        print("CROSS-DIMENSION CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Create pivot table for dimension correlation analysis
        # Each dimension becomes a variable, annotations are observations
        dimension_pivot = self.df.pivot_table(
            index=['annotator', 'module_type'], 
            columns='dimension_id', 
            values='percentage_score', 
            aggfunc='mean'
        )
        
        print("\nDimension Data Matrix:")
        print("-" * 50)
        print(dimension_pivot.round(2))
        
        if len(dimension_pivot.columns) >= 2:
            print("\n1. PEARSON CORRELATIONS BETWEEN DIMENSIONS:")
            print("-" * 50)
            corr_matrix = dimension_pivot.corr()
            print(corr_matrix.round(3))
            
            print("\n2. SPEARMAN CORRELATIONS BETWEEN DIMENSIONS:")
            print("-" * 50)
            spearman_corr_matrix = dimension_pivot.corr(method='spearman')
            print(spearman_corr_matrix.round(3))
            
            print("\n3. DIMENSION CORRELATION ANALYSIS BY MODULE:")
            print("-" * 50)
            
            # Module-specific dimension correlations
            for module_type in self.df['module_type'].unique():
                module_data = self.df[self.df['module_type'] == module_type]
                module_dimension_pivot = module_data.pivot_table(
                    index='annotator', 
                    columns='dimension_id', 
                    values='percentage_score', 
                    aggfunc='mean'
                )
                
                if len(module_dimension_pivot.columns) >= 2:
                    module_name = module_data['module_name'].iloc[0]
                    print(f"\n{module_name}:")
                    
                    module_corr_pearson = module_dimension_pivot.corr()
                    module_corr_spearman = module_dimension_pivot.corr(method='spearman')
                    
                    print("Pearson Correlations:")
                    print(module_corr_pearson.round(3))
                    print("\nSpearman Correlations:")
                    print(module_corr_spearman.round(3))
            
            print("\n4. STRONGEST CORRELATIONS:")
            print("-" * 50)
            
            # Find strongest positive and negative correlations
            corr_values = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    dim1 = corr_matrix.columns[i]
                    dim2 = corr_matrix.columns[j]
                    pearson_val = corr_matrix.iloc[i, j]
                    spearman_val = spearman_corr_matrix.iloc[i, j]
                    corr_values.append({
                        'dim1': dim1, 'dim2': dim2,
                        'pearson': pearson_val, 'spearman': spearman_val
                    })
            
            # Sort by absolute Pearson correlation
            corr_values.sort(key=lambda x: abs(x['pearson']), reverse=True)
            
            print("Top correlations by absolute Pearson value:")
            for item in corr_values[:3]:  # Top 3
                print(f"  {item['dim1']} ↔ {item['dim2']}:")
                print(f"    Pearson r={item['pearson']:.3f}, Spearman ρ={item['spearman']:.3f}")
            
            # Strongest negative correlations
            negative_corrs = [item for item in corr_values if item['pearson'] < -0.5]
            if negative_corrs:
                print(f"\nStrong negative correlations (>-0.5):")
                for item in negative_corrs:
                    print(f"  {item['dim1']} ↔ {item['dim2']}: Pearson r={item['pearson']:.3f}")
            
            # Strongest positive correlations  
            positive_corrs = [item for item in corr_values if item['pearson'] > 0.5]
            if positive_corrs:
                print(f"\nStrong positive correlations (>0.5):")
                for item in positive_corrs:
                    print(f"  {item['dim1']} ↔ {item['dim2']}: Pearson r={item['pearson']:.3f}")
        
        elif len(dimension_pivot.columns) == 1:
            print("Only one dimension available - cannot calculate correlations between dimensions.")
        
        return dimension_pivot
    
    def calculate_part_c_average_scores(self):
        """Calculate comprehensive average scores for Part C data across metrics and annotators."""
        print("=" * 80)
        print("PART C PROFILE MODULE ANALYSIS")
        print("=" * 80)
        
        # Filter for Part C data
        part_c_df = self.df[self.df['module_type'].isin(['part_c_initial', 'part_c_adapted'])]
        
        # Overall averages across all annotators
        overall_stats = part_c_df.groupby(['module_type', 'dimension_id']).agg({
            'raw_score': ['mean', 'std', 'min', 'max'],
            'percentage_score': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        print("\n1. Overall Averages Across All Annotators:")
        print("-" * 50)
        
        # Flatten column names for easier reading
        overall_stats.columns = ['_'.join(col).strip() for col in overall_stats.columns]
        
        # Sort by module and dimension
        module_order = ['part_c_initial', 'part_c_adapted']
        dimension_order = ['CC', 'PE', 'PS', 'CE', 'MA']
        
        for module_type in module_order:
            if module_type in overall_stats.index.get_level_values('module_type'):
                print(f"\n{module_type.upper()}:")
                module_data = overall_stats.loc[module_type]
                for dim_id in dimension_order:
                    if dim_id in module_data.index:
                        row = module_data.loc[dim_id]
                        dim_name = part_c_df[part_c_df['dimension_id'] == dim_id]['dimension_name'].iloc[0]
                        print(f"  {dim_id} ({dim_name}):")
                        print(f"    Raw Score: {row['raw_score_mean']:.2f} ± {row['raw_score_std']:.2f} (Range: {row['raw_score_min']}-{row['raw_score_max']})")
                        print(f"    Percentage: {row['percentage_score_mean']:.2f}% ± {row['percentage_score_std']:.2f}%")
        
        # Dimension correlation summary for Part C
        print("\n2. PART C DIMENSION CORRELATION SUMMARY")
        print("=" * 70)
        
        print(f"{'Dimension':<10} {'Pearson r':<10} {'p-value':<10} {'Spearman ρ':<10} {'p-value':<10} {'Samples':<8}")
        print(f"-" * 70)
        
        part_c_dimensions = ['CC', 'PE', 'PS', 'CE', 'MA']
        part_c_correlation_summary = []
        
        for dimension in part_c_dimensions:
            # For each dimension, compare annotator ratings across all samples
            dim_data = self.df[self.df['dimension_id'] == dimension]
            dim_pivot = dim_data.pivot_table(
                index='record_id',
                columns='annotator',
                values='percentage_score',
                aggfunc='mean'
            )
            
            # Filter out annotators with empty names (just spaces)
            valid_annotators = [col for col in dim_pivot.columns if col.strip()]
            
            if len(valid_annotators) == 2:  # Two valid annotators
                annotator1, annotator2 = valid_annotators[0], valid_annotators[1]
                valid_pivot = dim_pivot[valid_annotators]
                valid_pairs = valid_pivot.dropna()
                
                if len(valid_pairs) >= 3:  # Need at least 3 pairs for meaningful correlation
                    values1 = valid_pairs[annotator1].values
                    values2 = valid_pairs[annotator2].values
                    
                    # Calculate Pearson and Spearman correlations
                    pearson_corr, pearson_p = pearsonr(values1, values2)
                    spearman_corr, spearman_p = spearmanr(values1, values2)
                    
                    part_c_correlation_summary.append({
                        'dimension': dimension,
                        'pearson': pearson_corr,
                        'spearman': spearman_corr,
                        'p_pearson': pearson_p,
                        'p_spearman': spearman_p,
                        'n_samples': len(valid_pairs)
                    })
                    
                    print(f"{dimension:<10} {pearson_corr:<10.3f} {pearson_p:<10.3f} {spearman_corr:<10.3f} {spearman_p:<10.3f} {len(valid_pairs):<8}")
        
        # Overall summary for Part C
        if part_c_correlation_summary:
            overall_pearson_c = sum(item['pearson'] for item in part_c_correlation_summary) / len(part_c_correlation_summary)
            overall_spearman_c = sum(item['spearman'] for item in part_c_correlation_summary) / len(part_c_correlation_summary)
            avg_samples_c = sum(item['n_samples'] for item in part_c_correlation_summary) / len(part_c_correlation_summary)
            
            print(f"-" * 70)
            print(f"{'OVERALL':<10} {overall_pearson_c:<10.3f} {'':<10} {overall_spearman_c:<10.3f} {'':<10} {avg_samples_c:<8.0f}")
        
        return overall_stats
    
    def run_complete_analysis(self, create_plots=False, output_dir='analysis_output'):
        """Run the complete analysis pipeline."""
        
        # Calculate averages
        self.calculate_average_scores()
        
        # Calculate correlations
        self.calculate_correlations()
        
        # Calculate Part C average scores
        self.calculate_part_c_average_scores()
        
        # Calculate dimension-annotator correlations
        self.calculate_dimension_annotator_correlations()

def main():
    """Main function to run the annotation evaluation analysis."""
    # Initialize the evaluator
    json_path = '/home/ljy/Pxplore/pxplore-algo/service/scripts/buffer/eval_annotation.json'
    evaluator = AnnotationEvaluator(json_path)
    
    # Run complete analysis (print-only mode)
    evaluator.run_complete_analysis(create_plots=False, output_dir='annotation_analysis_output')

if __name__ == "__main__":
    main()
