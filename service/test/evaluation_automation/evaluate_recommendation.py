import json
import math
from collections import defaultdict
from itertools import combinations

def aggregate_part_b_selected_by_record_id(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    agg = defaultdict(list)
    for entry in data:
        record_id = entry.get('record_id')
        part_b_selected_str = entry.get('part_b_selected', '[]')
        try:
            part_b_selected = json.loads(part_b_selected_str)
        except Exception:
            part_b_selected = []
        agg[record_id].append(part_b_selected)
    return dict(agg)

def kendalls_tau_for_selections(selection1, selection2):
    """
    Compute Kendall's tau for two selection lists.
    Since part_b_selected contains selected items (not necessarily ranked),
    we'll treat the order in the list as the ranking.
    """
    # Convert to rankings if they're just selections
    rank1 = list(selection1) if isinstance(selection1, list) else []
    rank2 = list(selection2) if isinstance(selection2, list) else []
    
    # Only consider items present in both rankings
    common_items = list(set(rank1) & set(rank2))
    if len(common_items) < 2:
        return None  # Not enough common items to compare

    # Build index maps for common items only
    idx1 = {item: rank1.index(item) for item in common_items}
    idx2 = {item: rank2.index(item) for item in common_items}

    n = len(common_items)
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i+1, n):
            item_a, item_b = common_items[i], common_items[j]
            order1 = idx1[item_a] - idx1[item_b]
            order2 = idx2[item_a] - idx2[item_b]
            if order1 * order2 > 0:
                concordant += 1
            elif order1 * order2 < 0:
                discordant += 1
            # If order1*order2==0, it's a tie in at least one ranking, ignore

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return None
    tau = (concordant - discordant) / total_pairs
    return tau

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def overlap_coefficient(set1, set2):
    """Calculate overlap coefficient (Szymkiewicz–Simpson coefficient)."""
    intersection = len(set1.intersection(set2))
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size > 0 else 0

def fleiss_kappa_for_selections(selections, all_items):
    """
    Calculate Fleiss' kappa for multi-annotator selection agreement.
    This is more appropriate for selection tasks than Kendall's tau.
    """
    n_annotators = len(selections)
    n_items = len(all_items)
    
    if n_annotators < 2:
        return None
    
    # Create agreement matrix: items x annotators
    # 1 if item was selected by annotator, 0 otherwise
    agreement_matrix = []
    for item in all_items:
        item_selections = [1 if item in selection else 0 for selection in selections]
        agreement_matrix.append(item_selections)
    
    # Calculate observed agreement
    p_obs = 0
    for item_selections in agreement_matrix:
        selected_count = sum(item_selections)
        if selected_count > 1:
            # Agreement occurs when multiple annotators select the same item
            p_obs += selected_count * (selected_count - 1) / (n_annotators * (n_annotators - 1))
    p_obs /= n_items
    
    # Calculate expected agreement by chance
    total_selections = sum(sum(item_selections) for item_selections in agreement_matrix)
    p_selected = total_selections / (n_items * n_annotators)
    p_not_selected = 1 - p_selected
    p_exp = 2 * p_selected * p_not_selected
    
    # Calculate Fleiss' kappa
    if p_exp == 1:
        return None
    kappa = (p_obs - p_exp) / (1 - p_exp)
    return kappa

def selection_agreement_metrics(selections):
    """
    Calculate multiple agreement metrics for selection tasks.
    Returns a dictionary with various metrics.
    """
    if len(selections) < 2:
        return None
    
    # Get all unique items across all selections
    all_items = set()
    for selection in selections:
        all_items.update(selection)
    all_items = list(all_items)
    
    # Pairwise metrics
    jaccard_scores = []
    overlap_scores = []
    kendall_taus = []
    
    for s1, s2 in combinations(selections, 2):
        set1, set2 = set(s1), set(s2)
        
        # Jaccard similarity
        jaccard = jaccard_similarity(set1, set2)
        jaccard_scores.append(jaccard)
        
        # Overlap coefficient
        overlap = overlap_coefficient(set1, set2)
        overlap_scores.append(overlap)
        
        # Kendall's tau (for comparison)
        tau = kendalls_tau_for_selections(s1, s2)
        if tau is not None:
            kendall_taus.append(tau)
    
    # Multi-annotator metrics
    fleiss_k = fleiss_kappa_for_selections(selections, all_items)
    
    return {
        'avg_jaccard': sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0,
        'avg_overlap_coefficient': sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0,
        'avg_kendall_tau': sum(kendall_taus) / len(kendall_taus) if kendall_taus else None,
        'fleiss_kappa': fleiss_k,
        'n_annotators': len(selections),
        'n_unique_items': len(all_items),
        'valid_kendall_pairs': len(kendall_taus)
    }

file_path = './all_annotations.json'
agg = aggregate_part_b_selected_by_record_id(file_path)

# Collect all metrics
all_jaccard = []
all_overlap = []
all_kendall = []
all_fleiss = []
valid_records = 0
total_records = len(agg)

for record_id, selections in agg.items():
    if len(selections) < 2:
        print(f"Record ID: {record_id}, Only {len(selections)} annotator(s) - cannot calculate agreement")
        continue
        
    metrics = selection_agreement_metrics(selections)
    # print(f"Record ID: {record_id}")
    # fleiss_str = f"{metrics['fleiss_kappa']:.3f}" if metrics['fleiss_kappa'] is not None else "None"
    # print(f"  Jaccard: {metrics['avg_jaccard']:.3f}, Overlap: {metrics['avg_overlap_coefficient']:.3f}, "
    #       f"Kendall: {metrics['avg_kendall_tau']}, Fleiss κ: {fleiss_str}")
    
    # Collect metrics for averaging
    all_jaccard.append(metrics['avg_jaccard'])
    all_overlap.append(metrics['avg_overlap_coefficient'])
    if metrics['avg_kendall_tau'] is not None:
        all_kendall.append(metrics['avg_kendall_tau'])
    if metrics['fleiss_kappa'] is not None:
        all_fleiss.append(metrics['fleiss_kappa'])
    valid_records += 1

print(f"Jaccard Similarity: {sum(all_jaccard)/len(all_jaccard):.4f}")
print(f"Overlap Coefficient: {sum(all_overlap)/len(all_overlap):.4f}")

def load_recommendation_data(result_json_path):
    """
    Load recommendation results and extract both recommend_content and first recommend_candidate.
    Returns a dictionary with both types of recommendations.
    """
    with open(result_json_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    recommend_content_data = []
    recommend_candidates_data = []
    
    for item in result_data:
        # Extract recommend_content (existing logic)
        if 'recommend_content' in item:
            recommend_content = item['recommend_content']
            
            if isinstance(recommend_content, dict):
                course_name = recommend_content.get('course_name', '')
                chapter_name = recommend_content.get('chapter_name', '')
                module_name = recommend_content.get('module_name', '')
                
                formatted_title = f"{course_name}_{chapter_name}_{module_name}"
                
                recommend_content_data.append({
                    'formatted_title': formatted_title,
                    'original_data': recommend_content,
                    'course_name': course_name,
                    'chapter_name': chapter_name,
                    'module_name': module_name,
                    'type': 'recommend_content'
                })
        
        # Extract first item from recommend_candidates (new baseline)
        if 'recommend_candidates' in item and isinstance(item['recommend_candidates'], list) and len(item['recommend_candidates']) > 0:
            first_candidate = item['recommend_candidates'][0]
            if 'metadata' in first_candidate and 'title' in first_candidate['metadata']:
                title = first_candidate['metadata']['title']
                
                # Use the title directly but replace dashes with underscores
                formatted_title = title.replace('-', '_')
                
                # Parse the title to extract course_name, chapter_name, module_name for consistency
                parts = title.split('-')
                if len(parts) >= 3:
                    course_name = parts[0]
                    chapter_name = parts[1]
                    module_name = '-'.join(parts[2:])  # Keep original format for internal use
                    
                    recommend_candidates_data.append({
                        'formatted_title': formatted_title,
                        'original_data': first_candidate,
                        'course_name': course_name,
                        'chapter_name': chapter_name,
                        'module_name': module_name,
                        'type': 'first_candidate'
                    })
    
    return {
        'recommend_content': recommend_content_data,
        'first_candidates': recommend_candidates_data
    }

def calculate_recommendation_scores(recommendations):
    """
    Calculate scores for recommended content based on partial order in annotation list.
    Takes the bigger score between annotators for each record.
    """
    # Group annotations by record_id
    agg = aggregate_part_b_selected_by_record_id('./all_annotations.json')
    
    # Create a mapping from formatted titles to check against annotations
    recommendation_titles = {rec['formatted_title'] for rec in recommendations}
    
    recommendation_scores = {}
    matched_recommendations = []
    
    for record_id, selections in agg.items():
        if len(selections) < 1:
            continue
        
        # Get all unique items selected by annotators for this record
        all_selected_items = set()
        for selection in selections:
            all_selected_items.update(selection)
        
        # Find which recommended items match the selected items
        # We need to match the recommendation titles with annotation titles
        matched_items = {}
        
        for selected_item in all_selected_items:
            for rec in recommendations:
                # Try different matching strategies
                formatted_title = rec['formatted_title']
                
                # Direct match
                if selected_item == formatted_title:
                    matched_items[selected_item] = rec
                    continue
                
                # Try matching with the original annotation format (with dashes)
                annotation_format = f"{rec['course_name']}-{rec['chapter_name']}-{rec['module_name']}"
                if selected_item == annotation_format:
                    matched_items[selected_item] = rec
                    continue
                
                # Try partial matching (contains)
                if (rec['course_name'] in selected_item and 
                    rec['chapter_name'] in selected_item and 
                    rec['module_name'] in selected_item):
                    matched_items[selected_item] = rec
                    continue
        
        # Calculate scores for matched items
        item_scores = {}
        
        for selected_item, rec_data in matched_items.items():
            # Calculate score based on position in each annotator's selection
            scores_for_item = []
            
            for selection in selections:
                if selected_item in selection:
                    # Score based on position: 1st = 1.0, 2nd = 0.8, 3rd = 0.6, etc.
                    position = selection.index(selected_item) + 1
                    if position == 1:
                        score = 1.0
                    elif position == 2:
                        score = 0.8
                    elif position == 3:
                        score = 0.6
                    else:
                        # For positions beyond 3rd, continue the pattern: decrease by 0.2 each position
                        # 4th = 0.4, 5th = 0.2, 6th and beyond = 0.0
                        score = max(0.0, 1.0 - (position - 1) * 0.1)
                    scores_for_item.append(score)
                else:
                    # Item not selected by this annotator = score 0
                    scores_for_item.append(0)
            
            # Take the maximum score across all annotators (as requested)
            max_score = max(scores_for_item) if scores_for_item else 0
            item_scores[selected_item] = {
                'score': max_score,
                'formatted_title': rec_data['formatted_title'],
                'recommendation_data': rec_data
            }
        
        if matched_items:
            recommendation_scores[record_id] = item_scores
            matched_recommendations.extend(matched_items.values())
    
    return recommendation_scores, list(set(rec['formatted_title'] for rec in matched_recommendations))

def analyze_recommendation_performance():
    """
    Analyze the performance of recommended content based on human annotations.
    """
    print(f"\n=== Recommendation Performance Analysis ===")
    
    # Load recommendations from the result JSON file
    recommendation_data = load_recommendation_data('/home/ljy/Pxplore/pxplore-algo/model/data/test/sft_qwen3_1e-5.json')
    
    # Use recommend_content for the general performance analysis
    recommendations = recommendation_data['recommend_content']
    
    # Calculate recommendation scores
    scores, matched_titles = calculate_recommendation_scores(recommendations)
    
    # Analyze overall performance
    all_scores = []
    record_performance = {}
    
    for record_id, item_scores in scores.items():
        if not item_scores:
            continue
            
        # Get statistics for this record (extract just the score values)
        scores_list = [item_data['score'] for item_data in item_scores.values()]
        avg_score = sum(scores_list) / len(scores_list)
        max_score = max(scores_list)
        min_score = min(scores_list)
        
        record_performance[record_id] = {
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'num_items': len(scores_list),
            'items': item_scores
        }
        
        all_scores.extend(scores_list)
        
        # Show detailed info for this record
        # print(f"\nRecord {record_id}: Avg={avg_score:.3f}, Max={max_score:.3f}, Min={min_score:.3f}, Items={len(scores_list)}")
        # for item_name, item_data in item_scores.items():
        #     print(f"  - {item_data['formatted_title']}: {item_data['score']:.3f}")
    
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        overall_max = max(all_scores)
        overall_min = min(all_scores)
        
        # Remove detailed output - keeping only essential metrics
        
    # Remove unnecessary output
    
    return record_performance

def load_expert_annotations(json_path):
    """
    Load expert annotations and extract ground truth rankings.
    Returns a dictionary mapping record_id to expert rankings.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    expert_rankings = defaultdict(list)
    for entry in data:
        record_id = entry.get('record_id')
        annotator = entry.get('annotator', 'unknown')
        
        # Get the sorted ranking (ground truth)
        part_b_sorted_str = entry.get('part_b_sorted', '[]')
        try:
            part_b_sorted = json.loads(part_b_sorted_str)
        except Exception:
            part_b_sorted = []
        
        expert_rankings[record_id].append({
            'annotator': annotator,
            'ranking': part_b_sorted
        })
    
    return dict(expert_rankings)

def calculate_precision_at_1(predicted_item, expert_rankings):
    """
    Calculate Precision@1: whether the model's top prediction matches any expert's "best" choice.
    """
    if not expert_rankings:
        return 0.0
    
    # Check if predicted_item matches the first item (best choice) in any expert ranking
    matches = 0
    total_experts = len(expert_rankings)
    
    for expert_data in expert_rankings:
        ranking = expert_data['ranking']
        if ranking and predicted_item == ranking[0]:  # First item is the "best" choice
            matches += 1
    
    return matches / total_experts if total_experts > 0 else 0.0

def calculate_dcg(relevance_scores, k=None):
    """
    Calculate Discounted Cumulative Gain (DCG) for a list of relevance scores.
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) is 0, we want log2(2) for position 1
    
    return dcg

def calculate_ndcg_at_k(predicted_ranking, expert_rankings, k=5):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        predicted_ranking: List of predicted items in order
        expert_rankings: List of expert ranking data
        k: Number of top items to consider
    
    Returns:
        NDCG@k score
    """
    if not predicted_ranking or not expert_rankings:
        return 0.0
    
    # Create relevance scores for predicted ranking
    relevance_scores = []
    
    for predicted_item in predicted_ranking[:k]:
        # Calculate relevance based on expert annotations
        relevance = 0.0
        total_experts = len(expert_rankings)
        
        for expert_data in expert_rankings:
            ranking = expert_data['ranking']
            if predicted_item in ranking:
                position = ranking.index(predicted_item)
                if position == 0:
                    # Best choice gets highest relevance
                    relevance += 3.0
                elif position < len(ranking):
                    # Acceptable choices get decreasing relevance
                    relevance += max(1.0, 2.0 - position * 0.5)
            # Items not in ranking get 0 relevance (implicitly "not recommended")
        
        # Average relevance across experts
        relevance_scores.append(relevance / total_experts if total_experts > 0 else 0.0)
    
    # Calculate DCG for predicted ranking
    dcg = calculate_dcg(relevance_scores, k)
    
    # Calculate Ideal DCG (IDCG) - best possible ranking
    ideal_relevance = []
    all_items = set()
    for expert_data in expert_rankings:
        all_items.update(expert_data['ranking'])
    
    # Calculate ideal relevance for each item
    item_relevances = {}
    for item in all_items:
        relevance = 0.0
        for expert_data in expert_rankings:
            ranking = expert_data['ranking']
            if item in ranking:
                position = ranking.index(item)
                if position == 0:
                    relevance += 3.0
                else:
                    relevance += max(1.0, 2.0 - position * 0.5)
        item_relevances[item] = relevance / len(expert_rankings)
    
    # Sort by relevance for ideal ranking
    ideal_relevance = sorted(item_relevances.values(), reverse=True)[:k]
    idcg = calculate_dcg(ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_recommendation_metrics():
    """
    Evaluate recommendation system using Precision@1 and NDCG@k metrics.
    Compares both recommend_content and first_candidates.
    """
    
    # Load recommendations and expert annotations
    recommendation_data = load_recommendation_data('/home/ljy/Pxplore/pxplore-algo/model/data/test/prompt_4o.json')
    expert_annotations = load_expert_annotations('./all_annotations.json')
    
    results = {}
    
    # Evaluate recommend_content (existing approach)
    print("=== Recommend Content Evaluation ===")
    scores_content, matched_titles_content = calculate_recommendation_scores(recommendation_data['recommend_content'])
    results['recommend_content'] = evaluate_single_recommendation_type(scores_content, expert_annotations, "Recommend Content")
    
    # Evaluate first candidates (new baseline)
    print("\n=== First Candidate Baseline Evaluation ===")
    scores_candidates, matched_titles_candidates = calculate_recommendation_scores(recommendation_data['first_candidates'])
    results['first_candidates'] = evaluate_single_recommendation_type(scores_candidates, expert_annotations, "First Candidate")
    
    return results

def evaluate_single_recommendation_type(scores, expert_annotations, method_name):
    """
    Evaluate a single recommendation type (either recommend_content or first_candidates).
    """
    precision_at_1_scores = []
    ndcg_1_scores = []
    ndcg_3_scores = []
    ndcg_5_scores = []
    ndcg_7_scores = []
    ndcg_10_scores = []
    
    evaluated_records = 0
    
    for record_id, expert_data in expert_annotations.items():
        if record_id not in scores:
            continue  # No matching recommendation for this record
        
        evaluated_records += 1
        
        # Get the recommended item for this record (top recommendation)
        record_recommendations = scores[record_id]
        if not record_recommendations:
            continue
        
        # Get top recommended item (highest scoring)
        top_item = max(record_recommendations.items(), key=lambda x: x[1]['score'])
        predicted_item = top_item[0]
        
        # Calculate Precision@1
        p1 = calculate_precision_at_1(predicted_item, expert_data)
        precision_at_1_scores.append(p1)
        
        # For NDCG, we need a predicted ranking
        # Sort recommendations by score to create predicted ranking
        predicted_ranking = sorted(record_recommendations.keys(), 
                                 key=lambda x: record_recommendations[x]['score'], reverse=True)
        
        # Calculate NDCG@1, @3, @5, @7, @10
        ndcg_1 = calculate_ndcg_at_k(predicted_ranking, expert_data, k=1)
        ndcg_3 = calculate_ndcg_at_k(predicted_ranking, expert_data, k=3)
        ndcg_5 = calculate_ndcg_at_k(predicted_ranking, expert_data, k=5)
        ndcg_7 = calculate_ndcg_at_k(predicted_ranking, expert_data, k=7)
        ndcg_10 = calculate_ndcg_at_k(predicted_ranking, expert_data, k=10)
        
        ndcg_1_scores.append(ndcg_1)
        ndcg_3_scores.append(ndcg_3)
        ndcg_5_scores.append(ndcg_5)
        ndcg_7_scores.append(ndcg_7)
        ndcg_10_scores.append(ndcg_10)
    
    # Calculate average metrics
    if precision_at_1_scores:
        avg_p1 = sum(precision_at_1_scores) / len(precision_at_1_scores)
        avg_ndcg1 = sum(ndcg_1_scores) / len(ndcg_1_scores)
        avg_ndcg3 = sum(ndcg_3_scores) / len(ndcg_3_scores)
        avg_ndcg5 = sum(ndcg_5_scores) / len(ndcg_5_scores)
        avg_ndcg7 = sum(ndcg_7_scores) / len(ndcg_7_scores)
        avg_ndcg10 = sum(ndcg_10_scores) / len(ndcg_10_scores)
        
        print(f"Precision@1: {avg_p1:.4f}")
        print(f"NDCG@1: {avg_ndcg1:.4f}")
        print(f"NDCG@3: {avg_ndcg3:.4f}")
        print(f"NDCG@5: {avg_ndcg5:.4f}")
        print(f"NDCG@7: {avg_ndcg7:.4f}")
        print(f"NDCG@10: {avg_ndcg10:.4f}")
    else:
        print("No valid evaluations found")
    
    return {
        'precision_at_1': avg_p1 if precision_at_1_scores else 0.0,
        'ndcg_at_1': avg_ndcg1 if ndcg_1_scores else 0.0,
        'ndcg_at_3': avg_ndcg3 if ndcg_3_scores else 0.0,
        'ndcg_at_5': avg_ndcg5 if ndcg_5_scores else 0.0,
        'ndcg_at_7': avg_ndcg7 if ndcg_7_scores else 0.0,
        'ndcg_at_10': avg_ndcg10 if ndcg_10_scores else 0.0,
        'evaluated_records': evaluated_records
    }

# Run the analyses
analyze_recommendation_performance()
evaluate_recommendation_metrics()
