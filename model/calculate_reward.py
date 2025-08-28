#!/usr/bin/env python3
"""
评测脚本：计算推荐系统中四个维度的对齐度
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict
import argparse

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """加载测试数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载数据，共 {len(data)} 个item")
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return []

def calculate_alignment_scores(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """计算四个维度的对齐度分数"""
    
    # 初始化统计变量
    dimension_stats = {
        'long_term_objective': {'total': 0, 'aligned': 0},
        'short_term_objective': {'total': 0, 'aligned': 0},
        'implicit_motivation': {'total': 0, 'aligned': 0},
        'explicit_motivation': {'total': 0, 'aligned': 0}
    }
    
    # 遍历每个item
    for item in data:
            
        profile = item['next_state']
        
        # 检查四个维度
        for dimension in dimension_stats.keys():
            if dimension in profile and isinstance(profile[dimension], list):
                for objective in profile[dimension]:
                    if isinstance(objective, dict) and 'is_aligned' in objective:
                        dimension_stats[dimension]['total'] += 1
                        if objective['is_aligned']:
                            dimension_stats[dimension]['aligned'] += 1
    
    # 计算平均值
    results = {}
    for dimension, stats in dimension_stats.items():
        if stats['total'] > 0:
            alignment_rate = stats['aligned'] / stats['total']
            results[dimension] = {
                'alignment_rate': alignment_rate,
                'aligned_count': stats['aligned'],
                'total_count': stats['total'],
                'percentage': f"{alignment_rate * 100:.2f}%"
            }
        else:
            results[dimension] = {
                'alignment_rate': 0.0,
                'aligned_count': 0,
                'total_count': 0,
                'percentage': "0.00%"
            }
    
    return results

def print_results(results: Dict[str, Dict[str, Any]]):
    """打印评测结果"""
    print("\n" + "="*60)
    print("推荐系统四个维度对齐度评测结果")
    print("="*60)
    
    for dimension, stats in results.items():
        print(f"\n{dimension.replace('_', ' ').title()}:")
        print(f"  对齐率: {stats['percentage']}")
        print(f"  对齐数量: {stats['aligned_count']}")
        print(f"  总数量: {stats['total_count']}")
    
    # 计算总体平均对齐率
    total_aligned = sum(stats['aligned_count'] for stats in results.values())
    total_count = sum(stats['total_count'] for stats in results.values())
    
    if total_count > 0:
        overall_rate = total_aligned / total_count
        print(f"\n总体平均对齐率: {overall_rate * 100:.2f}%")
        print(f"总体对齐数量: {total_aligned}")
        print(f"总体总数量: {total_count}")
    
    print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Calculate alignment scores for a given setting.")
    parser.add_argument('--setting', type=str, default="reward_steering_4o")
    args = parser.parse_args()
    setting = args.setting

    data_file = f"./model/data/eval/{setting}.json"
    
    # 加载数据
    data = load_test_data(data_file)
    if not data:
        return
    
    # 计算对齐度分数
    results = calculate_alignment_scores(data)
    
    # 打印结果
    print_results(results)

if __name__ == "__main__":
    main()
