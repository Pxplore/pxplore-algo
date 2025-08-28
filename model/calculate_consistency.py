#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测脚本：对比两个JSON文件中相同item的recommend_content是否一致，并计算一致性
"""

import json
import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import difflib


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return []

def compare_recommend_content(item1: Dict[str, Any], item2: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """对比两个item的recommend_content是否一致 - 只检查label.summary和label.bloom_level"""
    content1 = item1.get('recommend_content', {})
    content2 = item2.get('recommend_content', {})
    
    if not content1 and not content2:
        return True, {}
    
    if not content1 or not content2:
        return False, {
            'reason': '其中一个item缺少recommend_content',
            'content1_exists': bool(content1),
            'content2_exists': bool(content2)
        }
    
    # 只对比关键字段：label.summary和label.bloom_level
    if type(content1) == str or type(content2) == str:
        return False, {
            'reason': 'recommend_content是字符串，无法对比'
        }
    
    label1 = content1.get('label', {})
    label2 = content2.get('label', {})
    
    comparison_result = {
        'label_summary': label1.get('summary') == label2.get('summary'),
        'label_bloom_level': label1.get('bloom_level') == label2.get('bloom_level')
    }
    
    # 检查是否所有字段都一致
    is_consistent = all(comparison_result.values())
    
    return is_consistent, comparison_result


def calculate_consistency(file1_path: str, file2_path: str) -> Dict[str, Any]:
    """计算两个文件的一致性"""
    print(f"正在加载文件...")
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)
    
    if not data1 or not data2:
        return {"error": "无法加载文件数据"}
    
    print(f"文件1包含 {len(data1)} 个item")
    print(f"文件2包含 {len(data2)} 个item")
    
    # 对比每个共同的item
    comparison_results = []
    consistent_count = 0
    
    for idx in range(len(data1)):
        item1 = data1[idx]
        item2 = data2[idx]
        
        is_consistent, details = compare_recommend_content(item1, item2)
        
        result = {
            'is_consistent': is_consistent,
            'details': details,
        }
        
        comparison_results.append(result)
        
        if is_consistent:
            consistent_count += 1
    
    # 计算一致性
    consistency_rate = consistent_count / len(data1) if data1 else 0
    
    return {
        'total_items': len(data1),
        'consistent_items': consistent_count,
        'inconsistent_items': len(data1) - consistent_count,
        'consistency_rate': consistency_rate,
        'comparison_results': comparison_results
    }


def print_detailed_report(result: Dict[str, Any]):
    print(f"item总数: {result['total_items']}")
    print(f"一致的item数: {result['consistent_items']}")
    print(f"不一致的item数: {result['inconsistent_items']}")
    print(f"一致性率: {result['consistency_rate']:.2%}")

def main():

    file1 = f"./model/data/test_reward_steering_qwen3.json"
    file2 = f"./model/data/test_reward_steering_4o.json"

    print(f"开始评测...")
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    
    # 计算一致性
    result = calculate_consistency(file1, file2)
    
    # 打印报告
    print_detailed_report(result)


if __name__ == "__main__":
    main()
