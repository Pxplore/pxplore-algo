#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本：将training.json中的input字段转换为messages格式
"""

import json
import os
from typing import List, Dict, Any


def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """加载训练数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条训练数据")
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return []


def transform_data_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换数据格式：将input字段转换为messages格式"""
    transformed_data = []
    
    for i, item in enumerate(data):
        if "input" in item:
            # 创建新的数据格式
            new_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": open("model/training/ms-swift/examples/train/grpo/qwen3/system_prompt.txt", "r").read()
                    },
                    {
                        "role": "user",
                        "content": item["input"]
                    }
                ]
            }
            transformed_data.append(new_item)
        else:
            print(f"警告：第 {i+1} 条数据缺少input字段，跳过")
    
    print(f"成功转换 {len(transformed_data)} 条数据")
    return transformed_data


def save_transformed_data(data: List[Dict[str, Any]], output_path: str) -> bool:
    """保存转换后的数据"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"保存数据失败: {e}")
        return False


def main():
    """主函数"""
    # 文件路径
    input_file = "training.json"
    output_file = "transformed_training.json"
    
    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, input_file)
    output_file = os.path.join(script_dir, output_file)
    
    print("开始数据处理...")
    
    # 1. 加载原始数据
    original_data = load_training_data(input_file)
    if not original_data:
        print("无法加载原始数据，程序退出")
        return
    
    # 2. 转换数据格式
    transformed_data = transform_data_format(original_data)
    if not transformed_data:
        print("数据转换失败，程序退出")
        return
    
    # 3. 保存转换后的数据
    if save_transformed_data(transformed_data, output_file):
        print("数据处理完成！")
        
        # 显示转换前后的对比
        print(f"\n数据转换统计:")
        print(f"原始数据条数: {len(original_data)}")
        print(f"转换后数据条数: {len(transformed_data)}")
        
        # 显示第一条转换后的数据作为示例
        if transformed_data:
            print(f"\n转换后的数据格式示例:")
            print(json.dumps(transformed_data[0], ensure_ascii=False, indent=2))
    else:
        print("数据处理失败！")


if __name__ == "__main__":
    main()
