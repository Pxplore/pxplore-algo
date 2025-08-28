import json
import os


def create_training_data(source_data):
    training_data = []
    
    for item in source_data:
        # 过滤掉recommend_snippet_id为空的记录
        if not item.get("recommend_snippet_id") or type(item.get("recommend_content")) != dict:
            continue
            
        # 提取学生画像作为推荐策略
        student_profile = item.get('student_profile', {})
        
        # 提取候选内容片段
        candidates = item.get('recommend_candidates', [])
        
        if not candidates:
            continue
            
        # 构造系统提示词
        instruction = open("service/scripts/prompts/snippet_selection.txt", "r", encoding="utf-8").read()
        
        # 构造用户输入
        user_input = {
            "recommendation_strategy": student_profile,
            "candidates": candidates
        }
        
        # 构造期望输出，使用源数据中的字段
        expected_output = {
            "selected_candidate": {
                "id": item.get("recommend_snippet_id", ""),
                "bloom_level": item.get("recommend_content", {}).get("label", {}).get("bloom_level", ""), 
                "summary": item.get("recommend_content", {}).get("label", {}).get("summary", ""), 
                "title": item.get("course", "")
            },
            "reason": item.get("recommend_reason", "")
        }
        
        # 构造训练数据项
        training_item = {
            "instruction": instruction,
            "input": json.dumps(user_input, ensure_ascii=False, indent=2),
            "output": json.dumps(expected_output, ensure_ascii=False, indent=2),
            "system": "",
            "history": []
        }
        
        training_data.append(training_item)
    
    return training_data

def save_training_data(training_data, output_path):
    """保存训练数据到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)

def main():
    # 文件路径
    source_file = "model/data/test_steering_4o_full.json"
    output_file = "model/training/LLaMA-Factory/data/pxplore.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"正在加载源数据: {source_file}")
    source_data = json.load(open(source_file, "r", encoding="utf-8"))
    print(f"源数据加载完成，共 {len(source_data)} 条记录")
    
    print("正在转换数据格式...")
    training_data = create_training_data(source_data)
    print(f"数据转换完成，共生成 {len(training_data)} 条训练数据")
    
    print(f"正在保存训练数据: {output_file}")
    save_training_data(training_data, output_file)
    print("训练数据保存完成！")

if __name__ == "__main__":
    main()
