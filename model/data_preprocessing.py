import pandas as pd
import re
import json
from typing import List, Dict, Any
from datetime import datetime
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_interaction_record(record_text: str) -> List[Dict[str, str]]:
    """
    解析交互记录，格式为[time][role][content]
    每行为一个记录，解析成dict格式
    
    Args:
        record_text: 交互记录文本
        
    Returns:
        解析后的交互记录列表
    """
    if pd.isna(record_text) or not record_text.strip():
        return []
    
    interactions = []
    # 按换行符分割每条记录
    lines = record_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 使用正则表达式匹配时间、角色和内容
        # 格式: 2024-04-29 00:14:51 王腾: hello，这门课的教授是谁
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.+?): (.+)'
        match = re.match(pattern, line)
        
        if match:
            time_str, role, content = match.groups()
            interactions.append({
                'time': time_str,
                'role': role,
                'content': content
            })
    
    return interactions


def process_student_data(file_path: str) -> List[Dict[str, Any]]:
    """
    处理学生交互数据，支持读取所有sheet
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        处理后的数据列表
    """
    # 读取Excel文件的所有sheet
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    print(f"发现 {len(sheet_names)} 个sheet: {sheet_names}")
    
    all_processed_data = []
    
    for sheet_name in sheet_names:
        print(f"\n正在处理sheet: {sheet_name}")
        
        # 读取当前sheet，不使用第一行作为列名
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 根据数据结构，找到实际的列名行（第2行）
        # 第0列：邮箱，第1列：姓名，第2列：角色，第3列：课程名，第4列：章节名
        # 第5列：模块名，第6列：已学完页数/模块总PPT页数，第7列：学习进度百分比
        # 第8列：答对题数/已作答题数，第9列：用户发言次数，第11列：交互记录
        
        processed_data = []
        
        # 从第3行开始处理数据（跳过标题行）
        for idx in range(3, len(df)):
            row = df.iloc[idx]
            
            # 检查是否为空行
            if pd.isna(row[0]) or str(row[0]).strip() == '':
                continue
                
            # 提取字段
            email = str(row[0]).strip()
            name = str(row[1]).strip() if not pd.isna(row[1]) else ''
            role = str(row[2]).strip() if not pd.isna(row[2]) else ''
            course_name = str(row[3]).strip() if not pd.isna(row[3]) else ''
            chapter_name = str(row[4]).strip() if not pd.isna(row[4]) else ''
            module_name = str(row[5]).strip() if not pd.isna(row[5]) else ''
            page_progress = str(row[6]).strip() if not pd.isna(row[6]) else ''
            progress_percentage = str(row[7]).strip() if not pd.isna(row[7]) else ''
            
            # 处理答对题数/已作答题数
            correct_ratio = str(row[8]).strip() if not pd.isna(row[8]) else '0'
            
            # 处理用户发言次数
            speech_count = str(row[9]).strip() if not pd.isna(row[9]) else '0'
            
            # 交互记录
            interaction_record = str(row[11]) if not pd.isna(row[11]) else ''
            
            # 过滤条件1：用户发言次数和答对题数/已作答题数都为0的数据
            try:
                speech_count_int = int(speech_count) if speech_count.isdigit() else 0
                correct_ratio_parts = correct_ratio.split('/')
                if len(correct_ratio_parts) == 2:
                    correct_count = int(correct_ratio_parts[0]) if correct_ratio_parts[0].isdigit() else 0
                    total_count = int(correct_ratio_parts[1]) if correct_ratio_parts[1].isdigit() else 0
                else:
                    correct_count = 0
                    total_count = 0
            except:
                speech_count_int = 0
                correct_count = 0
                total_count = 0
            
            # 如果用户发言次数和答对题数都为0，跳过这条记录
            if speech_count_int == 0 and correct_count == 0:
                continue
            
            # 解析交互记录
            interactions = []
            if speech_count_int > 0 and interaction_record:
                interactions = parse_interaction_record(interaction_record)
            
            # 构建数据记录，添加sheet信息
            record = {
                'sheet_name': sheet_name,  # 添加sheet名称
                'email': email,
                'name': name,
                'role': role,
                'course_name': course_name,
                'chapter_name': chapter_name,
                'module_name': module_name,
                'page_progress': page_progress,
                'progress_percentage': progress_percentage,
                'correct_ratio': correct_ratio,
                'speech_count': speech_count_int,
                'correct_count': correct_count,
                'total_count': total_count,
                'interactions': interactions
            }
            
            processed_data.append(record)
        
        print(f"  Sheet '{sheet_name}' 处理完成，获得 {len(processed_data)} 条有效记录")
        all_processed_data.extend(processed_data)
    
    print(f"\n所有sheet处理完成，总共获得 {len(all_processed_data)} 条记录")
    return all_processed_data


def save_processed_data(data: List[Dict[str, Any]], output_file: str):
    """
    保存处理后的数据到JSON文件
    
    Args:
        data: 处理后的数据
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到: {output_file}")
    print(f"总共处理了 {len(data)} 条记录")


def create_sample_dataset(data: List[Dict[str, Any]], num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    从处理后的数据中随机选择指定数量的样本，并保存到testset.json
    
    Args:
        data: 处理后的数据
        num_samples: 要选择的样本数量
        
    Returns:
        随机选择的样本列表
    """
    if not data:
        print("没有数据可供采样。")
        return []

    # 直接从全集中随机采样
    if len(data) >= num_samples:
        sample_data = random.sample(data, num_samples)
    else:
        # 如果总记录数不足num_samples，则全部采样
        sample_data = data.copy()

    # 保存采样数据到testset.json
    sample_output_file = os.path.join(BASE_DIR, f"data/student_data_{num_samples}.json")
    save_processed_data(sample_data, sample_output_file)
    print(f"\n从 {len(data)} 条记录中采样 {len(sample_data)} 条记录到 {sample_output_file}")
    
    # 显示采样统计信息
    sheet_stats = {}
    for record in sample_data:
        sheet_name = record['sheet_name']
        if sheet_name not in sheet_stats:
            sheet_stats[sheet_name] = 0
        sheet_stats[sheet_name] += 1
    
    print(f"采样数据中各sheet分布:")
    for sheet_name, count in sheet_stats.items():
        print(f"  {sheet_name}: {count} 条")
    
    return sample_data


def main():
    """
    主函数
    """
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 文件路径
    input_file = os.path.join(BASE_DIR, "data/student_interaction_data.xlsx")
    output_file = os.path.join(BASE_DIR, "data/student_data.json")
    
    try:
        # 处理数据
        print("开始处理学生交互数据...")
        processed_data = process_student_data(input_file)
        
        # 打印统计信息
        print(f"\n数据统计:")
        print(f"- 总记录数: {len(processed_data)}")
        
        # 按sheet统计记录数
        sheet_stats = {}
        for record in processed_data:
            sheet_name = record['sheet_name']
            if sheet_name not in sheet_stats:
                sheet_stats[sheet_name] = 0
            sheet_stats[sheet_name] += 1
        
        print(f"- 各sheet记录数:")
        for sheet_name, count in sheet_stats.items():
            print(f"  {sheet_name}: {count} 条")
        
        # 统计有交互记录的用户数
        users_with_interactions = sum(1 for record in processed_data if record['interactions'] and record['speech_count'] > 5)
        print(f"- 交互记录数大于5的用户数: {users_with_interactions}")
        
        processed_data = [record for record in processed_data if record['speech_count'] > 5]
        
        # 创建样本数据集
        create_sample_dataset(processed_data)

        # 保存数据
        save_processed_data(processed_data, output_file)
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
