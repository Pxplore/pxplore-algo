import json
import numpy as np
from datetime import datetime


def load_json_file(filename):
    """加载JSON文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到文件 '{filename}'")
        return None
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
        return None


def calculate_part_a_score(annotations):
    """计算Part A得分"""
    scores = []
    counts = {"合理": 0, "不合理": 0}

    for annotation in annotations:
        judgement = annotation.get("part_a_judgement", "")
        if judgement == "合理":
            scores.append(1)
            counts["合理"] += 1
        elif judgement == "不合理":
            scores.append(0)
            counts["不合理"] += 1

    avg_score = np.mean(scores) if scores else 0
    return avg_score, counts, len(scores)


def calculate_part_c_score(annotations):
    """计算Part C得分"""
    scores = []
    counts = {"合理": 0, "不合理": 0}

    for annotation in annotations:
        judgement = annotation.get("part_c_judgement", "")
        if judgement == "合理":
            scores.append(1)
            counts["合理"] += 1
        elif judgement == "不合理":
            scores.append(0)
            counts["不合理"] += 1

    avg_score = np.mean(scores) if scores else 0
    return avg_score, counts, len(scores)


def calculate_part_b_score(annotations, recommendations):
    """计算Part B得分 - 修正版：区分无效数据和0分数据"""
    # 创建record_id到推荐课程的映射
    rec_map = {}
    for rec in recommendations:
        record_id = rec.get("record id")
        if record_id is not None:
            if record_id not in rec_map:
                rec_map[record_id] = []
            rec_map[record_id].append(rec.get("name", ""))

    scores = []
    score_details = []  # 存储每个记录的详细评分信息
    detailed_issues = []  # 存储详细问题信息
    invalid_records = []  # 存储无效记录ID

    for annotation in annotations:
        record_id = annotation.get("record_id")
        sorted_list_str = annotation.get("part_b_sorted", "[]")
        annotator = annotation.get("annotator", "unknown")

        # 检查是否为无效数据（空排序列表）
        if sorted_list_str == "[]" or sorted_list_str.strip() == "":
            issue = f"record_id {record_id} (标注者: {annotator}) - 无效数据: part_b_sorted 为空"
            detailed_issues.append(issue)
            invalid_records.append(record_id)
            continue  # 跳过这条记录，不参与评分

        # 初始化记录信息
        record_info = {
            "record_id": record_id,
            "annotator": annotator,
            "score": 0,
            "recommended_courses": [],
            "found_courses": [],
            "missing_courses": [],
            "positions": []
        }

        try:
            sorted_list = json.loads(sorted_list_str)
        except json.JSONDecodeError:
            issue = f"record_id {record_id} (标注者: {annotator}) - part_b_sorted 格式异常: {sorted_list_str}"
            detailed_issues.append(issue)
            invalid_records.append(record_id)
            continue  # 跳过这条记录，不参与评分

        if record_id not in rec_map:
            issue = f"record_id {record_id} (标注者: {annotator}) - 在推荐课程文件中不存在"
            detailed_issues.append(issue)
            invalid_records.append(record_id)
            continue  # 跳过这条记录，不参与评分

        # 获取该record_id的所有推荐课程
        rec_courses = rec_map[record_id]
        record_info["recommended_courses"] = rec_courses.copy()

        record_score = 0
        course_count = len(rec_courses)

        for course in rec_courses:
            if course in sorted_list:
                position = sorted_list.index(course)
                # 新的评分规则：1, 0.8, 0.6, 0.4, 0.2, 其余0分
                if position == 0:
                    course_score = 1.0
                elif position == 1:
                    course_score = 0.8
                elif position == 2:
                    course_score = 0.6
                elif position == 3:
                    course_score = 0.4
                elif position == 4:
                    course_score = 0.2
                else:
                    course_score = 0  # 位置5及以后得0分

                record_score += course_score
                record_info["found_courses"].append(course)
                record_info["positions"].append(position)
            else:
                # 推荐课程不在排序列表中，得0分
                record_info["missing_courses"].append(course)

        # 计算平均分（即使所有课程都不在列表中，也得0分）
        if course_count > 0:
            final_score = record_score / course_count
        else:
            final_score = 0  # 没有推荐课程也得0分

        scores.append(final_score)
        record_info["score"] = final_score
        score_details.append(record_info)

        # 记录详细问题（如果有）
        if record_info["missing_courses"]:
            issue = f"record_id {record_id} (标注者: {annotator}) - 部分推荐课程不在排序列表中: {record_info['missing_courses']}"
            detailed_issues.append(issue)

    avg_score = np.mean(scores) if scores else 0
    valid_records = len(scores)  # 有效的评分记录数量

    return avg_score, valid_records, len(annotations), invalid_records, score_details, detailed_issues


def generate_report(results, output_prefix):
    """生成文本报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 计算0分记录的数量
    zero_scores = sum(1 for detail in results['part_b']['score_details'] if detail['score'] == 0)

    report = f"""个性化算法评估报告
生成时间: {timestamp}
总记录数: {results['overall']['total_records']}

Part A 评估结果:
  平均得分: {results['part_a']['average_score']:.4f} ({results['part_a']['average_score'] * 100:.2f}%)
  合理记录: {results['part_a']['counts']['合理']}
  不合理记录: {results['part_a']['counts']['不合理']}

Part B 评估结果:
  平均得分: {results['part_b']['average_score']:.4f} ({results['part_b']['average_score'] * 100:.2f}%)
  有效记录: {results['part_b']['valid_records']}/{results['part_b']['total_records']}
  无效记录: {len(results['part_b']['invalid_records'])}


Part C 评估结果:
  平均得分: {results['part_c']['average_score']:.4f} ({results['part_c']['average_score'] * 100:.2f}%)
  合理记录: {results['part_c']['counts']['合理']}
  不合理记录: {results['part_c']['counts']['不合理']}


总结:
  Part A 合理率: {results['part_a']['average_score'] * 100:.2f}%
  Part B 平均分: {results['part_b']['average_score'] * 100:.2f}%
  Part C 合理率: {results['part_c']['average_score'] * 100:.2f}%
"""

    # 保存报告
    with open(f'{output_prefix}_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"文本报告已保存为 {output_prefix}_report.txt")

    # 在控制台输出报告
    print(report)


def main():
    """主函数"""
    # 加载数据
    annotations = load_json_file('all_annotations（标注结果）.json')
    if annotations is None:
        return

    recommendations = load_json_file('recommend_content_ezversion.json')
    if recommendations is None:
        return

    # 计算各部分得分
    part_a_avg, part_a_counts, part_a_total = calculate_part_a_score(annotations)
    # 修改这一行，接收6个返回值
    part_b_avg, part_b_valid, part_b_total, part_b_invalid, part_b_details, part_b_issues = calculate_part_b_score(annotations, recommendations)
    part_c_avg, part_c_counts, part_c_total = calculate_part_c_score(annotations)

    # 准备结果
    results = {
        "part_a": {
            "average_score": part_a_avg,
            "counts": part_a_counts,
            "total_records": part_a_total
        },
        "part_b": {
            "average_score": part_b_avg,
            "valid_records": part_b_valid,
            "total_records": part_b_total,
            "invalid_records": part_b_invalid,  # 新增
            "score_details": part_b_details,
            "detailed_issues": part_b_issues
        },
        "part_c": {
            "average_score": part_c_avg,
            "counts": part_c_counts,
            "total_records": part_c_total
        },
        "overall": {
            "total_records": len(annotations)
        }
    }

    # 生成输出文件前缀（使用当前时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"evaluation_results_{timestamp}"

    # 生成报告
    generate_report(results, output_prefix)



if __name__ == "__main__":
    main()
