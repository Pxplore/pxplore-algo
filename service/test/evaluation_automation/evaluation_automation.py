import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
from datetime import datetime
import pandas as pd

# Set style
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")


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

    # 提取所有课程名称用于后续分析
    all_courses = set()
    for courses in rec_map.values():
        all_courses.update(courses)

    scores = []
    score_details = []  # 存储每个记录的详细评分信息
    detailed_issues = []  # 存储详细问题信息
    invalid_records = []  # 存储无效记录ID
    positions = []  # 存储所有有效课程的位置信息

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
                positions.append(position)  # 添加到全局位置列表
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

    return avg_score, valid_records, len(annotations), invalid_records, score_details, detailed_issues, positions


def bootstrap_ci(data, stat_func=np.mean, n_bootstraps=10000, ci=95):
    """Calculate confidence interval using bootstrap method"""
    if not data:
        return (0, 0)

    bootstrapped_stats = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_stats.append(stat_func(sample))

    lower = np.percentile(bootstrapped_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_stats, ci + (100 - ci) / 2)
    return lower, upper


def binomial_ci(successes, trials, ci=0.95):
    """Calculate binomial confidence interval"""
    if trials == 0:
        return (0, 0)

    alpha = 1 - ci
    lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return lower, upper


def calculate_top_k_hit_rate(positions, k):
    """Calculate Top-k hit rate"""
    if not positions:
        return 0
    return np.mean([1 if pos < k else 0 for pos in positions])


def generate_visualization(results, positions, output_path):
    """生成可视化报告"""
    # 提取数据
    part_a_data = {
        'score': results['part_a']['average_score'],
        'reasonable': results['part_a']['counts']['合理'],
        'unreasonable': results['part_a']['counts']['不合理'],
        'total': results['part_a']['total_records']
    }

    part_b_data = {
        'score': results['part_b']['average_score'],
        'valid': results['part_b']['valid_records'],
        'invalid': len(results['part_b']['invalid_records']),
        'total': results['part_b']['total_records']
    }

    part_c_data = {
        'score': results['part_c']['average_score'],
        'reasonable': results['part_c']['counts']['合理'],
        'unreasonable': results['part_c']['counts']['不合理'],
        'total': results['part_c']['total_records']
    }

    # 计算Top-k命中率
    top_rates = {}
    top_cis = {}
    if positions:
        for k in [1, 3, 5, 10]:
            hit_rate = calculate_top_k_hit_rate(positions, k)
            top_rates[k] = hit_rate
            top_cis[k] = binomial_ci(int(hit_rate * len(positions)), len(positions))

    # 计算置信区间
    a_ci = binomial_ci(part_a_data['reasonable'], part_a_data['total'])
    b_ci = binomial_ci(part_b_data['valid'], part_b_data['total'])
    c_ci = binomial_ci(part_c_data['reasonable'], part_c_data['total'])

    # 创建图表
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Algorithm Evaluation Results', fontsize=20, fontweight='bold', y=0.98)

    # 使用GridSpec创建布局
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # 1. 各部分得分对比图
    ax1 = fig.add_subplot(gs[0, :])
    parts = ['Part A (Relevance)', 'Part B (Ranking)', 'Part C (Overall)']
    scores = [part_a_data['score'], part_b_data['score'], part_c_data['score']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = ax1.bar(parts, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Score', fontsize=14)
    ax1.set_title('Overall Scores Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=10)

    # 在柱状图上添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{score:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # 2. Part A 和 Part C 的合理性分布
    ax2 = fig.add_subplot(gs[1, 0])
    labels = ['Relevant', 'Irrelevant']
    a_sizes = [part_a_data['reasonable'], part_a_data['unreasonable']]
    c_sizes = [part_c_data['reasonable'], part_c_data['unreasonable']]
    colors = ['#66BB6A', '#EF5350']

    wedges, texts, autotexts = ax2.pie(a_sizes, labels=labels, autopct='%1.1f%%',
                                       startangle=90, colors=colors, explode=(0.05, 0))
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('Part A Relevance Distribution', fontsize=14, fontweight='bold', pad=20)

    ax3 = fig.add_subplot(gs[1, 2])
    wedges, texts, autotexts = ax3.pie(c_sizes, labels=labels, autopct='%1.1f%%',
                                       startangle=90, colors=colors, explode=(0.05, 0))
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax3.set_title('Part C Relevance Distribution', fontsize=14, fontweight='bold', pad=20)

    # 3. Part B 有效性分布
    ax4 = fig.add_subplot(gs[1, 1])
    labels = ['Valid', 'Invalid']
    sizes = [part_b_data['valid'], part_b_data['invalid']]
    colors = ['#66BB6A', '#EF5350']

    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       startangle=90, colors=colors, explode=(0.05, 0))
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax4.set_title('Part B Validity Distribution', fontsize=14, fontweight='bold', pad=20)

    # 4. Part B Top-k 命中率
    if positions:
        ax5 = fig.add_subplot(gs[2, 0])
        top_ks = [f'Top-{k}' for k in [1, 3, 5, 10]]
        hit_rates = [top_rates[k] for k in [1, 3, 5, 10]]
        cis = [top_cis[k] for k in [1, 3, 5, 10]]

        x_pos = np.arange(len(top_ks))
        bars = ax5.bar(x_pos, hit_rates, color=['#3498DB', '#9B59B6', '#E74C3C', '#F39C12'],
                       alpha=0.8, edgecolor='black', linewidth=1.5,
                       yerr=[(hit_rates[i] - cis[i][0]) for i in range(len(hit_rates))], capsize=10)

        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(top_ks)
        ax5.set_ylabel('Hit Rate', fontsize=14)
        ax5.set_title('Part B Top-k Hit Rates', fontsize=14, fontweight='bold', pad=20)
        ax5.set_ylim(0, 1)

        # 添加数值标签
        for i, (bar, rate) in enumerate(zip(bars, hit_rates)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
    else:
        # 如果没有位置数据，显示占位符
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.text(0.5, 0.5, 'No ranking position data available',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Part B Ranking Analysis', fontsize=14, fontweight='bold', pad=20)
        ax5.set_xticks([])
        ax5.set_yticks([])

    # 5. 置信区间分析 - 修复yerr不能为负的问题
    ax6 = fig.add_subplot(gs[2, 1])

    parts = ['Part A', 'Part B', 'Part C']
    scores = [part_a_data['score'], part_b_data['score'], part_c_data['score']]
    cis = [a_ci, b_ci, c_ci]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    x_pos = np.arange(len(parts))

    for i, (score, ci, color) in enumerate(zip(scores, cis, colors)):
        # 确保yerr值不为负
        yerr_lower = max(0, score - ci[0])
        yerr_upper = max(0, ci[1] - score)

        ax6.errorbar(x_pos[i], score, yerr=[[yerr_lower], [yerr_upper]],
                     fmt='o', color=color, markersize=10, capsize=8, capthick=2,
                     elinewidth=2, label=parts[i])

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(parts)
    ax6.set_ylabel('Score', fontsize=14)
    ax6.set_title('Scores with 95% Confidence Intervals', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # 6. 统计摘要
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = f"""
Evaluation Results Summary

Part A (Relevance):
  Average Score: {part_a_data['score']:.4f} ({part_a_data['score'] * 100:.2f}%)
  Relevant Records: {part_a_data['reasonable']}
  Irrelevant Records: {part_a_data['unreasonable']}
  95% CI: [{a_ci[0]:.4f}, {a_ci[1]:.4f}]

Part B (Ranking Quality):
  Average Score: {part_b_data['score']:.4f} ({part_b_data['score'] * 100:.2f}%)
  Valid Records: {part_b_data['valid']}/{part_b_data['total']}
  Invalid Records: {part_b_data['invalid']}
  95% CI: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]
"""

    if positions:
        summary_text += f"  Top-1 Hit Rate: {top_rates[1]:.2%} (95% CI: [{top_cis[1][0]:.3f}, {top_cis[1][1]:.3f}])\n"
        summary_text += f"  Top-3 Hit Rate: {top_rates[3]:.2%} (95% CI: [{top_cis[3][0]:.3f}, {top_cis[3][1]:.3f}])\n"
        summary_text += f"  Top-5 Hit Rate: {top_rates[5]:.2%} (95% CI: [{top_cis[5][0]:.3f}, {top_cis[5][1]:.3f}])\n"
        summary_text += f"  Top-10 Hit Rate: {top_rates[10]:.2%} (95% CI: [{top_cis[10][0]:.3f}, {top_cis[10][1]:.3f}])\n"

    summary_text += f"""
Part C (Overall Relevance):
  Average Score: {part_c_data['score']:.4f} ({part_c_data['score'] * 100:.2f}%)
  Relevant Records: {part_c_data['reasonable']}
  Irrelevant Records: {part_c_data['unreasonable']}
  95% CI: [{c_ci[0]:.4f}, {c_ci[1]:.4f}]
"""

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 添加时间戳
    fig.text(0.02, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=10, style='italic', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_report(results, output_prefix):
    """生成文本报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    part_b_avg, part_b_valid, part_b_total, part_b_invalid, part_b_details, part_b_issues, positions = calculate_part_b_score(
        annotations, recommendations)
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
            "invalid_records": part_b_invalid,
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

    # 生成文本报告
    generate_report(results, output_prefix)

    # 生成可视化报告
    try:
        visualization_path = generate_visualization(results, positions, f"{output_prefix}_visualization.png")
        print(f"可视化报告已保存为 {visualization_path}")
    except Exception as e:
        print(f"生成可视化报告时出错: {e}")
        print("将只生成文本报告")


if __name__ == "__main__":
    main()
