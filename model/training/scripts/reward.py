from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_snippet
from pathlib import Path
from typing import Dict, Any
import json

BASE_DIR = Path(__file__).parent

class PxploreReward:

    def __init__(self):
        self.prompt = open(BASE_DIR / "reward_prompt.txt", "r", encoding="utf-8").read()

    def call(self, initial_state: Dict[str, Any], next_lesson_content: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": json.dumps({"初始学习状态": initial_state, "下一课内容": next_lesson_content})},
        ]

        job_id = OPENAI_SERVICE.trigger(
            parent_service="Pxplore-Reward",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=messages
        )

        response = OPENAI_SERVICE.get_response_sync(job_id)
        response = OPENAI_SERVICE.parse_json_response(response)
        return response

    def calculate_reward(self, next_state) -> float:
        dimension_stats = {
            'long_term_objective': {'total': 0, 'aligned': 0},
            'short_term_objective': {'total': 0, 'aligned': 0},
            'implicit_motivation': {'total': 0, 'aligned': 0},
            'explicit_motivation': {'total': 0, 'aligned': 0}
        }
        for dimension in dimension_stats.keys():
            if dimension in next_state and isinstance(next_state[dimension], list):
                for objective in next_state[dimension]:
                    if isinstance(objective, dict) and 'is_aligned' in objective:
                        dimension_stats[dimension]['total'] += 1
                        if objective['is_aligned']:
                            dimension_stats[dimension]['aligned'] += 1
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

        self.print_reward(results)

        total_aligned = sum(stats['aligned_count'] for stats in results.values())
        total_count = sum(stats['total_count'] for stats in results.values())
        reward = total_aligned / total_count

        return reward

    def print_reward(self, results: Dict[str, Dict[str, Any]]):
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

    def run(self, initial_state: Dict[str, Any], next_lesson_id: str) -> float:
        next_lesson = get_snippet(next_lesson_id)
        if next_lesson is None:
            return -1
        next_lesson_content = "\n".join([lesson['children'][1]["script"] for lesson in next_lesson["children"]])
        next_state = self.call(initial_state, next_lesson_content)
        reward = self.calculate_reward(next_state)
        return reward

if __name__ == "__main__":

    data = json.load(open("./model/data/test/steering_4o.json", "r"))
    initial_state = data[0]["student_profile"]
    next_lesson = "\n".join([lesson['children'][1]["script"] for lesson in data[0]["recommend_content"]["children"]])
    
    reward = PxploreReward()
    next_state = reward.call(initial_state, next_lesson)
    reward = reward.calculate_reward(next_state)
    print(reward)
