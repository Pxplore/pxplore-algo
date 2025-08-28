from typing import Dict, Any, List
from service.llm.openai import OPENAI_SERVICE
from pathlib import Path
import json
import argparse
from tqdm import tqdm

BASE_DIR = Path(__file__).parent

class ProfileReward:
    def __init__(self):
        prompt_path = BASE_DIR / "prompts" / "eval_profile.txt"
        self.prompt = open(prompt_path, "r", encoding="utf-8").read()

    def handle_prompt(self, initial_state: Dict[str, Any], next_lesson_content: str):
        return f'''### 初始学习状态：
{initial_state}

### 下一课内容：
{next_lesson_content}'''

    def process_profiling(self, initial_state: Dict[str, Any], next_lesson_content: str):

        try:
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.handle_prompt(initial_state, next_lesson_content)}
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
        
        except Exception as e:
            print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate alignment scores for a given setting.")
    parser.add_argument('--setting', type=str, default="reward_steering_4o")
    args = parser.parse_args()
    setting = args.setting

    profile_reward = ProfileReward()
    recommendation_data = json.load(open(BASE_DIR / "data" / "test" / f"{setting}.json", "r"))
    print(f"recommendation_data: {len(recommendation_data)}")
    student_profile_reward = []
    for item in tqdm(recommendation_data):
        initial_state = item["student_profile"]
        if item["recommend_content"] is None or type(item["recommend_content"]) == str:
            response = initial_state
        else:
            next_lesson = "\n".join([lesson['children'][1]["script"] for lesson in item["recommend_content"]["children"]])
            response = profile_reward.process_profiling(initial_state, next_lesson)
        student_profile_reward.append({
            "course": item["course"],
            "initial_state": initial_state,
            "next_lesson": item["recommend_content"],
            "next_state": response
        })
        json.dump(student_profile_reward, open(BASE_DIR / "data" / "eval" / f"{setting}.json", "w"), indent=4, ensure_ascii=False)
    print(f"Saved {len(student_profile_reward)} items to {setting}.json")

