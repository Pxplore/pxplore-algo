from typing import Dict, Any, List
from service.llm.openai import OPENAI_SERVICE
from pathlib import Path
import json
from tqdm import tqdm

BASE_DIR = Path(__file__).parent

class StudentProfiling:
    def __init__(self):
        prompt_path = BASE_DIR / "prompts" / "gen_profile.txt"
        self.prompt = open(prompt_path, "r", encoding="utf-8").read()

    def process_profiling(self, student_name: str, interaction_history: List[Dict[str, Any]]):
        try:
            messages = [
                {"role": "system", "content": self.prompt.replace("{student_name}", student_name)},
                {"role": "user", "content": "\n".join([f"{item['time']} {item['role']}: {item['content']}" for item in interaction_history])}
            ]

            job_id = OPENAI_SERVICE.trigger(
                parent_service="Pxplore",
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
    student_profiling = StudentProfiling()
    interaction_history = json.load(open(BASE_DIR / "data" / "student_data.json", "r"))
    print(f"interaction_history: {len(interaction_history)}")
    student_profile = []
    for item in tqdm(interaction_history):
        interactions = item["interactions"]
        response = student_profiling.process_profiling(item["name"], interactions)
        student_profile.append({
            "name": item["name"],
            "course_name": item["course_name"],
            "chapter_name": item["chapter_name"],
            "module_name": item["module_name"],
            "interactions": interactions,
            "profile": response
        })
        json.dump(student_profile, open(BASE_DIR / "data" / "student_profile.json", "w"), indent=4, ensure_ascii=False)

