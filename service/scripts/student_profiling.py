from typing import Dict, Any
import asyncio
from datetime import datetime
from service.llm.openai import OPENAI_SERVICE
from service.utils.episodes_processor import build_episodes
from data.task import add_task, get_task, update_task
from pathlib import Path
from tqdm import tqdm
import time
import json
BASE_DIR = Path(__file__).parent

STATUS_PENDING = "pending"
STATUS_COMPLETED_LANGUAGE = "completed_language"
STATUS_COMPLETED_BEHAVIOR = "completed_behavior"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class StudentProfiling:
    def __init__(self):
        prompt_language = BASE_DIR / "prompts" / "student_language.txt"
        prompt_behavior = BASE_DIR / "prompts" / "student_behavior.txt"
        prompt_finalize = BASE_DIR / "prompts" / "student_finalize.txt"
        self.prompt_language = open(prompt_language, "r", encoding="utf-8").read()
        self.prompt_behavior = open(prompt_behavior, "r", encoding="utf-8").read()
        self.prompt_finalize = open(prompt_finalize, "r", encoding="utf-8").read()

    def process_episodes(self, processed_behavioral_data: Dict[str, Any]):
        analysis = build_episodes(processed_behavioral_data)
        return {
            "meta": processed_behavioral_data.get("meta", {}),
            "processed_analysis": analysis
        }

    async def process_language_analysis(self, task_id: str, behavioral_data: Dict[str, Any]):
        try:
            messages = [
                {"role": "system", "content": self.prompt_language},
                {"role": "user", "content": str(behavioral_data)}
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

            update_task(task_id, {"status": STATUS_COMPLETED_LANGUAGE, "language_analysis": response})
            return response
        
        except Exception as e:
            update_task(task_id, {"status": STATUS_FAILED, "error": str(e)})

    async def process_behavior_analysis(self, task_id: str, behavioral_data: Dict[str, Any]):
        try:
            messages = [
                {"role": "system", "content": self.prompt_behavior},
                {"role": "user", "content": str(behavioral_data)}
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

            update_task(task_id, {"status": STATUS_COMPLETED_BEHAVIOR, "behavior_analysis": response})

            return response
        
        except Exception as e:
            update_task(task_id, {"status": STATUS_FAILED, "error": str(e)})

    async def process_finalize(self, task_id: str, behavioral_data: Dict[str, Any]):
        try:
            messages = [
                {"role": "system", "content": self.prompt_finalize},
                {"role": "user", "content": str(behavioral_data)}
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

            update_task(task_id, {"status": STATUS_COMPLETED, "finalize_analysis": response})
            return response
        except Exception as e:
            update_task(task_id, {"status": STATUS_FAILED, "error": str(e)})

    async def process_profiling(self, task_id: str, behavioral_data: Dict[str, Any]):
        if behavioral_data.get("discussion_threads"):
            response = await self.process_language_analysis(task_id, behavioral_data)
            behavioral_data["discussion_threads"] = response.get("discussion_threads")

        response = await self.process_behavior_analysis(task_id, behavioral_data)
        for key in response.keys():
            behavioral_data[key] = response.get(key)

        processed_episodes = self.process_episodes(behavioral_data)
        update_task(task_id, {"processed_episodes": processed_episodes})

        response = await self.process_finalize(task_id, behavioral_data)
        return response

    async def run(self, behavioral_data: Dict[str, Any]) -> str:
        task_id = add_task({
            "status": STATUS_PENDING,
            "task": "student_profiling",
            "student_profile": behavioral_data,
            "language_analysis": None,
            "behavior_analysis": None,
            "finalize_analysis": None,
            "processed_episodes": None,
            "error": None
        })

        asyncio.create_task(self.process_profiling(task_id, behavioral_data))

        return str(task_id)
    
    async def get_profiling_task(self, task_id: str) -> Dict[str, Any]:
        task = get_task(task_id)
        if not task:
            return {"status": STATUS_FAILED, "error": f"Task {task_id} not found."}
        
        return {
            "status": task["status"],
            "error": task.get("error"),
            "language_analysis": task.get("language_analysis"),
            "behavior_analysis": task.get("behavior_analysis"),
            "finalize_analysis": task.get("finalize_analysis")
        }

if __name__ == "__main__":
    processed_student_profiles = json.load(open("./service/test/test_output/processed_student_profiles.json", "r", encoding="utf-8"))
    result = []
    for profile in tqdm(processed_student_profiles):
        taskid = asyncio.run(StudentProfiling().run(profile))

        while True:
            task = get_task(taskid)
            if task["status"] == STATUS_COMPLETED:
                result.append({
                    "task_id": taskid,
                    "behavioral_data": profile,
                    "finalize_analysis": task["finalize_analysis"]
                })
                break
            time.sleep(5)

        json.dump(result, open("./service/test/test_output/result_student_profiling.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
