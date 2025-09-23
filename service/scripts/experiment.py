from pathlib import Path
import sys
from typing import Dict, Any
import asyncio
from datetime import datetime
from service.llm.openai import OPENAI_SERVICE
from service.scripts.student_profiling import StudentProfiling
from service.scripts.snippet_recommender import SnippetRecommender
from service.scripts.style_adaptation import StyleAdapter
from data.task import add_task, get_task, update_task
from data.snippet import get_snippet, parse_snippet
from data.session import add_session
from time import sleep

BASE_DIR = Path(__file__).parent

STATUS_PENDING = "pending"
STATUS_PROFILE = "profiling"
STATUS_RECOMMEND = "recommending"
STATUS_ADAPT = "adapting"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

snippet_recommender = SnippetRecommender()
style_adaptater = StyleAdapter()

class ExperimentController:

	async def process_profiling(self, task_id: str, interaction_history: str):
		prompt_text = open(BASE_DIR / "prompts" / "student_simple.txt", "r", encoding="utf-8").read()
		prompt_text = prompt_text.replace("[interaction_history]", interaction_history)

		job_id = OPENAI_SERVICE.trigger(
			parent_service="Pxplore",
			parent_job_id=None,
			use_cache=True,
			model="gpt-4o",
			messages=[
				{"role": "system", "content": prompt_text}
			]
		)
		response = OPENAI_SERVICE.get_response_sync(job_id)
		response = OPENAI_SERVICE.parse_json_response(response)
		return response

	async def process_experiment(self, task_id: str, interaction_history: str, src_snippet_id: str):
		try:
			src_snippet = get_snippet(src_snippet_id)
			title = f'{src_snippet["course_name"]}-{src_snippet["chapter_name"]}-{src_snippet["module_name"]}'
			history_scripts = parse_snippet(src_snippet)

			# student profiling
			student_profile = await self.process_profiling(task_id, interaction_history)
			update_task(task_id, {"status": STATUS_PROFILE, "student_profile": student_profile})

			# snippet recommendation
			recommend_task_id = snippet_recommender.run(student_profile, interaction_history, title)
			recommend_task_status = None
			while True:
				recommend_task_status = snippet_recommender.get_recommendation_task(recommend_task_id)
				if recommend_task_status["status"] == STATUS_COMPLETED:
					break
				elif recommend_task_status["status"] == STATUS_FAILED:
					update_task(task_id, {"status": STATUS_FAILED, "error": recommend_task_status["error"]})
					return
				sleep(3)
			recommend_id = recommend_task_status["recommend_snippet_id"]
			recommend_reason = recommend_task_status["recommend_reason"]
			recommend_snippet = get_snippet(recommend_id)
			update_task(task_id, {"status": STATUS_RECOMMEND, "recommend_snippet": recommend_snippet, "recommend_reason": recommend_reason})

			# style adaptation
			adapt_task_id = style_adaptater.run(history_scripts, title, recommend_id, recommend_reason)
			adapt_task_status = None
			while True:
				adapt_task_status = style_adaptater.get_adaptation_task(adapt_task_id)
				if adapt_task_status["status"] == STATUS_COMPLETED:
					break
				elif adapt_task_status["status"] == STATUS_FAILED:
					update_task(task_id, {"status": STATUS_FAILED, "error": adapt_task_status["error"]})
					return
				sleep(3)
			adaptation_result = adapt_task_status["adaptation_result"]
			update_task(task_id, {"status": STATUS_ADAPT, "adaptation_result": adaptation_result})

			# save session
			session_id = add_session({
				"title": title,
				"student_profile": student_profile,
				"recommend_snippet": recommend_snippet,
				"recommend_reason": recommend_reason,
				"adaptation_result": adaptation_result
			})
			update_task(task_id, {"status": STATUS_COMPLETED, "session_id": session_id})

		except Exception as e:
			update_task(task_id, {"status": STATUS_FAILED, "error": str(e)})
	
	async def run(self, interaction_history: str, src_snippet_id: str) -> str:

		task_id = add_task({
			"status": STATUS_PENDING,
			"task": "style_adaptation",
			"src_snippet_id": src_snippet_id,
            "interaction_history": interaction_history,
            "student_profile": None,
            "recommend_snippet": None,
            "recommend_reason": None,
            "adaptation_result": None,
            "session_id": None,
            "error": None
		})

		asyncio.create_task(self.process_experiment(task_id, interaction_history, src_snippet_id))

		return str(task_id)
	
	async def get_experiment_task(self, task_id: str) -> Dict[str, Any]:
		task = get_task(task_id)
		if not task:
			return {"status": STATUS_FAILED, "error": f"Task {task_id} not found."}
		
		return {
			"status": task["status"],
			"session_id": task.get("session_id")
		}


