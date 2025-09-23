from pathlib import Path
import sys
from typing import Dict, Any
import asyncio
from datetime import datetime
from service.llm.openai import OPENAI_SERVICE
from service.llm.volcark import VOLCARK_SERVICE
from data.snippet import get_snippet, parse_snippet
from data.task import add_task, get_task, update_task

BASE_DIR = Path(__file__).parent

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class StyleAdapter:

	def __init__(self) -> None:
		prompt_path = BASE_DIR / "prompts" / "style_adaptation.txt"
		self.prompt_text = open(prompt_path, "r", encoding="utf-8").read()
		summarize_prompt_path = BASE_DIR / "prompts" / "style_adaptation_summarize.txt"
		self.summarize_prompt_text = open(summarize_prompt_path, "r", encoding="utf-8").read()

	def handle_prompt(self, prompt_text: str, history_content: str, recommend_id: str, recommend_reason: str) -> str:
		recommend_snippet = get_snippet(recommend_id)
		recommend_content = parse_snippet(recommend_snippet)

		return prompt_text.replace("[history_content]", history_content).replace("[recommend_content]", str(recommend_content.split("\n\n"))).replace("[recommend_reason]", recommend_reason)

	async def process_summarize(self, history_content: str) -> str:
		job_id = OPENAI_SERVICE.trigger(
			parent_service="Pxplore",
			parent_job_id=None,
			use_cache=True,
			model="gpt-4o",
			messages=[
				{"role": "system", "content": self.summarize_prompt_text.replace("[history_content]", history_content)}
			]
		)
		response = OPENAI_SERVICE.get_response_sync(job_id)
		return response

	async def process_adaptation(self, task_id: str, history_content: str, recommend_id: str, recommend_reason: str):
		try:

			history_content_summary = await self.process_summarize(history_content)
			print(history_content_summary)
			system_prompt = self.handle_prompt(self.prompt_text, history_content_summary, recommend_id, recommend_reason)

			job_id = OPENAI_SERVICE.trigger(
				parent_service="Pxplore",
				parent_job_id=None,
				use_cache=False,
				model="gpt-4o",
				messages=[
					{"role": "system", "content": system_prompt}
				]
			)
			response = OPENAI_SERVICE.get_response_sync(job_id)
			response = OPENAI_SERVICE.parse_json_response(response)

			update_task(task_id, {"status": STATUS_COMPLETED, "adaptation_result": response, "history_content_summary": history_content_summary})
		
		except Exception as e:
			update_task(task_id, {"status": STATUS_FAILED, "error": str(e)})
	
	async def run(self, history_content: str, title: str, recommend_id: str, recommend_reason: str) -> str:

		task_id = add_task({
			"status": STATUS_PENDING,
			"task": "style_adaptation",
			"history_content": history_content,
			"title": title,
			"recommend_id": recommend_id,
			"recommend_reason": recommend_reason,
			"adaptation_result": None,
			"history_content_summary": None,
			"error": None
		})

		asyncio.create_task(self.process_adaptation(task_id, history_content, recommend_id, recommend_reason))

		return str(task_id)
	
	async def get_adaptation_task(self, task_id: str) -> Dict[str, Any]:
		task = get_task(task_id)
		if not task:
			return {"status": STATUS_FAILED, "error": f"Task {task_id} not found."}
		
		return {
			"status": task["status"],
			"error": task.get("error"),
			"adaptation_result": task.get("adaptation_result")
		}


