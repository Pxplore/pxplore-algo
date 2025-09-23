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
STATUS_SUGGESTION = "suggestion"
STATUS_TRANSITION = "transition"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class StyleAdapter:

	def __init__(self) -> None:
		prompt_path = BASE_DIR / "prompts" / "style_adaptation.txt"
		transition_prompt_path = BASE_DIR / "prompts" / "style_adaptation_transition.txt"
		suggestion_prompt_path = BASE_DIR / "prompts" / "style_adaptation_suggestion.txt"

		self.prompt_text = open(prompt_path, "r", encoding="utf-8").read()
		self.transition_prompt_text = open(transition_prompt_path, "r", encoding="utf-8").read()
		self.suggestion_prompt_text = open(suggestion_prompt_path, "r", encoding="utf-8").read()

	def parse_recommended_snippet(self, recommend_id: str) -> str:
		recommend_snippet = get_snippet(recommend_id)
		recommend_summary = recommend_snippet.get("label",{}).get("summary", "")
		recommend_content = parse_snippet(recommend_snippet)
		return recommend_summary, recommend_content

	async def process_suggestion(self, task_id: str, history_content: str, recommend_content_summary: str, recommendation_reason: str) -> str:
		job_id = OPENAI_SERVICE.trigger(
			parent_service="Pxplore",
			parent_job_id=None,
			use_cache=True,
			model="gpt-4o",
			messages=[
				{"role": "system", "content": self.suggestion_prompt_text.replace("[history_content]", history_content).replace("[recommend_content_summary]", recommend_content_summary).replace("[recommend_reason]", recommendation_reason)}
			]
		)
		response = OPENAI_SERVICE.get_response_sync(job_id)
		update_task(task_id, {"status": STATUS_SUGGESTION, "adaptation_suggestion": response})
		return response

	async def process_transition(self, task_id: str, recommend_content: str, adaptation_suggestion: str) -> str:
		job_id = OPENAI_SERVICE.trigger(
			parent_service="Pxplore",
			parent_job_id=None,
			use_cache=True,
			model="gpt-4o",
			messages=[
				{"role": "system", "content": self.transition_prompt_text.replace("[recommend_content]", recommend_content).replace("[adaptation_suggestion]", adaptation_suggestion)}
			]
		)
		response = OPENAI_SERVICE.get_response_sync(job_id)
		response = OPENAI_SERVICE.parse_json_response(response)
		update_task(task_id, {"status": STATUS_TRANSITION, "adaptation_result": response})
		return response["start_speech"], response["end_speech"]

	async def process_adaptation(self, task_id: str, history_content: str, recommend_id: str, recommend_reason: str):
		try:
			recommend_summary, recommend_content = self.parse_recommended_snippet(recommend_id)
			adaptation_suggestion = await self.process_suggestion(task_id, history_content, recommend_summary, recommend_reason)
			start_speech, end_speech = await self.process_transition(task_id, recommend_content, adaptation_suggestion)
			system_prompt = self.prompt_text.replace("[recommend_content]", str(recommend_content.split("\n\n"))).replace("[adaptation_suggestion]", adaptation_suggestion)
			scripts = recommend_content.split("\n\n")
			grouped_content = [
				"\n\n".join(scripts[i:i+3])
				for i in range(0, len(scripts), 3)
			]
			refined_scripts = []
			for chunk in grouped_content:
				system_prompt = self.prompt_text.replace("[recommend_content]", str(chunk.split("\n\n"))).replace("[adaptation_suggestion]", adaptation_suggestion)
				job_id = OPENAI_SERVICE.trigger(
					parent_service="Pxplore",
					parent_job_id=None,
					use_cache=True,
					model="gpt-4o",
					messages=[
						{"role": "system", "content": system_prompt}
					]
				)
				response = OPENAI_SERVICE.get_response_sync(job_id)
				response = OPENAI_SERVICE.parse_json_response(response)
				response_scripts = response.get("refined_scripts")
				if response_scripts and len(response_scripts) == len(chunk.split("\n\n")):
					refined_scripts += response_scripts
				else:
					refined_scripts += chunk.split("\n\n")

			update_task(task_id, {"status": STATUS_COMPLETED, "adaptation_result": {"start_speech": start_speech, "end_speech": end_speech, "refined_scripts": refined_scripts}})
		
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
			"adaptation_suggestion": None,
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


