from pathlib import Path
from pymongo import MongoClient
from typing import Dict, Any
import uuid
import asyncio
from datetime import datetime
from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_snippet, parse_snippet
from config import MONGO

BASE_DIR = Path(__file__).parent

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class StyleAdapter:

	def __init__(self) -> None:
		prompt_path = BASE_DIR / "prompts" / "style_adaptation.txt"
		self.prompt_text = open(prompt_path, "r", encoding="utf-8").read()
		self.tasks_collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.tasks

	def process_adaptation(self, task_id: str, src_snippet: Dict[str, Any], dst_snippet: Dict[str, Any]):
		try:
			src_scripts = parse_snippet(src_snippet)
			dst_scripts = parse_snippet(dst_snippet)

			prompt_text = self.prompt_text.replace("{src_scripts}", src_scripts).replace("{dst_scripts}", dst_scripts)

			job_id = OPENAI_SERVICE.trigger(
				parent_service="Pxplore",
				parent_job_id=None,
				use_cache=True,
				model="gpt-4o",
				messages=[
					{
						"role": "system",
						"content": prompt_text
					}
				]
			)
			response = OPENAI_SERVICE.get_response_sync(job_id)
			response = OPENAI_SERVICE.parse_json_response(response)

			self.tasks_collection.update_one(
				{"task_id": task_id},
				{"$set": {"status": STATUS_COMPLETED, "adaptation_result": response}}
			)

		
		except Exception as e:
			self.tasks_collection.update_one(
				{"task_id": task_id},
				{"$set": {"status": STATUS_FAILED, "error": str(e)}}
			)
	
	def run(self, src_snippet: Dict[str, Any], dst_snippet: Dict[str, Any]):

		task_id = str(uuid.uuid4())

		self.tasks_collection.insert_one({
			"task_id": task_id,
			"status": STATUS_PENDING,
			"created_at": datetime.now(),
			"updated_at": datetime.now(),	
			"src_snippet_id": src_snippet["_id"],
			"dst_snippet_id": dst_snippet["_id"],
			"adaptation_result": None,
			"error": None
		})

		asyncio.create_task(self.process_adaptation(task_id, src_snippet, dst_snippet))

		return task_id
	
	def get_adaptation_task(self, task_id: str) -> Dict[str, Any]:
		task = self.tasks_collection.find_one({"task_id": task_id})
		if not task:
			return {"status": STATUS_FAILED, "reason": f"Task {task_id} not found."}
		
		return {
			"id": task["task_id"],
			"status": task["status"],
			"created_at": task["created_at"].isoformat() if isinstance(task["created_at"], datetime) else task["created_at"],
			"updated_at": task["updated_at"].isoformat() if isinstance(task["updated_at"], datetime) else task["updated_at"],
			"error": task.get("error"),
			"adaptation_result": task.get("adaptation_result")
		}


if __name__ == "__main__":
	src_snippet = get_snippet("68808c0001fa9003100443f4")
	dst_snippet = get_snippet("68808c0001fa9003100443fa")

	results = StyleAdapter().run(src_snippet, dst_snippet)
	print(results)
