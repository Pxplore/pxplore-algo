from typing import List, Dict, Any
from pymongo import MongoClient
from datetime import datetime
import uuid
import asyncio
from service.scripts.hybrid_retriever import HybridRetriever
from config import QDRANT, MONGO
from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_snippet, parse_snippet


STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class SnippetRecommender:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.system_prompt = open("service/scripts/prompts/snippet_rerank.txt", "r").read()
        self.tasks_collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.tasks

    def parse_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Parse candidates into a string format for LLM input.
        """
        candidates_text = "\n\n".join([str(item["metadata"]) for item in candidates])
        return candidates_text

    def rank_snippets(self, content: str, student_profile: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to rerank and select the most suitable snippet.
        """
        candidates_text = self.parse_candidates(candidates)
        prompt_text = self.system_prompt.replace("{content}", content).replace("{student_profile}", str(student_profile)).replace("{candidates}", candidates_text)
        job_id = OPENAI_SERVICE.trigger(
            parent_service="Pxplore",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_text}],
        )
        response = OPENAI_SERVICE.get_response_sync(job_id)
        response = OPENAI_SERVICE.parse_json_response(response)
        return response
    
    def process_recommendation(self, task_id: str, key_snippet: Dict[str, Any], student_profile: Dict[str, Any]):

        try:
            content = parse_snippet(key_snippet)
            candidates = self.retriever.search(content)

            if not candidates:
                return {"snippet": None, "reason": "No candidates retrieved."}

            best_snippet = self.rank_snippets(content, student_profile, candidates)
            # best_snippet = get_snippet(best_snippet["id"])

            self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": {"status": STATUS_COMPLETED, "recommend_snippet_id": best_snippet["id"]}}
            )

        except Exception as e:
            self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": {"status": STATUS_FAILED, "error": str(e)}}
            )

    def run(self, key_snippet: Dict[str, Any], student_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Two-stage recommendation algorithm:
        1. First stage: Use HybridRetriever to retrieve Top-K candidates;
        2. Second stage: Use LLM to rerank and select the most suitable snippet.
        """

        task_id = str(uuid.uuid4())

        self.tasks_collection.insert_one({
            "task_id": task_id,
            "status": STATUS_PENDING,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "student_profile": student_profile,
            "key_snippet_id": key_snippet["_id"],
            "recommend_snippet_id": None,
            "error": None
        })

        asyncio.create_task(self.process_recommendation(task_id, key_snippet, student_profile))

        return task_id
    
    def get_recommendation_task(self, task_id: str) -> Dict[str, Any]:
        task = self.tasks_collection.find_one({"task_id": task_id})
        if not task:
            return {"status": STATUS_FAILED, "reason": f"Task {task_id} not found."}
        
        return {
            "id": task["task_id"],
            "status": task["status"],
            "created_at": task["created_at"].isoformat() if isinstance(task["created_at"], datetime) else task["created_at"],
            "updated_at": task["updated_at"].isoformat() if isinstance(task["updated_at"], datetime) else task["updated_at"],
            "error": task.get("error"),
            "recommend_snippet_id": task.get("recommend_snippet_id")
        }


if __name__ == "__main__":
    test_case = {
    "key_snippet": get_snippet("68808c0001fa9003100443f4"),
    "student_profile": {
        "grade": "大二",
        "major": "光电信息科学与工程",
        "feedback": "学生基本掌握了干涉仪结构与光程差原理，对干涉图样形成理解较好，但对干涉信号的处理方式不太清晰。"
        }
    }
    snippet_recommender = SnippetRecommender()
    target_snippet = snippet_recommender.run(test_case["key_snippet"], test_case["student_profile"])
    print(target_snippet)

