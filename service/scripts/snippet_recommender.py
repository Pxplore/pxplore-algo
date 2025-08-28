from typing import List, Dict, Any
from datetime import datetime
import uuid
import asyncio
from service.scripts.hybrid_retriever import HybridRetriever
from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_snippet, parse_snippet
from data.task import add_task, get_task, update_task
from pathlib import Path

BASE_DIR = Path(__file__).parent

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class SnippetRecommender:
    def __init__(self):
        self.retriever = HybridRetriever()
        prompt_path = BASE_DIR / "prompts" / "snippet_selection.txt"
        self.system_prompt = open(prompt_path, "r", encoding="utf-8").read()

    def parse_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Parse candidates into a string format for LLM input.
        """
        candidates_text = "\n\n".join([str(item["metadata"]) for item in candidates])
        return candidates_text

    def handle_prompt(self, student_profile: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
        candidates_text = self.parse_candidates(candidates)
        return f'''### recommendation_strategy
{str(student_profile)}

### candidates
{candidates_text}
'''

    async def rank_snippets(self, student_profile: Dict[str, Any], candidates: List[Dict[str, Any]], model: str = "gpt-4o") -> Dict[str, Any]:
        """
        Use LLM to rerank and select the most suitable snippet.
        """
        user_prompt = self.handle_prompt(student_profile, candidates)
        job_id = OPENAI_SERVICE.trigger(
            parent_service="Pxplore",
            parent_job_id=None,
            use_cache=True,
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt}, 
                {"role": "user", "content": user_prompt}
            ],
        )
        response = OPENAI_SERVICE.get_response_sync(job_id)
        response = OPENAI_SERVICE.parse_json_response(response)
        return response 
    
    async def process_recommendation(self, task_id: str, student_profile: Dict[str, Any], interaction_history: str, title: str = None, model: str = None):

        try:        
            candidates = self.retriever.search(interaction_history, title)

            if not candidates:
                return {"snippet": None, "reason": "No candidates retrieved."}

            update_task(task_id, {"recommend_candidates": candidates})

            response = await self.rank_snippets(student_profile, candidates, model)

            update_task(task_id, {"status": STATUS_COMPLETED, "llm_response": response, "recommend_snippet_id": response.get("selected_candidate", {}).get("id")})

        except Exception as e:
            update_task(task_id, {"status": STATUS_FAILED, "error": str(e)})

    async def run(self, student_profile: Dict[str, Any], interaction_history: str, title: str = None, model: str = None) -> str:
        """
        Two-stage recommendation algorithm:
        1. First stage: Use HybridRetriever to retrieve Top-K candidates;
        2. Second stage: Use LLM to rerank and select the most suitable snippet.
        """
        task_id = add_task({
            "status": STATUS_PENDING,
            "task": "snippet_selection",
            "created_at": datetime.now().isoformat(), 
            "student_profile": student_profile,
            "interaction_history": interaction_history,
            "title": title,
            "recommend_candidates": [],
            "llm_response": None,
            "recommend_snippet_id": None,
            "error": None
        })

        asyncio.create_task(self.process_recommendation(task_id, student_profile, interaction_history, title, model))

        return str(task_id)
    
    async def get_recommendation_task(self, task_id: str) -> Dict[str, Any]:
        task = get_task(task_id)
        if not task:
            return {"status": STATUS_FAILED, "error": f"Task {task_id} not found."}
        
        return {
            "status": task["status"],
            "created_at": task["created_at"].isoformat() if isinstance(task["created_at"], datetime) else task["created_at"],
            "error": task.get("error"),
            "student_profile": task.get("student_profile"),
            "interaction_history": task.get("interaction_history"),
            "title": task.get("title"),
            "recommend_snippet_id": task.get("recommend_snippet_id"),
            "recommend_candidates": task.get("recommend_candidates"),
            "recommend_reason": task.get("llm_response", {}).get("reason")
        }

