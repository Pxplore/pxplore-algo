from typing import List, Dict, Any
from service.scripts.hybrid_retriever import HybridRetriever
from config import QDRANT
from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_snippet
import json

class SnippetReranker:
    def __init__(self, content: str, user_profile: Dict[str, Any]):
        self.content = content
        self.user_profile = user_profile
        self.retriever = HybridRetriever()
        self.system_prompt = open("service/scripts/prompts/snippet_rerank.txt", "r").read().replace("{content}", content).replace("{user_profile}", str(user_profile))
    
    def parse_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Parse candidates into a string format for LLM input.
        """
        candidates_text = "\n\n".join([str(item["metadata"]) for item in candidates])
        return candidates_text


    def llm_rank_snippets(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to rerank and select the most suitable snippet.
        """
        candidates_text = self.parse_candidates(candidates)
        job_id = OPENAI_SERVICE.trigger(
            parent_service="Pxplore",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=[{"role": "user", "content": self.system_prompt.replace("{candidates}", candidates_text)}],
        )
        response = OPENAI_SERVICE.get_response_sync(job_id)
        response = json.loads(response.replace("```json", "").replace("```", "")) if response.startswith("```json") else json.loads(response)
        return response
    
    def run(self) -> Dict[str, Any]:
        """
        Two-stage recommendation algorithm:
        1. First stage: Use HybridRetriever to retrieve Top-K candidates;
        2. Second stage: Use LLM to rerank and select the most suitable snippet.
        """

        candidates = self.retriever.search(self.content)

        if not candidates:
            return {"snippet": None, "reason": "No candidates retrieved."}

        best_snippet = self.llm_rank_snippets(candidates)
        best_snippet = get_snippet(best_snippet["id"])

        return best_snippet


if __name__ == "__main__":
    test_case = {
    "content": "本节课我们学习了泰曼干涉仪的基本构造和工作原理。\n\n它主要包括光源、分束器、参考镜和测量镜等核心部件。\n\n通过分束与合束的干涉过程，我们可以测量微小的位移变化。\n\n学生学习了干涉图样的形成机理，以及相位差与光程差的定量关系。\n\n课堂上还讨论了干涉信号的输出形式与典型误差来源。",
    "user_profile": {
        "grade": "大二",
        "major": "光电信息科学与工程",
        "feedback": "学生基本掌握了干涉仪结构与光程差原理，对干涉图样形成理解较好，但对干涉信号的处理方式不太清晰。"
    }
    }
    snippet_reranker = SnippetReranker(test_case["content"], test_case["user_profile"])
    target_snippet = snippet_reranker.run()
    print(target_snippet)