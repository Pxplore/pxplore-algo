from pathlib import Path
import json
from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_snippet
from typing import Dict, Any

BASE_DIR = Path(__file__).parent

class StyleAdapter:

	def __init__(self, src_snippet: str, dst_snippet: str) -> None:
		self.src_scripts = self.parse_snippet(src_snippet)
		self.dst_scripts = self.parse_snippet(dst_snippet)

		self.style_adaptation_prompt_path = BASE_DIR / "prompts" / "style_adaptation.txt"
		with open(self.style_adaptation_prompt_path, "r", encoding="utf-8") as f:
			prompt_text = f.read()
		
		self.system_prompt = [
			{
				"role": "system",
				"content": prompt_text.replace("{src_scripts}", self.src_scripts).replace("{dst_scripts}", self.dst_scripts)
			}
		]

	def parse_snippet(self, snippet: Dict[str, Any]) -> str:
		content_list = [item['children'][1]['script'].replace('\n', '').strip() for item in snippet['children']]
		return "\n".join(content_list)

	def generate(self):
		job_id = OPENAI_SERVICE.trigger(
			parent_service="Pxplore",
			parent_job_id=None,
			use_cache=True,
			model="gpt-4o",
			messages=self.system_prompt
		)
		response = OPENAI_SERVICE.get_response_sync(job_id)
		response = json.loads(response.replace("```json", "").replace("```", "")) if response.startswith("```json") else json.loads(response)
		return response

if __name__ == "__main__":
	src_snippet = get_snippet("68808c0001fa9003100443f4")
	dst_snippet = get_snippet("68808c0001fa9003100443fa")

	results = StyleAdapter(src_snippet=src_snippet, dst_snippet=dst_snippet).generate()
	print(results)
