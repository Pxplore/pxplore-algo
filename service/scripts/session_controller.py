from typing import List, Dict
from data.session import add_session, get_session
from service.llm.openai import OPENAI_SERVICE
from pathlib import Path

BASE_DIR = Path(__file__).parent

class SessionController:
    def __init__(self):
        prompt_path = BASE_DIR / "prompts" / "teacher_agent.txt"
        self.system_prompt = open(prompt_path, "r", encoding="utf-8").read()

    def get_session(self, session_id: str):
        return get_session(session_id)

    def handle_message(self, session_id: str, scripts: List[Dict], history: List[Dict], message: str):
        prompt_text = self.system_prompt.replace("{scripts}", str(scripts)).replace("{history}", str(history)).replace("{message}", message)
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
        return response

if __name__ == "__main__":
    session_controller = SessionController()
    session_controller.handle_message("6883487a2958b1ab3fd82a70", [], [], "你好")