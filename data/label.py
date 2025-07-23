from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_all_snippets, add_label
import json

class DataLabeler:
    def __init__(self, snippet):
        self.snippet = snippet
        self.script_texts = [item['children'][1]['script'].replace('\n', '').strip() for item in snippet['children']]
        prompt = open("data/prompts/label.txt", "r").read()
        self.system_prompt = dict(
            role="system",
            content=prompt.replace("{script_text}", "\n\n".join(self.script_texts))
        )

    def label(self):
        if 'label' in self.snippet:
            return self.snippet['label']
        
        job_id = OPENAI_SERVICE.trigger(
            parent_service="Pxplore",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=[self.system_prompt]
        )
        response = OPENAI_SERVICE.get_response_sync(job_id)
        response = json.loads(response.replace("```json", "").replace("```", "")) if response.startswith("```json") else json.loads(response)
        add_label(self.snippet['_id'], response)
        return response 

