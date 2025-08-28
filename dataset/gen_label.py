from service.llm.openai import OPENAI_SERVICE
from data.snippet import get_all_snippets, add_label
import json

class DataLabeler:
    def __init__(self, snippet):
        self.snippet = snippet
        self.script_texts = [item['children'][1]['script'].replace('\n', '').strip() for item in snippet['children']]
        prompt = open("dataset/prompts/label.txt", "r").read()
        self.system_prompt = dict(
            role="system",
            content=prompt.replace("{script_text}", "\n\n".join(self.script_texts))
        )

    def label(self):
        if 'label' in self.snippet:
            print(f"Snippet already has label: {self.snippet['_id']}")
            return self.snippet['label']
        
        job_id = OPENAI_SERVICE.trigger(
            parent_service="Pxplore",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=[self.system_prompt]
        )
        response = OPENAI_SERVICE.get_response_sync(job_id)
        response = OPENAI_SERVICE.parse_json_response(response)
        add_label(self.snippet['_id'], response)
        return response 

if __name__ == "__main__":
    snippets = get_all_snippets()
    for snippet in snippets:
        label = DataLabeler(snippet).label()
        print(label)
