from data.snippet import get_all_snippets, get_snippet
from service.llm.openai import OPENAI_SERVICE
import requests
import time
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Calculate alignment scores for a given setting.")
parser.add_argument('--setting', type=str, default="steering_4o")
parser.add_argument('--model', type=str, default="gpt-4o")
args = parser.parse_args()
setting = args.setting
model = args.model

student_profile = json.load(open("./model/data/testset.json", "r"))
print(f"Found {len(student_profile)} student profiles")

def get_recommend_content(student_profile, candidates, model):
    
    system_prompt = open("./service/scripts/prompts/snippet_selection.txt", "r", encoding="utf-8").read()
    user_input = {
        "recommendation_strategy": student_profile,
        "candidates": candidates
    }

    job_id = OPENAI_SERVICE.trigger(
        parent_service="Pxplore-Model",
        parent_job_id=None,
        use_cache=True,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": json.dumps(user_input, ensure_ascii=False, indent=2)}
        ],
    )
    response = OPENAI_SERVICE.get_response_sync(job_id)
    response = OPENAI_SERVICE.parse_json_response(response)

    snippet_id = response.get("selected_candidate", {}).get("id")
    if snippet_id is None:
        snippet_id = response.get("selected_candidate", {}).get("metadata", {}).get("id")

    return {
        "recommend_snippet_id": snippet_id,
        "recommend_reason": response.get("reason", "No reason provided")
    }

result = []
for index, item in enumerate(tqdm(student_profile)):
    student_profile = item["student_profile"] if "prompt" not in setting else {"interaction_history": [f"{subitem['role']}: {subitem['content']}" for subitem in item["interaction_history"]]}
    candidates = item["recommend_candidates"]

    response = get_recommend_content(student_profile, candidates, model)

    try:
        recommend_content = get_snippet(response["recommend_snippet_id"]) if response.get("recommend_snippet_id") else ""
    except Exception as e:
        recommend_content = f'Error: {e}'

    item["recommend_snippet_id"] = response["recommend_snippet_id"]
    item["recommend_content"] = recommend_content
    item["recommend_reason"] = response["recommend_reason"]

    result.append(item)

    json.dump(result, open(f"./model/data/test/{setting}.json", "w"), indent=4, ensure_ascii=False)


