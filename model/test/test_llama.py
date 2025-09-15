from data.snippet import get_all_snippets, get_snippet
from service.llm.llama import LLaMA
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--setting', type=str, default="prompt_llama")
args = parser.parse_args()
setting = args.setting

student_profile = json.load(open("./model/data/testset.json", "r"))
print(f"Found {len(student_profile)} student profiles")

def get_recommend_content(student_profile, candidates):
    
    system_prompt = open("./service/scripts/prompts/snippet_selection.txt", "r", encoding="utf-8").read()
    user_input = {
        "recommendation_strategy": student_profile,
        "candidates": candidates
    }

    response = LLaMA.call_model([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False, indent=2)}
    ])
    try:
        response = json.loads(response)
    except Exception as e:
        response = {
            "selected_candidate": {
                "id": "error",
            },
            "reason": response,
        }

    return {
        "recommend_snippet_id": response["selected_candidate"]["id"],
        "recommend_reason": response["reason"]
    }

result = []
for index, item in enumerate(tqdm(student_profile)):
    student_profile = item["student_profile"] if "prompt" not in setting else {"interaction_history": [f"{subitem['role']}: {subitem['content']}" for subitem in item["interaction_history"]]}
    candidates = item["recommend_candidates"]

    response = get_recommend_content(student_profile, candidates)
    try:
        recommend_content = get_snippet(response["recommend_snippet_id"])
    except Exception as e:
        recommend_content = f'Error: {e}'

    item["recommend_snippet_id"] = response["recommend_snippet_id"]
    item["recommend_content"] = recommend_content
    item["recommend_reason"] = response["recommend_reason"]

    result.append(item)

    json.dump(result, open(f"./model/data/test/{setting}.json", "w"), indent=4, ensure_ascii=False)


