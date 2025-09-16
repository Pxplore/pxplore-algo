import requests
import time
import json

student_profile = json.load(open("./service/scripts/buffer/processed_student_profile.json", "r"))
print(f"Found {len(student_profile)} student profiles")

recommendation_results = json.load(open("./model/data/test/grpo_qwen3_v0.json", "r"))

results = []
for profile, recommend_content in zip(student_profile, recommendation_results): 
    interaction_history = [item["page_script"] for item in profile["page_interactions"]]
    recommend_id = recommend_content["recommend_snippet_id"]
    recommend_content = recommend_content["recommend_content"]
    recommend_reason = [item["children"][1]["script"] for item in recommend_content["children"]]
    title = f"{profile["meta"]["module_id"]}_{profile["meta"]["student_id"]}"
    
    response = requests.post("http://localhost:8899/adapt", json={"interaction_history": interaction_history, "title": title, "recommend_id": recommend_id, "recommend_reason": recommend_reason})

    task_id = response.json()["task_id"]

    while True:
        response = requests.get(f"http://localhost:8899/adapt/status/{task_id}")
        if response.json()["status"] == "completed":
            break
        time.sleep(5)

    adaptation_result = response.json()["adaptation_result"]
    results.append({
        "title": title,
        "original_content": interaction_history,
        "recommend_content": recommend_content,
        "recommend_reason": recommend_reason,
        "adaptation_result": adaptation_result
    })

    json.dump(results, open("./service/scripts/buffer/result_style_adaptation.json", "w"), indent=4, ensure_ascii=False)

print(f"Saved {len(results)} student profiles")

