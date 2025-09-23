import requests
import time
import json
from data.snippet import get_snippet


recommendation_results = json.load(open("./service/scripts/buffer/result_recommendation.json", "r"))

results = []
for item in recommendation_results: 
    profile = item["student_profile"]
    history_content = item["original_content"]
    recommend_id = item["recommend_snippet_id"]
    recommend_content = get_snippet(recommend_id)
    selected_content = [child["children"][1]["script"] for child in recommend_content["children"]]
    recommend_reason = item["recommend_reason"]
    title = item["title"]
    
    response = requests.post("http://localhost:8899/adapt", json={"history_content": "\n\n".join(history_content), "title": title, "recommend_id": recommend_id, "recommend_reason": recommend_reason})
    task_id = response.json()["task_id"]
    print(response.json())

    while True:
        response = requests.get(f"http://localhost:8899/adapt/status/{task_id}")
        if response.json()["status"] == "completed" or  response.json()["status"] == "failed":
            break
        time.sleep(5)

    adaptation_result = response.json()["adaptation_result"]
    results.append({
        "title": title,
        "student_profile": profile,
        "recommend_reason": recommend_reason,
        "recommend_content": selected_content,
        "adaptation_result": adaptation_result
    })
    print(title)
    json.dump(results, open("./service/scripts/buffer/result_style_adaptation.json", "w"), indent=4, ensure_ascii=False)



