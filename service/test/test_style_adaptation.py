import requests
import time
import json

student_profile = json.load(open("./service/scripts/buffer/processed_student_profiles.json", "r"))
print(f"Found {len(student_profile)} student profiles")

recommendation_results = json.load(open("./model/data/test/steering_4o.json", "r"))

results = []
for profile, recommend_content in zip(student_profile, recommendation_results): 
    history_content = [item["page_script"] for item in profile["page_interactions"]]
    recommend_id = recommend_content["recommend_snippet_id"]
    selected_content = [item["children"][1]["script"] for item in recommend_content["recommend_content"]["children"]]
    recommend_reason = recommend_content["recommend_reason"]
    title = f"{profile["meta"]["module_id"]}_{profile["meta"]["student_id"]}"
    
    response = requests.post("http://localhost:8899/adapt", json={"history_content": "\n".join(history_content), "title": title, "recommend_id": recommend_id, "recommend_reason": recommend_reason})
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
        "original_content": history_content,
        "recommend_content": selected_content,
        "recommend_reason": recommend_reason,
        "adaptation_result": adaptation_result
    })

    print(adaptation_result["start_speech"])
    print(adaptation_result["refined_scripts"])
    print(adaptation_result["end_speech"])
    json.dump(results, open("./service/scripts/buffer/result_style_adaptation.json", "w"), indent=4, ensure_ascii=False)
    exit()

print(f"Saved {len(results)} student profiles")

