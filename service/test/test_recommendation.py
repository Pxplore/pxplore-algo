import requests
import time
import json

student_profile = json.load(open("./service/scripts/buffer/result_student_profiling.json", "r"))
print(f"Found {len(student_profile)} student profiles")

results = []
for item in student_profile: 
    profile = item["finalize_analysis"]
    interaction_history = [interaction["page_script"] for interaction in item["behavioral_data"]["page_interactions"]]
    title = f'{item["behavioral_data"]["meta"]["student_id"]}_{item["behavioral_data"]["meta"]["session_id"]}_{item["behavioral_data"]["meta"]["module_id"]}'
    
    response = requests.post("http://localhost:8899/recommend", json={"student_profile": profile, "interaction_history": "\n\n".join(interaction_history), "title": title})
    task_id = response.json()["task_id"]
    print(response.json())

    while True:
        response = requests.get(f"http://localhost:8899/recommend/status/{task_id}")
        if response.json()["status"] == "completed" or  response.json()["status"] == "failed":
            break
        time.sleep(3)

    print(profile)
    print(response.json()["recommend_reason"])

    results.append({
        "title": title,
        "student_profile": profile,
        "original_content": interaction_history,
        "recommend_candidates": response.json()["recommend_candidates"],
        "recommend_snippet_id": response.json()["recommend_snippet_id"],
        "recommend_reason": response.json()["recommend_reason"]
    })

    json.dump(results, open("./service/scripts/buffer/result_recommendation.json", "w"), indent=4, ensure_ascii=False)

print(f"Saved {len(results)} student profiles")

