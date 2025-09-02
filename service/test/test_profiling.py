import requests
import time
import json

raw_student_profile = json.load(open("./service/scripts/buffer/raw_student_profile.json", "r"))
print(f"Found {len(raw_student_profile)} student profiles")

results = []
for profile in raw_student_profile: 
    response = requests.post("http://localhost:8899/student_profile", json={"behavioral_data": profile})
    print(response.json())

    task_id = response.json()["task_id"]

    while True:
        response = requests.get(f"http://localhost:8899/student_profile/status/{task_id}")
        if response.json()["status"] == "completed":
            break
        time.sleep(5)

    results.append({
        "raw_data": profile,
        "language_analysis": response.json()["language_analysis"],
        "behavior_analysis": response.json()["behavior_analysis"],
        "processed_episodes": response.json()["processed_episodes"],
        "finalize_analysis": response.json()["finalize_analysis"]
    })

    json.dump(results, open("./service/scripts/buffer/processed_student_profile.json", "w"), indent=4, ensure_ascii=False)

print(f"Saved {len(results)} student profiles")

