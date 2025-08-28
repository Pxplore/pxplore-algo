from data.snippet import get_all_snippets, get_snippet
from data.profile import add_profile
from data.session import get_session, add_session
import requests
import time
import json

raw_student_profile = json.load(open("./service/scripts/buffer/test_profile.json", "r"))

### Test Student Profiling

# response = requests.post("http://localhost:8899/student_profile", json={"behavioral_data": raw_student_profile})
# print(response.json())

# task_id = response.json()["task_id"]

# while True:
#     response = requests.get(f"http://localhost:8899/student_profile/status/{task_id}")
#     if response.json()["status"] == "completed":
#         break
#     time.sleep(5)

# student_profile_language = response.json()["language_analysis"]
# student_profile_behavior = response.json()["behavior_analysis"]

# student_profile = student_profile_language | student_profile_behavior
# student_profile["interaction_history"] = interaction_history

### Test Snippet Recommendation

student_profile = {"recommendation_strategy": raw_student_profile["recommendations"]}
interaction_history = "\n".join([f"小刘老师: {item['page_script'].replace('\n', '')}" for item in raw_student_profile["page_interactions"]])
title = "迈向通用的人工智能_第5讲_AI+X初探_第5讲_AI+_part1"

response = requests.post("http://localhost:8899/recommend", json={"student_profile": student_profile, "interaction_history": interaction_history, "title": title, "model": "gpt-4o"})
print(response.json())

task_id = response.json()["task_id"]

while True:
    response = requests.get(f"http://localhost:8899/recommend/status/{task_id}")
    if response.json()["status"] == "completed":
        break
    time.sleep(5)

recommend_id = response.json()["recommend_snippet_id"]
recommend_reason = response.json()["recommend_reason"]

### Test Style Adaptation

response = requests.post("http://localhost:8899/style_adapt", json={"student_profile": student_profile, "interaction_history": interaction_history, "title": title, "recommend_id": recommend_id, "recommend_reason": recommend_reason})
print(response.json())

task_id = response.json()["task_id"]
while True:
    response = requests.get(f"http://localhost:8899/style_adapt/status/{task_id}")
    if response.json()["status"] == "completed":
        break  
    time.sleep(5)

adaptation_result = response.json()["adaptation_result"]
print(adaptation_result)

add_session({
    "student_profile": raw_student_profile,
    "title": title,
    "recommend_snippet": get_snippet(recommend_id),
    "recommend_reason": recommend_reason,
    "adaptation_result": adaptation_result
})