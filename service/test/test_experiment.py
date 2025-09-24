import requests
import time
from data.snippet import get_snippet, parse_snippet

# 接口输入：src_snippet_id（当前上课snippet）
src_snippet_id = "6889c25b0b0dcac94374c598"

# 接口输入：interaction_history（学生和教师Agent交互历史）
# 我这里暂时用讲稿来做测试，实际调用时要改成每次学生提问Agent回复的交互历史（不包含讲稿）
# 格式类似于“学生: ...\n\n教师Agent: ...”
src_snippet = get_snippet(src_snippet_id)
src_content = parse_snippet(src_snippet)
src_title = f'{src_snippet["course_name"]}-{src_snippet["chapter_name"]}-{src_snippet["module_name"]}'
interactions = src_content.split("\n\n")

response = requests.post("http://localhost:8899/experiment", json={"src_snippet_id": src_snippet_id, "interaction_history": interactions})
print(response.json())

task_id = response.json()["task_id"]

while True:
    response = requests.get(f"http://localhost:8899/experiment/status/{task_id}")
    print(response.json())
    if response.json()["status"] == "completed":
        break
    time.sleep(5)

# 接口输出：session_id
session_id = response.json()["session_id"]
print(session_id)

