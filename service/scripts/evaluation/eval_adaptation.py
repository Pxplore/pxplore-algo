from service.llm.openai import OPENAI_SERVICE
from service.llm.volcark import VOLCARK_SERVICE
from service.llm.anthropic import ANTHROPIC_SERVICE
from data.snippet import get_snippet, parse_snippet
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent

prompt_text = open(BASE_DIR / "eval_adaptation.txt", "r", encoding="utf-8").read()
history_contents = json.load(open(BASE_DIR / ".." / "buffer" / "result_recommendation.json", "r", encoding="utf-8"))
result_adaptation = json.load(open(BASE_DIR / ".." / "buffer" / "result_style_adaptation.json", "r", encoding="utf-8"))

def eval_content(user_input):
    job_id_4o = OPENAI_SERVICE.trigger(
        parent_service="Pxplore",
        parent_job_id=None,
        use_cache=True,
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt_text}, {"role": "user", "content": user_input}]
    )
    response_4o = OPENAI_SERVICE.get_response_sync(job_id_4o)
    response_4o = OPENAI_SERVICE.parse_json_response(response_4o)

    job_id_r1 = VOLCARK_SERVICE.trigger(
        parent_service="Pxplore",
        parent_job_id=None,
        use_cache=True,
        model="deepseek-r1",
        messages=[{"role": "system", "content": prompt_text}, {"role": "user", "content": user_input}]
    )
    response_r1, reasoning = VOLCARK_SERVICE.get_response_sync(job_id_r1)
    response_r1 = OPENAI_SERVICE.parse_json_response(response_r1)

    job_id_claude = ANTHROPIC_SERVICE.trigger(
        parent_service="Pxplore",
        parent_job_id=None,
        use_cache=True,
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "system", "content": prompt_text}, {"role": "user", "content": user_input}]
    )
    response_claude = ANTHROPIC_SERVICE.get_response_sync(job_id_claude)
    response_claude = OPENAI_SERVICE.parse_json_response(response_claude)

    return response_4o, response_r1, response_claude

def calculate_score(response_4o, response_r1, response_claude):
    error = None
    try:
        avg_cc = round((response_4o["CC"]["score"] + response_r1["CC"]["score"] + response_claude["CC"]["score"]) / 3, 2)
        avg_pc = round((response_4o["PC"]["score"] + response_r1["PC"]["score"] + response_claude["PC"]["score"]) / 3, 2)
        avg_pe = round((response_4o["PE"]["score"] + response_r1["PE"]["score"] + response_claude["PE"]["score"]) / 3, 2)
        avg_ps = round((response_4o["PS"]["score"] + response_r1["PS"]["score"] + response_claude["PS"]["score"]) / 3, 2)
        avg_ce = round((response_4o["CE"]["score"] + response_r1["CE"]["score"] + response_claude["CE"]["score"]) / 3, 2)
        avg_ma = round((response_4o["MA"]["score"] + response_r1["MA"]["score"] + response_claude["MA"]["score"]) / 3, 2)
        avg_rs = round((response_4o["RS"]["score"] + response_r1["RS"]["score"] + response_claude["RS"]["score"]) / 3, 2)
        avg_overall = round((avg_cc + avg_pc + avg_pe + avg_ps + avg_ce + avg_ma + avg_rs) / 7, 2)
        average_score = {
            "CC": avg_cc,
            "PC": avg_pc,
            "PE": avg_pe,
            "PS": avg_ps,
            "CE": avg_ce,
            "MA": avg_ma,
            "RS": avg_rs,
            "overall": avg_overall
        }
    except Exception as e:
        average_score = {"CC": 0, "PC": 0, "PE": 0, "PS": 0, "CE": 0, "MA": 0, "RS": 0, "overall": 0}
        error = str(e)
    return average_score, error

results = []
for history, adapt in tqdm(zip(history_contents, result_adaptation)):
    history_content = parse_snippet(get_snippet(history["recommend_snippet_id"]))
    recommend_reason = history["recommend_reason"]

    original_content = "\n\n".join(adapt["recommend_content"])
    adaptation_content = "\n\n".join([adapt["adaptation_result"]["start_speech"]] + adapt["adaptation_result"]["refined_scripts"] + [adapt["adaptation_result"]["end_speech"]])

    original_user_input = str(dict(history_content=history_content, recommend_reason=recommend_reason, recommend_content=original_content))
    adaptation_user_input = str(dict(history_content=history_content, recommend_reason=recommend_reason, recommend_content=adaptation_content))

    original_response_4o, original_response_r1, original_response_claude = eval_content(original_user_input)
    adaptation_response_4o, adaptation_response_r1, adaptation_response_claude = eval_content(adaptation_user_input)

    original_average_score, original_error = calculate_score(original_response_4o, original_response_r1, original_response_claude)
    adaptation_average_score, adaptation_error = calculate_score(adaptation_response_4o, adaptation_response_r1, adaptation_response_claude)

    results.append({
        "average_score": {
            "original": original_average_score,
            "adaptation": adaptation_average_score
        },
        "evaluation_gpt-4o": {
            "original": original_response_4o,
            "adaptation": adaptation_response_4o
        },
        "evaluation_deepseek-r1": {
            "original": original_response_r1,
            "adaptation": adaptation_response_r1
        },
        "evaluation_claude-3-7-sonnet": {
            "original": original_response_claude,
            "adaptation": adaptation_response_claude
        },
        "error": {
            "original": original_error,
            "adaptation": adaptation_error
        },
    })

    json.dump(results, open(BASE_DIR / ".." / "buffer" / "eval_adaptation.json", "w"), indent=4, ensure_ascii=False)
