from service.llm.openai import OPENAI_SERVICE
from service.llm.volcark import VOLCARK_SERVICE
from service.llm.anthropic import ANTHROPIC_SERVICE
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent

prompt_text = open(BASE_DIR / "../prompts/eval_profile.txt", "r", encoding="utf-8").read()
result_profiling = json.load(open(BASE_DIR / ".." / "buffer" / "result_student_profiling.json", "r", encoding="utf-8"))

results = []
for profile in tqdm(result_profiling):
    raw_interaction_logs = profile["behavioral_data"]
    learner_profile = profile["finalize_analysis"]
    user_input = str(dict(raw_interaction_logs=raw_interaction_logs, learner_profile=learner_profile))

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

    error = None
    try:
        avg_iia = round((response_4o["IIA"]["score"] + response_r1["IIA"]["score"] + response_claude["IIA"]["score"]) / 3, 2)
        avg_csa = round((response_4o["CSA"]["score"] + response_r1["CSA"]["score"] + response_claude["CSA"]["score"]) / 3, 2)
        avg_lni = round((response_4o["LNI"]["score"] + response_r1["LNI"]["score"] + response_claude["LNI"]["score"]) / 3, 2)
        avg_opc = round((response_4o["OPC"]["score"] + response_r1["OPC"]["score"] + response_claude["OPC"]["score"]) / 3, 2)
        avg_tcg = round((response_4o["TCG"]["score"] + response_r1["TCG"]["score"] + response_claude["TCG"]["score"]) / 3, 2)
        avg_au = round((response_4o["AU"]["score"] + response_r1["AU"]["score"] + response_claude["AU"]["score"]) / 3, 2)
        avg_ie = round((response_4o["IE"]["score"] + response_r1["IE"]["score"] + response_claude["IE"]["score"]) / 3, 2)
        avg_overall = round((avg_iia + avg_csa + avg_lni + avg_opc + avg_tcg + avg_au + avg_ie) / 7, 2)
        average_score = {
            "IIA": avg_iia,
            "CSA": avg_csa,
            "LNI": avg_lni,
            "OPC": avg_opc,
            "TCG": avg_tcg,
            "AU": avg_au,
            "IE": avg_ie,
            "overall": avg_overall
        }
    except Exception as e:
        average_score = {"IIA": 0, "CSA": 0, "LNI": 0, "OPC": 0, "TCG": 0, "AU": 0, "IE": 0, "overall": 0}
        error = str(e)

    results.append({
        "average_score": average_score,
        "evaluation_gpt-4o": response_4o,
        "evaluation_deepseek-r1": response_r1,
        "evaluation_claude-3-7-sonnet": response_claude,
        "error": error
    })

    json.dump(results, open(BASE_DIR / ".." / "buffer" / "eval_profile.json", "w"), indent=4, ensure_ascii=False)
