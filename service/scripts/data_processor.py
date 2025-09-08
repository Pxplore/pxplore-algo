import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import re
import sys
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取项目根目录的路径（假设项目根目录是 pxplore-algo）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
# 将项目根目录添加到 Python 路径
sys.path.append(project_root)
# 现在可以导入 service 模块了

# 导入LLM服务
from service.llm.openai import OPENAI_SERVICE


class DataProcessor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.prompt_dir = self.base_dir / "prompts"

        # 加载所有prompt模板
        self.prompts = {}
        prompt_files = ["meta", "page_timeline", "discussion_threads", "review_threads", "quizzes"]
        for file in prompt_files:
            try:
                with open(self.prompt_dir / f"{file}.txt", "r", encoding="utf-8") as f:
                    self.prompts[file] = f.read()
            except FileNotFoundError:
                print(f"警告: 未找到prompt文件 {file}.txt")
                self.prompts[file] = ""

    def load_data(self, lecture_path: str, testset_path: str) -> Dict[str, Any]:
        """加载并合并数据"""
        with open(lecture_path, 'r', encoding='utf-8') as f:
            lecture_data = json.load(f)

        with open(testset_path, 'r', encoding='utf-8') as f:
            testset_data = json.load(f)

        return {
            "lecture_data": lecture_data,
            "testset_data": testset_data
        }

    async def extract_meta(self, testset_item: Dict[str, Any]) -> Dict[str, Any]:
        """使用AI模型提取元数据，包括时间处理"""
        input_data = {
            "course": testset_item["course"],
            "interactions": testset_item["interaction_history"]
        }

        messages = [
            {"role": "system", "content": self.prompts["meta"]},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)}
        ]

        job_id = OPENAI_SERVICE.trigger(
            parent_service="DataProcessor",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=messages
        )

        response = OPENAI_SERVICE.get_response_sync(job_id)
        meta = OPENAI_SERVICE.parse_json_response(response)

        return meta

    async def process_page_timeline(self, lecture_item: Dict[str, Any], testset_item: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """处理页面时间线"""
        input_data = {
            "pages": lecture_item["children"],
            "interactions": testset_item["interaction_history"]
        }

        messages = [
            {"role": "system", "content": self.prompts["page_timeline"]},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)}
        ]

        job_id = OPENAI_SERVICE.trigger(
            parent_service="DataProcessor",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=messages
        )

        response = OPENAI_SERVICE.get_response_sync(job_id)
        return OPENAI_SERVICE.parse_json_response(response)

    async def process_discussion_threads(self, lecture_item: Dict[str, Any], testset_item: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """处理讨论线程"""
        input_data = {
            "pages": lecture_item["children"],
            "interactions": testset_item["interaction_history"]
        }

        messages = [
            {"role": "system", "content": self.prompts["discussion_threads"]},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)}
        ]

        job_id = OPENAI_SERVICE.trigger(
            parent_service="DataProcessor",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=messages
        )

        response = OPENAI_SERVICE.get_response_sync(job_id)
        return OPENAI_SERVICE.parse_json_response(response)

    async def process_review_threads(self, lecture_item: Dict[str, Any], testset_item: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """处理复习线程"""
        input_data = {
            "pages": lecture_item["children"],
            "interactions": testset_item["interaction_history"]
        }

        messages = [
            {"role": "system", "content": self.prompts["review_threads"]},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)}
        ]

        job_id = OPENAI_SERVICE.trigger(
            parent_service="DataProcessor",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=messages
        )

        response = OPENAI_SERVICE.get_response_sync(job_id)
        return OPENAI_SERVICE.parse_json_response(response)

    async def process_quizzes(self, lecture_item: Dict[str, Any], testset_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理测验数据"""
        input_data = {
            "pages": lecture_item["children"],
            "interactions": testset_item["interaction_history"]
        }

        messages = [
            {"role": "system", "content": self.prompts["quizzes"]},
            {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)}
        ]

        job_id = OPENAI_SERVICE.trigger(
            parent_service="DataProcessor",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=messages
        )

        response = OPENAI_SERVICE.get_response_sync(job_id)
        return OPENAI_SERVICE.parse_json_response(response)

    async def process_single_item(self, lecture_item: Dict[str, Any], testset_item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个数据项"""
        # 提取元数据
        meta = await self.extract_meta(testset_item)

        # 并行处理各个部分
        results = await asyncio.gather(
            self.process_page_timeline(lecture_item, testset_item),
            self.process_discussion_threads(lecture_item, testset_item),
            self.process_review_threads(lecture_item, testset_item),
            self.process_quizzes(lecture_item, testset_item)
        )

        page_interactions, discussion_threads, review_threads, quizzes = results

        return {
            "meta": meta,
            "page_interactions": page_interactions,
            "discussion_threads": discussion_threads,
            "review_threads": review_threads,
            "quizzes": quizzes
        }

    async def process_all_data(self, lecture_path: str, testset_path: str, output_dir: str):
        """处理所有数据"""
        data = self.load_data(lecture_path, testset_path)

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 找出匹配的课程
        lecture_map = {item["course"]: item for item in data["lecture_data"]}

        # 处理每个testset项目
        for testset_item in data["testset_data"]:
            course_name = testset_item["course"]
            if course_name in lecture_map:
                lecture_item = lecture_map[course_name]
                result = await self.process_single_item(lecture_item, testset_item)

                # 保存结果
                student_id = result["meta"]["student_id"]
                # 清理文件名中的非法字符
                safe_course_name = re.sub(r'[<>:"/\\|?*]', '_', course_name)
                output_file = output_path / f"{safe_course_name}_{student_id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"Processed and saved: {output_file}")
            else:
                print(f"No lecture data found for course: {course_name}")


# 示例用法
async def main():
    processor = DataProcessor()

    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建相对于脚本位置的路径
    base_dir = os.path.join(script_dir, "..", "test")  # 上一级目录中的test文件夹

    lecture_path = os.path.join(base_dir, "lecture_snippets_end.json")
    testset_path = os.path.join(base_dir, "testset.json")
    output_dir = os.path.join(base_dir, "test_output")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    await processor.process_all_data(lecture_path, testset_path, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
