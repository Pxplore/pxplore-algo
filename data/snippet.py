from pymongo import MongoClient
from config import MONGO
from bson.objectid import ObjectId
from typing import Dict, Any
from datetime import datetime

collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.lecture_snippets

def add_snippet(snippet):
    if "children" not in snippet or len(snippet["children"]) == 0:
        print(f"Snippet has no children: {snippet["module_name"]}")
        return None

    existing_snippets = get_snippets_by_module(snippet["module_id"])
    for item in existing_snippets:
        if len(item["children"]) == len(snippet["children"]) and item["children"][0]["index"] == snippet["children"][0]["index"] and item["children"][-1]["index"] == snippet["children"][-1]["index"]:
            print(f"Snippet already exists: {item['_id']}")
            return item['_id']
    
    snippet["created_at"] = datetime.now().isoformat()
    _id = collection.insert_one(snippet).inserted_id
    return _id
    

def get_snippet(snippet_id):
    return collection.find_one({"_id": ObjectId(snippet_id)}, {"_id": 0, "created_at": 0})

def get_all_snippets():
    return list(collection.find({}))

def get_snippets_by_course(course_id):
    return list(collection.find({"course_id": course_id}))

def get_snippets_by_chapter(chapter_id):
    return list(collection.find({"chapter_id": chapter_id}))

def get_snippets_by_module(module_id):
    return list(collection.find({"module_id": module_id}))

def add_label(snippet_id, label):
    collection.update_one({"_id": snippet_id}, {"$set": {"label": label}})

def parse_snippet(snippet: Dict[str, Any]) -> str:
	content_list = [item['children'][1]['script'].replace('\n', '').strip() for item in snippet['children']]
	return "\n".join(content_list)

def parse_data(course, chapter, module):
    snippet_data = []
    for idx, snippet in enumerate(module['knowledge_snippet'], 1):
        start = snippet['start_index'] - 1  # 转为0基
        end = snippet['end_index']          # 切片右开
        snippet_data.append({
            "course_name": course['course_name'],
            "course_id": course['course_id'],
            "chapter_name": chapter['chapter_name'],
            "chapter_id": chapter['chapter_id'],
            "module_name": module['module_name'],
            "module_id": module['module_id'],
            "ppt_file_id": module['ppt_file_id'],
            "ppt": module['ppt'],
            "children": module['children'][start:end]
        })
    return snippet_data

def remove_snippet(snippet_id):
    collection.delete_one({"_id": snippet_id})