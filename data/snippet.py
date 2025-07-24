import json
import os
from pymongo import MongoClient
from config import MONGO
from bson.objectid import ObjectId
collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.lecture_snippets

def add_snippet(snippet):
    _id = collection.insert_one(snippet).inserted_id
    return _id

def get_snippet(snippet_id):
    return collection.find_one({"_id": ObjectId(snippet_id)})

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

def save_data(data):
    knowledge_snippets = []
    for course in data:
        for chapter in course['children']:
            for module in chapter['children']:
                snippets = parse_data(course, chapter, module)
                knowledge_snippets.extend(snippets)
    return knowledge_snippets

if __name__ == '__main__':
    with open('./data/output/result.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for snippet in save_data(data):
        add_snippet(snippet)
