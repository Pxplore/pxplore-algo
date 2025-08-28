import json
import os
from pymongo import MongoClient
from config import MONGO
from bson.objectid import ObjectId
from typing import Dict, Any
from data.snippet import parse_data, add_snippet, get_all_snippets, remove_snippet

collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.lecture_snippets

def save_data(data):
    knowledge_snippets = []
    for course in data:
        for chapter in course['children']:
            for module in chapter['children']:
                snippets = parse_data(course, chapter, module)
                knowledge_snippets.extend(snippets)
    return knowledge_snippets

if __name__ == '__main__':
    with open('./dataset/output/results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for snippet in save_data(data):
        add_snippet(snippet)
