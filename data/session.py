from pymongo import MongoClient
from config import MONGO
from bson.objectid import ObjectId
from typing import Dict, Any

collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.classroom_session

def add_session(session):
    existing_session = collection.find_one({"student_profile": session["student_profile"], "title": session["title"]})
    if existing_session:
        collection.update_one({"_id": existing_session["_id"]}, {"$set": session})
        return existing_session["_id"]
    else:
        _id = collection.insert_one(session).inserted_id
        return _id

def get_session(session_id):
    return collection.find_one({"_id": ObjectId(session_id)})

def get_all_sessions():
    return list(collection.find({}))

def update_session(session_id, session):
    collection.update_one({"_id": ObjectId(session_id)}, {"$set": session})

if __name__ == "__main__":
    
    add_session({
        "student_profile": {
            "bloom_level": "分析",
            "learning_status": "该学生具备较强的问题意识，能主动区分相关概念并探究系统机制，表现出较高的理解与分析能力。",
            "interaction_history": []
        },
        "recommend_snippet_id": "688314455286c4247fa68ec8"
    })