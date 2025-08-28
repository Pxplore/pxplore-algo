from pymongo import MongoClient
from config import MONGO
from bson.objectid import ObjectId
from typing import Dict, Any

collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.tasks

def add_task(task):
    existing_task = collection.find_one({"task": task["task"], "student_profile": task["student_profile"]})
    if existing_task:
        return str(existing_task["_id"])
    _id = collection.insert_one(task).inserted_id
    return str(_id)

def get_task(task_id):
    return collection.find_one({"_id": ObjectId(task_id)})

def update_task(task_id, task):
    collection.update_one({"_id": ObjectId(task_id)}, {"$set": task})

def delete_task(task_id):
    collection.delete_one({"_id": ObjectId(task_id)})

