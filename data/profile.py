from pymongo import MongoClient
from config import MONGO
from bson.objectid import ObjectId
from typing import Dict, Any

collection = MongoClient(MONGO.HOST, MONGO.PORT).pxplore.student_profile

def add_profile(profile):
    _id = collection.insert_one(profile).inserted_id
    return _id

def get_profile(profile_id):
    return collection.find_one({"_id": ObjectId(profile_id)})

def get_all_profiles():
    return list(collection.find({}))

def update_profile(profile_id, profile):
    collection.update_one({"_id": ObjectId(profile_id)}, {"$set": profile})

def delete_profile(profile_id):
    collection.delete_one({"_id": ObjectId(profile_id)})

def get_interaction_history(profile_id):
    profile = get_profile(profile_id)
    return profile.get("interaction_history", [])


