from pydantic import BaseModel
from typing import Dict, Any, List

class TaskResponse(BaseModel):
    task_id: str
    message: str

class RecommendRequest(BaseModel):
    student_profile: Dict[str, Any]
    interaction_history: str
    title: str = None
    model: str = None

class RecommendResponse(BaseModel):
    status: str
    recommend_snippet_id: str
    recommend_candidates: List[Dict[str, Any]]
    recommend_reason: str
    student_profile: Dict[str, Any]
    interaction_history: str
    title: str
    message: str

class AdaptRequest(BaseModel):
    interaction_history: str
    title: str
    recommend_id: str
    recommend_reason: str

class AdaptResponse(BaseModel):
    status: str
    adaptation_result: Dict[str, Any]
    message: str

class SessionRequest(BaseModel):
    session_id: str
    scripts: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    message: str

class SessionResponse(BaseModel):
    student_profile: Dict[str, Any]
    title: str
    recommend_snippet: Dict[str, Any]
    recommend_reason: str
    adaptation_result: Dict[str, Any]

class StudentProfileRequest(BaseModel):
    behavioral_data: Dict[str, Any]

class StudentProfileResponse(BaseModel):
    status: str
    language_analysis: Dict[str, Any]
    behavior_analysis: Dict[str, Any]
    finalize_analysis: Dict[str, Any]
    processed_episodes: Dict[str, Any]
    message: str
