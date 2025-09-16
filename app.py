import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from utils import get_logger
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from service.scripts.snippet_recommender import SnippetRecommender
from service.scripts.style_adaptation import StyleAdapter
from service.scripts.session_controller import SessionController
from service.scripts.student_profiling import StudentProfiling
import json
from base import *

logger = get_logger(__name__, __file__)

snippet_recommender = None
style_adapter = None
session_controller = None
student_profiling = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global snippet_recommender, style_adapter, session_controller, student_profiling
    try:
        snippet_recommender = SnippetRecommender()
        style_adapter = StyleAdapter()
        session_controller = SessionController()
        student_profiling = StudentProfiling()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    yield

app = FastAPI(
    title="Pxplore API",
    description="API for Pxplore",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/recommend", response_model=TaskResponse)
async def snippet_recommender(request: RecommendRequest):
    if snippet_recommender is None:
        raise HTTPException(status_code=503, detail="SnippetRecommender service not ready")
    try:
        task_id = await snippet_recommender.run(request.student_profile, request.interaction_history, request.title, request.model)
        return {
            "task_id": task_id,
            "message": "recommendation algorithm started successfully"
        }
    except Exception as e:
        logger.error(f"Error starting recommendation algorithm: {e}, {request}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/recommend/status/{task_id}", response_model=RecommendResponse)
async def get_recommendation_status(task_id: str):
    if snippet_recommender is None:
        raise HTTPException(status_code=503, detail="SnippetRecommender service not ready")
    try:
        task_status = await snippet_recommender.get_recommendation_task(task_id)
        if task_status.get("status") == "completed":
            message = "recommendation algorithm completed successfully"
        elif task_status.get("status") == "failed":
            message = task_status.get("error") 
        else:
            message = "recommendation algorithm is in progress"
        return {
            "status": task_status.get("status", "failed"),
            "recommend_snippet_id": task_status.get("recommend_snippet_id", ""),
            "recommend_candidates": task_status.get("recommend_candidates", []),
            "recommend_reason": task_status.get("recommend_reason", ""),
            "student_profile": task_status.get("student_profile", {}),
            "interaction_history": task_status.get("interaction_history", ""),
            "title": task_status.get("title", ""),
            "message": message
        }
    except Exception as e:
        logger.error(f"Error getting recommendation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/adapt", response_model=TaskResponse)
async def style_adaptation(request: AdaptRequest):
    if style_adapter is None:
        raise HTTPException(status_code=503, detail="StyleAdapter service not ready")
    try:
        task_id = await style_adapter.run(request.history_content, request.title, request.recommend_id, request.recommend_reason)
        return {
            "task_id": task_id,
            "message": "style adaptation started successfully"
        }
    except Exception as e:
        logger.error(f"Error starting style adaptation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/adapt/status/{task_id}", response_model=AdaptResponse)
async def get_adaptation_status(task_id: str):
    if style_adapter is None:
        raise HTTPException(status_code=503, detail="StyleAdapter service not ready")
    try:
        task_status = await style_adapter.get_adaptation_task(task_id)
        if task_status.get("status") == "completed":
            message = "style adaptation completed successfully"
        elif task_status.get("status") == "failed":
            message = task_status.get("error") 
        else:
            message = "style adaptation is in progress"
        return {
            "status": task_status.get("status"),
            "adaptation_result": task_status.get("adaptation_result"),
            "message": message
        }
    except Exception as e:
        logger.error(f"Error getting adaptation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/handle_message", response_model=str)
async def handle_message(request: SessionRequest):
    if session_controller is None:
        raise HTTPException(status_code=503, detail="SessionController service not ready")
    try:
        message = session_controller.handle_message(request.session_id, request.scripts, request.history, request.message)
        return message
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session_data(session_id: str):
    if session_controller is None:
        raise HTTPException(status_code=503, detail="SessionController service not ready")
    try:
        session = session_controller.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/student_profile", response_model=TaskResponse)
async def student_profile(request: StudentProfileRequest):
    if student_profiling is None:
        raise HTTPException(status_code=503, detail="StudentProfiling service not ready")
    try:
        task_id = await student_profiling.run(request.behavioral_data)
        return {
            "task_id": task_id,
            "message": "student profiling started successfully"
        }
    except Exception as e:
        logger.error(f"Error starting student profiling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/student_profile/status/{task_id}", response_model=StudentProfileResponse)
async def get_student_profile_status(task_id: str):
    if student_profiling is None:
        raise HTTPException(status_code=503, detail="StudentProfiling service not ready")
    try:
        task_status = await student_profiling.get_profiling_task(task_id)
        if task_status.get("status") == "completed":
            message = "student profiling completed successfully"
        elif task_status.get("status") == "failed":
            message = task_status.get("error") 
        else:
            message = "student profiling is in progress"
        return {
            "status": task_status.get("status"),
            "language_analysis": task_status.get("language_analysis"),
            "behavior_analysis": task_status.get("behavior_analysis"),
            "message": message
        }
    except Exception as e:
        logger.error(f"Error getting student profiling status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """
    Start the FastAPI application with uvicorn.
    """
    uvicorn.run("app:app", host="0.0.0.0", port=8899, reload=True)

if __name__ == "__main__":
    start() 