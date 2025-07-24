import uvicorn
from fastapi import FastAPI, HTTPException
from utils import get_logger
from typing import Dict, Any
from pydantic import BaseModel, Field
from service.scripts.snippet_recommender import SnippetRecommender
from service.scripts.style_adaptation import StyleAdapter

logger = get_logger(__name__)

app = FastAPI(
    title="Pxplore API",
    description="API for Pxplore",
    version="1.0.0"
)

recommender = None
adapter = None

@app.on_event("startup")
async def startup_db_client():
    global recommender, adapter
    try:
        recommender = SnippetRecommender()
        adapter = StyleAdapter()
        logger.info("SnippetRecommender and StyleAdapter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

class TaskResponse(BaseModel):
    task_id: str
    message: str

class RecommendRequest(BaseModel):
    key_snippet: Dict[str, Any]
    student_profile: Dict[str, Any]

class AdaptRequest(BaseModel):
    src_snippet: Dict[str, Any]
    dst_snippet: Dict[str, Any]

@app.post("/recommend", response_model=TaskResponse)
async def snippet_recommender(request: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="SnippetRecommender service not ready")
    try:
        task_id = await recommender.run(request.key_snippet, request.student_profile)
        return {
            "task_id": task_id,
            "message": "recommendation algorithm started successfully"
        }
    except Exception as e:
        logger.error(f"Error starting recommendation algorithm: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/recommend/status", response_model=TaskResponse)
async def get_recommendation_status(task_id: str):
    if recommender is None:
        raise HTTPException(status_code=503, detail="SnippetRecommender service not ready")
    try:
        return await recommender.get_recommendation_task(task_id)
    except Exception as e:
        logger.error(f"Error getting recommendation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/style_adapt", response_model=TaskResponse)
async def style_adaptation(request: AdaptRequest):
    if adapter is None:
        raise HTTPException(status_code=503, detail="StyleAdapter service not ready")
    try:
        task_id = await adapter.run(request.src_snippet, request.dst_snippet)
        return {
            "task_id": task_id,
            "message": "style adaptation started successfully"
        }
    except Exception as e:
        logger.error(f"Error starting style adaptation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/style_adapt/status", response_model=TaskResponse)
async def get_adaptation_status(task_id: str):
    if adapter is None:
        raise HTTPException(status_code=503, detail="StyleAdapter service not ready")
    try:
        return await adapter.get_adaptation_task(task_id)
    except Exception as e:
        logger.error(f"Error getting adaptation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """
    Start the FastAPI application with uvicorn.
    """
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start() 