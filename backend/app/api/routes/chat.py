from fastapi import APIRouter, HTTPException
from uuid import uuid4
from datetime import datetime

from app.models.chat import ChatRequest, ChatResponse, Message

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def create_chat_message(request: ChatRequest):
    """
    Create a new chat message and trigger agent processing.
    
    For streaming responses, use the WebSocket endpoint instead.
    """
    thread_id = request.thread_id or uuid4()
    
    # Create user message
    user_message = Message(
        role="user",
        content=request.message,
    )
    
    # For now, return a mock response
    # Real implementation would trigger the orchestrator
    response_message = Message(
        role="assistant",
        content=f"Processing your query: '{request.message}' in {request.mode} mode...",
    )
    
    return ChatResponse(
        thread_id=thread_id,
        message=response_message,
        agents_triggered=["yutori", "fabricate", "freepik"] if request.mode == "paper" else [],
    )


@router.get("/threads")
async def list_threads():
    """List all conversation threads."""
    # Mock implementation
    return {"threads": []}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get a specific conversation thread."""
    # Mock implementation
    return {
        "id": thread_id,
        "title": "Mock Thread",
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
    }
