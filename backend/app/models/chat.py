from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from uuid import UUID, uuid4


class Message(BaseModel):
    """A chat message."""
    id: UUID = Field(default_factory=uuid4)
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_data: Optional["AgentMessageData"] = None


class AgentMessageData(BaseModel):
    """Agent-specific data attached to a message."""
    agent_id: str
    agent_name: str
    status: Literal["pending", "running", "completed", "error"]
    progress: Optional[float] = None


class Thread(BaseModel):
    """A conversation thread."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    messages: list[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    thread_id: Optional[UUID] = None
    message: str
    mode: Literal["scout", "simulate", "paper"] = "paper"
    use_web_search: bool = True


class ChatResponse(BaseModel):
    """Response containing chat result."""
    thread_id: UUID
    message: Message
    agents_triggered: list[str] = []


class StreamEvent(BaseModel):
    """A streaming event from the backend."""
    event_type: Literal["message", "agent_update", "simulation", "paper", "error", "done"]
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


Message.model_rebuild()
