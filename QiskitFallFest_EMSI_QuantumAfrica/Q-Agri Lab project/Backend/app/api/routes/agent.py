import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.agents.research_agent import ResearcherAgent

router = APIRouter(prefix="/agent", tags=["AI Agent"])

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

@router.post("/chat")
def chat(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY is not configured; dataset and experiment endpoints remain available.")
    try:
        response = ResearcherAgent().run(request.message)
        return {"response": response, "status": "success"}
    except Exception as exc:
        # Keep credential failures actionable; the key itself is never returned.
        try:
            from openai import AuthenticationError, PermissionDeniedError
            if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
                raise HTTPException(
                    status_code=503,
                    detail="OpenAI rejected this API key or its permissions. Enable model requests and Responses API access for the key, then restart the backend.",
                ) from exc
        except ImportError:
            pass
        raise HTTPException(502, f"OpenAI request failed: {exc}") from exc
