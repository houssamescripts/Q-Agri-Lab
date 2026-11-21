import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/agent", tags=["AI Agent"])

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

@router.post("/chat")
def chat(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY is not configured; dataset and experiment endpoints remain available.")
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions="You are Q-Agri Lab, a research assistant for the supplied Moroccan crop classification challenge. Classes: alfalfa, maize, wheat. Official splits: 914 train, 150 validation, 221 unlabeled test. Never claim experimental results unless they are supplied. Preserve spatial splits; use validation for model selection and test only for final predictions.",
            input=request.message,
        )
        return {"response": response.output_text, "status": "success"}
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
