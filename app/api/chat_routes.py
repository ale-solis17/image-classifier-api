from fastapi import APIRouter, HTTPException

from app.schemas.chat import BacteriaChatRequest, BacteriaChatResponse
from app.services.bacteria_chat import (
    BacteriaChatValidationError,
    OpenAIConfigurationError,
    OpenAIServiceError,
    OpenAITimeoutError,
    chat_about_bacteria,
)


router = APIRouter(tags=["chat"])


@router.post("/chat/bacteria", response_model=BacteriaChatResponse)
async def bacteria_chat(chat_request: BacteriaChatRequest):
    try:
        return await chat_about_bacteria(chat_request)
    except BacteriaChatValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAITimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (OpenAIConfigurationError, OpenAIServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
