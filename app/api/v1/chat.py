import uuid

from fastapi import APIRouter, Depends

from app.api.deps import get_chat_service, get_current_user_id
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def ask_chat(
    body: ChatRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.ask(user_id, body.message)
