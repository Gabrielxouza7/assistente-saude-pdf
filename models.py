from typing import Dict, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pergunta do usuário")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Histórico recente da conversa [{'role': 'user'|'assistant', 'content': '...'}]"
    )