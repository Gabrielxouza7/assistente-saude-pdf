from typing import Dict, List
from pydantic import BaseModel, Field

# Define the request payload schema for incoming user chat messages and conversation history
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pergunta do usuário")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Histórico recente da conversa [{'role': 'user'|'assistant', 'content': '...'}]"
    )

# Define the request payload schema for logging negative user feedback on generated answers
class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pergunta original avaliada negativamente")
    answer: str = Field(..., min_length=1, description="Resposta gerada avaliada negativamente")