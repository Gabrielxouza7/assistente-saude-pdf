from pydantic import BaseModel

class ChatRequest(BaseModel):
    # Define o formato da requisição /chat
    question: str
