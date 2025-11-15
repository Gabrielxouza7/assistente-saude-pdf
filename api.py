import os
from fastapi import APIRouter, HTTPException
from typing import Dict
from models import ChatRequest
from app_state import app_state
from langchain.chains import RetrievalQA 

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "API do Assistente de Saúde Informativo"}

@router.post("/chat")
async def handle_chat(request: ChatRequest) -> Dict:
    
    llm = app_state.get("llm")
    vector_store = app_state.get("vector_store")
    prompt = app_state.get("prompt")
    
    if not all([llm, vector_store, prompt]):
        raise HTTPException(status_code=503, detail="Serviço indisponível.")


    retriever = vector_store.as_retriever(
        search_kwargs={'k': 3}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    response = qa_chain.invoke(request.question)
        
    answer = response.get('result', "Desculpe, não consegui processar a resposta.")
    
    source_documents = response.get('source_documents', [])
    real_sources = list(set([os.path.basename(doc.metadata.get('source', 'N/A')) for doc in source_documents]))

    if not real_sources and "Desculpe, não encontrei" not in answer:
        answer = "Desculpe, não encontrei nenhuma informação sobre isso nos meus documentos."
    elif real_sources:
        answer += f"\n\nFontes: {', '.join(real_sources)}"

    aviso_medico = "Esta informação não substitui uma consulta médica profissional. Consulte seu médico."
    if aviso_medico not in answer and "Desculpe, não encontrei" not in answer:
        answer += f"\n\n{aviso_medico}"

    return {
        "answer": answer
    }