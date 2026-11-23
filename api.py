import os
from fastapi import APIRouter, HTTPException, Request
from typing import Dict
from models import ChatRequest
from langchain.chains import RetrievalQA 
from prometheus_client import Counter

router = APIRouter()

guardrail_counter = Counter(
    "guardrail_interventions_total",
    "Number of times the guardrail intercepted or modified an LLM response"
)

@router.get("/")
def read_root():
    return {"message": "API do Assistente de Saúde Informativo"}

@router.post("/chat")
async def handle_chat(request: Request, body: ChatRequest) -> Dict:
    llm = getattr(request.app.state, "llm", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    prompt = getattr(request.app.state, "prompt", None)
    
    if not all([llm, vector_store, prompt]):
        raise HTTPException(status_code=503, detail="Serviço indisponível ou base de documentos não inicializada.")

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

    try:
        response = qa_chain.invoke(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar modelo: {str(e)}")
        
    answer = response.get('result', "Desculpe, não consegui processar a resposta.")
    
    source_documents = response.get('source_documents', [])
    real_sources = list(set([os.path.basename(doc.metadata.get('source', 'N/A')) for doc in source_documents]))

    if not real_sources and "Desculpe, não encontrei" not in answer:
        answer = "Desculpe, não encontrei nenhuma informação sobre isso nos meus documentos."
        guardrail_counter.inc()
    elif real_sources:
        answer += f"\n\nFontes: {', '.join(real_sources)}"

    aviso_medico = "Esta informação não substitui uma consulta médica profissional. Consulte seu médico."
    if aviso_medico not in answer and "Desculpe, não encontrei" not in answer:
        answer += f"\n\n{aviso_medico}"

    return {
        "answer": answer
    }
