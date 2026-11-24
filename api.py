import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from langchain.chains import RetrievalQA
from prometheus_client import Counter
from models import ChatRequest, FeedbackRequest

router = APIRouter()

guardrail_counter = Counter(
    "guardrail_interventions_total",
    "Number of times the guardrail intercepted or modified an LLM response"
)

AUDIT_LOG_PATH = "eval/audit_log.jsonl"
FAILED_PROD_PATH = "eval/production_failures.json"


def rewrite_standalone_question(llm, question: str, history: List[Dict[str, str]]) -> str:
    """Condensa a pergunta atual com os últimos 2 turnos do histórico em uma pergunta autossuficiente."""
    recent_history = history[-4:]
    formatted_history = "\n".join(
        f"{'Usuário' if msg.get('role') == 'user' else 'Assistente'}: {msg.get('content', '')}"
        for msg in recent_history
    )

    condense_prompt = (
        "Dada a conversa abaixo e uma pergunta de acompanhamento, reescreva a pergunta de acompanhamento "
        "para que ela seja uma pergunta independente e completa em português, substituindo pronomes como "
        "'isso', 'dela', 'esse problema' pelo termo médico exato citado anteriormente.\n"
        "Se a pergunta já for independente ou mudar totalmente de assunto, retorne-a exatamente como está.\n"
        "Retorne APENAS a pergunta reescrita, sem explicações ou aspas.\n\n"
        f"Histórico recente:\n{formatted_history}\n\n"
        f"Pergunta de acompanhamento: {question}\n"
        "Pergunta independente:"
    )

    try:
        res = llm.invoke(condense_prompt)
        rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def log_audit_and_eval(
    question: str,
    answer: str,
    cached: bool,
    guardrail_triggered: bool,
    latency_ms: float,
    sources: List[str],
    failure_reason: Optional[str] = None,
    standalone_question: Optional[str] = None
):
    os.makedirs("eval", exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    audit_entry = {
        "timestamp": timestamp,
        "question": question,
        "standalone_question": standalone_question or question,
        "answer": answer,
        "cached": cached,
        "guardrail_triggered": guardrail_triggered,
        "latency_ms": round(latency_ms, 2),
        "retrieved_sources": sources,
        "failure_reason": failure_reason
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

    if failure_reason:
        records = []
        if os.path.exists(FAILED_PROD_PATH):
            try:
                with open(FAILED_PROD_PATH, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        records.append({
            "timestamp": timestamp,
            "question": question,
            "standalone_question": standalone_question or question,
            "generated_answer": answer,
            "reason": failure_reason,
            "retrieved_sources": sources
        })

        with open(FAILED_PROD_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


@router.get("/")
def read_root():
    return {"message": "API do Assistente de Saúde Informativo"}


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        log_audit_and_eval,
        question=body.question,
        answer=body.answer,
        cached=False,
        guardrail_triggered=False,
        latency_ms=0.0,
        sources=[],
        failure_reason="user_negative_feedback"
    )
    return {"status": "Feedback registrado com sucesso"}


@router.post("/chat")
async def handle_chat(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks
) -> Dict:
    start_time = time.perf_counter()

    llm = getattr(request.app.state, "llm", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    prompt = getattr(request.app.state, "prompt", None)
    semantic_cache = getattr(request.app.state, "semantic_cache", None)

    if not all([llm, vector_store, prompt]):
        raise HTTPException(
            status_code=503,
            detail="Serviço indisponível ou base de documentos não inicializada."
        )

    # 1. Se houver histórico recente, reescreve a pergunta para torná-la autossuficiente
    effective_question = (
        rewrite_standalone_question(llm, body.question, body.history)
        if body.history
        else body.question
    )

    # 2. Consulta o cache semântico usando a pergunta efetiva
    if semantic_cache:
        cached_answer = semantic_cache.get(effective_question)
        if cached_answer:
            latency_ms = (time.perf_counter() - start_time) * 1000
            background_tasks.add_task(
                log_audit_and_eval,
                question=body.question,
                answer=cached_answer,
                cached=True,
                guardrail_triggered=False,
                latency_ms=latency_ms,
                sources=["semantic_cache"],
                failure_reason=None,
                standalone_question=effective_question
            )
            return {
                "answer": cached_answer,
                "cached": True
            }

    # 3. Retriever vetorial (k=4 chunks mais próximos)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    try:
        response = qa_chain.invoke(effective_question)
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        background_tasks.add_task(
            log_audit_and_eval,
            question=body.question,
            answer=str(e),
            cached=False,
            guardrail_triggered=False,
            latency_ms=latency_ms,
            sources=[],
            failure_reason="llm_runtime_error",
            standalone_question=effective_question
        )
        raise HTTPException(status_code=500, detail=f"Erro ao processar modelo: {str(e)}")

    answer = response.get("result", "Desculpe, não consegui processar a resposta.")
    source_documents = response.get("source_documents", [])
    real_sources = list(set([
        os.path.basename(doc.metadata.get("source", "N/A"))
        for doc in source_documents
    ]))

    # 4. Pós-processamento e detecção de abstenção
    guardrail_triggered = False
    failure_reason = None
    abstention_phrases = (
        "desculpe, não encontrei",
        "não contém informações",
        "não encontrei nenhuma informação",
        "não tenho informações",
        "não disponho de informações",
        "não há menção",
        "não há informações",
        "não menciona",
        "não especifica",
        "não aborda",
        "não estabelece",
        "não foi possível encontrar",
        "não consta",
    )

    if any(phrase in answer.lower() for phrase in abstention_phrases):
        guardrail_counter.inc()
        guardrail_triggered = True
        failure_reason = "abstention_or_out_of_scope"
    elif real_sources:
        answer += f"\n\nFontes: {', '.join(real_sources)}"

    aviso_medico = "Esta informação não substitui uma consulta médica profissional. Consulte seu médico."
    if aviso_medico not in answer:
        answer += f"\n\n{aviso_medico}"

    # 5. Salva no cache semântico apenas respostas válidas baseadas nos documentos
    if semantic_cache and not guardrail_triggered:
        semantic_cache.set(effective_question, answer)

    # 6. Registra trilha de auditoria e fila de avaliação em segundo plano
    latency_ms = (time.perf_counter() - start_time) * 1000
    background_tasks.add_task(
        log_audit_and_eval,
        question=body.question,
        answer=answer,
        cached=False,
        guardrail_triggered=guardrail_triggered,
        latency_ms=latency_ms,
        sources=real_sources,
        failure_reason=failure_reason,
        standalone_question=effective_question
    )

    return {
        "answer": answer,
        "cached": False
    }