import os
import json
import time
from datetime import datetime, timezone

# Import typing utilities, FastAPI components, LangChain QA chain, Prometheus counter, and models
from typing import Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from langchain.chains import RetrievalQA
from prometheus_client import Counter
from models import ChatRequest, FeedbackRequest

# Initialize API router and Prometheus counter for tracking guardrail interventions
router = APIRouter()
guardrail_counter = Counter(
    "guardrail_interventions_total",
    "Number of times the guardrail intercepted or modified an LLM response"
)

# Define file paths for audit logs and production failure records
AUDIT_LOG_PATH = "eval/audit_log.jsonl"
FAILED_PROD_PATH = "eval/production_failures.json"

# Condense the current question with recent conversation history into a standalone query
def rewrite_standalone_question(llm, question: str, history: List[Dict[str, str]]) -> str:
    recent_history = history[-4:]
    formatted_history = "\n".join(
        f"{'Usuário' if msg.get('role') == 'user' else 'Assistente'}: {msg.get('content', '')}"
        for msg in recent_history
    )

    # Build the prompt instructing the LLM to resolve pronouns using recent context
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

    # Invoke the LLM to rewrite the question and extract the stripped text response
    try:
        res = llm.invoke(condense_prompt)
        rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
        return rewritten if rewritten else question

    # Fallback to the original question if the LLM rewriting call fails
    except Exception:
        return question

# Persist request audit metadata and log failed interactions for evaluation
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

    # Construct the structured audit record with latency, sources, and outcome flags
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

    # Append the serialized audit entry to the JSONL log file
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

    # Record failed interactions into the production failures dataset when a failure reason exists
    if failure_reason:
        records = []

        # Load existing failure records from disk if the file already exists
        if os.path.exists(FAILED_PROD_PATH):

            # Attempt to parse the existing JSON failure log file
            try:
                with open(FAILED_PROD_PATH, "r", encoding="utf-8") as f:
                    records = json.load(f)

            # Reset records to an empty list if reading or parsing fails
            except Exception:
                records = []

        # Append the new failure event metadata to the loaded records list
        records.append({
            "timestamp": timestamp,
            "question": question,
            "standalone_question": standalone_question or question,
            "generated_answer": answer,
            "reason": failure_reason,
            "retrieved_sources": sources
        })

        # Write the updated failure records list back to disk in formatted JSON
        with open(FAILED_PROD_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

# Return a health check welcome message at the root endpoint
@router.get("/")
def read_root():
    return {"message": "API do Assistente de Saúde Informativo"}

# Schedule a background task to log negative user feedback for evaluation
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

# Process user chat questions using semantic caching, RAG retrieval, and guardrails
@router.post("/chat")
async def handle_chat(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks
) -> Dict:
    start_time = time.perf_counter()

    # Retrieve initialized RAG components and semantic cache from application state
    llm = getattr(request.app.state, "llm", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    prompt = getattr(request.app.state, "prompt", None)
    semantic_cache = getattr(request.app.state, "semantic_cache", None)

    # Raise a 503 error if required RAG components are missing or uninitialized
    if not all([llm, vector_store, prompt]):
        raise HTTPException(
            status_code=503,
            detail="Serviço indisponível ou base de documentos não inicializada."
        )

    # Rewrite the question into a standalone query when conversation history is provided
    effective_question = (
        rewrite_standalone_question(llm, body.question, body.history)
        if body.history
        else body.question
    )

    # Check the semantic cache for an existing answer to the effective question
    if semantic_cache:
        cached_answer = semantic_cache.get(effective_question)

        # Log audit metrics and return immediately when a cached answer is found
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

    # Configure the vector store retriever to fetch the top 4 most relevant chunks
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    # Build the RetrievalQA chain with the custom prompt and source document tracking
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Execute the QA chain against the standalone question
    try:
        response = qa_chain.invoke(effective_question)

    # Log runtime errors and raise an HTTP 500 exception if QA chain execution fails
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

    # Extract the generated answer and unique source filenames from the QA response
    answer = response.get("result", "Desculpe, não consegui processar a resposta.")
    source_documents = response.get("source_documents", [])
    real_sources = list(set([
        os.path.basename(doc.metadata.get("source", "N/A"))
        for doc in source_documents
    ]))

    # Initialize guardrail tracking flags and define Portuguese abstention phrases
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

    # Trigger guardrail metrics when the model abstains due to missing context
    if any(phrase in answer.lower() for phrase in abstention_phrases):
        guardrail_counter.inc()
        guardrail_triggered = True
        failure_reason = "abstention_or_out_of_scope"

    # Append cited document filenames to the answer when valid sources were retrieved
    elif real_sources:
        answer += f"\n\nFontes: {', '.join(real_sources)}"

    # Ensure the mandatory medical disclaimer is appended to the final answer
    aviso_medico = "Esta informação não substitui uma consulta médica profissional. Consulte seu médico."
    if aviso_medico not in answer:
        answer += f"\n\n{aviso_medico}"

    # Cache the valid response in the semantic cache if no guardrail was triggered
    if semantic_cache and not guardrail_triggered:
        semantic_cache.set(effective_question, answer)

    # Calculate total request latency and schedule background audit logging
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

    # Return the final formatted response and cache status
    return {
        "answer": answer,
        "cached": False
    }