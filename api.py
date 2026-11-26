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

# Condense conversation history and translate the retrieval query to Spanish to match the MedlinePlus index
def build_retrieval_query(llm, question: str, history: List[Dict[str, str]]) -> str:
    formatted_history = ""
    if history:
        recent_history = history[-4:]
        formatted_history = (
            "Histórico recente:\n"
            + "\n".join(
                f"{'Usuário' if msg.get('role') == 'user' else 'Assistente'}: {msg.get('content', '')}"
                for msg in recent_history
            )
            + "\n\n"
        )

    # Build the prompt instructing the LLM to resolve pronouns and translate using MedlinePlus lay terminology
    condense_prompt = (
        "Você é um otimizador de busca para a enciclopédia médica MedlinePlus en español.\n"
        "Dada a pergunta do usuário (e o histórico recente, se houver), resolva quaisquer pronomes "
        "(como 'isso', 'dela', 'esse problema') usando o histórico e traduza a pergunta resultante "
        "para uma consulta clara em ESPANHOL usando a terminologia leiga padrão do MedlinePlus:\n"
        "- infarto / ataque cardíaco -> ataque al corazón\n"
        "- hipertensão -> presión arterial alta\n"
        "- AVC / derrame -> accidente cerebrovascular\n"
        "- DPOC -> enfermedad pulmonar obstructiva crónica\n"
        "Se a pergunta contiver dois termos equivalentes, preserve o termo leigo do MedlinePlus.\n"
        "Retorne APENAS a consulta final em espanhol, sem explicações, prefixos ou aspas.\n\n"
        f"{formatted_history}"
        f"Pergunta atual: {question}\n"
        "Consulta em espanhol:"
    )

    # Invoke the LLM to generate the Spanish retrieval query and extract the stripped response
    try:
        res = llm.invoke(condense_prompt)
        rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
        return rewritten if rewritten else question

    # Fallback to the original question if the LLM call fails
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

    # Resolve conversation context and translate query to Spanish for accurate vector retrieval
    retrieval_query = build_retrieval_query(llm, body.question, body.history)

    # Check the semantic cache for an existing answer to the normalized query
    if semantic_cache:
        cached_answer = semantic_cache.get(retrieval_query)

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
                standalone_question=retrieval_query
            )
            return {
                "answer": cached_answer,
                "cached": True
            }

    # Configure similarity retriever to pull top chunks for the target Spanish medical topic
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 8}
    )

    # Build the RetrievalQA chain with the custom prompt and source document tracking
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Execute the QA chain using the Spanish retrieval query
    try:
        response = qa_chain.invoke(retrieval_query)

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
            standalone_question=retrieval_query
        )
        raise HTTPException(status_code=500, detail=f"Erro ao processar modelo: {str(e)}")

    # Extract the generated answer and unique MedlinePlus citations from the QA response
    answer = response.get("result", "Desculpe, não consegui processar a resposta.")
    source_documents = response.get("source_documents", [])

    formatted_sources = []
    for doc in source_documents:
        title = doc.metadata.get("title", "Guia MedlinePlus")
        source_url = doc.metadata.get("source", "https://medlineplus.gov/spanish/")
        formatted_sources.append(f"{title} ({source_url})")

    real_sources = list(dict.fromkeys(formatted_sources))

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

    # Append cited official MedlinePlus guides to the answer when valid sources were retrieved
    elif real_sources:
        sources_block = "\n".join(f"- {src}" for src in real_sources)
        answer += f"\n\nFontes (MedlinePlus / NIH):\n{sources_block}"

    # Ensure the mandatory medical disclaimer is appended once regardless of line breaks
    aviso_medico = "Esta informação não substitui uma consulta médica profissional. Consulte seu médico."
    normalized_answer = " ".join(answer.split())
    if aviso_medico not in normalized_answer:
        answer += f"\n\n{aviso_medico}"

    # Cache the valid response in the semantic cache if no guardrail was triggered
    if semantic_cache and not guardrail_triggered:
        semantic_cache.set(retrieval_query, answer)

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
        standalone_question=retrieval_query
    )

    # Return the final formatted response and cache status
    return {
        "answer": answer,
        "cached": False
    }