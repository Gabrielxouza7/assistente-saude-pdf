from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router
from rag import load_components
from semantic_cache import SemanticCache
from prometheus_fastapi_instrumentator import Instrumentator, metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm, vector_store, prompt, embeddings = load_components()
    
    app.state.llm = llm
    app.state.vector_store = vector_store
    app.state.prompt = prompt
    app.state.semantic_cache = SemanticCache(embeddings=embeddings)
    
    print("--- API Pronta ---")
    
    yield
    
    print("--- Evento de Desligamento (Shutdown) ---")
    app.state.llm = None
    app.state.vector_store = None
    app.state.prompt = None
    app.state.semantic_cache = None

app = FastAPI(
    title="API RAG de Saúde (LangChain)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument Prometheus com buckets calibrados para latência de inferência LLM (0.1s até 30s)
instrumentator = Instrumentator()
instrumentator.add(
    metrics.default(
        latency_lowr_buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0)
    )
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Inclui as rotas do api.py
app.include_router(api_router)