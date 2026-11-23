from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router
from rag import load_components 
from prometheus_fastapi_instrumentator import Instrumentator

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm, vector_store, prompt = load_components()
    
    app.state.llm = llm
    app.state.vector_store = vector_store
    app.state.prompt = prompt
    
    print("--- API Pronta ---")
    
    yield
    
    print("--- Evento de Desligamento (Shutdown) ---")
    app.state.llm = None
    app.state.vector_store = None
    app.state.prompt = None

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

# Instrument Prometheus
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Inclui as rotas do api.py
app.include_router(api_router)
