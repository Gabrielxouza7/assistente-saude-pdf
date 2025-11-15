from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router
from rag import load_components 
from app_state import app_state

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    llm, vector_store, prompt = load_components()
    
    app_state["llm"] = llm
    app_state["vector_store"] = vector_store
    app_state["prompt"] = prompt
    
    print("--- API Pronta ---")
    
    yield
    
    print("--- Evento de Desligamento (Shutdown) ---")
    app_state["llm"] = None
    app_state["vector_store"] = None
    app_state["prompt"] = None

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

# Inclui as rotas do api.py
app.include_router(api_router)