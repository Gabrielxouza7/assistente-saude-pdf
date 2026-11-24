from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router

# Import RAG loader, semantic cache, and Prometheus instrumentation modules
from rag import load_components
from semantic_cache import SemanticCache
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# Manage application startup and shutdown lifecycle to initialize and release shared resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    llm, vector_store, prompt, embeddings = load_components()

    # Store initialized RAG components and semantic cache in the application state
    app.state.llm = llm
    app.state.vector_store = vector_store
    app.state.prompt = prompt
    app.state.semantic_cache = SemanticCache(embeddings=embeddings)

    # Notify startup completion and yield control back to the FastAPI runtime
    print("--- API Pronta ---")
    yield

    # Log shutdown and clear all heavy resources from the application state
    print("--- Evento de Desligamento (Shutdown) ---")
    app.state.llm = None
    app.state.vector_store = None
    app.state.prompt = None
    app.state.semantic_cache = None

# Initialize the FastAPI application instance with the configured lifespan handler
app = FastAPI(
    title="API RAG de Saúde (LangChain)",
    lifespan=lifespan
)

# Configure CORS middleware to allow cross-origin requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Prometheus metrics with latency buckets calibrated for LLM inference times
instrumentator = Instrumentator()
instrumentator.add(
    metrics.default(
        latency_lowr_buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0)
    )
)

# Attach Prometheus instrumentation to the app and expose the metrics endpoint
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Register API routes defined in the api router module
app.include_router(api_router)