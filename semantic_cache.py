import time
import uuid
from typing import Optional
from prometheus_client import Counter
from langchain_community.vectorstores import Chroma

# Initialize Prometheus counter to track semantic cache hits and misses
cache_counter = Counter(
    "semantic_cache_requests_total",
    "Total semantic cache lookups by outcome",
    ["result"]
)

# Manage semantic caching using Chroma vector store and similarity thresholds
class SemanticCache:

    # Initialize the similarity threshold and the persistent Chroma vector store
    def __init__(self, embeddings, persist_directory: str = "./storage_chroma", threshold: float = 0.8):
        self.threshold = threshold

        # Configure the Chroma collection with cosine similarity metric
        self.store = Chroma(
            collection_name="semantic_cache",
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    # Search for a semantically similar cached question and return its answer if valid
    def get(self, question: str) -> Optional[str]:
        results = self.store.similarity_search_with_score(question, k=1)

        # Increment miss counter and return None when no similar entries are found
        if not results:
            cache_counter.labels(result="miss").inc()
            return None

        # Extract the closest document and convert cosine distance to similarity score
        doc, distance = results[0]
        similarity = 1.0 - distance

        # Increment hit counter and return cached answer if similarity meets threshold
        if similarity >= self.threshold:
            cache_counter.labels(result="hit").inc()
            return doc.metadata.get("answer")

        # Increment miss counter and return None when similarity is below threshold
        cache_counter.labels(result="miss").inc()
        return None

    # Store a new question and its corresponding answer with a timestamp and unique ID
    def set(self, question: str, answer: str) -> None:
        self.store.add_texts(
            texts=[question],
            metadatas=[{
                "answer": answer,
                "timestamp": time.time()
            }],
            ids=[str(uuid.uuid4())]
        )