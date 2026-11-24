import time
import uuid
from typing import Optional
from prometheus_client import Counter
from langchain_community.vectorstores import Chroma

cache_counter = Counter(
    "semantic_cache_requests_total",
    "Total semantic cache lookups by outcome",
    ["result"]  # "hit" ou "miss"
)

class SemanticCache:
    def __init__(self, embeddings, persist_directory: str = "./storage_chroma", threshold: float = 0.8):
        self.threshold = threshold
        self.store = Chroma(
            collection_name="semantic_cache",
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def get(self, question: str) -> Optional[str]:
        results = self.store.similarity_search_with_score(question, k=1)
        if not results:
            cache_counter.labels(result="miss").inc()
            return None

        doc, distance = results[0]
        similarity = 1.0 - distance

        if similarity >= self.threshold:
            cache_counter.labels(result="hit").inc()
            return doc.metadata.get("answer")

        cache_counter.labels(result="miss").inc()
        return None

    def set(self, question: str, answer: str) -> None:
        self.store.add_texts(
            texts=[question],
            metadatas=[{
                "answer": answer,
                "timestamp": time.time()
            }],
            ids=[str(uuid.uuid4())]
        )