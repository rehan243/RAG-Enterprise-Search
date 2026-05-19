"""enterprise hybrid rag stack: ingest, retrieve, rerank."""

from src.retriever import HybridRetriever, HybridRetrieverConfig
from src.reranker import CrossEncoderReranker, RerankerConfig
from typing import List

__all__ = [
    "HybridRetriever",
    "HybridRetrieverConfig",
    "CrossEncoderReranker",
    "RerankerConfig",
]

def initialize_components() -> List[str]:
    """initialize components of the rag system"""
    try:
        # assuming these classes have an init that could raise
        retriever = HybridRetriever()
        config = HybridRetrieverConfig()
        reranker = CrossEncoderReranker()
        reranker_config = RerankerConfig()
    except Exception as e:
        # log the error, could be a logger setup here
        print(f"Failed to initialize components: {e}")
        return []
    
    return [retriever, config, reranker, reranker_config]  # returning initialized components for further use

# TODO: consider adding logging instead of print statements