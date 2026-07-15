"""enterprise hybrid rag stack: ingest, retrieve, rerank."""

from src.retriever import HybridRetriever, HybridRetrieverConfig
from src.reranker import CrossEncoderReranker, RerankerConfig
from typing import List, Union, Optional

__all__ = [
    "HybridRetriever",
    "HybridRetrieverConfig",
    "CrossEncoderReranker",
    "RerankerConfig",
]

def initialize_components() -> Optional[List[Union[HybridRetriever, HybridRetrieverConfig, CrossEncoderReranker, RerankerConfig]]]:
    """initialize components of the rag system"""
    try:
        retriever = HybridRetriever()
        config = HybridRetrieverConfig()
        reranker = CrossEncoderReranker()
        reranker_config = RerankerConfig()
    except Exception as e:
        print(f"failed to initialize components: {e}")
        return None  # returning None to indicate failure
    
    return [retriever, config, reranker, reranker_config]  # returning initialized components for further use

# TODO: consider adding logging instead of print statements