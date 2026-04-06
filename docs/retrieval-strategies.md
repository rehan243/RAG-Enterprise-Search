# Retrieval Strategy Comparison

| Strategy | Recall | Latency | Best For |
|----------|--------|---------|----------|
| Dense (vector) | High | Medium | Semantic similarity |
| Sparse (BM25) | Medium | Low | Keyword matching |
| Hybrid (dense+sparse) | Highest | Medium | Production RAG |
| Reranked | Highest | Higher | Precision-critical |

## Recommended Pipeline
1. Hybrid retrieval (dense + BM25) for initial candidate set
2. Cross-encoder reranking for top-k refinement
3. Metadata filtering for access control and freshness