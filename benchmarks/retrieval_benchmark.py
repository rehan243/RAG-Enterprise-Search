"""RAG Retrieval Quality Benchmark - Rehan Malik"""

import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    name: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    latency_ms: float
    ndcg: float


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / k if k > 0 else 0.0


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(relevant: set, retrieved: list) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    import math
    dcg = sum(
        (1.0 if retrieved[i] in relevant else 0.0) / math.log2(i + 2)
        for i in range(min(k, len(retrieved)))
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def run_benchmark(name, relevant, retrieved, k=5):
    start = time.time()
    result = BenchmarkResult(
        name=name,
        precision_at_k=precision_at_k(relevant, retrieved, k),
        recall_at_k=recall_at_k(relevant, retrieved, k),
        mrr=mean_reciprocal_rank(relevant, retrieved),
        latency_ms=(time.time() - start) * 1000,
        ndcg=ndcg_at_k(relevant, retrieved, k),
    )
    return result


if __name__ == "__main__":
    relevant = {"doc1", "doc3", "doc5"}
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    r = run_benchmark("hybrid_search", relevant, retrieved)
    print(f"P@5={r.precision_at_k:.3f} R@5={r.recall_at_k:.3f} "
          f"MRR={r.mrr:.3f} NDCG={r.ndcg:.3f}")
