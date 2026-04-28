"""
Dense + sparse + fusion. The cross-encoder lives in reranker.py because mixing concerns
makes tests miserable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _rrf_scores(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return scores


@dataclass
class HybridRetrieverConfig:
    dense_k: int = 50
    sparse_k: int = 50
    fuse_top_k: int = 20
    rrf_k: int = 60
    min_rrf_score: float = 0.0
    query_expansion_templates: Tuple[str, ...] = (
        "{q}",
        "explain {q}",
        "{q} definition",
    )


class HybridRetriever:
    def __init__(
        self,
        index: faiss.Index,
        doc_ids: Sequence[str],
        dense_vectors: np.ndarray,
        bm25_retriever: BM25Retriever,
        cfg: Optional[HybridRetrieverConfig] = None,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        chunk_text_by_id: Optional[Dict[str, str]] = None,
    ) -> None:
        if len(doc_ids) != dense_vectors.shape[0]:
            raise ValueError("doc_ids length must match dense_vectors rows")
        self._index = index
        self._ids = list(doc_ids)
        self._dense = dense_vectors.astype("float32", copy=False)
        faiss.normalize_L2(self._dense)
        self._bm25 = bm25_retriever
        self._cfg = cfg or HybridRetrieverConfig()
        self._embed_fn = embed_fn
        self._text = chunk_text_by_id or {}
        if self._embed_fn is None:
            raise ValueError("embed_fn is required for dense retrieval")

    def update_corpus_text(self, mapping: Dict[str, str]) -> None:
        """Hydrate chunk bodies for downstream rerankers (optional)."""
        self._text.update(mapping)

    def _expand_queries(self, q: str) -> List[str]:
        q = q.strip()
        if not q:
            return []
        return [t.format(q=q) for t in self._cfg.query_expansion_templates]

    def _dense_search(self, query_vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        q = query_vec.astype("float32", copy=False).reshape(1, -1)
        faiss.normalize_L2(q)
        sims, idxs = self._index.search(q, min(k, len(self._ids)))
        out: List[Tuple[str, float]] = []
        for score, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            out.append((self._ids[idx], float(score)))
        return out

    def _sparse_docs(self, q: str, k: int) -> List[Document]:
        self._bm25.k = k
        try:
            return self._bm25.get_relevant_documents(q)
        except Exception as e:
            logger.warning("BM25 retrieval failed: %s", e)
            return []

    def retrieve(self, query: str) -> List[Document]:
        subqs = self._expand_queries(query)
        if not subqs:
            return []
        dense_ids_nested: List[List[str]] = []
        for sq in subqs:
            emb = self._embed_fn([sq])
            if emb.shape[0] != 1:
                raise RuntimeError("embed_fn must return one row per query")
            ranked = self._dense_search(emb[0], self._cfg.dense_k)
            dense_ids_nested.append([d for d, _ in ranked])

        sparse_ids_nested: List[List[str]] = []
        for sq in subqs:
            docs = self._sparse_docs(sq, self._cfg.sparse_k)
            sparse_ids_nested.append([d.metadata.get("doc_id", d.page_content[:40]) for d in docs])

        fused = _rrf_scores(dense_ids_nested + sparse_ids_nested, k=self._cfg.rrf_k)
        ranked = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
        top_ids = [i for i in ranked if fused[i] >= self._cfg.min_rrf_score][: self._cfg.fuse_top_k]
        if not top_ids:
            logger.info("hybrid retrieve: no candidates (empty corpus or dead indexes)")
        else:
            logger.debug(
                "hybrid retrieve: %s expansions, fused=%s, returning=%s",
                len(subqs),
                len(fused),
                len(top_ids),
            )
        return [
            Document(
                page_content=self._text.get(did, ""),
                metadata={"doc_id": did, "rrf": fused[did]},
            )
            for did in top_ids
        ]

    def retrieve_with_trace(self, query: str) -> Tuple[List[Document], Dict[str, float]]:
        """Returns (docs, rrf_scores_for_returned_ids) — enough for tracing without recomputing fusion."""
        docs = self.retrieve(query)
        trace = {d.metadata["doc_id"]: float(d.metadata["rrf"]) for d in docs}
        return docs, trace
