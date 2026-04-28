"""
Cross-encoder rerank. Batch scoring + tiny LRU-ish cache because users love repeating queries.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_cache: int = 512
    min_score: Optional[float] = None
    score_clip: float = 12.0


class CrossEncoderReranker:
    def __init__(self, cfg: Optional[RerankerConfig] = None) -> None:
        self._cfg = cfg or RerankerConfig()
        self._model = CrossEncoder(self._cfg.model_name)
        self._cache: "OrderedDict[str, float]" = OrderedDict()

    def _cache_key(self, query: str, text: str) -> str:
        h = hashlib.sha256()
        h.update(query.encode())
        h.update(b"\0")
        h.update(text.encode(errors="ignore"))
        return h.hexdigest()

    def _get_cached(self, key: str) -> Optional[float]:
        val = self._cache.get(key)
        if val is None:
            return None
        self._cache.move_to_end(key)
        return val

    def _set_cached(self, key: str, score: float) -> None:
        self._cache[key] = score
        self._cache.move_to_end(key)
        while len(self._cache) > self._cfg.max_cache:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _clip(self, x: float) -> float:
        c = self._cfg.score_clip
        return max(-c, min(c, x))

    def score_pairs(self, query: str, texts: Sequence[str]) -> List[float]:
        scores: List[float] = []
        pending_idx: List[int] = []
        pending_texts: List[str] = []
        keys: List[str] = []

        for i, t in enumerate(texts):
            k = self._cache_key(query, t)
            c = self._get_cached(k)
            if c is not None:
                scores.append(c)
            else:
                scores.append(float("nan"))
                pending_idx.append(i)
                pending_texts.append(t)
                keys.append(k)

        if pending_texts:
            pairs = [[query, t] for t in pending_texts]
            bs = self._cfg.batch_size
            computed: List[float] = []
            for j in range(0, len(pairs), bs):
                batch = pairs[j : j + bs]
                raw = self._model.predict(batch, convert_to_numpy=True, show_progress_bar=False)
                computed.extend(float(x) for x in np.asarray(raw).reshape(-1))
            for idx, key, sc in zip(pending_idx, keys, computed):
                self._set_cached(key, self._clip(sc))
                scores[idx] = self._clip(sc)
        return [self._clip(float(s)) for s in scores]

    def rerank(self, query: str, docs: List[Document], top_k: int = 10) -> List[Document]:
        if not docs:
            return []
        texts = [d.page_content or "" for d in docs]
        scores = self.score_pairs(query, texts)
        thresh = self._cfg.min_score
        candidates = list(range(len(docs)))
        if thresh is not None:
            candidates = [i for i in candidates if scores[i] >= thresh]
            if not candidates:
                logger.info("reranker: all candidates below min_score=%s", thresh)
        order = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
        out: List[Document] = []
        for i in order:
            d = docs[i]
            meta = dict(d.metadata or {})
            meta["cross_score"] = scores[i]
            out.append(Document(page_content=d.page_content, metadata=meta))
        return out

    def explain(self, query: str, docs: List[Document]) -> List[Dict[str, float]]:
        """Scores without mutating metadata — handy for A/B against BM25-only."""
        texts = [d.page_content or "" for d in docs]
        scores = self.score_pairs(query, texts)
        return [{"idx": i, "score": scores[i]} for i in range(len(docs))]

    def rerank_fused(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 10,
        prior_weight: float = 0.18,
    ) -> List[Document]:
        """Blend cross-encoder with HybridRetriever's RRF prior when metadata has `rrf`."""
        if not docs:
            return []
        texts = [d.page_content or "" for d in docs]
        cross = np.asarray(self.score_pairs(query, texts), dtype=np.float64)
        priors = np.array(
            [float((d.metadata or {}).get("rrf") or 0.0) for d in docs],
            dtype=np.float64,
        )
        if priors.size and priors.max() > 0:
            priors = priors / (priors.max() + 1e-9)
        fused = (1.0 - prior_weight) * cross + prior_weight * priors
        order = np.argsort(-fused)[:top_k]
        out: List[Document] = []
        for i in order:
            d = docs[int(i)]
            meta = dict(d.metadata or {})
            meta["cross_score"] = float(cross[int(i)])
            meta["fused_score"] = float(fused[int(i)])
            out.append(Document(page_content=d.page_content, metadata=meta))
        return out

    def cache_stats(self) -> Dict[str, int]:
        return {"entries": len(self._cache), "max_entries": self._cfg.max_cache}
