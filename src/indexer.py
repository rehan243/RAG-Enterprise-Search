"""
Ingest: split, embed in batches, build FAISS. IVF+PQ keeps memory sane on big corpora.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _retry(callable_fn, attempts: int = 5, base_sleep: float = 0.5):
    last = None
    for i in range(attempts):
        try:
            return callable_fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2**i))
            logger.warning("retry %s/%s after %s", i + 1, attempts, e)
    raise last  # type: ignore[misc]


@dataclass
class IndexerConfig:
    chunk_size: int = 900
    chunk_overlap: int = 120
    batch_size: int = 64
    nlist: int = 4096
    m_pq: int = 64  # must divide embedding dim
    nprobe: int = 32


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    embed_batches: int = 0
    wall_seconds: float = 0.0


class DocumentIngestor:
    def __init__(
        self,
        cfg: IndexerConfig,
        embed_fn: Callable[[List[str]], np.ndarray],
    ) -> None:
        self._cfg = cfg
        self._embed_fn = embed_fn
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

    def chunk(self, doc_id: str, text: str) -> List[Tuple[str, str]]:
        parts = self._splitter.split_text(text)
        return [(f"{doc_id}::{i}", p) for i, p in enumerate(parts)]

    def embed_batches(self, texts: Sequence[str]) -> Tuple[np.ndarray, int]:
        out: List[np.ndarray] = []
        bs = self._cfg.batch_size
        batches = 0
        for i in range(0, len(texts), bs):
            batch = list(texts[i : i + bs])
            arr = _retry(lambda: self._embed_fn(batch))
            if arr.dtype != np.float32:
                arr = arr.astype("float32")
            out.append(arr)
            batches += 1
        return (np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)), batches

    def build_ivf_pq_index(self, vectors: np.ndarray) -> faiss.Index:
        if vectors.size == 0:
            raise ValueError("no vectors")
        dim = vectors.shape[1]
        if dim % self._cfg.m_pq != 0:
            raise ValueError("embedding dim must be divisible by m_pq")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, self._cfg.nlist, self._cfg.m_pq, 8)
        index.nprobe = self._cfg.nprobe
        if not index.is_trained:
            index.train(vectors)
        index.add(vectors)
        return index

    def ingest_corpus(
        self,
        corpus: Iterable[Tuple[str, str]],
    ) -> Tuple[faiss.Index, List[str], np.ndarray, IngestStats]:
        t0 = time.perf_counter()
        chunk_ids: List[str] = []
        chunk_texts: List[str] = []
        doc_count = 0
        for doc_id, body in corpus:
            doc_count += 1
            if not body.strip():
                logger.warning("skipping empty document %s", doc_id)
                continue
            for cid, chunk in self.chunk(doc_id, body):
                chunk_ids.append(cid)
                chunk_texts.append(chunk)
        logger.info("ingest: %s docs -> %s chunks", doc_count, len(chunk_texts))
        vecs, batches = self.embed_batches(chunk_texts)
        if vecs.size == 0:
            raise ValueError("nothing to index after chunking")
        faiss.normalize_L2(vecs)
        index = self.build_ivf_pq_index(vecs)
        stats = IngestStats(
            documents=doc_count,
            chunks=len(chunk_texts),
            embed_batches=batches,
            wall_seconds=time.perf_counter() - t0,
        )
        return index, chunk_ids, vecs, stats

    def write_bundle(self, out_dir: Path, index: faiss.Index, chunk_ids: List[str], vectors: np.ndarray) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        np.save(out_dir / "vectors.npy", vectors)
        (out_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

    @staticmethod
    def read_bundle(in_dir: Path) -> Tuple[faiss.Index, List[str], np.ndarray]:
        index = faiss.read_index(str(in_dir / "index.faiss"))
        vectors = np.load(in_dir / "vectors.npy")
        chunk_ids = json.loads((in_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        return index, chunk_ids, vectors
