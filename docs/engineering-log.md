# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-21

I've been experimenting with different chunking strategies for my retrieval-augmented generation (RAG) setup. I found that using a fixed-size sliding window with 50% overlap significantly improved the relevance of retrieved chunks, but it also increased the index size by about 30%. This tradeoff is crucial to consider when balancing retrieval quality and storage efficiency.

### 2026-07-23

Reviewed retrieval quality, chunking, and hybrid search today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-07-30

Noticed that smaller chunk sizes (e.g., 200 tokens) improve retrieval precision but significantly increase the number of chunks, which impacts search latency. Balancing chunk size is key: larger chunks reduce total chunks and improve speed but sometimes hurt retrieval granularity, especially with hybrid search combining dense and sparse vectors.

### 2026-08-04

Reviewed retrieval quality, chunking, and hybrid search today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-10

Reviewed retrieval quality, chunking, and hybrid search today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-12

Reviewed retrieval quality, chunking, and hybrid search today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-17

Reviewed retrieval quality, chunking, and hybrid search today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.
