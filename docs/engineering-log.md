# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-21

I've been experimenting with different chunking strategies for my retrieval-augmented generation (RAG) setup. I found that using a fixed-size sliding window with 50% overlap significantly improved the relevance of retrieved chunks, but it also increased the index size by about 30%. This tradeoff is crucial to consider when balancing retrieval quality and storage efficiency.
