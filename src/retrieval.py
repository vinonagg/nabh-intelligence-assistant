from collections import defaultdict
import re
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class HybridRetriever:
    """Reciprocal-rank fusion over FAISS semantic and BM25 keyword results."""
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        self.vectorstore = None
        self.chunks = []
        self.bm25 = None
        self.tokens = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+|[\u0B80-\u0BFF]+", text.lower())

    def build(self, chunks):
        if not chunks:
            raise ValueError("No text passages are available to index.")
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        self.vectorstore = FAISS.from_texts(
            texts,
            embedding=self.embeddings,
            metadatas=[{"page_start": c.page_start, "page_end": c.page_end, "chunk_id": c.chunk_id} for c in chunks],
        )
        self.tokens = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int = 7):
        if self.vectorstore is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")
        pool = max(top_k * 3, 15)
        vector_hits = self.vectorstore.similarity_search_with_score(query, k=min(pool, len(self.chunks)))
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        bm25_ranked = np.argsort(bm25_scores)[::-1][:min(pool, len(self.chunks))]
        fused, details = defaultdict(float), {}
        for rank, (doc, raw_score) in enumerate(vector_hits, start=1):
            key = doc.metadata["chunk_id"]
            fused[key] += 1.0 / (60 + rank)
            details[key] = {
                "text": doc.page_content, "page_start": doc.metadata["page_start"],
                "page_end": doc.metadata["page_end"], "chunk_id": key, "vector_raw_score": float(raw_score),
            }
        for rank, index in enumerate(bm25_ranked, start=1):
            chunk = self.chunks[int(index)]
            fused[chunk.chunk_id] += 1.0 / (60 + rank)
            details[chunk.chunk_id] = {
                "text": chunk.text, "page_start": chunk.page_start, "page_end": chunk.page_end, "chunk_id": chunk.chunk_id,
            }
        results = []
        for key in sorted(fused, key=fused.get, reverse=True)[:top_k]:
            item = details[key]
            item["score"] = float(fused[key])
            a, b = item["page_start"], item["page_end"]
            item["label"] = f"PDF page {a}" if a == b else f"PDF pages {a}–{b}"
            results.append(item)
        return results

    @staticmethod
    def merge_graph_evidence(evidence, graph_evidence):
        merged = list(evidence)
        seen = {item.get("text", "") for item in merged}
        for item in graph_evidence:
            if item.get("text") not in seen:
                item.setdefault("score", 0.0)
                item.setdefault("label", "Neo4j knowledge graph")
                merged.append(item)
                seen.add(item.get("text", ""))
        return merged
