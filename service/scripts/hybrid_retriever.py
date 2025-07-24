import json
import argparse
from typing import List, Dict
import numpy as np
from qdrant_client import QdrantClient
from service.utils.dense_embedding import DenseEmbedding
from rank_bm25 import BM25Okapi
from config import QDRANT, EMBED


class HybridRetriever:
    def __init__(self):
        """
        alpha: BM25 与 Dense 检索结果融合权重，范围 [0, 1]
        top_k: 返回的文档数量
        """
        self.alpha = QDRANT.ALPHA
        self.top_k = QDRANT.TOP_K

        self.embedder = DenseEmbedding.from_setting()
        self.qdrant = QdrantClient(
            url=f"http://{QDRANT.HOST}",
            port=QDRANT.PORT,
            grpc_port=QDRANT.GRPC_PORT
        )

        self.payloads, self.corpus_texts = self._load_corpus_from_qdrant()
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus_texts])

    def _load_corpus_from_qdrant(self) -> (List[Dict], List[str]):
        print("→ Loading full corpus from Qdrant...")
        points = self.qdrant.scroll(
            collection_name=QDRANT.COLLECTION,
            scroll_filter=None,
            limit=10_000,  # 可调整上限
            with_payload=True,
            with_vectors=False
        )[0]

        print(f"→ Raw points from Qdrant: {len(points)}")

        payloads = []
        texts = []
        for p in points:
            content = p.payload.get("text") or p.payload.get("content")
            if content:
                texts.append(content)
                payloads.append(p.payload)
        print(f"→ Loaded {len(payloads)} documents.")
        return payloads, texts

    def search(self, query: str) -> List[Dict]:
        print(f"🔍 Searching for: {query}")
        query_embedding = self.embedder._get_text_embedding(query)

        # Step 1: dense retrieval from Qdrant
        dense_hits = self.qdrant.search(
            collection_name=QDRANT.COLLECTION,
            query_vector=query_embedding,
            limit=self.top_k * 2,
            with_payload=True
        )

        dense_scores = {}
        for point in dense_hits:
            text = point.payload.get("text") or point.payload.get("content")
            if text:
                dense_scores[text] = 1 - point.score  # Qdrant uses cosine distance

        # Step 2: bm25 retrieval (local)
        bm25_scores = dict(zip(
            self.corpus_texts,
            self.bm25.get_scores(query.split())
        ))

        # Step 3: hybrid score = alpha * bm25 + (1-alpha) * dense
        hybrid_scores = {}
        for text in bm25_scores:
            bm25_score = bm25_scores[text]
            dense_score = dense_scores.get(text, 0)
            hybrid_scores[text] = self.alpha * bm25_score + (1 - self.alpha) * dense_score

        # Step 4: sort by hybrid score
        sorted_texts = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        # Return top-k payloads
        results = []
        for text, score in sorted_texts:
            idx = self.corpus_texts.index(text)
            payload = self.payloads[idx]
            results.append({
                "content": text,
                "score": round(score, 4),
                "metadata": payload
            })

        return results


if __name__ == "__main__":
    query = "双频干涉 位移测量"
    hits = HybridRetriever().search(query)

    print("\n🎯 Top Results:")
    for i, hit in enumerate(hits):
        print(f"[{i+1}] Score={hit['score']:.4f}")
        print(hit["metadata"])
        print("-" * 40)