from typing import List, Dict
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

        # 临时跳过嵌入和Qdrant，使用模拟数据
        print("→ Using mock data for testing...")
        self.payloads, self.corpus_texts = self._load_mock_corpus()
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus_texts])

    def _load_mock_corpus(self) -> (List[Dict], List[str]):
        """临时模拟数据，用于测试"""
        print("→ Loading mock corpus for testing...")
        mock_payloads = [
            {
                "id": "mock_1",
                "title": "机器学习基础",
                "summary": "介绍机器学习的基本概念和算法",
                "bloom_level": "理解",
                "content": "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出决策。"
            },
            {
                "id": "mock_2", 
                "title": "深度学习原理",
                "summary": "深入讲解深度学习的核心原理",
                "bloom_level": "应用",
                "content": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。"
            },
            {
                "id": "mock_3",
                "title": "神经网络架构",
                "summary": "各种神经网络架构的设计和应用",
                "bloom_level": "分析",
                "content": "神经网络架构包括前馈网络、卷积网络、循环网络等，每种架构都有其特定的应用场景。"
            }
        ]
        
        mock_texts = [payload["content"] for payload in mock_payloads]
        print(f"→ Loaded {len(mock_payloads)} mock documents.")
        return mock_payloads, mock_texts

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

    def search(self, query: str, title: str = None) -> List[Dict]:
        print(f"🔍 Searching for: {query[:20]}...({len(query)})...")
        
        # 模拟模式：只使用BM25检索
        bm25_scores = dict(zip(
            self.corpus_texts,
            self.bm25.get_scores(query.split())
        ))

        # 模拟dense分数（随机分数）
        import random
        dense_scores = {text: random.random() for text in self.corpus_texts}

        # 混合评分
        hybrid_scores = {}
        for text in bm25_scores:
            bm25_score = bm25_scores[text]
            dense_score = dense_scores.get(text, 0)
            hybrid_scores[text] = self.alpha * bm25_score + (1 - self.alpha) * dense_score

        # 排序并返回结果
        sorted_texts = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        results = []
        for text, score in sorted_texts:
            idx = self.corpus_texts.index(text)
            payload = self.payloads[idx].copy()

            if title and payload.get("title") == title:
                print(f"⏭️ Skipping key snippet: {title}")
                continue

            if "content" in payload:
                del payload["content"]

            results.append({
                "content": text,
                "score": round(score, 4),
                "metadata": payload
            })
            
            if len(results) >= self.top_k:
                break

        return results


if __name__ == "__main__":
    query = "双频干涉 位移测量"
    hits = HybridRetriever().search(query)

    print("\n🎯 Top Results:")
    for i, hit in enumerate(hits):
        print(f"[{i+1}] Score={hit['score']:.4f}")
        print(hit["metadata"])
        print("-" * 40)