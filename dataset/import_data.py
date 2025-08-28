import json
import uuid
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from service.utils.dense_embedding import DenseEmbedding
from service.utils.data_transformer import DataTransformer
from data.snippet import get_all_snippets
from config import EMBED, QDRANT

class DataImporter:
    def __init__(self, data: List[Dict[str, Any]]):
        self.collection_name = QDRANT.COLLECTION
        self.documents = self.parse_data(data)

    def parse_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        docs = []
        for record in data:
            content_list = [item['children'][1]['script'].replace('\n', '').strip() for item in record['children']]
            if not content_list:
                continue

            full_text = "\n".join(content_list).strip()
            if not full_text:
                continue

            title_parts = [
                record.get("course_name", ""),
                record.get("chapter_name", ""),
                record.get("module_name", "")
            ]
            title = "-".join([t for t in title_parts if t])

            # 将ObjectId转换为字符串
            record_id = str(record.get("_id", ""))

            docs.append({
                "id": record_id,
                "title": title,
                "content": full_text,
                "metadata": {
                    "summary": record.get("label", {}).get("summary"),
                    "keywords_tags": record.get("label", {}).get("keywords_tags"),
                    "bloom_level": record.get("label", {}).get("bloom_level")
                }
            })

        return docs


    def clear_collection(self):
        print(f"🗑️ Clearing collection '{self.collection_name}'...")
        client = QdrantClient(url=f"http://{QDRANT.HOST}", port=QDRANT.PORT, grpc_port=QDRANT.GRPC_PORT)
        try:
            # 检查collection是否存在
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                # 删除collection
                client.delete_collection(self.collection_name)
                print(f"✅ Collection '{self.collection_name}' deleted successfully")
            else:
                print(f"ℹ️ Collection '{self.collection_name}' does not exist, nothing to clear")
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")


    def import_data(self, batch_size: int = 128, clear_existing: bool = False):

        if clear_existing:
            self.clear_collection()

        print("Step 1: Splitting content into nodes…")
        transformer = DataTransformer(included_metadata_keys=["id", "title", "summary", "keywords_tags", "bloom_level"])
        nodes = transformer.run_docs(self.documents)
        print(f"→ Transformed into {len(nodes)} nodes")

        # 设置文本和 payload
        texts = [node.text for node in nodes]
        payloads = []
        for i, node in enumerate(nodes):
            payload = {
                "content": node.text,  # payload只包含content
                **node.metadata  # metadata包含id, title, summary, keywords_tags, bloom_level
            }
            payloads.append(payload)

        print("Step 2: Initializing embedding model…")
        embedder = DenseEmbedding.from_setting(EMBED.MODEL, EMBED.PROTOCOL, EMBED.HOST, EMBED.PORT)
        vector_dim = len(embedder._get_text_embedding("test"))
        print(f"→ Vector dimension: {vector_dim}")

        print("Step 3: Connecting to Qdrant…")
        client = QdrantClient(url=f"http://{QDRANT.HOST}", port=QDRANT.PORT, grpc_port=QDRANT.GRPC_PORT)
        if QDRANT.COLLECTION not in [c.name for c in client.get_collections().collections]:
            print(f"→ Creating collection '{QDRANT.COLLECTION}' with dim={vector_dim}")
            client.create_collection(
                collection_name=QDRANT.COLLECTION,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
        else:
            print(f"→ Using existing collection '{QDRANT.COLLECTION}'")

        print("Step 4: Embedding and uploading to Qdrant…")
        for i in tqdm(range(0, len(nodes), batch_size), desc="Uploading"):
            batch_texts = texts[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batch_vectors = embedder._get_text_embeddings(batch_texts)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=np.asarray(vec, dtype=np.float32),
                    payload=pl
                )
                for vec, pl in zip(batch_vectors, batch_payloads)
            ]

            client.upsert(collection_name=QDRANT.COLLECTION, points=points)

        print("✅ Finished uploading all data to Qdrant!")

if __name__ == "__main__":
    DataImporter(get_all_snippets()).import_data(clear_existing=True)
