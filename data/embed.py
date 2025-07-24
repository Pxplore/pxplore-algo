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
from config import EMBED, QDRANT

def parse_data(input_path: str) -> List[Dict[str, Any]]:
    """
    读取并解析 JSON 文件（数组格式），只保留有意义的字段。
    
    参数:
        input_json: JSON 文件路径
    
    返回:
        List[Dict]，每个 dict 包含 'content'（纯文本）和 'title'（元信息）
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    docs = []
    for record in raw_data:
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
        title = " - ".join([t for t in title_parts if t])

        docs.append({
            "content": full_text,
            "title": title,
            "metadata": {
                "summary": record.get("label", {}).get("summary"),
                "keywords_tags": record.get("label", {}).get("keywords_tags"),
                "bloom_level": record.get("label", {}).get("bloom_level")
            }
        })

    return docs

def embed_json(input_path: str, batch_size: int = 128):
    print("Step 0: Parsing JSON data…")
    docs = parse_data(input_path)
    print(f"→ Parsed {len(docs)} documents")

    print("Step 1: Splitting content into nodes…")
    transformer = DataTransformer(included_metadata_keys=["course_name", "chapter_name"])
    nodes = transformer.run_docs(docs)
    print(f"→ Transformed into {len(nodes)} nodes")

    # 设置文本和 payload
    texts = [node.text for node in nodes]
    payloads = [node.metadata for node in nodes]

    print("Step 2: Initializing embedding model…")
    embedder = DenseEmbedding.from_setting(EMBED.MODEL, EMBED.PROTOCOL, EMBED.HOST, EMBED.PORT)
    vector_dim = len(embedder._get_text_embedding("test"))

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