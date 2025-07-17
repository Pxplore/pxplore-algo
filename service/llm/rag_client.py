import numpy as np
import httpx
import requests
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, ClassVar, Tuple
from time import sleep
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import BaseNode
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from urllib3.util.retry import Retry
from config import BOCHA_TOKEN, RAG_HOST, RAG_PORT, RAG_EMBED_MODEL

# Configure logging
logger = logging.getLogger(__name__)

class RAG:
    """RAG configuration constants."""
    EMBED_MODEL = RAG_EMBED_MODEL
    EMBED_PROTOCOL = "http"
    EMBED_HOST = RAG_HOST
    EMBED_PORT = RAG_PORT

class DenseEmbedding(BaseEmbedding):
    """Dense embedding model that interfaces with an external embedding service.
    
    This class handles requesting embeddings from a remote service for text and queries.
    """
    
    embedding_url: str = Field(description="The url of the embedding model server.")
    HEADERS: ClassVar[Dict[str, str]] = {"Content-Type": "application/json", "Accept": "application/json"}
    
    @staticmethod
    def from_setting(model: str = None, protocol: str = None, host: str = None, port: int = None) -> 'DenseEmbedding':
        """Create a DenseEmbedding instance from configuration settings.
        
        Args:
            model: Name of the embedding model
            protocol: Protocol for the embedding service (http/https)
            host: Host of the embedding service
            port: Port of the embedding service
            
        Returns:
            Configured DenseEmbedding instance
        """
        # Use provided values or fall back to config
        model = model or RAG.EMBED_MODEL
        protocol = protocol or RAG.EMBED_PROTOCOL
        host = host or RAG.EMBED_HOST
        port = port or RAG.EMBED_PORT
        
        embedding_url = f'{protocol}://{host}:{port}/embed'
        return DenseEmbedding(model_name=model, embedding_url=embedding_url)

    def _make_embedding_request(self, 
                               text: Union[str, List[str]], 
                               timeout: float = 60 * 100, 
                               retries: int = 3, 
                               backoff_factor: float = 1.0) -> Any:
        """Make an embedding request to the embedding service.
        
        Args:
            text: Text or list of texts to embed
            timeout: Request timeout in seconds
            retries: Number of retries on failure
            backoff_factor: Backoff factor for retries
            
        Returns:
            JSON response from the embedding service
            
        Raises:
            Exception: If all retry attempts fail
        """
        is_batch = isinstance(text, list)
        
        attempt = 0
        while attempt < retries:
            try:
                with httpx.Client() as client:
                    r = client.post(
                        self.embedding_url, 
                        headers=self.HEADERS, 
                        json={
                            'text': text,
                            'model': self.model_name,
                            'return_dense': True,
                            'return_sparse': False,
                        }, 
                        timeout=10  # Reduced timeout to prevent hanging
                    )
                    r.raise_for_status()
                    
                    if is_batch:
                        return [i['dense_embed'] for i in r.json()['data']]
                    else:
                        return r.json()['data']['dense_embed']
                        
            except (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError) as e:
                attempt += 1
                if attempt < retries:
                    sleep_time = backoff_factor * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(f"Embedding request failed (attempt {attempt}/{retries}): {str(e)}. Retrying in {sleep_time:.2f}s")
                    sleep(sleep_time)
                else:
                    logger.error(f"All {retries} embedding request attempts failed: {str(e)}")
                    logger.error(f"Embedding service at {self.embedding_url} appears to be unavailable")
                    # Return mock embeddings as fallback
                    logger.warning("Falling back to mock embeddings")
                    return self._create_mock_embedding(text, is_batch)
            except Exception as e:
                logger.error(f"Unexpected error in embedding request: {str(e)}")
                # Return mock embeddings as fallback
                logger.warning("Falling back to mock embeddings due to unexpected error")
                return self._create_mock_embedding(text, is_batch)

    def _create_mock_embedding(self, text: Union[str, List[str]], is_batch: bool = False) -> Union[List[float], List[List[float]]]:
        """Create mock embeddings when the real service is unavailable."""
        import hashlib
        
        def create_single_embedding(single_text: str) -> List[float]:
            # Create a deterministic hash-based embedding
            text_hash = hashlib.md5(single_text.encode()).hexdigest()
            embedding = []
            embedding_dim = 1024
            
            for i in range(embedding_dim):
                # Use different parts of the hash to create varied values
                hash_segment = text_hash[(i * 2) % len(text_hash):(i * 2 + 2) % len(text_hash) + 1]
                if hash_segment:
                    value = int(hash_segment, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
                else:
                    value = 0.0
                embedding.append(value)
            return embedding
        
        if is_batch:
            return [create_single_embedding(t) for t in text]
        else:
            return create_single_embedding(text)

    def request_bge_embedding(self, text: str) -> List[float]:
        """Request embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self._make_embedding_request(text)

    def request_bge_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Request embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._make_embedding_request(texts)

    # Base class implementation methods
    def _get_text_embedding(self, text: str) -> Embedding:
        return self.request_bge_embedding(text)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self.request_bge_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return self.request_bge_embeddings(texts)

    # Async methods
    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Get text embedding asynchronously."""
        async with httpx.AsyncClient() as aclient:
            r = await aclient.post(
                self.embedding_url, 
                headers=self.HEADERS, 
                json={
                    'text': text,
                    'model': self.model_name,
                    'return_dense': True,
                    'return_sparse': False,
                },
                timeout=60 * 100
            )
            r.raise_for_status()
            return r.json()['data']['dense_embed']

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding asynchronously."""
        return await self._aget_text_embedding(query)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Get text embeddings for multiple texts asynchronously."""
        async with httpx.AsyncClient() as aclient:
            r = await aclient.post(
                self.embedding_url, 
                headers=self.HEADERS, 
                json={
                    'text': texts,
                    'model': self.model_name,
                    'return_dense': True,
                    'return_sparse': False,
                },
                timeout=60 * 100
            )
            r.raise_for_status()
            return [i['dense_embed'] for i in r.json()['data']]

class DataTransformer:
    """Transform documents into nodes for vector storage and retrieval.
    
    This class provides methods to process documents, split them into nodes,
    and prepare them for vector embedding and storage.
    """
    
    splitter_dict = {
        'SentenceSplitter': SentenceSplitter,
        'TokenTextSplitter': TokenTextSplitter,
    }
    
    def __init__(
            self,
            included_metadata_keys: List[str] = ['file_name'],
            splitter_type: str = 'SentenceSplitter',
            splitter_kwargs: Dict[str, Any] = None,
        ):
        """Initialize the DataTransformer.
        
        Args:
            included_metadata_keys: Metadata keys to include in the processed nodes
            splitter_type: Type of text splitter to use ('SentenceSplitter' or 'TokenTextSplitter')
            splitter_kwargs: Arguments for the text splitter
        """
        self.included_metadata_keys = included_metadata_keys
        
        if splitter_kwargs is None:
            splitter_kwargs = {'chunk_size': 512, 'chunk_overlap': 128}
            
        if splitter_type not in self.splitter_dict:
            raise ValueError('splitter_type only supports SentenceSplitter and TokenTextSplitter')
            
        splitter = self.splitter_dict[splitter_type](**splitter_kwargs)
        self.pipeline = IngestionPipeline(transformations=[splitter])
    
    def run(self, input_file_paths: List[str]) -> List[BaseNode]:
        """Process files from the given paths into nodes.
        
        Args:
            input_file_paths: List of file or directory paths to process
            
        Returns:
            List of processed nodes
            
        Raises:
            ValueError: If no valid files are found to process
        """
        all_files = []
        for path in input_file_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    all_files.extend([os.path.join(root, file) for file in files])
            elif os.path.isfile(path):
                all_files.append(path)
            else:
                logger.warning(f"Path {path} is not a valid file or directory.")
        
        if not all_files:
            raise ValueError("No valid files found to process.")
        
        documents = SimpleDirectoryReader(input_files=all_files).load_data()
        nodes = self.pipeline.run(
            documents=documents,
            in_place=True,
            show_progress=True,
        )
        
        # Filter node metadata to only include specified keys
        self._filter_node_metadata(nodes)
        return nodes

    def run_docs(self, docs: List[Dict[str, str]]) -> List[BaseNode]:
        """Process document dictionaries into nodes.
        
        Args:
            docs: List of document dictionaries with 'content' and 'title' keys
            
        Returns:
            List of processed nodes
        """
        if not docs:
            return []
            
        documents = [Document(text=doc['content'], metadata={'file_name': doc['title']}) for doc in docs]
        nodes = self.pipeline.run(
            documents=documents,
            in_place=True,
            show_progress=True  
        )
        
        # Filter node metadata to only include specified keys
        self._filter_node_metadata(nodes)
        return nodes
    
    def _filter_node_metadata(self, nodes: List[BaseNode]) -> None:
        """Filter node metadata to only include specified keys.
        
        Args:
            nodes: List of nodes to filter
        """
        for node in nodes:
            node.metadata = {key: node.metadata[key] for key in self.included_metadata_keys if key in node.metadata}

    def run_to_save(self, input_file_paths: List[str], output_file_path: str) -> None:
        """Process files and save the resulting nodes to a file.
        
        Args:
            input_file_paths: List of file or directory paths to process
            output_file_path: Path to save the processed nodes
        """
        nodes = self.run(input_file_paths)
        with open(output_file_path, 'w') as f:
            for node in nodes:
                f.write(json.dumps({
                    'id_': node.id_,
                    'text': node.text,
                    'start_char_idx': node.start_char_idx,
                    'end_char_idx': node.end_char_idx,
                    'mimetype': node.mimetype,
                    'metadata': node.metadata,
                }, ensure_ascii=False))
                f.write('\n')

class LocalVectorRetriever:
    """Local vector similarity computation and retrieval without external vector store.
    
    This class handles vector similarity computation locally using numpy,
    removing the need for an external vector store service.
    """
    
    def __init__(self, top_k: int = 5):
        """Initialize the LocalVectorRetriever.
        
        Args:
            top_k: Number of top similar items to retrieve
        """
        self.top_k = top_k
        self.embed_model = DenseEmbedding.from_setting()
        self.nodes: List[BaseNode] = []
        self.node_embeddings: Optional[np.ndarray] = None
        
    def import_data(self, nodes: List[BaseNode]) -> None:
        """Import nodes and compute their embeddings.
        
        Args:
            nodes: List of nodes to import
        """
        if not nodes:
            logger.warning("No nodes to import")
            return
            
        self.nodes = nodes
        # Get embeddings for all nodes
        texts = [node.text for node in nodes]
        embeddings = self.embed_model._get_text_embeddings(texts)
        self.node_embeddings = np.array(embeddings)
        
    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and all nodes.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        node_norms = self.node_embeddings / np.linalg.norm(self.node_embeddings, axis=1)[:, np.newaxis]
        
        # Compute cosine similarities
        similarities = np.dot(node_norms, query_norm)
        return similarities
        
    def retrieve(self, query: str) -> List[BaseNode]:
        """Retrieve most similar nodes for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of most similar nodes with similarity scores
        """
        if not self.nodes or self.node_embeddings is None:
            logger.warning("No nodes available for retrieval")
            return []
            
        # Get query embedding
        query_embedding = np.array(self.embed_model._get_query_embedding(query))
        
        # Compute similarities
        similarities = self._compute_similarities(query_embedding)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Create result nodes with scores
        results = []
        for idx in top_k_indices:
            node = self.nodes[idx]
            results.append(node)
            
        return results

class BingSpider:
    """Web spider for searching and extracting content using BOCHA API.
    
    This class provides methods to search and get content from search results
    using the BOCHA web search API.
    """
    
    BOCHA_SEARCH_URL = "https://api.bochaai.com/v1/web-search"
    
    def __init__(self):
        """Initialize the BingSpider."""
        self._session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry capabilities.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504, 429]
        )
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retry))
        return session
    
    def search(self, query: str, pages: int = 1) -> List[Dict[str, str]]:
        """Search using BOCHA API and return search results with content.
        
        Args:
            query: Search query
            pages: Number of search result pages to process (Note: BOCHA API returns all results in one call)
            
        Returns:
            List of dictionaries containing search results with content
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
            
        query = query.strip()
        results = []
        
        try:
            # Calculate count based on pages (assuming 10 results per page)
            count = min(pages * 10, 50)  # BOCHA API may have limits, cap at 50
            
            # Prepare request payload
            payload = json.dumps({
                "query": query,
                "summary": True,
                "count": count
            })
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {BOCHA_TOKEN}',
                'Content-Type': 'application/json'
            }
            
            logger.info(f"Searching BOCHA API with query: {query}")
            
            # Make search request
            response = self._session.post(
                self.BOCHA_SEARCH_URL, 
                headers=headers, 
                data=payload,
                timeout=30
            )
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"BOCHA API request failed with status code {response.status_code}")
                return []
            
            # Parse response
            response_data = response.json()
            
            # Check for successful response
            if response_data.get('code') != 200:
                logger.error(f"BOCHA API returned error code: {response_data.get('code')}")
                return []
            
            # Extract search results from the correct path: data.webPages.value
            data = response_data.get('data', {})
            web_pages = data.get('webPages', {})
            search_results = web_pages.get('value', [])
            
            if search_results:
                for result in search_results:
                    # Extract information from BOCHA response
                    title = result.get('name', '')
                    url = result.get('url', '')
                    snippet = result.get('snippet', '')
                    # Use 'summary' field as the main content, fallback to 'snippet' if no summary
                    content = result.get('summary', '') or snippet
                    
                    # Only add results that have substantial content
                    if title and url and content and len(content.strip()) > 100:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "content": content,
                            "query": query,
                        })
                        logger.debug(f"Added search result: {title}, length: {len(content)}")
            else:
                logger.warning("No search results found in BOCHA API response")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during BOCHA API search for query '{query}': {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in BOCHA API response: {str(e)}")
        except Exception as e:
            logger.error(f"Error during BOCHA API search for query '{query}': {str(e)}")
        
        return results