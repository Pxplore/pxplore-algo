import numpy as np
import httpx
import requests
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, ClassVar, Tuple
from time import sleep
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field
from config import EMBED

# Configure logging
logger = logging.getLogger(__name__)

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
        model = model or EMBED.MODEL
        protocol = protocol or EMBED.PROTOCOL
        host = host or EMBED.HOST
        port = port or EMBED.PORT
        
        embedding_url = f'{protocol}://{host}:{port}/embed'
        return DenseEmbedding(model_name=model, embedding_url=embedding_url)

    def _make_embedding_request(self, 
                                text: Union[str, List[str]], 
                                timeout: float = 10.0,  
                                retries: int = 3, 
                                backoff_factor: float = 1.0) -> Any:
        """Make an embedding request to the embedding service.
        
        Args:
            text: Text or list of texts to embed
            timeout: Request timeout in seconds (increased to 120s for model loading)
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
                        timeout=timeout  # Use the passed timeout parameter
                    )
                    r.raise_for_status()
                    
                    response_data = r.json()
                    if response_data.get('message') is not None:
                        logger.error(f"Embedding service returned error: {response_data.get('message')}")
                        raise Exception(f"Embedding service error: {response_data.get('message')}")
                    
                    if is_batch:
                        return [i['dense_embed'] for i in response_data['data']]
                    else:
                        return response_data['data']['dense_embed']
                        
            except (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError) as e:
                attempt += 1
                if attempt < retries:
                    sleep_time = backoff_factor * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(f"Embedding request failed (attempt {attempt}/{retries}): {str(e)}. Retrying in {sleep_time:.2f}s")
                    sleep(sleep_time)
                else:
                    logger.error(f"All {retries} embedding request attempts failed: {str(e)}")
                    logger.error(f"Embedding service at {self.embedding_url} appears to be unavailable")
                    raise e
            except Exception as e:
                raise e

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
                timeout=120.0  # Increased timeout for async requests too
            )
            r.raise_for_status()
            response_data = r.json()
            if response_data.get('message') is not None:
                raise Exception(f"Embedding service error: {response_data.get('message')}")
            return response_data['data']['dense_embed']

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
                timeout=120.0  # Increased timeout for async requests too
            )
            r.raise_for_status()
            response_data = r.json()
            if response_data.get('message') is not None:
                raise Exception(f"Embedding service error: {response_data.get('message')}")
            return [i['dense_embed'] for i in response_data['data']]
