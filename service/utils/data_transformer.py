import json
import os
import logging
from typing import List, Dict, Any, Optional, Union, ClassVar, Tuple
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline

# Configure logging
logger = logging.getLogger(__name__)

splitter_dict = {
    'SentenceSplitter': SentenceSplitter,
    'TokenTextSplitter': TokenTextSplitter,
}

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
            docs: List of document dictionaries with 'content', 'title', 'id' and 'metadata' keys
            
        Returns:
            List of processed nodes
        """
        if not docs:
            return []
            
        documents = []
        for doc in docs:
            # 合并title、id和metadata到metadata中
            metadata = doc.get('metadata', {}).copy()
            metadata['title'] = doc.get('title', '')
            metadata['id'] = doc.get('id', '')
            
            documents.append(Document(
                text=doc['content'], 
                metadata=metadata
            ))
            
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
