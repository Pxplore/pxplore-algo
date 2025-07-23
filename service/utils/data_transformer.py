import json
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline

splitter_dict = {
    'SentenceSplitter': SentenceSplitter,
    'TokenTextSplitter': TokenTextSplitter,
}
import os
class DataTransformer:
    def __init__(
            self,
            included_metadata_keys: list[str] = ['file_name'],
            splitter_type: str = 'SentenceSplitter',
            splitter_kwargs: dict = dict(chunk_size=512, chunk_overlap=128),
        ):
        self.included_metadata_keys = included_metadata_keys
        assert splitter_type in splitter_dict, 'splitter_type only support SentenceSplitter and TokenTextSplitter'
        splitter = splitter_dict[splitter_type](**splitter_kwargs)
        self.pipeline = IngestionPipeline(transformations=[splitter])

    def run(self, input_file_paths):
        print('start to load files')
        all_files = []
        for path in input_file_paths:
            if os.path.isdir(path):
                # 如果是目录，则递归找到目录下的所有文件
                for root, dirs, files in os.walk(path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        all_files.append(full_path)
            elif os.path.isfile(path):
                # 如果是文件，直接添加到文件列表
                all_files.append(path)
            else:
                print(f"Path {path} is not a valid file or directory.")
        
        if not all_files:
            raise ValueError("No valid files found to process.")
        
        print(f"Files to be processed: {all_files}")  # 输出处理的文件列表以进行调试
        documents = SimpleDirectoryReader(input_files=all_files).load_data()
        print('start to do transform')
        nodes = self.pipeline.run(
            documents=documents,
            in_place=True,
            show_progress=True,
        )
        for node in nodes:
            node.metadata = {key: node.metadata[key] for key in self.included_metadata_keys if key in node.metadata}

        return nodes

    def run_to_save(self, input_file_paths, output_file_path):
        nodes = self.run(input_file_paths)
        print('start to save')
        with open(output_file_path, 'w') as f:
            for node in nodes:
                f.write(json.dumps({
                    'id:':              node.id_,
                    'text:':            node.text,
                    'start_char_idx':   node.start_char_idx,
                    'end_char_idx':     node.end_char_idx,
                    'mimetype':         node.mimetype,
                    'metadata':         node.metadata,
                }, ensure_ascii=False))
                f.write('\n')
