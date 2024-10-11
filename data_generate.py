# coding=utf-8
import json
import os
import argparse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI
from llm import MyLLM
from common.config import qa_generate_prompt_tmpl
from common.util import setup_logger
from llama_index.llms.openai import OpenAI


class DataGenerate:
    def __init__(self, train_files: str, val_files: str, dataset_name: str):
        self.train_files = [os.path.join('docs', file) for file in train_files.split(',')]
        self.val_files = [os.path.join('docs', file) for file in val_files.split(',')]
        self.logger = setup_logger()
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join("data", self.dataset_name)

    def load_corpus(self, files: list, verbose=False):
        if verbose:
            self.logger.info(f"Loading files {files}")

        reader = SimpleDirectoryReader(input_files=files, encoding='utf-8')
        docs = reader.load_data()
        if verbose:
            self.logger.info(f"Loaded {len(docs)} docs")

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=0)
        nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

        if verbose:
            self.logger.info(f"Parsed {len(nodes)} nodes")

        return nodes

    def run(self):
        train_nodes = self.load_corpus(self.train_files, verbose=True)
        val_nodes = self.load_corpus(self.val_files, verbose=True)
        custom_llm = MyLLM()

        if not os.path.exists(self.dataset_path):
            self.logger.info(f"New datasets.")
            os.mkdir(self.dataset_path)

        train_dataset = generate_qa_embedding_pairs(nodes=train_nodes, llm=custom_llm, num_questions_per_chunk=1,
                                                    qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
                                                    output_path=os.path.join(self.dataset_path, "train_dataset.json"))

        val_dataset = generate_qa_embedding_pairs(nodes=val_nodes, llm=custom_llm, num_questions_per_chunk=1,
                                                  qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
                                                  output_path=os.path.join(self.dataset_path, "val_dataset.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--tfs', type=str, required=True, help='train files')
    parser.add_argument('--vfs', type=str, required=True, help='val files')
    parser.add_argument('--ds_name', type=str, required=True, help='dataset name')

    args = parser.parse_args()
    data_generator = DataGenerate(args.tfs, args.vfs, args.ds_name)
    data_generator.run()
