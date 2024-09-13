# coding=utf-8
import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI
from llm import MyLLM

import os


TRAIN_FILES = ["docs/train.txt"]
VAL_FILES = ["docs/test.txt"]
EXP_FILES = ["docs/exp.txt"]

TRAIN_CORPUS_FPATH = "train_corpus.json"
VAL_CORPUS_FPATH = "val_corpus.json"

qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination in Chinese. The questions should be diverse in nature \
across the document in Chinese. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.
The length of each question should not beyond 20 chars.
Directly give the question without start with any instruction.
"""

def load_corpus(files:list, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files, encoding='utf-8')
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter(chunk_size=250, chunk_overlap=0)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)
llm = MyLLM()

train_dataset = generate_qa_embedding_pairs(nodes=train_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
val_dataset = generate_qa_embedding_pairs(nodes=val_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)

train_dataset.save_json("data/train_dataset.json")
val_dataset.save_json("data/val_dataset.json")
