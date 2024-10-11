# -*- coding: utf-8 -*-
import os
import json
import time
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
import logging
import pandas as pd
from datetime import datetime
import argparse

# 添加模型和数据集
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='Model to use', required=True)
parser.add_argument('--dataset', type=str, help='Dataset to use', required=True)

# 选择模型和数据集
args = parser.parse_args()
model_name = args.model
data = args.dataset
# model_name = "bge-base-zh-v1.5"
# data = "doc_qa_dataset.json"

# 日志打印
project_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()
logger.handlers.clear()  # 清除所有已存在的 handlers
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# data process
# load dataset, get corpus, queries, relevant_docs
with open(os.path.join(project_dir, f"data/{data}"), "r", encoding="utf-8") as f:
    content = json.loads(f.read())

corpus = content['corpus']
queries = content['queries']
relevant_docs = content['relevant_docs']

# Load a model
model_path = os.path.join(project_dir, f"model/{model_name}")
model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Model loaded:{model_name}")

s_time = time.time()

# # Evaluate the model
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    # name=f"{os.path.basename(model_path)}",
    score_functions={"cos_sim": cos_sim},
    show_progress_bar=True
)

# Evaluate the model
result = evaluator(model)
logger.info("Evaluation end")

res_df = pd.DataFrame([result])
data = '_'.join(data.split('/'))
print(os.path.join(project_dir, "result", f"{model_name}_{data}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv"))
output_path = os.path.join(project_dir, "result", f"{model_name}_{data}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv")
res_df.to_csv(output_path)
logger.info(f"Time cost: {time.time() - s_time:.2f}s")