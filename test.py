from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import os
import random
import jsonlines
from tqdm import tqdm
import joblib
import logging
from datasets import Dataset

# 初始化日志
logger = logging.getLogger()
logger.handlers.clear()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class BuildTrainData:
    def __init__(self, model_path, train_data, file_name):
        logging.info("加载原始数据...")
        self.data = train_data
        logging.info(f"从 {model_path} 加载向量化模型...")
        self.model = SentenceTransformer(model_path)
        self.model.eval()
        self.batch_size = 32
        self.faiss_measure = faiss.METRIC_L2
        self.index_type = "HNSW64"

        save_dir = "index"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.embedding_path = f"{save_dir}/embedding_{file_name}.pkl"
        self.faiss_index_path = f"{save_dir}/faiss_{file_name}.index"
        self.bge_train_data_path = f"data/{file_name}_hnm_train.jsonl"

    def embedding(self, text_list):
        logging.info("向量化...")
        embeddings = self.model.encode(text_list, batch_size=self.batch_size, show_progress_bar=True)
        return embeddings

    def embedding_mul_gpu(self, text_list):
        logging.info("多GPU并行向量化...")
        # 通过target_devices指定GPU，如target_devices=['cuda:0', 'cuda:1']
        pool = self.model.start_multi_process_pool()
        embeddings = self.model.encode_multi_process(text_list, pool, batch_size=self.batch_size)
        self.model.stop_multi_process_pool(pool)
        return embeddings

    def build_faiss_index(self):
        if os.path.exists(self.faiss_index_path):
            logging.info(f"{self.faiss_index_path}已存在...")
            faiss_index = faiss.read_index(self.faiss_index_path)
            embeddings = joblib.load(self.embedding_path)
            return faiss_index, embeddings

        logging.info("从本地加载向量化的数据...")
        embeddings = joblib.load(self.embedding_path)
        dim = embeddings.shape[1]
        faiss_index = faiss.index_factory(dim, self.index_type, self.faiss_measure)
        logging.info("构建索引...")
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, self.faiss_index_path)
        return faiss_index, embeddings

    def compute_retrival(self, mul_gpus=None, retrival_topk=100):
        logging.info("挖掘困难样本...")
        query_list = self.data["anchor"]

        # query = "为这个句子生成表示以用于检索相关文章：" + row["query"]
        if not os.path.exists(self.embedding_path):
            logging.info("embedding 文件不存在, 重新embedding...")
            if not mul_gpus:
                logging.info("只使用一个GPU...")
                query_embedding = self.embedding(self.data["positive"])
            else:
                logging.info("多GPU加速...")
                query_embedding = self.embedding_mul_gpu(self.data["positive"])
            joblib.dump(query_embedding, self.embedding_path)
        faiss_index, query_embedding = self.build_faiss_index()

        logging.info("开始处理数据...")
        distances, indexs = faiss_index.search(query_embedding, retrival_topk)

        anchor, positive, negative = [], [], []
        for idx, query in enumerate(tqdm(query_list, desc="挖掘困难样本")):
            answer = self.data["positive"][idx]
            target_answers = []

            # dist越小越相似
            neg_samples_tune = []
            for dist, df_idx in zip(*[distances[idx], indexs[idx]]):
                if df_idx == -1:
                    logging.info(f"bade index {df_idx}")
                    continue

                target_query = self.data["anchor"][df_idx]
                if target_query == query:
                    continue
                target_answer = self.data["positive"][df_idx]
                if target_answer == answer:
                    continue

                if dist > 0.4 and dist <= 0.7:
                    target_answers.append(target_answer)
                elif dist > 0.7:
                    neg_samples_tune.append(target_answer)

            if len(target_answers) == 0:
                logging.info(f"query： {query} 无负样本")
                target_answers = neg_samples_tune
                if len(target_answers) == 0:
                    logging.info(f"query： {query} 无负样本")
                    continue
            elif len(target_answers) > 10:
                target_answers = random.sample(target_answers, 10)

            anchor.append(query)
            positive.append(answer)
            negative.append(random.sample(target_answers, 1)[0])

        return Dataset.from_dict({"anchor": anchor, "positive": positive, "negative": negative})