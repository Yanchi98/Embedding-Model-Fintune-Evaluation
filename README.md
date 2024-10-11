# Embedding_Fintuning_evaluation

🤖sentence-transformers for embedding model evaluation and finetuning.

🦙llama-index for training data augmentation.

# Requirements
```
Python 3.12.3
PyTorch: 2.4.0+cu121
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
pip install llama-index-finetuning
pip install sentence-transformers==3.0.1
pip install transformers==4.43.2
pip install accelerate==0.34.2
pip install datasets==2.21.0
```

# Evaluation Steps

step 1: Put embedding model to \model (run ```bash model_download.sh```), put evaluation datasets to \data.

step 2: run ```python embedding_evaluate.py --model bge-base-zh-v1.5 --dataset doc_qa_dataset.json```.


# Fine-Tuning

step 1 (optional): Data Augment if needed. Put corpus.txt to \docs. run ```python data_generate.py```.

step 2: Put training dataset and evaluation dataset to \data.

step 3: run ```python finetune.py --model bge-large-zh-v1.5 --train Dragon/train_dataset.json --val Dragon/val_dataset.json --hnm True```.

llama-index provides a data augment method：

Generate Corpus: https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/#finetune-embeddings



# 优化点
## 生成语料

1、自定义MyLLM类, 替换openai chatgpt3.5（主要是考虑到🪜什么的很麻烦）

LLM是用的qwen2， 部署方法参考项目(https://github.com/Yanchi98/Flask-vllm-qwen-)

重写方法见llm.py，要实现complete方法

2、修改prompt，让query不要超过20char(考虑到用户输入长query的概率较低)，并且不要带引导语

修改之前：

prompt为

```
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
```

输出结果为："1、请从下面的文档中回答，根据相关机构的统计，2022年全球半导体设备销售额中，哪个地区的销售额同比增长幅度最大？"  -> 问题冗长、带instruction

修改后，增加两条约束：

```
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
**The length of each question should not beyond 20 chars.
Directly give the question without start with any instruction.**
```

修改以后的效果比较好：

![image](https://github.com/user-attachments/assets/642715e4-ea9a-46cc-91af-74ae7a2b9d3e)

3、中文存在编码问题，将生成的dataset保存为json格式时，调用的是lama_index/finetuning/embeddings/common.py中的save_json方法，需要加上ensure_ascii=False
```
 def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4, ensure_ascii=False)
```

