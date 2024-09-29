pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download BAAI/bge-large-zh-v1.5 --local-dir model/bge-large-zh-v1.5

#BAAI/bge-large-zh-v1.5
#BAAI/bge-m3