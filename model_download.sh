pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download BAAI/bge-m3 --local-dir model/bge-m3/