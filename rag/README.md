# RAG

Here we will create a llama 3 based RAG application to do context based question answering, where the context is from the content of a webpage.

## Setup

In addition to setting up the Python environment with either conda, pip, or docker; you will also need to download the underlying model. For us this will be llama 3 and can be done with:

```bash
python download_model.py --model_name llama3
```

### Conda
```bash
conda create -n llama-rag python==3.10
conda activate llama-rag
pip install -r requirements.txt
```

### Docker