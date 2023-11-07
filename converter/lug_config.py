# This is used for configure the project.

class LugConfig:
    embedding_device = "cpu"
    embedding_model = 'BAAI/bge-base-en-v1.5'
    embedding_desc_model = ""
    embedding_desc_prompt = "baai-desc"
    embedding_exemplar_prompt = "baai-exemplar"
    retriever_mode = "embedding"
    desc_retrieve_topk = 8
    exemplar_retrieve_topk = 16
    llm_device = "cpu"
