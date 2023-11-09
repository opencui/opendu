# This is used for configure the project.

class LugConfig:
    embedding_device = "cpu"
    embedding_model = 'BAAI/bge-base-en-v1.5'
    embedding_desc_model = ""
    embedding_desc_prompt = "baai_desc"
    embedding_exemplar_prompt = "baai_exemplar"

    desc_retriever_mode = "embedding"
    exemplar_retriever_mode = "OR"
    desc_retrieve_topk = 4
    exemplar_retrieve_topk = 16
    llm_device = "cpu"
