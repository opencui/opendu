# This is used for configure the project.

class Config:
    embedding_device = "cpu"
    embedding_model = 'BAAI/llm-embedder'
    embedding_desc_prompt = "tools"
    embedding_exemplar_prompt = "irda"
    retriever_mode = "embedding"
    desc_retrieve_topk = 8
    exemplar_retrieve_topk = 16
    llm_device = "cpu"
