# This is used for configure the project.

class LugConfig:
    embedding_device = "cpu"
    embedding_model = 'BAAI/llm-embedder'
    embedding_desc_prompt = "tool"
    embedding_exemplar_prompt = "irda"
    retriever_mode = "embedding"
    desc_retrieve_topk = 8
    exemplar_retrieve_topk = 32
    llm_device = "cpu"
