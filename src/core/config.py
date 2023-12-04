# This is used for configure the project during the index and training.
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
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
    exemplar_retrieve_arity = 1
    exemplar_combined_topk = 4

    # multiclass, classification, simple
    skill_mode = "simple"
    skill_arity = 1
    llm_device = "cpu"
    skill_full_prompt = "default"
    extractive_slot_prompt = "default"

    skill_model = "OpenCUI/sgd-skill-tinyllama-2t-1.0"
    extractive_slot_model = "OpenCUI/sgd-slot-tinyllama-2t-1.0"
    abstractive_slot_model = ""

    converter_debug = True