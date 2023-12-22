# This is used for configure the project during the index and training.
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LugConfig:
    embedding_device = "cpu"
    embedding_model = "BAAI/bge-base-en-v1.5"
    embedding_desc_model = ""
    embedding_desc_prompt = "baai_desc"
    embedding_exemplar_prompt = "baai_exemplar"

    desc_retrieve_topk = 6
    exemplar_retrieve_topk = 32
    exemplar_retrieve_arity = 6

    skill_arity = 1
    llm_device = "cuda:0"

    skill_prompt = "struct-token1"
    slot_prompt = "default"
    nli_prompt = "default"
    bool_prompt = "default"

    # We will append instance.desc/instance.exemplar to this.
    generator = "FftGenerator"
    model="./output/tinyllama2.5t-st/checkpoint-5580/"

    skill_model = "OpenCUI/skill-tinyllama-0.1"
    extractive_slot_model = "OpenCUI/extractive-tinyllama2.5t-1.0"
    nli_model = ""
    converter_debug = True
