# This is used for configure the project during the index and training.

from pydantic import BaseModel

DEVICE="cpu"

class LugConfig:
    _instance = None

    @classmethod
    def init(cls, jsonobj):
        LugConfig._instance = InferenceConfig(**jsonobj)

    @classmethod
    def get(cls):
        if LugConfig._instance is None:
            LugConfig._instance = InferenceConfig()
        return LugConfig._instance


class InferenceConfig(BaseModel):
    embedding_device: str = DEVICE
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    #embedding_model: str = "dunzhang/stella_en_400M_v5"


    # We might not want to touch this, without rerun find_k
    desc_retrieve_topk: int = 8
    exemplar_retrieve_topk: int = 8
    exemplar_retrieve_arity: int = 8

    skill_arity: int = 1
    llm_device: str = DEVICE

    skill_prompt: str = "structural"
    slot_prompt: str = "structural"
    yni_prompt: str = "default"
    bool_prompt: str = "plain"

    eval_mode: bool = True

    # We will append instance.desc/instance.exemplar to this.
    generator: str = "FftGenerator"
    model: str = "OpenCUI/dug-t5base-0.1"

    # When we use one lora for each task, which we should not.
    skill_model: str = ""
    extractive_slot_model: str = ""
    nli_model: str = ""
    converter_debug: bool = True
