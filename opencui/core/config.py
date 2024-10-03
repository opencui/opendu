# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.

from pydantic import BaseModel
from enum import Enum


# This is used for configure the project during the index and training.
class ModelType(Enum):
    t5 = 1
    gpt = 2
    llama = 3

    # This normalizes type to t5/gpt/bert (potentially)
    @staticmethod
    def normalize(model_in_str):
        if ModelType[model_in_str] == ModelType.llama:
            return ModelType.gpt
        return ModelType[model_in_str]


DEVICE="cuda:0"


class RauConfig:
    _instance = None

    @classmethod
    def init(cls, jsonobj):
        RauConfig._instance = InferenceConfig(**jsonobj)

    @classmethod
    def get(cls):
        if RauConfig._instance is None:
            RauConfig._instance = InferenceConfig()
        return RauConfig._instance


class InferenceConfig(BaseModel):
    embedding_device: str = DEVICE
    #embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_model: str = "dunzhang/stella_en_400M_v5"


    # We might not want to touch this, without rerun find_k
    desc_retrieve_topk: int = 8
    exemplar_retrieve_topk: int = 8
    exemplar_retrieve_arity: int = 8

    skill_arity: int = 1
    llm_device: str = DEVICE

    # this is used
    skill_modes: list = ["both"]

    skill_prompt: str = "skill-knn-structural"
    skill_desc_prompt: str = "skill-desc-structural"
    slot_prompt: str = "slot-qa-structural"
    yni_prompt: str = "yni-default"
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
