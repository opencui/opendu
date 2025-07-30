# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
from dataclasses import field
from pydantic import BaseModel, Field
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
        RauConfig._instance = BcSsYniFullConfig(**jsonobj)

    @classmethod
    def get(cls):
        if RauConfig._instance is None:
            RauConfig._instance = BcSsYniFullConfig()
        return RauConfig._instance




class BcSsYniFullConfig(BaseModel):
    embedding_device: str = DEVICE
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"


    # We might not want to touch this, without rerun find_k
    desc_retrieve_topk: int = 8
    exemplar_retrieve_topk: int = 8
    exemplar_retrieve_arity: int = 8

    skill_arity: int = 1
    llm_device: str = DEVICE


    # Append input or output suffix, we get the actual prompt template.
    # id -> intent identification, bc -> binary class (could be multi class).
    # sf -> slot filling, ss -> single slot (could be frame or multi slots).
    # yni-bethere -> boolean gate, yes/no/irrelevant
    # the last part is to identify prompt template.
    prompt: dict = Field(default_factory=lambda: {"id_bc": "id_bc_literal", "yni": "yni_default"})

    # All task should share the same base model
    base_model: str = "Qwen/Qwen3-4B"
    eval_mode: bool = True

    # We will append instance.desc/instance.exemplar to this.
    generator: str = "BcSsYniGenerator"

    # When we use one lora for each task, which we should not.
    #id_bc_model: str = "OpenCUI/IdBc-Qwen2.5-7B-Instruct-0.1"
    #sf_ss_model: str = "OpenCUI/SfSs-Qwen2.5-7B-Instruct-0.1"
    # yni_model: str = "OpenCUI/Yni-Qwen2.5-7B-Instruct-0.1"
    converter_debug: bool = True
