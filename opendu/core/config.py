# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
from dataclasses import field
from pydantic import BaseModel, Field
from enum import Enum


# We only work with well-defined task.
class Task(Enum):
    IdBc = "id_bc",
    SfSs = "sf_ss",
    Yni = "yni"


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


GeneratorType = Enum("Generator", ["FftGenerator", "LoraGenerator"])

class RauConfig:
    _instance = None
    model_caching_path = "/data/models/"

    @classmethod
    def init(cls, jsonobj):
        RauConfig._instance = BcSsYniFullConfig(**jsonobj)

    @classmethod
    def get(cls):
        if RauConfig._instance is None:
            RauConfig._instance = BcSsYniFullConfig()
        return RauConfig._instance

    @classmethod
    def get_model_path(cls, model_name):
        return f"{RauConfig.model_caching_path}/{model_name.split('/')[-1]}"

    @classmethod
    def get_embedding_model(cls):
        return RauConfig.get_model_path(RauConfig.get().embedding_model)

    @classmethod
    def get_generator_model(cls):
        return RauConfig.get_model_path(RauConfig.get().base_model)



# This is configueration for treating the du as 3 tasks.
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
    # yni -> boolean gate, yes/no/irrelevant
    # the last part is to identify prompt template.
    prompt: dict[Task, str] = Field(default_factory=lambda: {Task.IdBc: "id_bc_literal", Task.SfSs: "sf_se_default", Task.Yni: "yni_default"})


    # All task should share the same base model
    generator: GeneratorType = GeneratorType.FftGenerator
    base_model: str = "Qwen/Qwen3-4B"
    eval_mode: bool = True

    # We will append instance.desc/instance.exemplar to this.

    # When we use one lora for each task, which we should not.
    #id_bc_model: str = "OpenCUI/IdBc-Qwen2.5-7B-Instruct-0.1"
    #sf_ss_model: str = "OpenCUI/SfSs-Qwen2.5-7B-Instruct-0.1"
    # yni_model: str = "OpenCUI/Yni-Qwen2.5-7B-Instruct-0.1"
    debug: bool = False
    converter_debug: bool = False
    id_debug: bool = False
    sf_debug: bool = True
    yni_debug: bool = False
