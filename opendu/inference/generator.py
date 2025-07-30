# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
from abc import ABC, abstractmethod
from enum import Enum

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModelForSeq2SeqLM, AutoConfig

from opendu import ModelType
from opendu.core.config import RauConfig
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# The modes that we will support.
# GeneratorType = Enum("Generator", ["FftGenerator", "LoraGenerator"])
# GenerateMode = Enum("GenerateMode", ["desc", "exemplar", "extractive", "nli"])

# In case you are curious about decoding: https://huggingface.co/blog/how-to-generate
# We are not interested in the variance, so we do not do sampling not beam search.
#
# Here are the work that client have to do.
# For now, assume single module. We can worry about multiple module down the road.
# 1. use description and exemplars to find built the prompt for the skill prompt.
# 2. use skill and recognizer to build prompt for slot.
# 3. stitch together the result.
# Generator is responsible for low level things, we will have two different implementation
# local/s-lora. Converter is built on top of generator.
class Generator(ABC):
    @abstractmethod
    def generate(self, input_texts: list[str]):
        pass

    def process_return(self, outputs: list[str], input_texts: list[str]):
        return outputs


class FftVllmGenerator(Generator, ABC):
    def __init__(self, model: str):
        self.model = LLM(model=model, enable_prefix_caching=True)

    def generate(self, input_texts: list[str]):
        outputs = self.model.generate(input_texts)
        return self.process_return(outputs, input_texts)




if __name__ == "__main__":

    llm = FftVllmGenerator(model="Qwen/Qwen3-4B")
    prompts = ["System: Summarize in JSON. Dialog: User: Help with order. Assistant: What's the issue?"]

    output = llm.generate(prompts)

    print(output)