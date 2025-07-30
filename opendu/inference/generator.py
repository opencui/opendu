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
from lmformatenforcer import Enforcer, AllowedTokensGuidance


# The modes that we will support.
# GeneratorType = Enum("Generator", ["FftGenerator", "LoraGenerator"])
GenerateMode = Enum("GenerateMode", ["desc", "exemplar", "extractive", "nli"])

class OutputExpectation(BaseModel):
    temperature: Float = Field(default=0.0, description="Temperature for the decoding process.")
    top_p: Float = Field(default=0.9, description="Top-p, e.g., 'The answer is {answer}'.")
    top_k: int = Field(default=50, description="Top-k, e.g., 'The answer is {answer}'.")
    max_new_tokens: int = Field(default=256, description="Maximum number of new tokens to generate.")
    repetition_penalty: Float = Field(default=1.0, description="Repetition penalty for the decoding process.")
    choices: list[str] = Field(default=[], description="List of expected outputs from the model.")

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
    def generate(self, input_texts: list[str], expectation: OutputExpectation = OutputExpectation):
        pass

    def process_return(self, outputs: list[str], input_texts: list[str]):
        return outputs


class FftVllmGenerator(Generator, ABC):
    def __init__(self, model: str):
        self.model = LLM(model=model, enable_prefix_caching=True)

    def generate(self, input_texts: list[str], expectation: OutputExpectation = None):
        samplingParams = SamplingParams(
            temperature=expectation.temperature if expectation else 0.0,
            top_p=expectation.top_p if expectation else 0.9,
            top_k=expectation.top_k if expectation else 50,
            max_tokens=expectation.max_new_tokens if expectation else 256,
            repetition_penalty=expectation.repetition_penalty if expectation else 1.0,
        )

        if (len(expectation.choices) > 0):
            samplingParams = SamplingParams(
                **samplingParams.model_dump(),
                guided_choice=["Yes", "No"],
                guided_decoding_backend="lm-format-enforcer"
            )

        outputs = self.model.generate(input_texts, sampling_params=samplingParams)
        return self.process_return(outputs, input_texts)




if __name__ == "__main__":

    llm = FftVllmGenerator(model="Qwen/Qwen3-4B")
    prompts = ["System: Summarize in JSON. Dialog: User: Help with order. Assistant: What's the issue?"]

    output = llm.generate(prompts)

    print(output)