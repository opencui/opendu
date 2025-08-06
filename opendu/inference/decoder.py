# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
from abc import ABC, abstractmethod
from typing import Union
from opendu.core.config import GeneratorType, RauConfig
from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field
from vllm.sampling_params import GuidedDecodingParams


# This is expectation for the output of the generator.
# different implementation may need to translate this into different format.
# For example, the vllm generator will use this to set the sampling parameters.
# The vllm generator will use this to set the sampling parameters.
class OutputExpectation(BaseModel):
    temperature: float = Field(
        default=0.0, description="Temperature for the decoding process."
    )
    top_p: float = Field(
        default=0.9, description="Top-p, e.g., 'The answer is {answer}'."
    )
    top_k: int = Field(default=50, description="Top-k, e.g., 'The answer is {answer}'.")
    repetition_penalty: float = Field(
        default=1.0, description="Repetition penalty for the decoding process."
    )
    choices: list[str] = Field(
        default_factory=list, description="List of expected outputs from the model."
    )
    json_schema: dict = Field(
        default_factory=dict, description="JSON schema for the expected output."
    )


#
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
# This is singlton instance. (Later we can separate the llm into a different process)
#
class Decoder(ABC):
    generator = None

    @staticmethod
    def get():
        if (
            RauConfig.get().generator == GeneratorType.FftGenerator
            and Decoder.generator is None
        ):
            Decoder.generator = FftVllmGenerator(RauConfig.get().base_model)
        return Decoder.generator

    @abstractmethod
    def generate(self, input_texts: list[str], expectation: OutputExpectation):
        pass

    def process_return(self, outputs: list[str], input_texts: list[str]):
        return outputs


# The caller need to set the output git diexpectation.
class FftVllmGenerator(Decoder):
    # qwen3 has context length of 32768, for dialog understanding, we should not need that long.
    def __init__(self, model: str):
        self.model = LLM(
            model=model,
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            max_model_len=16384,
            gpu_memory_utilization=0.8
        )

    def generate(
        self,
        prompts: list[str],
        expectation: Union[
            OutputExpectation, list[OutputExpectation]
        ] = OutputExpectation(),
    ):
        # Check if expectation is a single instance or a list
        if isinstance(expectation, OutputExpectation):
            # Single expectation - apply to all inputs
            sampling_kwargs = {
                "temperature": expectation.temperature,
                "top_p": expectation.top_p,
                "top_k": expectation.top_k,
                "repetition_penalty": expectation.repetition_penalty,
            }

            if expectation.choices:
                sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                    choice=expectation.choices
                )
            if expectation.json_schema:
                sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                    json=expectation.json_schema
                )

            samplingParams = SamplingParams(**sampling_kwargs)
            outputs = self.model.generate(prompts, sampling_params=samplingParams)

        else:
            # List of expectations - one per input
            if len(expectation) != len(prompts):
                raise ValueError(
                    f"Number of expectations ({len(expectation)}) must match number of input texts ({len(prompts)})"
                )

            # Create a SamplingParams for each expectation
            sampling_params_list = []
            for exp in expectation:
                sampling_kwargs = {
                    "temperature": exp.temperature,
                    "top_p": exp.top_p,
                    "top_k": exp.top_k,
                    "repetition_penalty": exp.repetition_penalty,
                }
                if exp.choices:
                    sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                        choice=exp.choices
                    )
                if exp.json_schema:
                    sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                        json=exp.json_schema
                    )
                sampling_params_list.append(SamplingParams(**sampling_kwargs))

            outputs = self.model.generate(prompts, sampling_params=sampling_params_list)

        return self.process_return(outputs, prompts)


if __name__ == "__main__":
    llm = FftVllmGenerator(model="Qwen/Qwen3-4B")
    prompts = [
        "System: Summarize in JSON. Dialog: User: Help with order. Assistant: What's the issue?"
    ]

    output = llm.generate(prompts)

    print(output)
