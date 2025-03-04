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


# The modes that we will support.
GeneratorType = Enum("Generator", ["FftGenerator", "LoraGenerator"])
GenerateMode = Enum("GenerateMode", ["desc", "exemplar", "extractive", "nli"])

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
    generator = None

    @staticmethod
    def build():
        if GeneratorType[RauConfig.get().generator] == GeneratorType.FftGenerator and Generator.generator is None:
            Generator.generator = FftGenerator()
        if GeneratorType[RauConfig.get().generator] == GeneratorType.LoraGenerator and Generator.generator is None:
            Generator.generator = LoraGenerator()
        return Generator.generator

    @staticmethod
    def get_model_type(model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        return config.model_type

    @staticmethod
    def from_pretrained(*args, **kwargs):
        config = AutoConfig.from_pretrained(args[0])
        # Check the model type
        print(f"loading model: {args[0]} with type: {config.model_type}")
        if ModelType.normalize(config.model_type) == ModelType.t5:
            return AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        if ModelType.normalize(config.model_type) == ModelType.gpt:
            return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

    @abstractmethod
    def generate(self, input_texts: list[str], mode: GenerateMode = None):
        pass

    def process_return(self, outputs: list[str], input_texts: list[str]):
        if ModelType.normalize(self.model_type) == ModelType.t5:
            return outputs
        if ModelType.normalize(self.model_type) == ModelType.gpt:
            return [output[len(input_texts[index]):] for index, output in enumerate(outputs)]


# This should be desc/exemplar based.
class LoraGenerator(Generator, ABC):
    def __init__(self):
        parts = RauConfig.get().skill_model.split("/")

        desc_model = f"{parts[0]}/desc-{parts[1]}"
        exemplar_model = f"{parts[0]}/exemplar-{parts[1]}"

        skill_config = PeftConfig.from_pretrained(desc_model)

        model_path = skill_config.base_model_name_or_path

        # Is this the right place to clean cache.
        torch.cuda.empty_cache()

        base_model = Generator.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map=RauConfig.get().llm_device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.models = {}

        self.lora_model = PeftModel.from_pretrained(
            base_model, desc_model, adapter_name=GenerateMode.desc.name)

        self.lora_model.load_adapter(
            exemplar_model, adapter_name=GenerateMode.exemplar.name)

        self.lora_model.load_adapter(
            RauConfig.get().extractive_slot_model, adapter_name=GenerateMode.extractive.name)

        if RauConfig.get().nli_model != "":
            self.lora_model.load_adapter(RauConfig.get().nli_model, adapter_name=GenerateMode.nli.name)

        # Move to device
        self.lora_model.to(RauConfig.get().llm_device)
        self.lora_model.eval()

    def generate(self, input_texts: list[str], mode: GenerateMode):
        self.lora_model.set_adapter(mode.name)
        encoding = self.tokenizer(
            input_texts, padding=True, return_tensors="pt"
        ).to(RauConfig.get().llm_device)

        with torch.no_grad():
            peft_outputs = self.lora_model.generate(
                input_ids=encoding.input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=encoding.attention_mask,
                    do_sample=False,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                ),
            )

        results = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)
        return Generator.process_return(results, input_texts)


#
# This is the triple task version, with binary class for intent detection, single slot for slot filling
# and yes/no/irrelevant for boolean gate, multi value and explicit confirmation.
#
class BcSsYniLoraGenerator(Generator, ABC):
    def __init__(self):
        parts = RauConfig.get().skill_model.split("/")

        desc_model = f"{parts[0]}/desc-{parts[1]}"
        exemplar_model = f"{parts[0]}/exemplar-{parts[1]}"

        skill_config = PeftConfig.from_pretrained(desc_model)

        model_path = skill_config.base_model_name_or_path

        # Is this the right place to clean cache.
        torch.cuda.empty_cache()

        base_model = Generator.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map=RauConfig.get().llm_device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.models = {}

        self.lora_model = PeftModel.from_pretrained(
            base_model, desc_model, adapter_name=GenerateMode.desc.name)

        self.lora_model.load_adapter(
            exemplar_model, adapter_name=GenerateMode.exemplar.name)

        self.lora_model.load_adapter(
            RauConfig.get().extractive_slot_model, adapter_name=GenerateMode.extractive.name)

        if RauConfig.get().nli_model != "":
            self.lora_model.load_adapter(RauConfig.get().nli_model, adapter_name=GenerateMode.nli.name)

        # Move to device
        self.lora_model.to(RauConfig.get().llm_device)
        self.lora_model.eval()

    def generate(self, input_texts: list[str], mode: GenerateMode):
        self.lora_model.set_adapter(mode.name)
        encoding = self.tokenizer(
            input_texts, padding=True, return_tensors="pt"
        ).to(RauConfig.get().llm_device)

        with torch.no_grad():
            peft_outputs = self.lora_model.generate(
                input_ids=encoding.input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=encoding.attention_mask,
                    do_sample=False,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                ),
            )

        results = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)
        return Generator.process_return(results, input_texts)




# Full finetuned generator
class FftGenerator(Generator, ABC):
    def __init__(self):
        # Is this the right place to clean cache.
        torch.cuda.empty_cache()
        self.model = Generator.from_pretrained(
            RauConfig.get().model,
            return_dict=True,
            device_map=RauConfig.get().llm_device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model_type = Generator.get_model_type(RauConfig.get().model)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(RauConfig.get().model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Move to device
        self.model.to(RauConfig.get().llm_device)
        self.model.eval()

    def generate(self, input_texts: list[str], mode: GenerateMode):
        # The tokenizer can not handle empty list, so we safeguard that.
        if len(input_texts) == 0:
            return []

        encoding = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(RauConfig.get().llm_device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoding.input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=encoding.attention_mask,
                    do_sample=False,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                ),
            )
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return self.process_return(results, input_texts)

