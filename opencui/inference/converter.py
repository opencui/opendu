import json
import re
from abc import ABC, abstractmethod
from enum import Enum

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from opencui.core.annotation import (CamelToSnake, DialogExpectation, EntityMetas, Exemplar, FrameValue, ListRecognizer,
                                     OwnerMode)
from opencui.core.config import LugConfig
from opencui.core.prompt import (BinarySkillPrompts, ExtractiveSlotPrompts, MulticlassSkillPrompts,
                                 NliPrompts, DescriptionPrompts, ExemplarPrompts)
from opencui.core.retriever import (ContextRetriever, load_context_retrievers)
from opencui.inference.schema_parser import load_all_from_directory


# The modes that we will support.
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
    @abstractmethod
    def generate(self, mode: GenerateMode, input_texts: list[str]):
        pass


# This should be desc/exemplar based.
class LoraGenerator(Generator, ABC):
    def __init__(self):
        parts = LugConfig.skill_model.split("/")

        desc_model = f"{parts[0]}/desc-{parts[1]}"
        exemplar_model = f"{parts[0]}/exemplar-{parts[1]}"

        skill_config = PeftConfig.from_pretrained(desc_model)

        model_path = skill_config.base_model_name_or_path

        # Is this the right place to clean cache.
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map=LugConfig.llm_device,
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
            LugConfig.extractive_slot_model, adapter_name=GenerateMode.extractive.name)

        if LugConfig.nli_model != "":
            self.lora_model.load_adapter(LugConfig.nli_model, adapter_name=GenerateMode.nli.name)

        # Move to device
        self.lora_model.to(LugConfig.llm_device)

    def generate(self, mode: GenerateMode, input_texts: list[str]):
        self.lora_model.set_adapter(mode.name)
        peft_encoding = self.tokenizer(
            input_texts, padding=True, return_tensors="pt"
        ).to(LugConfig.llm_device)

        with torch.no_grad():
            peft_outputs = self.lora_model.generate(
                input_ids=peft_encoding.input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=32,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=peft_encoding.attention_mask,
                    do_sample=False,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                ),
            )

        outputs = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)
        return [
            output[len(input_texts[index]):] for index, output in enumerate(outputs)
        ]


# Full finetuned generator
class FftGenerator(Generator, ABC):
    def __init__(self):
        # Is this the right place to clean cache.
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            LugConfig.model,
            return_dict=True,
            device_map=LugConfig.llm_device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(LugConfig.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Move to device
        self.model.to(LugConfig.llm_device)

    def generate(self, mode: GenerateMode, input_texts: list[str]):
        encoding = self.tokenizer(
            input_texts, padding=True, return_tensors="pt"
        ).to(LugConfig.llm_device)

        with torch.no_grad():
            peft_outputs = self.model.generate(
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
        outputs = self.tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)
        return [
            output[len(input_texts[index]):] for index, output in enumerate(outputs)
        ]


GeneratorType = Enum("Generator", ["FftGenerator", "LoraGenerator"])


class SkillConverter(ABC):
    @abstractmethod
    def get_skill(self, text) -> list[str]:
        pass

    @abstractmethod
    def grade(self, text, owner):
        pass


def parse_json_from_string(text, default=None):
    try:
        return json.loads(text)
    except ValueError as e:
        return default


class ISkillConverter(SkillConverter, ABC):
    def __init__(self, retriever: ContextRetriever, generator):
        self.retrieve = retriever
        self.generator = generator
        self.desc_prompt = DescriptionPrompts[LugConfig.skill_prompt]
        self.example_prompt = ExemplarPrompts[LugConfig.skill_prompt]
        self.use_exemplar = False
        self.use_desc = True
        assert self.use_desc or self.use_exemplar

    def build_prompts_by_examples(self, text, nodes, to_snake):
        skill_prompts = []
        owners = []

        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        exemplars = [
            Exemplar(
                owner=to_snake.encode(node.metadata["owner"]),
                template=node.text,
                owner_mode=node.metadata["owner_mode"]
            )
            for node in nodes
        ]

        for exemplar in exemplars:
            input_dict = {"utterance": text, "template": exemplar.template}
            skill_prompts.append(self.example_prompt(input_dict))
            owners.append(exemplar.owner)

        return skill_prompts, owners

    def build_prompts_by_desc(self, text, skills, to_snake):
        skill_prompts = []
        owners = []

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        # for now, we process it once.
        for skill in skills:
            input_dict = {"utterance": text, "skill": skill}
            skill_prompts.append(self.desc_prompt(input_dict))
            owners.append(skill["name"])
        return skill_prompts, owners

    @staticmethod
    def parse_results(skill_prompts, owners, skill_outputs):
        if LugConfig.converter_debug:
            print(json.dumps(skill_prompts, indent=2))
            print(json.dumps(skill_outputs, indent=2))

        flags = [
            parse_json_from_string(raw_flag, None)
            for index, raw_flag in enumerate(skill_outputs)
        ]
        return [owners[index] for index, flag in enumerate(flags) if flag]

    def get_skill(self, text):
        to_snake = CamelToSnake()
        # nodes owner are always included in the
        skills, nodes = self.retrieve(text)

        if self.use_exemplar:
            skill_prompts, owners = self.build_prompts_by_examples(text, nodes, to_snake)
            skill_outputs = self.generator.generate(GenerateMode.exemplar, skill_prompts)
            functions = self.parse_results(skill_prompts, owners, skill_outputs)
            if len(functions) != 0:
                return functions

        if self.use_desc:
            skill_prompts, owners = self.build_prompts_by_desc(text, skills, to_snake)
            skill_outputs = self.generator.generate(GenerateMode.desc, skill_prompts)
            return self.parse_results(skill_prompts, owners, skill_outputs)

        return []

    @staticmethod
    def update(preds, truth, counts, skill_prompts, skill_outputs):
        pairs = zip(preds, truth)
        for index, pair in enumerate(pairs):
            if pair[0] != pair[1]:
                print(f"{skill_prompts[index]} {skill_outputs[index]}, not correct.")

        for pair in pairs:
            index = 2 if pair[0] else 0
            index += 1 if pair[1] else 0
            counts[index] += 1

    def grade(self, text, owner, owner_mode, counts):
        to_snake = CamelToSnake()
        # nodes owner are always included in the
        skills, nodes = self.retrieve(text)

        # for examplar
        skill_prompts, owners = self.build_prompts_by_examples(text, nodes, to_snake)
        skill_outputs = self.generator.generate(GenerateMode.exemplar, skill_prompts)
        preds = [
            parse_json_from_string(raw_flag, None)
            for index, raw_flag in enumerate(skill_outputs)
        ]
        truth = [owner == lowner and OwnerMode[owner_mode] == OwnerMode.normal for lowner in owners]
        assert len(preds) == len(truth)

        if "exemplar" in counts:
            self.update(preds, truth, counts["exemplar"], skill_prompts, skill_outputs)

        # for desc
        skill_prompts, owners = self.build_prompts_by_desc(text, skills, to_snake)
        skill_outputs = self.generator.generate(GenerateMode.desc, skill_prompts)
        preds = [
            parse_json_from_string(raw_flag, None)
            for index, raw_flag in enumerate(skill_outputs)
        ]
        truth = [owner == lowner and OwnerMode[owner_mode] == OwnerMode.normal for lowner in owners]
        assert len(preds) == len(truth)
        if "desc" in counts:
            self.update(preds, truth, counts["desc"], skill_prompts, skill_outputs)


class Converter:
    def __init__(
            self,
            retriever: ContextRetriever,
            entity_metas: EntityMetas = None,
            with_arguments=True,
    ):
        self.retrieve = retriever
        self.recognizer = None
        if entity_metas is not None:
            self.recognizer = ListRecognizer(entity_metas)

        self.generator = Converter.generator()
        self.slot_prompt = ExtractiveSlotPrompts[LugConfig.slot_prompt]
        self.nli_prompt = NliPrompts[LugConfig.nli_prompt]
        self.with_arguments = with_arguments
        self.bracket_match = re.compile(r"\[([^]]*)\]")
        self.skill_converter = None

        self.skill_converter = ISkillConverter(retriever, self.generator)
        self.nli_labels = {"entailment": True, "neutral": None, "contradiction": False}

    @staticmethod
    def generator():
        if GeneratorType[LugConfig.generator] == GeneratorType.FftGenerator:
            return FftGenerator()
        if GeneratorType[LugConfig.generator] == GeneratorType.LoraGenerator:
            return LoraGenerator()

    def understand(
            self, text: str, expectation: DialogExpectation = None
    ) -> FrameValue:
        # low level get skill.
        func_names = self.skill_converter.get_skill(text)

        if len(func_names) == 0:
            return None

        # For now, just return the first one.
        func_name = func_names[0]

        if not self.retrieve.module.has_module(func_name):
            print(f"{func_name} is not recognized.")
            return None

        if not self.with_arguments:
            return FrameValue(name=func_name, arguments={})

        # We assume the function_name is global unique for now. From UI perspective, I think
        module = self.retrieve.module.get_module(func_name)
        slot_labels_of_func = module.skills[func_name]["slots"]

        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slot_labels_of_func:
            values = []
            if self.recognizer is not None:
                values = self.recognizer.extract_values(slot, text)
            slot_input_dict = {"utterance": text, "values": values}
            slot_input_dict.update(module.slots[slot].to_dict())
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        if LugConfig.converter_debug:
            print(json.dumps(slot_prompts, indent=2))
        slot_outputs = self.generator.generate(GenerateMode.extractive, slot_prompts)

        if LugConfig.converter_debug:
            print(json.dumps(slot_outputs, indent=2))

        slot_values = [parse_json_from_string(seq) for seq in slot_outputs]
        slot_values = dict(zip(slot_labels_of_func, slot_values))
        slot_values = {
            key: value for key, value in slot_values.items() if value is not None
        }

        final_name = func_name
        if module.backward is not None:
            final_name = module.backward[func_name]

        return FrameValue(name=final_name, arguments=slot_values)

    # There are three different
    def decide(self, question, utterance, lang="en") -> bool:
        # For now, we ignore the language
        input_dict = {"premise": utterance, "hypothesis": f"{question}."}
        input_prompt = self.nli_prompt(input_dict)
        output = self.generator.for_nli(input_prompt)
        if LugConfig.converter_debug:
            print(f"{input_prompt} {output}")
        if output not in self.nli_labels:
            return None
        return self.nli_labels[output]

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented


def load_converter(module_path, index_path):
    # First load the schema info.
    module_schema, examplers, recognizers = load_all_from_directory(module_path)
    # Then load the retriever by pointing to index directory
    context_retriever = load_context_retrievers(
        {module_path: module_schema}, index_path
    )
    # Finally build the converter.
    return Converter(context_retriever)
