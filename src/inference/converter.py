import re
from abc import abstractmethod, ABC

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from core.config import LugConfig
from core.annotation import FrameValue, Exemplar, DialogExpectation, CamelToSnake
from core.prompt import SkillPrompts, SlotPrompts, ClassificationPrompts
from core.retriever import ContextRetriever, load_context_retrievers
from inference.schema_parser import load_all_from_directory


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
    def for_skill(self, input_texts):
        pass

    @abstractmethod
    def for_extractive_slot(self, input_texts):
        pass

    @abstractmethod
    def for_abstractive_slot(self, input_texts):
        pass


class LocalGenerator(Generator, ABC):
    def __init__(self):
        skill_config = PeftConfig.from_pretrained(LugConfig.skill_model)

        model_path = skill_config.base_model_name_or_path

        skill_base_model = AutoModelForCausalLM.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.skill_model = PeftModel.from_pretrained(skill_base_model, LugConfig.skill_model)

        extractive_slot_config = PeftConfig.from_pretrained(LugConfig.extractive_slot_model)
        if model_path != extractive_slot_config.base_model_name_or_path:
            raise RuntimeError("Only support same base model")

        # For now, we load the base model multiple times.
        slot_base_model = AutoModelForCausalLM.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.extractive_slot_model = PeftModel.from_pretrained(slot_base_model, LugConfig.extractive_slot_model)

    @classmethod
    def generate(cls, peft_model, peft_tokenizer, input_text):
        peft_encoding = peft_tokenizer(input_text, padding=True, return_tensors="pt").to("cuda:0")
        peft_outputs = peft_model.generate(
            input_ids=peft_encoding.input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=128,
                pad_token_id=peft_tokenizer.eos_token_id,
                eos_token_id=peft_tokenizer.eos_token_id,
                attention_mask=peft_encoding.attention_mask,
                do_sample=False,
                repetition_penalty=1.2,
                num_return_sequences=1, ))

        return peft_tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)

    def for_skill(self, input_texts):
        outputs = LocalGenerator.generate(self.skill_model, self.tokenizer, input_texts)
        return [output[len(input_texts[index]):] for index, output in enumerate(outputs)]

    def for_abstractive_slot(self, input_texts):
        pass

    def for_extractive_slot(self, input_texts):
        outputs = LocalGenerator.generate(self.extractive_slot_model, self.tokenizer, input_texts)
        return [output[len(input_texts[index]):] for index, output in enumerate(outputs)]


class SkillConverter(ABC):
    @abstractmethod
    def get_skill(self, text) -> list[str]:
        pass


def parse_json_from_string(text, default=None):
    try:
        return json.loads(text)
    except ValueError as e:
        return default


class MSkillConverter(SkillConverter):
    def __init__(self, retriever: ContextRetriever, generator=LocalGenerator()):
        self.retrieve = retriever
        self.generator = generator
        self.skill_prompt = SkillPrompts[LugConfig.skill_full_prompt]

    def get_skill(self, text):
        to_snake = CamelToSnake()

        # first we figure out what is the
        skills, nodes = self.retrieve(text)
        exemplars = [Exemplar(owner=to_snake.encode(node.metadata["owner"]), template=node.text) for node in nodes]

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        skill_input_dict = {"utterance": text.strip(), "examples": exemplars, "skills": skills}
        skill_prompt = self.skill_prompt([skill_input_dict])

        skill_outputs = self.generator.for_skill(skill_prompt)

        if LugConfig.converter_debug:
            print(skill_prompt)
            print(skill_outputs)

        func_name = parse_json_from_string(skill_outputs[0])
        if LugConfig.converter_debug:
            print(f"{skill_outputs} is converted to {func_name}, valid: {self.retrieve.module.has_module(func_name)}")

        return [func_name]


class BSkillConverter(SkillConverter):
    def __init__(self, retriever: ContextRetriever, generator=LocalGenerator()):
        self.retrieve = retriever
        self.generator = generator
        self.prompt = ClassificationPrompts[LugConfig.skill_full_prompt]

    def get_skill(self, text):
        to_snake = CamelToSnake()

        # nodes owner are always included in the
        skills, nodes = self.retrieve(text)
        exemplars = [Exemplar(owner=to_snake.encode(node.metadata["owner"]), template=node.text) for node in nodes]

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        skill_map = {skill["name"]: skill for skill in skills}

        skill_prompts = []
        owners = []
        processed = set()
        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        for o_exemplar in exemplars:
            target = o_exemplar.owner
            # Try not to have more than two examples.
            exemplar_dicts = [
                {"template": exemplar.template, "target": target, "decision": target == exemplar.owner}
                for exemplar in exemplars]

            input_dict = {"utterance": text, "examples": exemplar_dicts, "skill": skill_map[target]}
            skill_prompts.append(self.prompt(input_dict))
            owners.append(target)

            processed.add(target)

        for skill in skills:
            if skill["name"] in processed:
                continue
            input_dict = {"utterance": text, "examples": [], "skill": skill}
            skill_prompts.append(self.prompt(input_dict))
            owners.append[skill["name"]]

        skill_outputs = self.generator.for_skill(skill_prompts)

        if LugConfig.converter_debug:
            print(skill_prompts)
            print(skill_outputs)

        flags = [parse_json_from_string(raw_flag, False) for index, raw_flag in enumerate(skill_outputs)]

        func_names = [owners[index] for index, flag in enumerate(flags) if flag]

        return func_names


class Converter:
    def __init__(self, retriever: ContextRetriever, recognizers=None, generator=LocalGenerator(), with_arguments=True):
        self.retrieve = retriever
        self.recognizers = recognizers
        self.generator = generator
        self.slot_prompt = SlotPrompts[LugConfig.extractive_slot_prompt]
        self.with_arguments = with_arguments
        self.bracket_match = re.compile(r'\[([^]]*)\]')
        self.skill_converter = None
        if LugConfig.classification_prompt:
            self.skill_converter = BSkillConverter(retriever, generator)
        else:
            self.skill_converter = MSkillConverter(retriever, generator)

    def understand(self, text: str, expectation: DialogExpectation = None) -> FrameValue:
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
        print(slot_labels_of_func)
        print(module.slots)
        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slot_labels_of_func:
            slot_input_dict = {"utterance": text}
            slot_input_dict.update(module.slots[slot].to_dict())
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        slot_outputs = self.generator.for_extractive_slot(slot_prompts)

        slot_values = [parse_json_from_string(seq) for seq in slot_outputs]
        slot_values = dict(zip(slot_labels_of_func, slot_values))
        slot_values = {key: value for key, value in slot_values.items() if value is not None}

        final_name = func_name
        if module.backward is not None:
            final_name = module.backward[func_name]

        return FrameValue(name=final_name, arguments=slot_values)

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented


def load_converter(module_path, index_path):
    # First load the schema info.
    module_schema, examplers, recognizers = load_all_from_directory(module_path)
    # Then load the retriever by pointing to index directory
    context_retriever = load_context_retrievers({module_path: module_schema}, index_path)
    # Finally build the converter.
    return Converter(context_retriever)