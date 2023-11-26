import re
from abc import abstractmethod, ABC

from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from core.config import LugConfig
from core.annotation import FrameValue, Exemplar, DialogExpectation, CamelToSnake, FrameSchema
from core.prompt import SkillPrompts, SlotPrompts
from core.retriever import ContextRetriever


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
    def for_skill(self, input_text):
        pass

    @abstractmethod
    def for_extractive_slot(self, input_texts):
        pass

    @abstractmethod
    def for_abstractive_slot(self, input_text):
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
        )
        self.extractive_slot_model = PeftModel.from_pretrained(slot_base_model, LugConfig.extractive_slot_model)

    @classmethod
    def generate(cls, peft_model, peft_tokenizer, input_text, batch):
        peft_encoding = peft_tokenizer(input_text, padding=True, return_tensors="pt").to("cuda:0")
        peft_outputs = peft_model.generate(
            input_ids=peft_encoding.input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=128,
                pad_token_id=peft_tokenizer.eos_token_id,
                eos_token_id=peft_tokenizer.eos_token_id,
                attention_mask=peft_encoding.attention_mask,
                repetition_penalty=1.2,
                num_return_sequences=1, ))
        if batch:
            return peft_tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)
        else:
            return peft_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)

    def for_skill(self, input_text):
        outputs = LocalGenerator.generate(self.skill_model, self.tokenizer, input_text, False)
        return outputs[len(input_text):]

    def for_abstractive_slot(self, input_text):
        pass

    def for_extractive_slot(self, input_texts):
        outputs = LocalGenerator.generate(self.extractive_slot_model, self.tokenizer, input_texts, True)
        return [output[len(input_texts[index]):] for index, output in enumerate(outputs)]


class Converter:
    def __init__(self, retriever: ContextRetriever, recognizers=None, generator=LocalGenerator(), with_arguments=True):
        self.retrieve = retriever
        self.recognizers = recognizers
        self.generator = generator
        self.skill_prompt = SkillPrompts[LugConfig.skill_prompt]
        self.slot_prompt = SlotPrompts[LugConfig.slot_prompt]
        self.with_arguments = with_arguments
        self.bracket_match = re.compile(r'\[([^]]*)\]')

    @staticmethod
    def parse_json_from_string(text):
        try:
            return json.loads(text)
        except ValueError as e:
            return None

    def understand(self, text: str, expectation: DialogExpectation = None) -> FrameValue:
        to_snake = CamelToSnake()

        # first we figure out what is the
        skills, nodes = self.retrieve(text)
        exemplars = [Exemplar(owner=to_snake.encode(node.metadata["owner"]), template=node.text) for node in nodes]

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        skill_input_dict = {"utterance": text.strip(), "examples": exemplars, "skills": skills}
        skill_prompt = self.skill_prompt(skill_input_dict)

        skill_outputs = self.generator.for_skill(skill_prompt)

        func_match = self.bracket_match.search(skill_outputs)
        if not func_match:
            return None

        func_name = func_match.group(1).strip()

        if not self.with_arguments:
            return FrameValue(name=func_name, arguments={})

        # We assume the function_name is global unique for now. From UI perspective, I think
        module = self.retrieve.module.get_module(func_name)
        slot_labels_of_func = module.skills[func_name]["slots"]
        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slot_labels_of_func:
            slot_input_dict = {"utterance": text}
            slot_input_dict.update(module.slots[slot])
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        slot_outputs = self.generator.for_extractive_slot(slot_prompts)

        slot_values = [self.parse_json_from_string(seq) for seq in slot_outputs]
        slot_values = dict(zip(slot_labels_of_func, slot_values))
        slot_values = {key: value for key, value in slot_values.items() if value is not None}
        return FrameValue(name=func_name, arguments=slot_values)

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented
