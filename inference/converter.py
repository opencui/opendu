import re

from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from core.config import LugConfig
from core.annotation import FrameValue, Exemplar, DialogExpectation, CamelToSnake, FrameSchema
from core.prompt import SkillPrompts, SlotPrompts
from core.retriever import ContextRetriever


#
# Here are the work that client have to do.
# For now, assume single module. We can worry about multiple module down the road.
# 1. use description and exemplars to find built the prompt for the skill prompt.
# 2. use skill and recognizer to build prompt for slot.
# 3. stitch together the result.
#
def generate(peft_model, peft_tokenizer, input_text):
    peft_encoding = peft_tokenizer(input_text, return_tensors="pt").to("cuda:0")
    peft_outputs = peft_model.generate(
        input_ids=peft_encoding.input_ids,
        generation_config=GenerationConfig(
            max_new_tokens=256,
            pad_token_id=peft_tokenizer.eos_token_id,
            eos_token_id=peft_tokenizer.eos_token_id,
            attention_mask=peft_encoding.attention_mask,
            temperature=0.1,
            top_p=0.1,
            repetition_penalty=1.2,
            num_return_sequences=1, ))
    peft_text_outputs = peft_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)
    return peft_text_outputs


class Converter:
    def __init__(self, retriever: ContextRetriever, recognizers=None, with_arguments=False):
        self.retrieve = retriever
        self.recognizers = recognizers

        skill_config = PeftConfig.from_pretrained(LugConfig.skill_model)

        model_path = skill_config.base_model_name_or_path

        base_model = AutoModelForCausalLM.from_pretrained(
            skill_config.base_model_name_or_path,
            return_dict=True,
            device_map="auto",
            trust_remote_code=True,
        )
        print(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.skill_model = PeftModel.from_pretrained(base_model, LugConfig.skill_model)

        extractive_slot_config = PeftConfig.from_pretrained(LugConfig.extractive_slot_model)
        if model_path != extractive_slot_config.base_model_name_or_path:
            raise RuntimeError("Only support same base model")

        self.extractive_slot_model = PeftModel.from_pretrained(base_model, LugConfig.extractive_slot_model)

        self.skill_prompt = SkillPrompts[LugConfig.skill_prompt]
        self.slot_prompt = SlotPrompts[LugConfig.slot_prompt]

        self.with_arguments = with_arguments

        self.bracket_match = re.compile(r'\[([^]]*)\]')

    def understand(self, text: str, expectation: DialogExpectation = None) -> FrameValue:
        to_snake = CamelToSnake()

        # first we figure out what is the
        skills, nodes = self.retrieve(text)
        exemplars = [Exemplar(owner=to_snake.encode(node.metadata["owner"]), template=node.text) for node in nodes]

        for skill in skills:
            skill["name"] = to_snake.encode(skill["name"])

        skill_input_dict = {"utterance": text.strip(), "examples": exemplars, "skills": skills}
        skill_prompt = self.skill_prompt(skill_input_dict)

        print(skill_prompt)

        skill_outputs = generate(self.skill_model, self.tokenizer, skill_prompt)

        print(f"Skill model generated: {skill_outputs}.")

        func_match = self.bracket_match.search(skill_outputs[len(skill_prompt):])
        if not func_match:
            return None

        func_name = func_match.group(1).strip()

        if not self.with_arguments:
            return FrameValue(name=func_name, arguments={})

        # We assume the function_name is global unique for now. From UI perspective, I think
        module = self.retrieve.module.get_module(func_name)
        slots_of_func = module.skills[func_name]
        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slots_of_func:
            slot_input_dict = {"utterance": text, "slot": slot}
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        slot_outputs = generate(self.extractive_slot_model, self.tokenizer, slot_prompts)

        print(f"Slot model generated: {slot_outputs}")

        for seq in slot_outputs:
            print(f"Result: {seq['generated_text']}")

        slot_jsons = [json.load(seq['generated_text']) for seq in slot_outputs]
        slot_values = {key: value for slot_obj in slot_jsons for key, value in slot_obj.items()}
        return FrameValue(name=func_name, arguments=slot_values)

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented
