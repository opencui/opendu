import torch
import transformers
from transformers import AutoTokenizer
import json
from core.lug_config import LugConfig
from core.annotation import FrameValue, Exemplar, DialogExpectation
from core.prompt import SkillPrompts, SlotPrompts
from core.retriever import ContextRetriever


#
# Here are the work that client have to do.
# For now, assume single module. We can worry about multiple module down the road.
# 1. use description and exemplars to find built the prompt for the skill prompt.
# 2. use skill and recognizer to build prompt for slot.
# 3. stitch together the result.
#
class Converter:
    def __init__(self, retriever: ContextRetriever, recognizers=None):
        self.retrieve = retriever
        self.recognizers = recognizers

        model = LugConfig.inference_model

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            return_full_text=False,
        )

        self.skill_prompt = SkillPrompts[LugConfig.skill_prompt]
        self.slot_prompt = SlotPrompts[LugConfig.slot_prompt]

    def understand(self, text: str, expectation: DialogExpectation = None) -> FrameValue:
        # first we figure out what is the
        skills, nodes = self.retrieve(text)
        exemplars = [Exemplar(owner=node.metadata["owner"], template=node.text) for node in nodes]

        print(exemplars)

        skill_input_dict = {"utterance": text.strip(), "examples": exemplars, "skills": skills}
        skill_prompt = self.skill_prompt(skill_input_dict)

        print(skill_prompt)
    
        skill_outputs = self.pipeline(
            skill_prompt,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1,
            repetition_penalty=1.1,
            max_new_tokens=16,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=32000
        )

        print(f"There are {len(skill_outputs)} generated text.")
        for seq in skill_outputs:
            print(f"Result: {seq['generated_text']}")

        # We assume the function_name is global unique for now. From UI perspective, I think
        #
        func_name = skill_outputs[0]["generated_text"].strip()
        module = self.retrieve.module.get_module(func_name)
        slots_of_func = module.skills[func_name]
        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slots_of_func:
            slot_input_dict = {"utterance": text, "slot": slot}
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        slot_outputs = self.pipeline(
            skill_prompt,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1,
            repetition_penalty=1.1,
            max_new_tokens=128,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=32000
        )

        for seq in slot_outputs:
            print(f"Result: {seq['generated_text']}")

        slot_jsons = [json.load(seq['generated_text']) for seq in slot_outputs]
        slot_values = {key: value for slot_obj in slot_jsons for key, value in slot_obj.items()}
        return FrameValue(name=func_name, arguments=slot_values)

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented


