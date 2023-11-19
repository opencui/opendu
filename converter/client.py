import torch
import transformers
from transformers import AutoTokenizer

from converter.lug_config import LugConfig
from converter.schema_parser import load_specs_and_recognizers_from_directory
from core.annotation import FrameValue, Exemplar, FrameSchema, DialogExpectation
from core.prompt import SkillPrompts, SlotPrompts
from core.retriever import load_context_retrievers, ContextRetriever


#
# Here are the work that client have to do.
# For now, assume single module. We can worry about multiple module down the road.
# 1. use description and exemplars to find built the prompt for the skill prompt.
# 2. use skill and recognizer to build prompt for slot.
# 3. stitch together the result.
#
class Converter:
    def __init__(self, retriever: ContextRetriever, recognizers, model):
        self.retrieve = retriever
        self.recognizers = recognizers

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.skill_prompt = SkillPrompts[LugConfig.skill_prompt]
        self.slot_prompt = SlotPrompts[LugConfig.slot_prompt]

    def understand(self, text: str, expectation: DialogExpectation = None) -> FrameValue:
        # first we figure out what is the
        skills, nodes = self.retrieve(text)
        exemplars = [Exemplar(owner=node.metadata["owner"], template=node.text) for node in nodes]
        input_dict = {"utterance": text, "examples": exemplars, "skills": skills}
        formatted_prompt = self.skill_prompt(input_dict)

        skill_outputs = self.pipeline(
            formatted_prompt,
            do_sample=True,
            top_k=50,
            top_p = 0.9,
            num_return_sequences=1,
            repetition_penalty=1.1,
            max_new_tokens=1024,
        )

        for seq in skill_outputs:
            print(f"Result: {seq['generated_text']}")

        func_name = skill_outputs[0]["generated_text"]

        # Then we need to create the prompt for the parameters.

        slot_values = None
        return FrameValue(name=func_name, arguments=slot_values)

    def generate(self, struct: FrameValue) -> str:
        llm = self.llm
        raise NotImplemented


def get_skill_infos(skills, nodes) -> list[FrameSchema]:
    funcset = {item.node.meta["owner"] for item in nodes}
    return [skills[func] for func in funcset]


def get_exemplars(nodes) -> list[Exemplar]:
    return [Exemplar(owner=item.node.meta["owner"]) for item in nodes]


def load_converter(specs: str, index: str) -> Converter:
    # We assume
    specs, recognizers = load_specs_and_recognizers_from_directory(specs)
    retrievers = load_context_retrievers(index)
    return Converter(specs, retrievers, recognizers)
