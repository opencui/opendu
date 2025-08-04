# Copyright 2024, OpenCUI
# Licensed under the Apache License, Version 2.0.
# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel
from opendu import FrameSchema
from opendu.core.annotation import SlotSchema, build_json_schema
from opendu.core.config import RauConfig, Task
from opendu.core.prompt import PromptManager
from opendu.core.retriever import ContextRetriever
from opendu.inference.decoder import Decoder


#
# There are two levels of the APIs here: a
# 1. assuming the entity recognizer is external,
# 2. assuming the entity recognizer is internel.
# The difference will be whether we include candidates in the API.
# We will be focusing on the #1 as #2 can be easily implemented with #1.
#
# There are many possible solutions for this:
# For example, question/answering based: https://arxiv.org/pdf/2406.08848
# For structure extraction bases: nuextract
# https://arxiv.org/pdf/2403.17536v1



class SlotMeta(BaseModel):
    description: str
    multi_value: bool

class EntitySlotMeta(SlotMeta):
    pass

class FrameSlotMeta(SlotMeta):
    slots: Dict[str, SlotMeta]

#
# The slot extractor takes task description as context, and slot schema, candidate values and
# generate the json representation of values.
#
class SlotExtractor(ABC):
    @abstractmethod
    def extract_values(self, utterance:str, frame: str, expectation:list[str], candidates: dict, debug=False) -> dict[str, Any]:
        """"""
        pass


class StructuredExtractor(SlotExtractor):
    def __init__(self, retriever:ContextRetriever=None):
        self.retriever = retriever
        self.decocer = Decoder.get()
        self.slot_prompt = PromptManager.get_builder(Task.Sfss)
        self.module = self.retriever.module if retriever else None
        self.debug = RauConfig().g.get().sf_debug or RauConfig.get().converter_debug or RauConfig.get().debug
    
    # For each slot, we can use a different extraction, regardless whether it is entity or structure,
    # single value or multiple value.                                         
    def extract_values(self, utterance:str, frame_name: str, expectations:list[str], candidates: dict):      
        frame_schema = self.module.get_skill(frame_name)
        slot_infos = [self.module.slots[slot_name] for slot_name in frame_schema.slots]
        slot_types = [build_json_schema(self.module.skills, self.module.slots, slot_schema.type, True, slot_schema.multi_value) for slot_schema in slot_infos]
        return self.raw_extract_values(utterance, frame_schema, slot_infos, slot_types, expectations, candidates)
    

    def raw_extract_values(self, utterance:str, frame: FrameSchema, slots: list[SlotSchema], slot_types: list[dict], expectations: list[str], candidates: dict):
        slot_prompts = []
        for idx in range(len(slots)):
            slot = slots[index]
            slot_type = slot_types[index]

            # For now, we do not have example.
            slot_prompts.append(self.slot_prompt({
                "utterance": utterance,
                "skill": frame,
                "slot": slot,
                "type_schema": slot_type,
                "candidates": candidates,
                "is_expected": slot.name in expectations
            }))
        
        slot_outputs = self.generator.generate(slot_prompts)

        if self.debug:
            print(json.dumps(slot_outputs, indent=2))

        results = {}
        for index, slot in enumerate(slots):
            # TODO(sean): this the source where we know what is the value, while we do not do
            # normalization here, we need to explicitly
            if slot_outputs[index] != "":
                results[slot["name"]] = {"values" : [slot_outputs[index]], "operator": "=="}
        return results

        print(f"parse for skill: {text} with {expectations} and {candidates}")
        # For now, we only pick one skill
        # TODO: try to use candidates. 
        skills, exemplar_nodes = self.retrieve(text)
        print(f"get_skills for {text} with {len(exemplar_nodes)} nodes\n")

        debug_infos = []

        skill_with_exemplars = self.build_skills(text, skills, exemplar_nodes)

        # Now we should use the expectation for improve node score, and filtering
        # the contextual template that is not match.
        skill_prompts = []
        build_prompt = PromptManager.get_builder(Task.IdBc, input_mode=True)
        for skill_demonstration in skill_with_exemplars:
            skill_prompts.append(
                build_prompt(
                    {
                        "skill": skill_demonstration.skill,
                        "exemplars": skill_demonstration.exemples,
                        "utterance": text,
                        "arguments": candidates
                    }
                )
            )
        
        skill_outputs = self.generator.generate(skill_prompts, OutputExpectation(choices=["True", "False"])).output
        # For now we assume single intent.
    
        zipped = list(zip(skills, skill_outputs))
        # Later we can run this twice, first with examples (more trustworthy), then without examples.
        label = next((paired[0].name for paired in zipped if paired[1].outputs == "True"), None)

        return label, list(map(node_to_exemplar, exemplar_nodes)), debug_infos


if __name__ == "__main__":

    skill = FrameSchema(
        name="build_reservation_module",
        description="build a reservation module")

    examples = [
        BcSkillExample(template="can you help me to build a table reservation module", label=True), 
        BcSkillExample(template="I like to reserve a table", label=False)
                ]
    build_prompt = PromptManager.get_builder(Task.IdBc, input_mode=True)
    prompt0 = build_prompt({"utterance": "can you help me to make a table reservation please?", "skill": skill, "examples": examples, "arguments": {}})
    prompt1 = build_prompt({"utterance": "I like to build a table reservation module, please", "skill": skill, "examples": examples, "arguments": {}})
    print(prompt0)

    generator = FftVllmGenerator(model="Qwen/Qwen3-4B")

    raw_output = generator.generate([prompt0, prompt1], OutputExpectation(choices=["True", "False"]))
    outputs = [output.outputs for output in raw_output]
    print(outputs)