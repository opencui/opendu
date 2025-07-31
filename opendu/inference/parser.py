# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
import json
import re
from enum import Enum
from typing import Dict, Any

from pydantic import BaseModel
from opendu.inference.intent_detector import BcIntentDetector
from opendu.core.annotation import (ListRecognizer, get_value, EntityStore)
from opendu.core.config import RauConfig
from opendu.core.prompt import (PromptManager, Task)
from opendu.core.retriever import (ContextRetriever, load_context_retrievers)
from opendu.core.schema_parser import load_all_from_directory
from opendu.inference.generator import GenerateMode, Generator

from opendu.utils.json_tools import parse_json_from_string

# The modes that we will support.
YesNoResult = Enum("YesNoResult", ["Affirmative", "Negative", "Indifferent", "Irrelevant"])


# Used for serving function calling API.
class FrameValue(BaseModel):
    name: str
    arguments: Dict[str, Any]

#
# This is the parser is used to convert natural language text into its semantic representation.
# Instead of using function calling enabled model, or prompt, this parser using an agentic
# approach, by separating intent detection and slot filling into two sub-tasks, and each
# is addressed by agentic RAG again.
#
class Parser:
    def __init__(
            self,
            retriever: ContextRetriever,
            entity_metas: EntityStore = None,
            with_arguments=True,
    ):
        self.retrieve = retriever
        self.recognizer = None
        if entity_metas is not None:
            self.recognizer = ListRecognizer(entity_metas)

        self.generator = Generator.build()
        self.slot_prompt = PromptManager.get_builder(Task.SLOT)
        self.yni_prompt = PromptManager.get_builder(Task.YNI)
        self.with_arguments = with_arguments
        self.bracket_match = re.compile(r"\[([^]]*)\]")

        self.skill_converter = KnnIntentDetector(retriever, self.generator)
        self.yni_results = {"Affirmative", "Negative", "Indifferent", "Irrelevant" }


    # Reference implementation for function calling.
    def understand(self, text: str) -> FrameValue:
        # low level get skill.
        func_name, _ = self.skill_converter.detect_intents(text)

        if not self.with_arguments:
            return FrameValue(name=func_name, arguments={})

        # We assume the function_name is global unique for now. From UI perspective, I think
        module = self.retrieve.module
        slot_labels_of_func = module.skills[func_name]["slots"]

        # Then we need to create the prompt for the parameters.
        slot_prompts = []
        for slot in slot_labels_of_func:
            values = []
            if self.recognizer is not None:
                values = self.recognizer.extract_values(slot, text)
            slot_input_dict = {"utterance": text, "values": values}
            slot_input_dict.update(module.slots[slot])
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        if RauConfig.get().converter_debug:
            print(json.dumps(slot_prompts, indent=2))
        slot_outputs = self.generator.generate(slot_prompts, GenerateMode.extractive)

        if RauConfig.get().converter_debug:
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


    def debug(self, utterance, expectations):
        _, _, debugs = self.skill_converter.detect_intents(utterance, expectations, True)
        # For now, we assume single intent.

        # TODO: figure out how to handle the multi intention utterance.
        return debugs


    def detect_triggerables(self, utterance, expectations, candidates = {}, debug=False):
        func_name, evidence, _ = self.skill_converter.detect_intents(utterance, expectations, debug)
        # For now, we assume single intent.
        result = {
            "owner": func_name,
            "utterance": utterance,
            "evidence": evidence
        }

        # TODO: figure out how to handle the multi intention utterance.
        return [result]


    def fill_slots(self, text, slots:list[dict[str, str]], candidates:dict[str, list[str]])-> dict[str, str]:
        slot_prompts = []
        for slot in slots:
            name = slot["name"]
            values = get_value(candidates, name, [])
            slot_input_dict = {"utterance": text, "name": name, "candidates": values}
            slot_prompts.append(self.slot_prompt(slot_input_dict))

        if RauConfig.get().converter_debug:
            print(json.dumps(slot_prompts, indent=2))
        slot_outputs = self.generator.generate(slot_prompts, GenerateMode.extractive)

        if RauConfig.get().converter_debug:
            print(json.dumps(slot_outputs, indent=2))

        results = {}
        for index, slot in enumerate(slots):
            # TODO(sean): this the source where we know what is the value, while we do not do
            # normalization here, we need to explicitly
            if slot_outputs[index] != "":
                results[slot["name"]] = {"values" : [slot_outputs[index]], "operator": "=="}
        return results

    def inference(self, utterance:str, questions:list[str]) -> list[str]:
        input_prompts = []
        for question in questions:
            # For now, we ignore the language
            input_dict = {"response": utterance, "question": f"{question}."}
            input_prompts.append(self.yni_prompt(input_dict))

        outputs = self.generator.generate(input_prompts, GenerateMode.nli)
        outputs = list(map(lambda x: x if (x in self.yni_results) else "Irrelevant", outputs))

        if RauConfig.get().converter_debug:
            print(f"{input_prompts} {outputs}")
        return outputs

    def generate(self, struct: FrameValue) -> str:
        raise NotImplemented


def load_parser(module_path, index_path):
    # First load the schema info.
    module_schema, _, _ = load_all_from_directory(module_path)

    # Then load the retriever by pointing to index directory
    context_retriever = load_context_retrievers(module_schema, index_path)

    # Finally build the converter.
    return Parser(context_retriever)
