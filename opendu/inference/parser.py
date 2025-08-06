# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
import json
import re
from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel
from opendu.inference.intent_detector import BcIntentDetector
from opendu.core.annotation import (ListRecognizer, get_value, EntityStore)
from opendu.core.config import RauConfig
from opendu.core.prompt import (PromptManager, Task)
from opendu.core.retriever import (ContextRetriever, load_context_retrievers)
from opendu.core.schema_parser import load_all_from_directory
from opendu.inference.decoder import Decoder

from opendu.inference.slot_filler import StructuredExtractor
from opendu.inference.yn_inferencer import YesNoInferencer, YesNoQuestion, YesNoResult
from opendu.utils.json_tools import parse_json_from_string


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

        self.skill_converter = BcIntentDetector(retriever)
        self.slot_extractor = StructuredExtractor(retriever)
        self.yn_decider = YesNoInferencer(retriever)

    # this is used to detect the intent, or skill, of the utterance.
    def detect_triggerables(self, utterance:str, candidates: dict[str, list[str]], expectedFrames: list[str] = []):
        func_name, evidence, _ = self.skill_converter.detect_intents(utterance, candidates, expectedFrames)
        # For now, we assume single intent.
        result = {
            "owner": func_name,
            "utterance": utterance,
            "evidence": evidence
        }

        # TODO: figure out how to handle the multi intention utterance.
        return [result]

    # This is used to extract the slots from the utterance for the given frame.
    def fill_slots(self, text, frame: str, candidates:dict[str, list[str]], expectedSlots: list[str] = [])-> dict[str, str]:
        return self.slot_extractor.extract_values(text, frame, candidates, expectedSlots)
    
    # This is used to understand the user response to yes no question.
    def decide(self, utterance:str, question: str, dialogActType: str = None, targetFrame: str = None, targetSlot: str = None) -> YesNoResult:
        return self.yn_decider.decide(utterance, question, dialogActType, targetFrame, targetSlot)



def load_parser(module_path, index_path):
    # First load the schema info.
    module_schema, _, _ = load_all_from_directory(module_path)

    # Then load the retriever by pointing to index directory
    context_retriever = load_context_retrievers(module_schema, index_path)

    # Finally build the converter.
    return Parser(context_retriever)
