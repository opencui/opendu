# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pydantic import BaseModel

from opendu.core.annotation import (CamelToSnake, Exemplar)
from opendu.core.config import RauConfig
from opendu.core.matcher import OwnerMode, ExactMatcher
from opendu.core.prompt import (PromptManager, Task)
from opendu.core.retriever import (ContextRetriever)
from opendu.inference.generator import GenerateMode

from opendu.utils.json_tools import parse_json_from_string


class FrameState(BaseModel):
    frame: str
    slot: str
    slotType: str


# How to present context is strictly state tracking implementation dependent.
class DialogExpectation(BaseModel):
    context: list[FrameState]


class SkillDemonstration(BaseModel):
    owner: str
    owner_mode: str = "normal"
    description: str = ""
    exemplars: list[Exemplar] = []


#
# The intent detector try to detect all triggerable intents from user utterance, with respect to
# existing conversational history, summarized in expectations. However, expectations are only used
# during the retrieval stage, not in the decision stage beyond that.
#
class IntentDetector(ABC):
    @abstractmethod
    def detect_intents(self, text, expectations=None, debug=False):
        pass

    @abstractmethod
    def grade(self, text, owner, owner_mode, counts_dict):
        pass




# Intent dectection cast as single class, binary classification problem.
class BcIntentDetector(IntentDetector, ABC):
    def __init__(self, retriever: ContextRetriever, generator):
        self.retrieve = retriever
        self.generator = generator

    def build_skill_prompts(self, text, skills, exemplar_nodes):
        skill_metas = []
        for skill in skills:
            skill_metas.append({
                "name": skill.name,
                "description": skill.description,
                "examples": skill.slots
            })

    def build_prompts_by_examples(self, text, nodes):
        skill_prompts = []
        owners = []
        owner_modes = []

        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        exemplars = [
            Exemplar(
                owner=node.metadata["owner"],
                template=node.text,
                owner_mode=node.metadata["owner_mode"]
            )
            for node in nodes
        ]

        for exemplar in exemplars:
            print(f"process template: {exemplar.template}")
            input_dict = {"utterance": text, "template": exemplar.template}
            skill_prompts.append(self.example_prompt(input_dict))
            owners.append(exemplar.owner)
            owner_modes.append(exemplar.owner_mode)

        return skill_prompts, owners, owner_modes

    def build_prompts_by_desc(self, text, skills):
        skill_prompts = []
        owners = []

        # first we try full prompts, if we get hit, we return. Otherwise, we try no spec prompts.
        # for now, we process it once.
        for skill in skills:
            print(f"process skill: {skill}")
            input_dict = {"utterance": text, "skill": skill}
            skill_prompts.append(self.desc_prompt(input_dict))
            owners.append(skill["name"])
        return skill_prompts, owners

    @staticmethod
    def parse_results(skill_prompts, owners, skill_outputs, owner_modes):
        if RauConfig.get().converter_debug:
            print(json.dumps(skill_prompts, indent=2))
            print(json.dumps(skill_outputs, indent=2))

        flags = [
            parse_json_from_string(raw_flag, None)
            for index, raw_flag in enumerate(skill_outputs)
        ]
        return [owners[index] for index, flag in enumerate(flags) if flag]


    @staticmethod
    def accumulate_debug_for_exemplars(preds, nodes, infos):
        assert len(preds) == len(nodes)
        for index in range(len(preds)):
            item = {
                "type": "exemplar",
                "owner": nodes[index].metadata["owner"],
                "text": nodes[index].text,
                "result": preds[index]
            }
            infos.append(item)

    @staticmethod
    def accumulate_debug_for_skills(preds, skills, infos):
        assert len(preds) == len(skills)
        for index in range(len(preds)):
            item = {
                "type": "desc",
                "owner": skills[index].name,
                "text": skills[index].description,
                "result": preds[index]
            }
            infos.append(item)

    def detect_intents(self, text, expectations, candidates, debug=False):
        print(f"parse for skill: {text} with {expectations} and {candidates}")
        # For now, we only pick one skill
        picker = SingleOwnerKnnPicker(expectations)
        # TODO: try to use candidates. 
        skills, exemplar_nodes = self.retrieve(text)
        print(f"get_skills for {text} with {len(exemplar_nodes)} nodes\n")

        debug_infos = []

        # for exemplar
        if self.use_exemplar:
            exemplar_prompts, owners, _ = self.build_prompts_by_examples(text, exemplar_nodes)
            exemplar_outputs = self.generator.generate(exemplar_prompts, GenerateMode.exemplar)

            exemplar_preds = [
                parse_json_from_string(raw_flag, raw_flag)
                for _, raw_flag in enumerate(exemplar_outputs)
            ]

            if debug:
                print(exemplar_prompts)
                print(exemplar_preds)
                self.accumulate_debug_for_exemplars(exemplar_preds, exemplar_nodes, debug_infos)

            picker.accumulate(exemplar_preds, owners, 1)

        # Now we should use the expectation for improve node score, and filtering
        # the contextual template that is not match.

        # for desc
        if self.use_desc:
            desc_prompts, owners = self.build_prompts_by_desc(text, skills)

            desc_outputs = self.generator.generate(desc_prompts, GenerateMode.desc)
            desc_preds = [
                parse_json_from_string(raw_flag, None)
                for index, raw_flag in enumerate(desc_outputs)
            ]

            if debug:
                print(desc_prompts)
                print(desc_preds)
                self.accumulate_debug_for_skills(desc_preds, skills, debug_infos)

            picker.accumulate(desc_preds, owners, 1)

        label = picker.decide()
        return label, list(map(node_to_exemplar, exemplar_nodes)), debug_infos


    @staticmethod
    def update(preds, truth, counts, skill_prompts, skill_outputs, output=True):
        pairs = list([str(item) for item in zip(preds, truth)])
        if output:
            print(json.dumps(skill_prompts, indent=2))
            print(json.dumps(pairs, indent=2))

        pairs = zip(preds, truth)
        for index, pair in enumerate(pairs):
            if pair[0] != pair[1] and output:
                print(f"At {index}, {skill_prompts[index]} : {skill_outputs[index]}, not correct.")

        pairs = zip(preds, truth)
        for pair in pairs:
            if pair[1] is None:
                continue
            index = 2 if pair[0] else 0
            index += 1 if pair[1] else 0
            counts[index] += 1

    def grade(self, text, owner, owner_mode, count_dict):
        if not self.matcher.is_good_mode(owner_mode):
            return

        picker = SingleOwnerKnnPicker([])
        # nodes owner are always included in the
        skills, nodes = self.retrieve(text, [])

        # for exemplar
        exemplar_prompts, owners, owner_modes = self.build_prompts_by_examples(text, nodes, CamelToSnake)
        exemplar_outputs = self.generator.generate(exemplar_prompts, GenerateMode.exemplar)
        exemplar_preds = [
            parse_json_from_string(raw_flag, raw_flag)
            for index, raw_flag in enumerate(exemplar_outputs)
        ]
        exemplar_truth = [
            self.matcher.agree(owner, owner_mode, lowner, owner_modes[index])
            for index, lowner in enumerate(owners)]

        assert len(exemplar_preds) == len(exemplar_truth)
        picker.accumulate(exemplar_preds, owners, 2)

        # for desc
        desc_prompts, owners = self.build_prompts_by_desc(text, skills, CamelToSnake)
        desc_outputs = self.generator.generate(desc_prompts, GenerateMode.desc)
        desc_preds = [
            parse_json_from_string(raw_flag, None)
            for index, raw_flag in enumerate(desc_outputs)
        ]
        desc_truth = [owner == lowner and OwnerMode[owner_mode] == OwnerMode.normal for lowner in owners]
        assert len(desc_preds) == len(desc_truth)

        picker.accumulate(desc_preds, owners, 1)
        counts = count_dict["skill"]
        predicted_owner = picker.decide()
        concrete = count_dict["skills"]
        if predicted_owner == owner and OwnerMode[owner_mode] in picker.modes:
            counts[1] += 1
            debug_output = False
            concrete[owner][1] += 1
        else:
            counts[0] += 1
            debug_output = True
            concrete[owner][0] += 1

        if debug_output:
            print(f"\n\nMade mistakes on: [ {text} ] expecting [{owner}] but get [{predicted_owner}].")
            print(json.dumps(picker.counts))

        # We only output when there is a need for study
        self.update(exemplar_preds, exemplar_truth, count_dict["exemplar"], exemplar_prompts, exemplar_outputs, debug_output)
        self.update(desc_preds, desc_truth, count_dict["desc"], desc_prompts, desc_outputs, debug_output)


def node_to_exemplar(node):
    meta = node.metadata
    result = {
        "type": "exemplar",
        "template": meta["template"],
        "ownerFrame": meta["owner"]
    }

    # The optional information. Kotlin side use label instead of owner_mode.
    if meta["owner_mode"] != "normal":
        result["label"] = meta["owner_mode"]
    if meta["context_frame"] != "":
        result["contextFrame"] = meta["context_frame"]
    if meta["context_slot"] != "":
        result["contextSlot"] = meta["context_slot"]

    return result