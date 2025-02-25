# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.
import abc
import json
import random
import re

from abc import ABC
from enum import Enum
from typing import Optional, Dict

from opendu import InstructBuilder, IOMode
from opendu.core.retriever import ContextRetriever
from opendu.core.annotation import Schema, Exemplar, ListRecognizer, OwnerMode, ExactMatcher
from opendu.core.prompt import (Task, promptManager)
from pydantic import BaseModel, Field


# This exemplar contains arguments, used for fine-tuning.
class FullExemplar(BaseModel):
    """
    Expression examples. If the expected_slots is empty, this can be used for both skills and slots.
    """

    id: str
    owner: str
    utterance: str  # useful for slot model
    arguments: Dict[str, str]  # Specify the type of the values in the dictionary if needed
    owner_mode: Optional[str] = Field("normal", description="The label for owner mode")
    template: Optional[str] = Field(None, description="Template for the exemplar")
    context_frame: Optional[str] = Field(None, description="Context frame associated with the exemplar")
    context_slot: Optional[str] = Field(None, description="Context slot associated with the exemplar")

    def flatten(self):
        return {
            "id": self.id,
            "owner": self.owner,
            "utterance": self.utterance,
            "arguments": str(self.arguments),
            "owner_mode": self.owner_mode,
            "template": self.template,
            "context_frame": self.context_frame,
            "context_slot": self.context_slot
        }

    @staticmethod
    def get_span(word, sentence):
        # Construct a regular expression pattern to find the word with boundaries and punctuation
        pattern = r'\b' + re.escape(word) + r'\b'

        # Search for the pattern in the sentence
        return re.findall(pattern, sentence), re.search(pattern, sentence)

    @staticmethod
    def extract_template(utterance, arguments):
        if len(arguments) == 0:
            return utterance

        single_dict = dict()
        spans = []
        for key, values in arguments.items():
            for value in values:
                single_dict[value] = key
                found, match = FullExemplar.get_span(value, utterance)
                if len(found) != 1:
                    return None
                spans.append(match.span())

        spans = sorted(spans, key=lambda x: x[0])
        res_utterance = utterance[: spans[0][0]]
        for i, (cur_start, cur_end) in enumerate(spans):
            # if len(string_list) >=2:
            #     print("sub string",utterance[cur_start:cur_end])
            res_utterance = (
                    res_utterance + " < " + single_dict[utterance[cur_start:cur_end]] + " > "
            )
            if i == len(spans) - 1:
                res_utterance = res_utterance + utterance[cur_end:]
            else:
                res_utterance = res_utterance + utterance[cur_end: spans[i + 1][0]]
        return res_utterance


# This inference is responsible for convert the exemplars in the original dataset into what is needed
# by generation fine-tuning. The assumed the columns are input and output, and we added id for debugging
# purpose.
# We assume that batch is AnnotatedExemplar in column form, this is what we get from pandas.
# Ins/Outs are used to collect generated instance that might be more or less than what is the in batch.
#
class TrainPhase1Converter(ABC):
    @abc.abstractmethod
    def __call__(self, batch, ins: list[str], outs: list[str]):
        return


class MultiClassSkillConverter(TrainPhase1Converter):
    def __init__(self, retriever: ContextRetriever):
        label = promptManager.get_task_label(Task.SKILL)
        assert label.startswith("skill-mc"), "need to be skill-mc prefix"
        self.prompt = promptManager.get_builder(Task.SKILL)
        self.context_retrieve = retriever

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # Working on the batched dataset, with first dimension is column then index.
        for idx, utterance in enumerate(batch["utterance"]):
            # We assume the input is dict version of AnnotatedExemplar
            skills, nodes = self.context_retrieve(utterance)
            # remove the identical exemplar
            nodes = [node for node in nodes if node.id_ != batch["id"][idx]]
            exemplars = [
                Exemplar(owner=node.metadata["owner"], template=node.text, owner_mode=node.metadata["owner_mode"])
                for node in nodes
            ]
            owner = batch["owner"][idx]

            # How can we reduce the need for
            neg_owners = [
                node.metadata["owner"]
                for node in nodes
                if node.metadata["owner"] != owner
            ]

            # randomly filter one neg skills and exemplars
            if len(neg_owners) != 0:
                neg_owner = random.choice(neg_owners)
                rm_neg_exemplars = [
                    exemplar for exemplar in exemplars if exemplar.owner != neg_owner
                ]
                rm_neg_skills = [
                    skill for skill in skills if skill["name"] != neg_owner
                ]

                # Without exemplars.
                random.shuffle(rm_neg_skills)
                input_dict = {
                    "utterance": utterance,
                    "examples": [],
                    "skills": rm_neg_skills,
                }
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(owner)}</s>")

                # With exemplars.
                if len(rm_neg_exemplars) != 0:
                    random.shuffle(rm_neg_exemplars)
                    input_dict = {
                        "utterance": utterance,
                        "examples": rm_neg_exemplars,
                        "skills": rm_neg_skills,
                    }
                    ins.append(self.prompt(input_dict))
                    outs.append(f"{json.dumps(owner)}</s>")

            # Try to filter the pos skills and exemplars
            rm_pos_exemplars = [
                exemplar for exemplar in exemplars if exemplar.owner != owner
            ]
            rm_pos_skills = [skill for skill in skills if skill["name"] != owner]

            random.shuffle(rm_pos_skills)
            input_dict = {
                "utterance": utterance,
                "examples": [],
                "skills": rm_pos_skills,
            }
            ins.append(self.prompt(input_dict))
            outs.append(f"{json.dumps(None)}</s>")

            if len(rm_pos_exemplars) != 0:
                random.shuffle(rm_pos_exemplars)
                input_dict = {
                    "utterance": utterance,
                    "examples": rm_pos_exemplars,
                    "skills": rm_pos_skills,
                }
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(None)}</s>")


class SingleClassSkillConverter(TrainPhase1Converter):
    def __init__(self, retriever: ContextRetriever):
        label = promptManager.get_task_label(Task.SKILL)
        assert label.startswith("skill-sc"), "need to be skill-sc prefix"
        self.prompt = promptManager.get_builder(Task.SKILL)
        self.context_retrieve = retriever
        self.neg_k = 1
        self.match_mode = "normal"

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # Working on the batched dataset, with first dimension is column then index.
        for idx, utterance in enumerate(batch["utterance"]):
            # We assume the input is dict version of AnnotatedExemplar
            skills, nodes = self.context_retrieve(utterance)
            # remove the identical exemplar
            nodes = [node for node in nodes if node.id_ != batch["id"][idx]]
            exemplars = [
                Exemplar(
                    owner=node.metadata["owner"],
                    template=node.text,
                    owner_mode=node.metadata["owner_mode"]
                )
                for node in nodes
            ]
            exampled = set([node.metadata["owner"] for node in nodes])
            owner = batch["owner"][idx]
            owner_mode = batch["owner_mode"][idx]

            skill_map = {}

            # Just using the skill name/descriptions
            for skill in skills:
                input_dict = {"utterance": utterance, "examples": [], "skill": skill}
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(owner == skill['name'] and OwnerMode[owner_mode] == OwnerMode.normal)}</s>")
                skill_map[skill["name"]] = skill

            # Add the examples for each skill.
            for skill in skills:
                # Need to project each examples in the view of this skill.
                target = skill["name"]
                # Should be somehow related.
                if target not in exampled:
                    continue

                # Try not to have more than two examples.
                exemplar_dicts = [
                    {
                        "template": exemplar.template,
                        "target": target,
                        "decision": target == exemplar.owner and OwnerMode[exemplar.owner_mode] == OwnerMode.normal
                    }
                    for exemplar in exemplars
                ]

                input_dict = {
                    "utterance": utterance,
                    "examples": exemplar_dicts,
                    "skill": skill_map[target],
                }
                ins.append(self.prompt(input_dict))
                outs.append(f"{json.dumps(owner == target and OwnerMode[owner_mode] == OwnerMode.normal)}</s>")


# This is needed to determine the intention, intended function or skill
# https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
# This only works with simple use case where we only match in normal/exact/literal sense.
class InstanceMode(Enum):
    desc = "desc"
    example = "example"
    both = "both"


def skill_converter(retriever: ContextRetriever, skill_mode):
    if skill_mode == "desc":
        return DescExemplarConverter(retriever, InstanceMode.desc)
    if skill_mode == "exemplar":
        return DescExemplarConverter(retriever, InstanceMode.example)
    if skill_mode == "rag":
        return RagSkillConverter(retriever, InstanceMode.both)


def suffix_sublists_with_empty(lst):
    return [lst[i:] for i in range(len(lst) + 1)]

#
# For this one, we retrieve based on both on description and then exemplar, we will create the prompt
# based on both list. We use json output.
#
class RagSkillConverter(TrainPhase1Converter):
    def __init__(self, retriever: ContextRetriever, mode=InstanceMode.both):
        # Make sure that we have the same key for Desc and exemplar prompt.
        label = promptManager.get_task_label(Task.SKILL)
        assert label.startswith("id_mc_full"), f"need to be skill-rag prefix: {label}"
        self.input_prompt = promptManager.get_builder(Task.SKILL, IOMode.INPUT)
        self.output_prompt = promptManager.get_builder(Task.SKILL, IOMode.OUTPUT)
        self.context_retrieve = retriever
        self.mode = mode
        self.matcher = ExactMatcher

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We need to make sure that
        assert self.mode == InstanceMode.both

        # Working on the batched dataset, with first dimension is column then index.
        for idx, utterance in enumerate(batch["utterance"]):
            # We assume the input is dict version of AnnotatedExemplar
            skills, nodes = self.context_retrieve(utterance, batch["id"][idx])

            exemplars = [
                Exemplar(owner=node.metadata["owner"], template=node.text, owner_mode=node.metadata["owner_mode"])
                for node in nodes
            ]
            owner = batch["owner"][idx]
            owner_mode = batch["owner_mode"][idx]

            # reverse both skills and exemplars
            skills.reverse()
            exemplars.reverse()

            print(f"skill = {skills}")
            print(f"exemplars = {exemplars}")
            print(f"utterance = {utterance}")

            # First positive.
            # We always have all the skills and descriptions, but remove exemplars one at a time to simulate
            # the cases where there are not exemplars.
            sublists = suffix_sublists_with_empty(exemplars)
            for sublist in sublists:
                ins.append(self.input_prompt.build(utterance=utterance,skills=skills, exemplars=sublist))
                outs.append(self.output_prompt.build(outputs=[owner]))

            neg_skills = filter(lambda x: x["name"] != owner, skills)
            neg_exemplars = filter(lambda x: x["owner"] != owner, exemplars)
            neg_sublists = suffix_sublists_with_empty(list(neg_exemplars))
            for sublist in neg_sublists:
                ins.append(self.input_prompt.build(utterance=utterance,skills=neg_skills, exemplars=sublist))
                outs.append(self.output_prompt.build(outputs=[]))


# For this one, we first use example based prediction, and then description based prediction.
class DescExemplarConverter(TrainPhase1Converter):
    def __init__(self, retriever: ContextRetriever, mode=InstanceMode.both):
        # Make sure that we have the same key for Desc and exemplar prompt.
        self.desc_prompt = promptManager.get_builder(Task.SKILL_DESC)
        self.example_prompt = promptManager.get_builder(Task.SKILL)
        self.context_retrieve = retriever
        self.neg_k = 1
        self.mode = mode
        self.matcher = ExactMatcher

    @staticmethod
    def label(value):
        label_dict = {"label": "true" if value else "false"}
        return promptManager.get_builder(Task.BOOL_VALUE)(label_dict)

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # Working on the batched dataset, with first dimension is column then index.
        for idx, utterance in enumerate(batch["utterance"]):
            # We assume the input is dict version of AnnotatedExemplar
            skills, nodes = self.context_retrieve(utterance)

            # remove the identical exemplar
            nodes = [node for node in nodes if node.id_ != batch["id"][idx]]

            exemplars = [
                Exemplar(owner=node.metadata["owner"], template=node.text, owner_mode=node.metadata["owner_mode"])
                for node in nodes
            ]
            owner = batch["owner"][idx]
            owner_mode = batch["owner_mode"][idx]

            # First handle exemplars.
            if self.mode != InstanceMode.desc:
                # Include pairing with itself
                input_dict = {"utterance": utterance, "template": batch["template"][idx]}
                ins.append(self.example_prompt(input_dict))
                outs.append(f"{self.label(True)}")
                for exemplar in exemplars:
                    # if there are more details in the templates, we ignore this pair, as we do not know.
                    match_status = self.matcher.agree(owner, owner_mode, exemplar.owner, exemplar.owner_mode)

                    # if matching strategy can not make a decision, ignore the pair.
                    if match_status is None:
                        print(f"Nothing normal here: {utterance} : {exemplar.template} ", flush=True)
                        continue

                    # Try not to have more than two examples.
                    input_dict = {"utterance": utterance, "template": exemplar.template}
                    ins.append(self.example_prompt(input_dict))
                    outs.append(self.label(match_status))

            # Then descriptions.
            if self.mode != InstanceMode.example:
                for skill in skills:
                    input_dict = {"utterance": utterance, "skill": skill}
                    ins.append(self.desc_prompt(input_dict))
                    outs.append(self.label(self.matcher.match(owner, skill['name'], owner_mode)))

#
#  We need to handle many different use case here:
#  premise is what user said, and hypothesis is what we want to know.
#
class NliConverter(TrainPhase1Converter, ABC):
    def __init__(self, prompt):
        self.prompt = prompt
        self.labels = ["entailment", "neutral", "contradiction"]

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, premise in enumerate(batch["premise"]):
            hypothesis = batch["hypothesis"][idx]
            label = self.labels[int(batch["label"][idx])]
            input_dict = {"premise": premise, "hypothesis": hypothesis}
            ins.append(self.prompt(input_dict))
            outs.append(f"{label}</s>")


#
# For the yes/no question, what does response imply: yes, not, don't care or not related.
#
class YniConverter(TrainPhase1Converter, ABC):
    def __init__(self):
        self.prompt = promptManager.get_builder(Task.YNI)

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, question in enumerate(batch["question"]):
            response = batch["response"][idx]
            label = batch["label"][idx]
            input_dict = {"question": question, "response": response}
            ins.append(self.prompt(input_dict))
            outs.append(f"{label}</s>")


# This is for slot.
# The slot converter need to have access to entities.
class SlotConverter(TrainPhase1Converter, ABC):
    entities: dict[str, re.Pattern]


#
# This is for extractive slot value understanding.
# This somewhat influenced by structured extraction from here.
# https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction
#
# One of the issue is how do we handle nested structures, while we still separate from
# how we prompt.
#
class RagStructExtractConverter(SlotConverter):
    def __init__(self, module: Schema, slot_prompt: InstructBuilder, entities):
        self.prompt = slot_prompt
        self.module = module
        self.include_negative = True
        # First try to be efficient.
        self.entities = entities
        self.patterns = {}
        for key, values in entities.items():
            strings_to_check = list(values)
            pattern = re.compile("|".join(map(re.escape, strings_to_check)))
            self.patterns[key] = pattern
        # This is
        self.input_builder = None

    @staticmethod
    def format_value(key, value=None):
        return f"{json.dumps(value)}</s>"

    def add_one_negative(self, slot_name, small_value_set):
        if slot_name not in self.entities:
            return

        picked = None
        candidates = self.entities[slot_name]

        while picked in small_value_set:
            picked = random.choice(candidates)

        if picked is not None:
            small_value_set.add(picked)


    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, sarguments in enumerate(batch["arguments"]):
            arguments = eval(sarguments)
            utterance = batch["utterance"][idx]
            owner = batch["owner"][idx]
            #
            # We will handle three different situations:
            # 1. with just slot schema,
            # 2. with some examples (not the related one).
            # 3. with candidates, provided by external recognizer.
            #
            # First get all the slots.
            slots = [self.module.slots[slot_label] for slot_label in self.module.skills[owner]["slots"]]

            for slot_label in self.module.skills[owner]["slots"]:
                slot = self.module.slots[slot_label]
                slot_name = slot["name"]

                # Now we need to select the value from entities
                # In addition to the true value, the best should be of the same type and
                # also the occurs in the utterance but not the value.
                values = set(
                    ListRecognizer.find_matches(self.patterns, slot_name, utterance)
                )
                # Most likely we do not need to add the negatives.
                # self.add_one_negative(slot_label, values)
                input_dict = {"utterance": utterance}
                input_dict.update(slot.to_dict())
                if slot_name in arguments:
                    value = arguments[slot_name]
                    # First without values. We assume that value is
                    input_dict["values"] = []
                    ins.append(self.prompt(input_dict))
                    if len(value) == 1:
                        outs.append(
                            self.format_value(slot_name, arguments[slot_name][0])
                        )
                    else:
                        outs.append(self.format_value(slot_name, arguments[slot_name]))
                    # then with values.
                    input_dict["values"] = values
                    ins.append(self.prompt(input_dict))
                    if len(value) == 1:
                        outs.append(
                            self.format_value(slot_name, arguments[slot_name][0])
                        )
                    else:
                        outs.append(self.format_value(slot_name, arguments[slot_name]))
                else:
                    input_dict["values"] = []
                    if self.include_negative:
                        ins.append(self.prompt(input_dict))
                        outs.append(self.format_value(slot_name, None))


#
# This is for extractive slot value understanding.
# For each slot, we create a question for the slot value. For some reason, it is not being used.
#
class IsolatedQAConverter(SlotConverter):
    def __init__(self, module: Schema, slot_prompt: InstructBuilder, entities):
        self.prompt = slot_prompt
        self.module = module
        self.include_negative = True
        # First try to be efficient.
        self.entities = entities
        self.patterns = {}
        for key, values in entities.items():
            strings_to_check = list(values)
            pattern = re.compile("|".join(map(re.escape, strings_to_check)))
            self.patterns[key] = pattern

    @staticmethod
    def format_value(key, value=None):
        return f"{json.dumps(value)}</s>"

    def add_one_negative(self, slot_name, small_value_set):
        if slot_name not in self.entities:
            return

        picked = None
        candidates = self.entities[slot_name]

        while picked in small_value_set:
            picked = random.choice(candidates)

        if picked is not None:
            small_value_set.add(picked)

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, sarguments in enumerate(batch["arguments"]):
            arguments = eval(sarguments)
            utterance = batch["utterance"][idx]
            owner = batch["owner"][idx]
            for slot_label in self.module.skills[owner]["slots"]:
                slot = self.module.slots[slot_label]
                slot_name = slot["name"]

                # Now we need to select the value from entities
                # In addition to the true value, the best should be of the same type and
                # also the occurs in the utterance but not the value.
                values = set(
                    ListRecognizer.find_matches(self.patterns, slot_name, utterance)
                )
                # Most likely we do not need to add the negatives.
                # self.add_one_negative(slot_label, values)
                input_dict = {"utterance": utterance}
                input_dict.update(slot.to_dict())
                if slot_name in arguments:
                    value = arguments[slot_name]
                    # First without values. We assume that value is
                    input_dict["values"] = []
                    ins.append(self.prompt(input_dict))
                    if len(value) == 1:
                        outs.append(
                            self.format_value(slot_name, arguments[slot_name][0])
                        )
                    else:
                        outs.append(self.format_value(slot_name, arguments[slot_name]))
                    # then with values.
                    input_dict["values"] = values
                    ins.append(self.prompt(input_dict))
                    if len(value) == 1:
                        outs.append(
                            self.format_value(slot_name, arguments[slot_name][0])
                        )
                    else:
                        outs.append(self.format_value(slot_name, arguments[slot_name]))
                else:
                    input_dict["values"] = []
                    if self.include_negative:
                        ins.append(self.prompt(input_dict))
                        outs.append(self.format_value(slot_name, None))

