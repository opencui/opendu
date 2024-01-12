import abc
import json
import random
import re

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from random import sample, seed
from typing import Optional

from dataclasses_json import dataclass_json
from datasets import Dataset, load_dataset
from llama_index.embeddings.base import BaseEmbedding
from llama_index.schema import TextNode

from opencui.core.prompt import (Prompt, MulticlassSkillPrompts, BinarySkillPrompts,
                                 ExemplarPrompts, DescriptionPrompts, BoolPrompts, YniPrompts, ExtractiveSlotPrompts)
from opencui.core.annotation import Schema, Exemplar, ListRecognizer, OwnerMode, ExactMatcher
from opencui.core.config import LugConfig
from opencui.core.retriever import create_index, ContextRetriever


# This inference is responsible for convert the exemplars in the original dataset into what is needed
# by generation fine-tuning. The assumed the columns are input and output, and we added id for debugging
# purpose.
class TrainConverter(ABC):
    @abc.abstractmethod
    def __call__(self, item: AnnotatedExemplar, ins: list[str], outs: list[str]):
        return


# The slot converter need to have access to entities.
class SlotExtractConverter(TrainConverter, ABC):
    entities: dict[str, re.Pattern]


# This is needed to determine the intention, intended function or skill
# https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
# This only works with simple use case where we only match in normal/exact/literal sense.
class SkillTrainConverter(TrainConverter):
    def __init__(self, retriever: ContextRetriever):
        self.prompt = MulticlassSkillPrompts[LugConfig.get().skill_prompt]
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


class OneSkillTrainConverter(TrainConverter):
    def __init__(self, retriever: ContextRetriever):
        self.prompt = BinarySkillPrompts[LugConfig.get().skill_prompt]
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


InstanceMode = Enum("InstanceMode", ["desc", "example", "both"])


# For this one, we first use example based prediction, and then description based prediction.
class InstanceTrainConverter(TrainConverter):
    def __init__(self, retriever: ContextRetriever, mode=InstanceMode.both):
        # Make sure that we have the same key for Desc and exemplar prompt.
        self.desc_prompt = DescriptionPrompts[LugConfig.get().skill_prompt]
        self.example_prompt = ExemplarPrompts[LugConfig.get().skill_prompt]
        self.context_retrieve = retriever
        self.neg_k = 1
        self.mode = mode
        self.matcher = ExactMatcher

    @staticmethod
    def label(value):
        label_dict = {"label": "true" if value else "false"}
        return BoolPrompts[LugConfig.get().bool_prompt](label_dict)

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
                    outs.append(f"{self.label(match_status)}")

            # Then descriptions.
            if self.mode != InstanceMode.example:
                for skill in skills:
                    input_dict = {"utterance": utterance, "skill": skill}
                    ins.append(self.desc_prompt(input_dict))
                    outs.append(f"{self.label(self.matcher.match(owner, skill['name'], owner_mode))}")


#
# This is for extractive slot value understanding.
# For now, we only get positive example.
class OneSlotExtractConverter(SlotExtractConverter):
    def __init__(self, module: Schema, slot_prompt: Prompt, entities):
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
                # This adds the slot info.
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


# We need to handle many different use case here: premise is what user said, and hypothesis is what we want to know.
class NliConverter(TrainConverter, ABC):
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


class YniConverter(TrainConverter, ABC):
    def __init__(self, prompt):
        self.prompt = prompt

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, question in enumerate(batch["question"]):
            response = batch["response"][idx]
            label = batch["label"][idx]
            input_dict = {"question": question, "response": response}
            ins.append(self.prompt(input_dict))
            outs.append(f"{label}</s>")


class ConllLabel:
    label_info = {
            "PER" : {"name": "person"},
            "LOC" : {"name": "location"},
            "ORG" : {"name": "organization"}
        }

    def __init__(self, label):
        self.labels = label.split("-")

    def is_payload(self):
        return len(self.labels) != 1

    def is_start(self):
        return self.is_payload() and self.labels[0] == "B"

    def payload(self):
        return self.labels[1]

    def is_close(self, last):
        if last is None:
            return True
        if not self.is_payload():
            return True
        if self.is_start():
            return True
        return False

    def get_name(self):
        return label_info[label.payload()]


class ConllLabelBuilder:
    def __init__(self, cares):
        self.sep = "|"
        self.start = "["
        self.end = ']'
        self.cares = cares

    def care(self, label: ConllLabel):
        return labe.payload() in self.cares

    def __call__(self, tokens, tags):
        check(len(tokens) == len(tags))
        out = []
        last_label = None
        for index, tag in enumerate(tags):
            label = ConllLabel(tag)
            # We need to make two decisions, whether to add start marker, whether to add end marker.
            if label.is_close(last) and last is not None and self.care(last):
                out.add(self.sep)
                out.add(last_label.get_name())
                out.add(self.end)

            if label.is_start() and self.care(last):
                out.add(self.start)

            out.add(tokens[index])
            last_label = label
        return " ".join(out)


class Conll03OneSlotConverter(TrainConverter, ABC):
    def __init__(self, prompt, care):
        self.prompt = prompt
        self.label_to_id = {
            "O": 0,
            "B-ORG": 1,
            "B-MISC": 2,
            "B-PER": 3,
            "I-PER": 4,
            "B-LOC": 5,
            "I-ORG": 6,
            "I-MISC": 7,
            "I-LOC": 8
        }
        pairs = list(self.label_to_id.items()).sort(lambda x: x[1])
        self.id_to_label = list(map(lambda x: x[0], pairs))

        self.care = care
        self.build_label = ConllLabelBuilder()

    def __call__(self, batch, ins: list[str], outs: list[str]):
        # We assume the input is dict version of AnnotatedExemplar
        for idx, tokens in enumerate(batch["tokens"]):
            tags = batch["tags"][idx]
            label = ConllLabel(tag)

            input_dict = {"utterance": " ".join(tokens)}
            input_dict.update(ConllLabel.label_info[self.care])

            build_label = ConllLabelBuilder(tokens, tags, [self.care])

            # without the values for conll.
            input_dict["values"] = []
            ins.append(self.prompt(input_dict))
            outs.append(f"{build_label()}</s>")