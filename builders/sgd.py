#!/usr/bin/env python
# coding: utf-8

import json
import os
import sys
from collections import defaultdict

from datasets import IterableDataset
from core.commons import DomainInfo, SkillInfo, SlotInfo, DatasetCreator, Expression

# pip install -U gin-config faiss-cpu scikit-learn sentence-transformers
# python3 generate_intent.py --input=/home/sean/src/dstc8-schema-guided-dialogue/train/ --output=./res/train


#
# We only care about the first in the same service family (by different business) so that the semantic competition
# replicate the use case that most businesses face. (Only gatekeeper need to deal with more than one business providing
# the same service.
#
def load_schema(base_path, split):
    domain = DomainInfo(skills={}, slots={})
    with open(f"{base_path}/{split}/schema.json", encoding='utf-8') as f:
        f = json.load(f)

        for service in f:
            service_name = service["service_name"]
            # handle intents
            intents = service["intents"]
            for intent in intents:
                intent_label = f"{service_name}.{intent['name']}"
                intent_name = intent['name']
                intent_desc = intent["description"]
                slots = intent["required_slots"]
                optional_slots = intent["optional_slots"].keys()
                slots.extend(list(optional_slots))
                slots = [f"{service_name}.{slot}" for slot in slots]
                domain.skills[intent_label] = SkillInfo(intent_name, intent_desc, slots).to_dict()
            slots = service["slots"]
            for slot in slots:
                slot_label = f"{service_name}.{slot['name']}"
                slot_name = slot['name']
                is_categorical = slot['is_categorical']
                possible_values = slot['possible_values']
                slot_description = slot["description"]
                domain.slots[slot_label] = SlotInfo(slot_name, slot_description, is_categorical, possible_values).to_dict()
    return domain


class SGD:
    @classmethod
    def extract_actions(cls, turn):
        if turn["speaker"] != "SYSTEM":
            return None
        return turn["frames"][0]["actions"]


class SGDSKills(DatasetCreator):

    def __init__(self, base_path, domain):
        self.base_path = base_path
        self.tag = "sgd/skill"
        self.counts = [0, 0, 0, 0]

    def build(self, split):
        if split == "validation":
            split = "dev"

        base_path = f"{self.base_path}/{split}/"

        def gen():
            """
            load original sgd data and create expression examples
            :param base_path: input path to original sgd dataset
            :return: expression examples
            """
            files = os.listdir(base_path)
            sentence_set = defaultdict(set)
            # For all files.
            for file in files:
                if file[:6] != 'dialog':
                    continue
                with open(base_path + file, encoding='utf-8') as f:
                    f = json.load(f)
                    # For all sessions.
                    for dialogue in f:
                        turns = dialogue["turns"]

                        # For each session with multiple turns.
                        pre_intents = set()
                        actions = None
                        for idx, turn in enumerate(turns):
                            # Getting actions.
                            if turn['speaker'] != 'USER':
                                actions = SGD.extract_actions(turn)
                                continue

                            active_intents = set()
                            for frame in turn['frames']:
                                active_intents.add(frame['state']['active_intent'])

                            if idx - 1 >= 0 and turns[idx - 1]["frames"][0]["actions"][0]["act"] == "OFFER_INTENT":
                                offered_intent = set(turns[idx - 1]["frames"][0]["actions"][0]["values"])
                            else:
                                offered_intent = set()

                            # if active_intents is carried over, we ignore.
                            if not (active_intents - pre_intents):
                                self.counts[0] += 1
                                continue

                            frame = turn['frames'][0]
                            if frame['state']['active_intent'] not in (active_intents - pre_intents):
                                self.counts[1] += 1
                                continue

                            if frame['state']['active_intent'] == 'NONE':
                                self.counts[2] += 1
                                continue

                            if frame['state']['active_intent'] in offered_intent:
                                self.counts[3] += 1
                                continue

                            skill_label = f"{frame['service']}.{frame['state']['active_intent']}"
                            spans = []
                            utterance = turn['utterance'].lower()
                            local_slots = defaultdict(list)
                            for _slot in frame['slots']:
                                local_slots[_slot['slot']].append(utterance[_slot['start']:_slot['exclusive_end']])
                                spans.append((_slot['start'], _slot['exclusive_end']))

                            # remember the active intents from last user turn.
                            pre_intents = active_intents
                            # yield the example
                            yield Expression(utterance, skill_label, local_slots, spans).to_dict()
        return IterableDataset.from_generator(gen)


class SGDSlots(DatasetCreator):

    def __init__(self, base_path, domain):
        self.base_path = base_path
        self.tag = "sgd/slot"
        self.counts = [0, 0, 0, 0]

    def build(self, split):
        if split == "validation":
            split = "dev"

        base_path = f"{self.base_path}/{split}/"

        def gen():
            """
            load original sgd data and create expression examples
            :param base_path: input path to original sgd dataset
            :return: expression examples
            """
            files = os.listdir(base_path)
            sentence_set = defaultdict(set)
            # For all files.
            for file in files:
                if file[:6] != 'dialog':
                    continue
                with open(base_path + file, encoding='utf-8') as f:
                    f = json.load(f)
                    # For all sessions.
                    for dialogue in f:
                        turns = dialogue["turns"]

                        # For each session with multiple turns.
                        pre_intents = set()
                        actions = None
                        for idx, turn in enumerate(turns):
                            # Getting actions.
                            if turn['speaker'] != 'USER':
                                actions = SGD.extract_actions(turn)
                                continue

                            active_intents = set()
                            for frame in turn['frames']:
                                active_intents.add(frame['state']['active_intent'])

                            if idx - 1 >= 0 and turns[idx - 1]["frames"][0]["actions"][0]["act"] == "OFFER_INTENT":
                                offered_intent = set(turns[idx - 1]["frames"][0]["actions"][0]["values"])
                            else:
                                offered_intent = set()

                            # if active_intents is carried over, we ignore.
                            if not (active_intents - pre_intents):
                                self.counts[0] += 1
                                continue

                            frame = turn['frames'][0]
                            if frame['state']['active_intent'] not in (active_intents - pre_intents):
                                self.counts[1] += 1
                                continue

                            if frame['state']['active_intent'] == 'NONE':
                                self.counts[2] += 1
                                continue

                            if frame['state']['active_intent'] in offered_intent:
                                self.counts[3] += 1
                                continue

                            skill_label = f"{frame['service']}.{frame['state']['active_intent']}"
                            spans = []
                            utterance = turn['utterance'].lower()
                            local_slots = defaultdict(list)
                            for _slot in frame['slots']:
                                local_slots[_slot['slot']].append(utterance[_slot['start']:_slot['exclusive_end']])
                                spans.append((_slot['start'], _slot['exclusive_end']))

                            # remember the active intents from last user turn.
                            pre_intents = active_intents
                            # yield the example
                            yield Expression(utterance, skill_label, local_slots, spans).to_dict()
        return IterableDataset.from_generator(gen)



if __name__ == '__main__':

    path = "/home/sean/src/dstc8-schema-guided-dialogue/"
    split = "train"

    domains = load_schema(path, split)

    skill = SGDSKills(path, domains)
    dataset = skill.build("train")
    count = 0
    for item in dataset:
        count += 1
        #print(item)
    print(skill.counts)
    print(count)