#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
import sys
from collections import defaultdict

from factories import IterableDataset
from core.commons import DomainInfo, SkillInfo, SlotInfo, DatasetFactory, Expression, Config
from core.retriever import build_desc_index, build_exemplar_index


# pip install -U gin-config faiss-cpu scikit-learn sentence-transformers
# python3 generate_intent.py --input=/home/sean/src/dstc8-schema-guided-dialogue/train/ --output=./res/train

class SGD:
    limited_to_first_service: bool = True

    @classmethod
    def extract_actions(cls, turn):
        if turn["speaker"] != "SYSTEM":
            return None
        return turn["frames"][0]["actions"]


#
# We only care about the first in the same service family (by different business) so that the semantic competition
# replicate the use case that most businesses face. (Only gatekeeper need to deal with more than one business providing
# the same service.
#
def load_schema_as_dict(full_path, suffix: str = "_1"):
    domain = DomainInfo(skills={}, slots={})
    with open(f"{full_path}/schema.json", encoding='utf-8') as f:
        f = json.load(f)

        for service in f:
            service_name = service["service_name"]

            if SGD.limited_to_first_service and service_name.endswith(suffix):
                continue

            # handle intents
            intents = service["intents"]
            for intent in intents:
                intent_name = intent['name']
                intent_desc = intent["description"]
                slots = intent["required_slots"]
                optional_slots = intent["optional_slots"].keys()
                slots.extend(list(optional_slots))
                slots = [f"{slot}" for slot in slots]
                domain.skills[intent_name] = SkillInfo(intent_name, intent_desc, slots).to_dict()
            slots = service["slots"]
            for slot in slots:
                slot_name = slot['name']
                is_categorical = slot['is_categorical']
                possible_values = slot['possible_values']
                slot_description = slot["description"]
                domain.slots[slot_name] = SlotInfo(slot_name, slot_description, is_categorical,
                                                   possible_values).to_dict()
    return domain


class SGDSkills(DatasetFactory):

    # Which schema do we use? Default to train.
    def __init__(self, base_path, domain="train", suffix: str = "_1"):
        self.base_path = base_path
        self.tag = "sgd/skill"
        self.suffix = suffix
        self.counts = [0, 0, 0, 0]
        self.domain = load_schema_as_dict(f"{base_path}/{domain}/")

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
                        dialogue_id = dialogue["dialogue_id"]

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
                            if SGD.limited_to_first_service and frame["service"].endswith(self.suffix):
                                continue

                            if frame['state']['active_intent'] not in (active_intents - pre_intents):
                                self.counts[1] += 1
                                continue

                            if frame['state']['active_intent'] in offered_intent:
                                self.counts[3] += 1
                                continue

                            skill_name = frame['state']['active_intent']
                            spans = []
                            utterance = turn['utterance'].lower()
                            local_slots = defaultdict(list)
                            for _slot in frame['slots']:
                                local_slots[_slot['slot']].append(utterance[_slot['start']:_slot['exclusive_end']])
                                spans.append((_slot['start'], _slot['exclusive_end']))

                            # remember the active intents from last user turn.
                            pre_intents = active_intents
                            id = f"sgd.{split}.{dialogue_id}.{idx}"
                            # yield the example
                            yield Expression(id, utterance, skill_name, local_slots, spans).to_dict()

        return IterableDataset.from_generator(gen)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    output = "./index/sgdskill/"
    dsc = SGDSkills("/home/sean/src/dstc8-schema-guided-dialogue/")

    print(f"there are {len(dsc.domain.skills)} skills.")
    build_desc_index(dsc, output)
    build_exemplar_index(dsc, output)
