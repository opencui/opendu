#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
from collections import defaultdict

from datasets import Dataset, Features, Value

from opencui.core.annotation import (CamelToSnake, FrameSchema, Schema, SlotSchema)
from opencui.finetune.commons import DatasetFactory, create_full_exemplar, purge_dataset


# pip install -U gin-config faiss-cpu scikit-learn sentence-transformers
# python3 generate_intent.py --input=/home/sean/src/dstc8-schema-guided-dialogue/train/ --output=./res/train
#
# We only care about the first in the same service family (by different business) so that the semantic competition
# replicate the use case that most businesses face. (Only gatekeeper need to deal with more than one business providing
# the same service.
#
class SGD(DatasetFactory):
    intent_taboo_word = ["SearchOnewayFlight"]

    # Which schema do we use? Default to train.
    def __init__(self, base_path, domain="train", suffix: str = "_1"):
        self.base_path = base_path
        self.tag = "sgd"
        self.suffix = suffix
        self.counts = [0, 0, 0, 0, 0, 0, 0]
        self.schema = load_schema_as_dict(f"{base_path}/{domain}")
        self.known_skills = set()
        self.bad_turns = set(
            [
                "sgd.train.95_00094.4",
                "sgd.train.95_00065.8",
                "sgd.train.95_00066.12",
                "sgd.train.93_00083.8",
                "sgd.train.8_00121.8",
                "sgd.train.54_00096.6",
                "sgd.train.121_00036.10",
                "sgd.train.52_00020.18",
                "sgd.train.74_00081.16",
                "sgd.train.90_00018.16",
                " " "sgd.train.27_00113.10",
                "sgd.train.75_00023.20",
                "sgd.train.96_00019.12",
                "sgd.train.96_00048.4",
                "sgd.train.96_00048.6",
                "sgd.train.96_00019.10",
                "sgd.train.42_00002.6",
                "sgd.dev.9_00058.0",
            ]
        )
        self.features = Features(
            {
                "title": Value(dtype="string", id=None),
                "utterance": Value(dtype="string", id=None),
                "template": Value(dtype="string", id=None),
                "owner": Value(dtype="string", id=None),
                "arguments": Value(dtype="string", id=None),
                "expectations": Value(dtype="string", id=None),
            }
        )

    def __getitem__(self, split):
        if split == "validation":
            split = "dev"

        base_path = f"{self.base_path}/{split}/"

        def gen_skills():
            """
            load original sgd data and create expression examples
            :param base_path: input path to original sgd dataset
            :return: expression examples
            """
            files = os.listdir(base_path)
            s_set = defaultdict(set)
            # For all files.
            for file in files:
                if file[:6] != "dialog":
                    continue
                with open(base_path + file, encoding="utf-8") as f:
                    f = json.load(f)
                    # For all sessions.
                    for dialogue in f:
                        turns = dialogue["turns"]
                        dialogue_id = dialogue["dialogue_id"]

                        # For each session with multiple turns.
                        pre_intents = set()
                        existing_intents = set()
                        actions = None
                        for idx, turn in enumerate(turns):
                            try:
                                # Getting actions.
                                if turn["speaker"] != "USER":
                                    actions = SGD.extract_actions(turn)
                                    continue

                                id = f"sgd.{split}.{dialogue_id}.{idx}"
                                if id in self.bad_turns:
                                    continue

                                active_intents = set()
                                for frame in turn["frames"]:
                                    active_intents.add(frame["state"]["active_intent"])

                                if len(active_intents) > 1:
                                    continue

                                # this is offered intent from system.
                                if (
                                    idx - 1 >= 0
                                    and turns[idx - 1]["frames"][0]["actions"][0]["act"] == "OFFER_INTENT"
                                ):
                                    offered_intent = set(turns[idx - 1]["frames"][0]["actions"][0]["values"])
                                else:
                                    offered_intent = set()

                                # if active_intents is carried over, we ignore.
                                if not (active_intents - pre_intents):
                                    self.counts[0] += 1
                                    # This might be the place where we create slot only examples.
                                    # Particularly assume there are reactive understanding.
                                    continue

                                frame = turn["frames"][0]
                                skill_name = frame["state"]["active_intent"]
                                # skill_name = self.forward[skill_name]
                                skill_name = CamelToSnake.convert(skill_name)

                                if skill_name not in self.schema.skills.keys():
                                    continue

                                if skill_name in existing_intents:
                                    # print(frame)
                                    continue

                                if not frame["service"].endswith(self.suffix):
                                    # Only care the servie with suffix _1
                                    continue

                                if frame["state"]["active_intent"] not in (
                                    active_intents - pre_intents
                                ):
                                    self.counts[1] += 1
                                    continue

                                if frame["state"]["active_intent"] in offered_intent:
                                    self.counts[3] += 1
                                    continue

                                if (
                                    frame["state"]["active_intent"]
                                    in SGD.intent_taboo_word
                                ):
                                    continue

                                spans = []
                                utterance = turn["utterance"].lower()
                                local_slots = defaultdict(list)
                                for _slot in frame["slots"]:
                                    local_slots[_slot["slot"]].append(
                                        utterance[
                                            _slot["start"] : _slot["exclusive_end"]
                                        ]
                                    )
                                    spans.append(
                                        (_slot["start"], _slot["exclusive_end"])
                                    )

                                exemplar = create_full_exemplar(
                                    id, utterance, skill_name, dict(local_slots), spans
                                )
                                # yield the example
                                yield exemplar.flatten()
                            finally:
                                # remember the active intents from last user turn.
                                pre_intents = active_intents
                                existing_intents.update(active_intents)

        return Dataset.from_generator(gen_skills)

    @classmethod
    def extract_actions(cls, turn):
        if turn["speaker"] != "SYSTEM":
            return None
        return turn["frames"][0]["actions"]


def load_schema_as_dict(full_path, suffix: str = "_1"):
    domain = Schema(skills={}, slots={})
    with open(f"{full_path}/schema.json", encoding="utf-8") as f:
        f = json.load(f)

        for service in f:
            service_name = service["service_name"]

            if service_name.endswith(suffix):
                continue

            # handle intents
            intents = service["intents"]
            for intent in intents:
                skill_name = intent["name"]
                intent_desc = intent["description"]
                slots = intent["required_slots"]
                optional_slots = intent["optional_slots"].keys()
                slots.extend(list(optional_slots))

                if skill_name in SGD.intent_taboo_word:
                    continue
                skill_name = CamelToSnake.convert(skill_name)
                domain.skills[skill_name] = FrameSchema(
                    skill_name, intent_desc, slots
                ).to_dict()
            slots = service["slots"]
            for slot in slots:
                slot_name = slot["name"]
                is_categorical = slot["is_categorical"]
                possible_values = slot["possible_values"]
                slot_description = slot["description"]
                domain.slots[slot_name] = SlotSchema(
                    slot_name, slot_description
                ).to_dict()
    return domain


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    output = "./index/sgdskill/"
    dsc = SGD("/home/sean/src/dstc8-schema-guided-dialogue/")

    with open(f"./datasets/sgd/schema.json", "w") as file:
        json.dump(dsc.schema.to_dict(), file, indent=2)

    print(f"there are {len(dsc.schema.skills)} skills")
    print(json.dumps(dsc.schema.skills, indent=2))
    # print(json.dumps(dsc.schema.slots, indent=2))

    for tag in ["train", "test", "validation"]:
        dataset = dsc[tag]
        examples = purge_dataset(dataset)
        with open(f"./datasets/sgd/{tag}.jsonl", "w") as file:
            print(f"there are {len(examples)} examples left for {tag}.")
            for example in examples:
                file.write(f"{json.dumps(example)}\n")

