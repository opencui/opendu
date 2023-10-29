#!/usr/bin/env python3
import json
from abc import ABC
from datasets import Dataset
from core.commons import SkillInfo, DatasetFactory, SlotInfo, DomainInfo, Exemplar


#
# This is used to create the DatasetCreator from OpenAI function descriptions.
#
# We assume that in each domain, the slot name are unique, and skill name are unique.
#
class OpenAIParser(DatasetFactory, ABC):
    def __init__(self, functions) -> None:
        self.exemplars = []
        skillInfos = {}
        slotInfos = {}
        for func in functions:
            f_name = func["name"]
            f_description = func["description"]
            f_slots = []
            parameters = func["parameters"]
            if parameters["type"] != "object":
                raise RuntimeError("Need to handle this case.")

            for key, slot in parameters["properties"].items():
                f_slots.append(key)
                if key in slotInfos:
                    continue
                else:
                    slot_name = key
                    slot_description = slot["description"]
                    slotInfos[slot_name] = SlotInfo(slot_name, slot_description)
            skillInfos[f_name] = SkillInfo(f_name, f_description, f_slots)
        self.domain = DomainInfo(skillInfos, slotInfos)

    def build(self, split) -> Dataset:
        return Dataset.from_list(self.exemplars)


if __name__ == "__main__":
    openaids = OpenAIParser(json.load(open("./converter/examples/openai_example.json")))
    print(openaids.domain)

