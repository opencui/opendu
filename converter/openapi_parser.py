#!/usr/bin/env python3

from abc import ABC
import json
import yaml
from typing import Dict, List, TypedDict, Union

from datasets import Dataset
from core.commons import SkillInfo, DatasetFactory, SlotInfo, ModelInfo


#
# This is used to create dataset need for build index from OpenAPI specs.
#
class OpenAPIParser(DatasetFactory, ABC):
    def __init__(self, specs) -> None:
        self.exemplars = []

        skills = {}
        slots = {}
        for path, v in specs.get("paths", {}).items():
            for op, _v in v.items():
                label = f"{path}.{op}"
                f = {}
                name = _v.get("operationId", "")
                description = _v.get("description", "")
                if description == "":
                    description = _v.get("summary", "")
                parameters = []
                for _p in _v.get("parameters", []):
                    slot_name = _p.get("name", "")
                    slot_description = _p.get("description", "")
                    if slot_name not in slots:
                        slots[slot_name] = SlotInfo(slot_name, slot_description)
                    parameters.append(slot_name)
                skills[label] = SkillInfo(name, description, parameters)
        self.domain = ModelInfo(skills, slots)

    def build(self, split) -> Dataset:
        return Dataset.from_list(self.exemplars)


if __name__ == "__main__":
    s = OpenAPIParser(json.load(open("./converter/openai_examples/openapi_petstore_v3.1.json")))
    print(s.domain)


