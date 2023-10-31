#!/usr/bin/env python3

import json
from core.annotation import ExemplarStore, SlotRecognizers
from core.commons import SkillInfo, DatasetFactory, SlotInfo, ModuleSchema


#
# This is used to create the DatasetCreator from OpenAI function descriptions.
#
# We assume that in each domain, the slot name are unique, and skill name are unique.
#
def from_openai(functions) -> ModuleSchema:
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
    return ModuleSchema(skillInfos, slotInfos)


def from_openapi(specs) -> ModuleSchema:
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
    return ModuleSchema(skills, slots)


if __name__ == "__main__":
    schema = from_openai(json.load(open("./converter/examples/schemas.json")))
    print(schema)
    print("\n")

    exemplars = ExemplarStore(**json.load(open("./converter/examples/exemplars.json")))
    print(exemplars)
    print("\n")

    recognizer = SlotRecognizers(**json.load(open("./converter/examples/recognizers.json")))
    print(recognizer)
    print("\n")
