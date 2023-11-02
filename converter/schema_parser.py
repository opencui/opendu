#!/usr/bin/env python3

import json
from core.annotation import ExemplarStore, SlotRecognizers, FrameSchema, SlotSchema, ModuleSchema


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
                slotInfos[slot_name] = SlotSchema(slot_name, slot_description)
        skillInfos[f_name] = FrameSchema(f_name, f_description, f_slots)
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
                    slots[slot_name] = SlotSchema(slot_name, slot_description)
                parameters.append(slot_name)
            skills[label] = FrameSchema(name, description, parameters)
    return ModuleSchema(skills, slots)


# This assumes that in a directory we have schemas.json in openai/openapi format, and then exemplars
# recognizers.
def load_schema_from_directory(path):
    schema_object = json.load(open(path))
    return from_openai(schema_object) if isinstance(schema_object, list) else from_openapi(schema_object)


def load_all_from_directory(input_path):
    module_schema = load_schema_from_directory(f"{input_path}/schemas.json")
    examplers = ExemplarStore(**json.load(open(f"{input_path}/exemplars.json")))
    recognizers = SlotRecognizers(**json.load(open(f"{input_path}/recognizers.json")))
    return module_schema, examplers, recognizers

def load_specs_and_recognizers_from_directory(input_path):
    module_schema = load_schema_from_directory(f"{input_path}/schemas.json")
    recognizers = SlotRecognizers(**json.load(open(f"{input_path}/recognizers.json")))
    return module_schema, recognizers



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
