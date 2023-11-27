#!/usr/bin/env python3

import json
from core.annotation import ExemplarStore, SlotRecognizers, FrameSchema, SlotSchema, Schema, get_value


#
# This is used to create the DatasetCreator from OpenAI function descriptions.
#
# We assume that in each domain, the slot name are unique, and skill name are unique.
#
def from_openai(functions) -> Schema:
    skill_infos = {}
    slot_infos = {}
    for func in functions:
        f_name = func["name"]
        f_description = func["description"]
        f_slots = []
        parameters = func["parameters"]
        if parameters["type"] != "object":
            raise RuntimeError("Need to handle this case.")

        for key, slot in parameters["properties"].items():
            f_slots.append(key)
            if key in slot_infos:
                continue
            else:
                slot_name = key
                slot_description = slot["description"]
                slot_infos[slot_name] = SlotSchema(slot_name, slot_description)
        skill_infos[f_name] = FrameSchema(f_name, f_description, f_slots)
    return Schema(skill_infos, slot_infos)


def from_openapi(specs) -> Schema:
    skills = {}
    slots = {}
    print(specs)
    for path, v in specs["paths"].items():
        for op, _v in v.items():
            name = _v["operationId"]

            description = get_value(_v, "description")
            if description is None:
                description = get_value(_v, "summary")
            assert name is not None and description is not None

            parameters = []
            for _p in get_value(_v, "parameters", []):
                slot_name = get_value(_p, "name")
                slot_description = get_value(_p, "description")
                if slot_name not in slots:
                    slots[slot_name] = SlotSchema(slot_name, slot_description)
                parameters.append(slot_name)
            skills[name] = FrameSchema(name, description, parameters).to_dict()
    return Schema(skills, slots)


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
    schema = from_openai(json.load(open("./examples/schemas.json")))
    print(schema)
    print("\n")

    schema = from_openapi(json.load(open("./examples/openapi_petstore_v3.1.json")))
    print(schema)
    print("\n")

    exemplars = ExemplarStore(**json.load(open("./examples/exemplars.json")))
    print(exemplars)
    print("\n")

    recognizer = SlotRecognizers(**json.load(open("./examples/recognizers.json")))
    print(recognizer)
    print("\n")
