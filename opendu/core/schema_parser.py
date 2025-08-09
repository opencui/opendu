#!/usr/bin/env python3
# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.

import json
import os
from opendu.core.annotation import (CamelToSnake, EntitySchema, FrameSchema, Schema, SlotSchema, get_value)


#
# This is used to create the DatasetCreator from OpenAI function descriptions.
# We assume that in each domain, the slot name are unique, and skill name are unique.
#
def from_openai(functions) -> Schema:
    skill_infos = {}
    slot_infos = {}
    for func in functions:
        o_name = func["name"]
        f_name = CamelToSnake.encode(o_name)
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
                slot_infos[slot_name] = SlotSchema(
                    name=slot_name, description=slot_description
                )
        skill_infos[f_name] = FrameSchema(name=f_name, description=f_description, slots=f_slots)
    return Schema(skills=skill_infos, slots=slot_infos)


def from_openapi(specs) -> Schema:
    skills = {}
    slots = {}
    print(specs)
    for path, v in specs["paths"].items():
        for op, _v in v.items():
            orig_name = _v["operationId"]
            name = CamelToSnake.encode(orig_name)
            description = get_value(_v, "description")
            if description is None:
                description = get_value(_v, "summary")
            assert name is not None and description is not None

            parameters = []
            for _p in get_value(_v, "parameters", []):
                slot_name = get_value(_p, "name")
                slot_description = get_value(_p, "description")
                if slot_name not in slots:
                    slots[slot_name] = SlotSchema(name=slot_name, description=slot_description)
                parameters.append(slot_name)
            skills[name] = FrameSchema(name=name, description=description, slots=parameters)
    return Schema(skills=skills, slots=slots)


# This assumes that in a directory we have schemas.json in openai/openapi format, and then exemplars
# recognizers.
def load_schema_from_directory(path):
    schema_object = json.load(open(path))
    if isinstance(schema_object, list):
        l_schema = from_openai(schema_object)
    elif "slots" in schema_object and "skills" in schema_object:
        l_schema = Schema(**schema_object)
    else:
        l_schema = from_openapi(schema_object)
    return l_schema


def load_all_from_directory(input_path):
    module_schema = load_schema_from_directory(f"{input_path}/schemas.json")
    exemplars = json.load(open(f"{input_path}/exemplars.json"))
    if os.path.exists(f"{input_path}/recognizers.json"):
        recognizers = json.load(open(f"{input_path}/recognizers.json"))
    else:
        recognizers = None
    module_schema.updateNameToBeLabel()    
    return module_schema, exemplars, recognizers


def load_specs_and_recognizers_from_directory(input_path):
    module_schema = load_schema_from_directory(f"{input_path}/schemas.json")
    recognizers = json.load(open(f"{input_path}/recognizers.json"))
    return module_schema, recognizers


if __name__ == "__main__":
    schema = from_openai(json.load(open("./examples/openai/schemas.json")))
    print(schema)
    print("\n")

    schema = from_openapi(json.load(open("./examples/openapi_petstore_v3.1.json")))
    print(schema)
    print("\n")

    recognizer = json.load(open("./examples/openai/recognizers.json"))
    print(recognizer)
    print("\n")

    exemplars = json.load(open("./examples/openai/exemplars.json"))
    print(exemplars)
    print("\n")


