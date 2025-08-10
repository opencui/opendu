# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.

import random
import re
from typing import Dict, List, TypedDict, Set, Any
from typing import Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum
from llama_index.core.schema import TextNode
import json


class Schema(BaseModel):
    name: str = Field(..., description="The name of the schema, human readable, only simple name if not provided", title="Name")
    label: Optional[str] = Field(None, description="The label of the frame, fully qualified", title="Label")
    description: str = Field(..., description="Description of the frame")

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


# During the understanding, we do not have concept of multivalued, as even when the slot
# is the single valued, user can still say multiple values.
class SlotSchema(Schema):
    multi_value: bool = Field(False, description="Whether the slot can have multiple values")
    type: Optional[str] = Field(None, description="The type of the slot")
    examples: Set[str] = Field(set(), description="Contextual example values for the slot.")


class TypeSchema(Schema):
    pass


class FrameSchema(TypeSchema):
    slots: List[str] = Field(default_factory=list, description="List of slot names in the frame")
    headSlot: Optional[str] = Field(None, description="Optional head slot")
    type: Optional[str] = Field(None, description="Optional type of the frame")


class EntityInstance(BaseModel):
    label: str = Field(description="the canonical form of the instance")
    expressions: List[str] = Field(
        description="the expressions used to identify this instance."
    )


class EntitySchema(TypeSchema):
    enumable: Optional[bool] = Field(True, description="whether this type is enumable.")
    instances: Optional[List[EntityInstance]]

    @computed_field
    @property
    def examples(self) -> Set[str]:
        """Random sample of expressions from instances."""
        if not self.instances:
            return set()
        
        all_expressions = [
            expr for instance in self.instances 
            for expr in instance.expressions
        ]
        if not all_expressions:
            return set()
            
        sample_size = min(5, len(all_expressions))
        return set(random.sample(all_expressions, sample_size))
    

# EntityStore just need to be:  Dict[str, List[EntityInstance]]
EntityStore = Dict[str, EntitySchema]


#
# Owner is not needed if exemplars are listed insider function specs.
# This exemplar is used for intent detection only, as the template
# already delegate the entity detection out.
#
class Exemplar(BaseModel):
    owner: str = Field(description="onwer of this exemplar.")
    template: str = Field(
        description="the example utterance that should trigger the given skill"
    )
    owner_mode: Optional[str] = Field(None,
        description="the matching mode between template and owner",
    )
    context_frame: Optional[str] = Field(None,
        description="the context slot under which this exemplar works.",
    )
    context_slot: Optional[str] = Field(None,
        description="the context slot under which this exemplar works.",
    )
    arguments: Optional[Dict[str, Any]] = Field(None)

    def __getitem__(self, key):
        return self.__dict__[key]


# The exemplar store should simply just be a dict.
ExemplarStore = Dict[str, list[Exemplar]]


# This name inside the FrameSchema and SlotSchema is semantic bearing.
# So there should not be overlapping between schema names.
# the key for skills and slots does not need to be.
class Schema(BaseModel):
    skills: Dict[str, FrameSchema]
    slots: Dict[str, SlotSchema]

    def get_skill(self, frame_id: str):
        return self.skills[frame_id]

    def has_skill(self, frame_id: str):
        return frame_id in self.skills

    def updateNameToBeSimpleLabelIfNeeded(self):
        # we need to make sure
        for key, value in self.skills.items():
            value["label"] = key
            if value["name"] == "":
                value["name"] = key.split(".")[-1]

        for key, value in self.slots.items():
            value["label"] = key
            if value["name"] == "":
                value["name"] = key.split(".")[-1]        

    def updateSlotExamples(self, pickValueExamples):
        for example in pickValueExamples:
            frame = example["context_frame"]
            slot = example["context_slot"]
            if frame and slot:
                label = f"{frame}.{slot}"
                if label in self.slots:
                    self.slots[label].examples.add(example.template)


    def get_slots_descriptions_in_dict(self, frame_name: str) -> dict:
        res = {}
        frame = self.skills[frame_name]
        for slot in frame.slots:
            slot_schema = self.slots[slot]
            if slot_schema.type not in self.skills:
                print(f"slot_schema.type: {slot_schema.type} is not a skill")
                res[slot_schema.name] = slot_schema.description
            else:
                print(f"slot_schema.type: {slot_schema.type} is a skill")
                res[slot_schema.name] = self.get_slots_descriptions_in_dict(slot_schema.type)
        return res

    def get_slots_examples_in_dict(self, frame_name: str) -> dict:
        res = {}
        frame = self.skills[frame_name]
        for slot in frame.slots:
            slot_schema = self.slots[slot]
            if slot_schema.type not in self.skills:
                res[slot_schema.name] = list(slot_schema.examples)[:1]
            else:
                res[slot_schema.name] = self.get_slots_examples_in_dict(slot_schema.type)
        return res





# For now, we do not handle normalization in understanding.
class ListRecognizer:
    def __init__(self, infos: Dict[str, EntitySchema]):
        self.infos = infos
        self.patterns = {}
        for key, info in infos.items():
            instances = [item for instance in info for item in instance.expressions]
            self.patterns[key] = re.compile("|".join(map(re.escape, instances)))

    @staticmethod
    def find_matches(patterns, slot, utterance):
        if slot not in patterns:
            return []

        pattern = patterns[slot]
        return pattern.findall(utterance)

    def extract_values(self, slot, text):
        return ListRecognizer.find_matches(self.patterns, slot, text)


def get_value(item, key, value=None):
    try:
        return item[key]
    except:
        return value



#
# This is need to convert the camel casing to snake casing.
#
class CamelToSnake:
    pattern = re.compile(r"(?<!^)(?=[A-Z])")

    @staticmethod
    def encode(text):
        return CamelToSnake.pattern.sub("_", text).lower()

    @staticmethod
    def decode(word):
        tkns = word.split('_')
        return tkns[0] + ''.join(x.capitalize() or '_' for x in tkns[1:])


# This try to replace label with name so that it is easy to change DU behavior without touch label.
class ToSlotName:
    def __init__(self, module:Schema, owner:str):
        self.module = module
        self.owner = owner

    def __call__(self, label):
        full_label = f"{self.owner}.{label}"
        slot_meta = self.module.slots[full_label]
        return slot_meta["name"]


class MatchReplace:
    def __init__(self, replace):
        self.replace = replace
    def __call__(self, match):
        label = match.group(1)
        slot_name = self.replace(label)
        return f"< {slot_name} >"


def build_nodes_from_exemplar_store(module_schema: Schema, store: Dict[str, List[Exemplar]], nodes: List[TextNode]):
    pattern = re.compile(r"<(.+?)>")
    for label, exemplars in store.items():
        label_to_name = MatchReplace(ToSlotName(module_schema, label))
        for exemplar in exemplars:
            template = exemplar["template"]
            text = pattern.sub(label_to_name, template)

            if text.strip() == "":
                continue

            context_frame = get_value(exemplar, "context_frame", None)
            full_text = f'{text} : {label}'
            hashed_id = str(hash(full_text))
            nodes.append(
                TextNode(
                    text=text,
                    id_=hashed_id,
                    metadata={
                        "owner": label,
                        "template": template,  # This is the original template
                        "context_frame": context_frame,
                        "context_slot": get_value(exemplar, "context_slot", None),
                        "owner_mode": get_value(exemplar, "owner_mode", "normal"),
                    },
                    excluded_embed_metadata_keys=["owner", "context_frame", "context_slot", "owner_mode", "template"],
                )
            )


#
# This is used to build the JSON schema for the given frame and slots.
#
def build_json_schema(
    frame_dict: Dict[str, FrameSchema],
    slot_dict: Dict[str, SlotSchema],
    root_frame_name: str,
    root_multi_value: bool = False,
    include_deps: bool = False) -> Dict[str, Any]:
    visited_frames = set()
    visited_slots = set()
    components = {}

    def resolve_slot_type(slot_name: str) -> Dict[str, Any]:
        visited_slots.add(slot_name)
        slot = slot_dict[slot_name]

        # Base schema for the slot's underlying type
        if slot.type in {"string", "number", "boolean", "integer"}:
            base_schema = {"type": slot.type}
        elif slot.type in frame_dict:
            visited_frames.add(slot.type)
            # Reference to reusable nested frame schema
            base_schema = {"$ref": f"#/components/schemas/{slot.type}"}
            if slot.type not in components:
                components[slot.type] = resolve_frame(frame_dict[slot.type])
        else:
            base_schema = {"type": "string", "description": slot.type}

        # Wrap in array if multi_value is True
        if getattr(slot, "multi_value", False):
            array_schema = {
                "type": "array",
                "items": base_schema,
                "description": slot.description,
            }
            # Optionally, you can add examples here as well (array examples)
            if slot.examples:
                array_schema["examples"] = [list(slot.examples)]
            return array_schema
        else:
            # Single value slot
            base_schema["description"] = slot.description
            if slot.examples:
                base_schema["examples"] = sorted(slot.examples)
            return base_schema

    def resolve_frame(frame_name: str) -> Dict[str, Any]:
        if frame_name in frame_dict:
            frame = frame_dict[frame_name]
            props = {}
            required = []
            for slot_name in frame.slots:
                if slot_name in slot_dict:
                    props[slot_name] = resolve_slot_type(slot_name)
                    required.append(slot_name)
            return {
                "type": "object",
                "description": frame.description,
                "properties": props,
                "required": required,
                "additionalProperties": False,
            }
        else:
            return  {"type": "string", "description": frame_name}

    visited_frames.add(root_frame_name)
    root_schema = resolve_frame(root_frame_name)

    if root_multi_value:
        root_schema = {
            "title": root_frame_name,
            "type": "array",
            "items": root_schema,
        }
    else:
        root_schema = {
            "title": root_frame_name,
            **root_schema,
        }

    result = root_schema
    if (len(components) != 0):
        result["components"] = {"schemas": {name: schema for name, schema in components.items()}}

    if include_deps:
        result["$deps"] = {
            "frames": sorted(visited_frames),
            "slots": sorted(visited_slots),
        }

    return result


if __name__ == "__main__":
    slots = [
        SlotSchema(name="lat", description="Latitude", type="number", multi_value=True),
        SlotSchema(name="lng", description="Longitude", type="number"),
        SlotSchema(name="loc", description="GPS location", type="Coordinates", multi_value=True),
        SlotSchema(name="time", description="Time of day", type="string"),
        SlotSchema(name="irrelevant", description="unused slot", type="string"),
    ]

    frames = [
        FrameSchema(name="Coordinates", description="GPS Coordinates", slots=["lat", "lng"]),
        FrameSchema(name="WeatherQuery", description="Ask about weather", slots=["loc", "time"]),
        FrameSchema(name="UnusedFrame", description="Should not appear", slots=["irrelevant"]),
    ]

    slot_dict = {slot.name: slot for slot in slots}
    frame_dict = {frame.name: frame for frame in frames}

    json_schema = build_json_schema(frame_dict, slot_dict, "WeatherQuery", True)
    print(json.dumps(json_schema, indent=2, sort_keys=True))
