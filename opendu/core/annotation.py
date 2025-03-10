# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
# This source code is licensed under the BeThere AI license.
# See LICENSE file in the project root for full license information.

import re
from typing import Dict, List, TypedDict, Set, Any
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
from llama_index.core.schema import TextNode


# During the understanding, we do not have concept of multivalued, as even when the slot
# is the single valued, user can still say multiple values.
class SlotSchema(BaseModel):
    name: str = Field(..., description="The name of the slot", title="Name")
    description: str = Field(..., description="Description of the slot")
    type: Optional[str] = Field(None, description="The type of the slot")
    label: Optional[str] = Field(None, description="Optional label for the slot")
    examples: Set[str] = Field(set(), description="Example values for the slot.")

    def __getitem__(self, item):
        return self.__dict__[item]

    def to_description_dict(self):
        return {field: self.model_field[field].field_info.description for field in self.model_fields}


class FrameSchema(BaseModel):
    name: str = Field(..., description="The name of the frame", title="Name")
    description: str = Field(..., description="Description of the frame")
    slots: List[str] = Field(default_factory=list, description="List of slot names in the frame")
    headSlot: Optional[str] = Field(None, description="Optional head slot")
    type: Optional[str] = Field(None, description="Optional type of the frame")

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


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


class EntityInstance(BaseModel):
    label: str = Field(description="the canonical form of the instance")
    expressions: List[str] = Field(
        description="the expressions used to identify this instance."
    )


class EntityType(BaseModel):
    name: Optional[str]= Field(None, description="language dependent name")
    description: Optional[str] = Field(None, description="define what is this type for.")
    instances: List[EntityInstance]

# EntityStore just need to be:  Dict[str, List[EntityInstance]]
EntityStore = Dict[str, EntityType]

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


# For now, we do not handle normalization in understanding.
class ListRecognizer:
    def __init__(self, infos: Dict[str, EntityType]):
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



