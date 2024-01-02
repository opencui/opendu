import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict, Union

from dataclasses_json import dataclass_json
from llama_index.schema import TextNode
from pydantic import BaseModel, Field
from enum import Enum


class ModelType(Enum):
    t5 = 1
    gpt = 2
    llama = 3

    # This normalizes type to t5/gpt/bert (potentially)
    @staticmethod
    def normalize(model_in_str):
        if ModelType[model_in_str] == ModelType.llama:
            return ModelType.gpt
        return ModelType[model_in_str]


@dataclass_json
@dataclass
class SlotSchema:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    type: str = field(metadata={"required": False}, default=None)

    def __getitem__(self, item):
        match item:
            case "description":
                return self.description
            case "name":
                return self.name
            case "type":
                return self.type
            case _:
                raise RuntimeError("wrong property.")


@dataclass_json
@dataclass
class FrameSchema:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    slots: list[str] = field(default_factory=list)

    def __getitem__(self, item):
        match item:
            case "description":
                return self.description
            case "name":
                return self.name
            case "slots":
                return self.slots
            case _:
                raise RuntimeError("wrong property.")

    def __setitem__(self, key, value):
        if key == "name":
            self.name = value
        else:
            raise RuntimeError("wrong property.")


@dataclass_json
@dataclass
class FrameId:
    module: str
    name: str


# This name inside the FrameSchema and SlotSchema is semantic bearing.
# So there should not be overlapping between schema names.
# the key for skills and slots does not need to be.
@dataclass_json
@dataclass
class Schema:
    skills: Dict[str, FrameSchema]
    slots: Dict[str, SlotSchema]
    # We use snake case inside, so we need this to get back the original name.
    backward: Dict[str, str]

    def __init__(self, skills, slots, backward=None):
        self.skills = skills
        self.slots = slots
        self.backward = backward


@dataclass_json
@dataclass
class SchemaStore:
    schemas: Dict[str, Schema]

    def __init__(self, schemas: dict[str, Schema]):
        self.schemas = schemas
        self.func_to_module = {
            skill["name"]: schema
            for schema in schemas.values()
            for skill in schema.skills.values()
        }

    def get_skill(self, frame_id: FrameId):
        module = self.schemas[frame_id.module]
        return module.skills[frame_id.name]

    def get_module(self, func_name):
        return self.func_to_module[func_name]

    def has_module(self, func_name):
        return func_name in self.func_to_module


OwnerMode = Enum('OwnerMode', ["normal", "extended"])


# This considers the match under the exact sense.
class ExactMatcher:
    @staticmethod
    def agree(owner, owner_mode, target, target_mode):
        label_match = owner == target
        if not label_match:
            return label_match

        # We should not use this example.
        if OwnerMode[owner_mode] != OwnerMode.normal and OwnerMode[target_mode] != OwnerMode.normal:
            return None

        # now we have match, but mode does not match.
        return OwnerMode[owner_mode] == OwnerMode.normal and OwnerMode[target_mode] == OwnerMode.normal

    @staticmethod
    def match(owner, target, mode_in_str):
        return owner == target and OwnerMode[mode_in_str] == OwnerMode.normal

    @staticmethod
    def is_good_mode(mode_in_str):
        return OwnerMode[mode_in_str] == OwnerMode.normal


@dataclass_json
@dataclass
class FrameValue:
    name: str
    arguments: TypedDict


@dataclass_json()
@dataclass
class FrameState:
    module: Schema
    frame: FrameSchema
    activated: list[str]


# How to present context is strictly state tracking implementation dependent.
@dataclass_json
@dataclass
class DialogExpectation(BaseModel):
    context: list[FrameState]


class EntityInstance(BaseModel):
    label: str = Field(description="the canonical form of the instance")
    expressions: List[str] = Field(
        description="the expressions used to identify this instance."
    )


class ListEntityInfo(BaseModel):
    rec_type: Literal["list"]
    name: str = Field(description="language dependent name")
    description: Optional[str] = Field(description="define what is this type for.")
    instances: List[EntityInstance]


# For now, we only worry about list entity in the python side, as it is mainly designed for function calling.
class EntityMetas(BaseModel):
    slots: Dict[str, str] = Field(
        description="the mapping from slot name to entity name"
    )
    recognizers: Dict[str, ListEntityInfo] = Field(description="the name to recognizer")


# For now, we do not handle normalization in understanding.
class ListRecognizer:
    def __init__(self, infos: Dict[str, ListEntityInfo]):
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


# owner is not needed if exemplars are listed insider function specs.
class Exemplar(BaseModel):
    owner: str = Field(description="onwer of this exemplar.")
    template: str = Field(
        description="the example utterance that should trigger the given skill"
    )
    owner_mode: str = Field(description="the matching mode between template and owner")
    context: Optional[str] = Field(
        description="the context under which this exemplar works.", default=None
    )


# There are two different use cases for exemplars:
# During fine-turning, we need both utterance and exemplars.
# During index, we only need exemplars.
class ExemplarStore(TypedDict):
    name: str
    exemplars: List[Exemplar]


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

    @classmethod
    def convert(cls, text):
        return CamelToSnake.pattern.sub("_", text).lower()

    def __init__(self):
        self.backward = {}
        self.forward = {}

    def encode(self, text):
        snake = CamelToSnake.pattern.sub("_", text).lower()
        self.backward[snake] = text
        self.forward[text] = snake
        return snake

    def decode(self, snake):
        return self.backward[snake]


def build_nodes_from_exemplar_store(module: str, store: ExemplarStore, nodes: List[TextNode]):
    to_snake = CamelToSnake()
    for label, exemplars in store.items():
        for exemplar in exemplars:
            label = to_snake.encode(label)
            nodes.append(
                TextNode(
                    text=exemplar["template"],
                    id_=str(hash(exemplar["template"]))[1:13],
                    metadata={
                        "owner": label,
                        "context": get_value(exemplar, "context", ""),
                        "owner_mode": get_value(exemplar, "owner_mode", "normal"),
                        "module": module,
                    },
                    excluded_embed_metadata_keys=["owner", "context", "module", "owner_mode"],
                )
            )


if __name__ == "__main__":
    print(json.dumps(ExemplarStore.model_json_schema(), indent=2))
