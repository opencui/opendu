import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict, Union
from typing import Optional
from dataclasses_json import dataclass_json
from pydantic import BaseModel, Field
from enum import Enum
from llama_index.core.schema import TextNode


@dataclass_json
@dataclass
class SlotSchema:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    type: str = field(metadata={"required": False}, default=None)
    label: Optional[str] = None

    def __getitem__(self, item):
        match item:
            case "description":
                return self.description
            case "name":
                return self.name
            case "type":
                return self.type
            case "label":
                return self.label
            case _:
                raise RuntimeError("wrong property.")


@dataclass_json
@dataclass
class FrameSchema:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    slots: list[str] = field(default_factory=list)
    headSlot: str = field(metadata={"required": False}, default=None)
    type: Optional[str] = None

    def __getitem__(self, item):
        match item:
            case "type":
                self.type
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
    name: str


# This name inside the FrameSchema and SlotSchema is semantic bearing.
# So there should not be overlapping between schema names.
# the key for skills and slots does not need to be.
@dataclass_json
@dataclass
class Schema:
    skills: Dict[str, FrameSchema]
    slots: Dict[str, SlotSchema]

    def __init__(self, skills, slots):
        self.skills = skills
        self.slots = slots

    def get_skill(self, frame_id: FrameId):
        return self.skills[frame_id.name]

    def has_skill(self, frame_id: FrameId):
        return frame_id.name in self.skills

    def get_slots_in_dict(self, frame_name: str) -> dict:
        res = {}
        frame = self.skills[frame_name]
        for slot in frame.slots:
            slot_schema = self.slots[slot]
            if slot_schema.type not in self.skills:
                res[slot_schema.name] = vars(slot_schema)
            else:
                res[slot_schema.name] = self.get_slots_in_dict(slot_schema.type)
        return res


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
    frame: str
    slot: str
    slotType: str


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
    owner_mode: Optional[str] = Field(
        description="the matching mode between template and owner",
        default=None
    )
    context_frame: Optional[str] = Field(
        description="the context slot under which this exemplar works.",
        default=None
    )
    context_slot: Optional[str] = Field(
        description="the context slot under which this exemplar works.",
        default=None
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
        return slot_meta.name


class MatchReplace:
    def __init__(self, replace):
        self.replace = replace
    def __call__(self, match):
        label = match.group(1)
        slot_name = self.replace(label)
        return f"< {slot_name} >"


def build_nodes_from_exemplar_store(module_schema: Schema, store: ExemplarStore, nodes: List[TextNode]):
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

if __name__ == "__main__":
    print(json.dumps(ExemplarStore.model_json_schema(), indent=2))


