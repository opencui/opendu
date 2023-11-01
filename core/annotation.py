import json
from dataclasses import field
from enum import Enum
from typing import Union, List, TypedDict, Optional, Dict, Literal
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from llama_index.schema import TextNode
from typing_extensions import Annotated
from pydantic import BaseModel, Field

@dataclass_json
@dataclass
class SlotSpec:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})


@dataclass_json
@dataclass
class SkillSpec:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    slots: list[str] = field(default_factory=list)


@dataclass_json
@dataclass
class ModuleSpec:
    skills: Dict[str, SkillSpec]
    slots: Dict[str, SlotSpec]


class EntityInstance(BaseModel):
    label: str = Field(description='the canonical form of the instance')
    expressions: List[str] = Field(description="the expressions used to identify this instance.")


class ListRecognizer(BaseModel):
    rec_type: Literal['list']
    name: str = Field(description="language dependent name")
    description: Optional[str] = Field(description="define what is this type for.")
    instances: List[EntityInstance]


class PatternRecognizer(BaseModel):
    rec_type: Literal['pattern']
    name: str = Field(description="language dependent name")
    description: Optional[str] = Field(description="define what is this type for.")
    pattern: str = Field(description="regex pattern to recognize the instance of this entity.")


class SlotRecognizers(BaseModel):
    slots: Dict[str, str] = Field(description="the mapping from slot name to entity name")
    recognizers: Dict[str, Annotated[Union[ListRecognizer, PatternRecognizer], Field(discriminator='rec_type')]] = Field(description="the name to recognizer")


# owner is not needed if exemplars are listed insider function specs.
class Exemplar(BaseModel):
    owner: Optional[str] = Field(description="onwer of this exemplar.")
    context: Optional[str] = Field(description="the context under which this exemplar works.")
    template: str = Field(description="the example utterance that should trigger the given skill")


# There are two different use cases for exemplars:
# During fine-turning, we need both utterance and exemplars.
# During index, we only need exemplars.
class ExemplarStore(TypedDict):
    name: str
    exemplars: List[Exemplar]


def build_nodes_from_exemplar_store(store: ExemplarStore) -> list[TextNode]:
    nodes = []
    for label, exemplars in store.items():
        for exemplar in exemplars:
            nodes.append(
                TextNode(
                    text=exemplar.template,
                    id_=str(hash(exemplar.template))[1:13],
                    metadata={"target_intent": label, "context": exemplar.context},
                    excluded_embed_metadata_keys=["target_intent", "context"]))
    return nodes


class SemanticStructure(BaseModel):
    name: str = Field(description="the name of the function/skill/intent")
    arguments: TypedDict = Field(description="argument for callable object")


if __name__ == "__main__":
    print(json.dumps(ExemplarStore.model_json_schema(), indent=2))
