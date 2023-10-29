import json
from enum import Enum
from typing import Union, List, TypedDict, Optional, Dict
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class AnnotatedModel(BaseModel):
    label: str = Field(description="language independent identifier in snake_case")
    domain: str = Field(description="module name")
    name: str = Field(description="language dependent name")
    description: str = Field(description="define what is this type for.")


class EntityInstance(BaseModel):
    label: str = Field(description='the canonical form of the instance')
    expressions: List[str] = Field(description="the expressions used to identify this instance.")


class ListEntity(AnnotatedModel):
    instances: List[EntityInstance]


class PatternEntity(AnnotatedModel):
    pattern: str = Field(description="regex pattern to recognize the instance of this entity.")


class SlotStore(BaseModel):
    slot_types: Dict[str, str] = Field(description="the mapping from slot name to entity name")
    entities: Dict[str, Union[ListEntity, PatternEntity]] =  Field(description="the name to recognizer")


# owner is not needed if exemplars are listed insider function specs.
class Exemplar(BaseModel):
    context: Optional[str] = Field(description="the context under which this exemplar works.")
    template: str = Field(description="the example utterance that should trigger the given skill")


class ExampledModel(BaseModel):
    owner: Optional[str] = Field(description="symbolic representation of the skill.")
    exemplars: List[Exemplar] = Field(description="The type with natural language example")


# There are two different use cases for exemplars:
# During fine-turning, we need both utterance and exemplars.
# During index, we only need exemplars.
class ExampleStore(BaseModel):
    examples: Dict[str, ExampledModel] = Field(description="this is for all the ")


if __name__ == "__main__":
    print(json.dumps(ExampledModel.model_json_schema(), indent=2))
