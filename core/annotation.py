import json
from enum import Enum
from typing import Union, List, TypedDict, Optional, Dict, Literal
from typing_extensions import Annotated
from pydantic import BaseModel, Field


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
    context: Optional[str] = Field(description="the context under which this exemplar works.")
    template: str = Field(description="the example utterance that should trigger the given skill")


# There are two different use cases for exemplars:
# During fine-turning, we need both utterance and exemplars.
# During index, we only need exemplars.
class ExemplarStore(TypedDict):
    name: str
    exemplars: List[Exemplar]


if __name__ == "__main__":
    print(json.dumps(ExemplarStore.model_json_schema(), indent=2))
