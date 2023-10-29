import json
from abc import ABC

import abc
from dataclasses import dataclass, field
from functools import reduce

from dataclasses_json import dataclass_json
from datasets import Dataset, concatenate_datasets


class Config:
    embedding_device = "cpu"
    embedding_model = 'BAAI/llm-embedder'
    retriever_mode = "embedding"
    desc_retrieve_topk = 8
    exemplar_retrieve_topk = 16
    llm_device = "cpu"


@dataclass
@dataclass_json
class Expression:
    """
    expression examples
    """
    id: str = field(metadata={"required": True})
    utterance: str = field(metadata={"required": True})
    target_intent: str = field(metadata={"required": True})
    target_slots: dict[str, str] = field(metadata={"required": True})
    spans: list[tuple[int, int]] = field(metadata={"required": True})
    exemplar: str = field(metadata={"required": True})

    def __init__(self, id, utterance, intent, slots, spans):
        self.id = id
        self.utterance = utterance
        self.target_intent = intent
        self.target_slots = slots  # dict to store slot, value pairs
        self.spans = spans
        self.exemplar = Expression.generate_expression_template(utterance, slots, spans)

    @classmethod
    def generate_expression_template(cls, utterance, slot_dict, spans):
        '''
        replacing the slot val with the slot name,to avoid match the short slot val which may be included in other
        long slot val, we need sort by the length of the slot val
        '''
        if spans == []:
            return utterance
        single_dict = dict()

        for key, values in slot_dict.items():
            for value in values:
                single_dict[value] = key

        spans = sorted(spans, key=lambda x: x[0])
        res_utterance = utterance[:spans[0][0]]
        for i, (cur_start, cur_end) in enumerate(spans):
            # if len(string_list) >=2:
            #     print("sub string",utterance[cur_start:cur_end])
            res_utterance = res_utterance + ' < ' + single_dict[utterance[cur_start:cur_end]] + ' > '
            if i == len(spans) - 1:
                res_utterance = res_utterance + utterance[cur_end:]
            else:
                res_utterance = res_utterance + utterance[cur_end:spans[i + 1][0]]

        return res_utterance


@dataclass
@dataclass_json
class Exemplar:
    """
    expression examples
    """
    target_intent: str = field(metadata={"required": True})
    exemplar: str = field(metadata={"required": True})

    def __init__(self, exemplar, intent):
        self.exemplar = exemplar
        self.target_intent = intent


@dataclass_json
@dataclass
class SlotInfo:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    closed: bool = field(metadata={"required": True}, default=False)
    possible_values: list[str] = field(metadata={"required": True}, default_factory=list)


@dataclass_json
@dataclass
class SkillInfo:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    slots: list[str] = field(metadata={"required": True})


@dataclass_json
@dataclass
class ModelInfo:
    skills: dict[str, SkillInfo]
    slots: dict[str, SlotInfo]


#
# This assumes the dataset has skills, skill_descriptions, slots, slot_descriptions
# Then user utterance as input, and output.
#
@dataclass
class Prompt:
    __metaclass__ = abc.ABCMeta
    extra_tokens: list[str] = field(default_factory=list)

    @abc.abstractmethod
    def __call__(self, item: dict[str, str]) -> str:
        # Expecting: utterance, [skills, slots, examples]
        return


#
# This is need to create the different dataset based on prompt templating.
# We expect the input dataset has utterance field.
# We need to make sure the output dataset has input/output field,
@dataclass
class DatasetFactory(ABC):
    __metaclass__ = abc.ABCMeta
    domain: ModelInfo

    @abc.abstractmethod
    def build(self, split: str = "train") -> Dataset:
        """This return the domain meta needed."""
        return


@dataclass
class DatasetsCreator(ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dscs=list[DatasetFactory]):
        self.domain = ModelInfo(
            skills=reduce(lambda x, y: {**x, **y}, [dsc.domain.skills for dsc in dscs]),
            slots=reduce(lambda x, y: {**x, **y}, [dsc.domain.target_slots for dsc in dscs])
        )
        self.dscs = dscs

    def build(self, split):
        datasets = [dsc.build(split) for dsc in self.dscs]
        return concatenate_datasets(**datasets)


if __name__ == "__main__":
    print(Config.embedding_model)

