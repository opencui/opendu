import json
from abc import ABC

import abc
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
from datasets import Dataset

@dataclass
@dataclass_json
class Expression:
    """
    expression examples
    """
    utterance: str = field(metadata={"required": True})
    skill_label: str = field(metadata={"required": True})
    slots: dict[str, str] = field(metadata={"required": True})
    spans: list[tuple[int, int]] = field(metadata={"required": True})
    exemplar: str = field(metadata={"required": True})

    def __init__(self, utterance, intent, slots, spans):
        self.utterance = utterance
        self.skill_label = intent
        self.slots = slots  # dict to store slot, value pairs
        self.spans = spans
        self.exemplar = Expression.generate_expression_template(utterance, slots, spans)

    @classmethod
    def generate_expression_template(cls, utterance, slot_dict,  spans):
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


@dataclass_json
@dataclass
class SlotInfo:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    is_categorical: bool = field(metadata={"required": True})
    possible_values: list[str] = field(metadata={"required": True})


@dataclass_json
@dataclass
class SkillInfo:
    name: str = field(metadata={"required": True})
    description: str = field(metadata={"required": True})
    slots: list[str] = field(metadata={"required": True})

@dataclass_json
@dataclass
class DomainInfo:
    skills: dict[str, SkillInfo]
    slots: dict[str, SlotInfo]



@dataclass
class Domain:
    skills: list[dict[str, str]]
    slots: list[dict[str, str]]


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
class DatasetCreator(ABC):
    __metaclass__ = abc.ABCMeta
    domain: Domain

    @abc.abstractmethod
    def build(self, split: str) -> Dataset:
        """This return the domain meta needed."""
        return


class DatasetWrapper(DatasetCreator, ABC):
    def __init__(self, creator: DatasetCreator, prompt: Prompt, mode: str = "full", num_procs: int = 4):
        self.creator = creator
        self.prompt = prompt
        self.mode = mode
        self.num_procs = num_procs
        self.domain = creator.domain

    def build(self, split: str) -> Dataset:
        dataset = self.creator.build(split)
        if self.mode == "full":
            dataset = dataset.map(lambda x: {"output": x['target_full']})
        else:
            dataset = dataset.map(lambda x: {"output": x['target_intent']})
        return dataset.map(lambda x: {"input": self.prompt(x)})



if __name__ == "__main__":
    skill = SkillInfo(name="test", description="what")
    print(skill.to_dict())
