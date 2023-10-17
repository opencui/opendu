from abc import ABC

import abc
from dataclasses import dataclass, field
from datasets import Dataset


@dataclass
class SlotInfo:
    name: str
    description: str

    def __repr__(self):
        return f"SlotInfo(name={self.name}, description={self.description})"


@dataclass
class SkillInfo:
    name: str
    description: str

    def __repr__(self):
        return f"SkillInfo(name={self.name}, description={self.description})"

@dataclass
class Domain:
    skills: list[SkillInfo] = field(metadata={"help": "a list of function names."})
    slots: list[SlotInfo] = field(metadata={"help": "a list of slot names."})


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
            dataset.map(lambda x: {"output": x['target_full']})
        else:
            dataset.map(lambda x: {"output": x['target_intent']})
        return dataset.map(lambda x: {"input": self.prompt(x)})

