from abc import ABC

from pybars import Compiler
import abc
from dataclasses import dataclass, field
from datasets import Dataset


@dataclass
class SlotInfo:
    name: str
    description: str = None
    type: str = None


@dataclass
class SkillInfo:
    name: str
    description: str = None
    slots: list[SlotInfo] = field(default_factory=list)


@dataclass
class Domain:
    skills: list[SkillInfo] = field(metadata={"help": "a list of function names."})
    slots: list[SlotInfo] = field(metadata={"help": "a list of slot names."})


#
# This assumes the dataset has skills, skill_descriptions, slots, slot_descriptions
# Then user utterance as input, and output.
#
class Prompt:
    __metaclass__ = abc.ABCMeta
    compiler = Compiler()

    @abc.abstractmethod
    def extra_tokens(self) -> list[str]:
        """This return the extra tokens the dataset builder will generate."""
        return []

    @abc.abstractmethod
    def __call__(self, utterance: str, output: str):
        """Method documentation"""
        return


#
# This is need to create the different dataset based on prompt templating.
#
class DatasetCreator(ABC):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_meta(self) -> Domain:
        """This return the domain meta needed."""
        return

    @abc.abstractmethod
    def build(self, split: str) -> Dataset:
        """This return the domain meta needed."""
        return


class DatasetWrapper(DatasetCreator, ABC):
    def __init__(self, creator: DatasetCreator, prompt: Prompt):
        self.creator = creator
        self.prompt = prompt

    def get_meta(self) -> Domain:
        return self.creator.get_meta()

    def build(self, split: str) -> Dataset:
        dataset = self.creator.build(split)
        return dataset.map(lambda x: {"input": self.prompt(x)})


