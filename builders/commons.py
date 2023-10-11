from abc import ABC

from pybars import Compiler
import abc
from dataclasses import dataclass, field
from datasets import Dataset


@dataclass
class Domain:
    skills: list[str] = field(metadata={"help": "a list of function names."})
    skill_descriptions: dict[str, str] = field(metadata={"help": "functions and their descriptions."})
    slots: list[str] = field(metadata={"help": "a list of slot names."})
    slot_descriptions: dict[str, str] = field(metadata={"help": "functions and their descriptions."})


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


class SimplePrompt(Prompt, ABC):
    def __init__(self, source):
        self.template = Prompt.compiler.compile(source)

    # Assume the item has ["utterance", "output"], and "utterance" is used to create input.
    def __call__(self, item):
        return self.template(item)


#
# This is need to create the different dataset based on prompt templating.
#
class DatasetCreator(ABC):
    __metaclass__ = abc.ABCMeta
    prompt: Prompt

    @abc.abstractmethod
    def get_meta(self) -> Domain:
        """This return the domain meta needed."""
        return

    @abc.abstractmethod
    def build(self, split: str) -> Dataset:
        """This return the domain meta needed."""
        return
