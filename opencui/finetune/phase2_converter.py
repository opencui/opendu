import abc
from abc import ABC
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from opencui.core.prompt import (promptManager, Task)


@dataclass_json
@dataclass
class LabeledMatchingData:
    """
    This is used to encode the labeled matching data
    """
    _id: str
    matchType: str
    reference: str
    utterance: str  # useful for slot model
    decision: bool


# We assume that batch is LabeledMatchingData in column form, this is what we get from pandas.
class TrainPhase2Converter(ABC):
    @abc.abstractmethod
    def __call__(self, batch, ins: list[str], outs: list[str]):
        return


class PromptConverter(TrainPhase2Converter):
    def __init__(self):
        self.prompts = {
            "desc": promptManager.get_builder(Task.SKILL_DESC),
            "exemplar": promptManager.get_builder(Task.SKILL)
        }

    @staticmethod
    def label(value):
        label_dict = {"label": "true" if value else "false"}
        return promptManager.get_builder(Task.BOOL_VALUE)(label_dict)

    def __call__(self, batch, ins: list[str], outs: list[str]):
        for idx, mode in enumerate(batch["matchType"]):
            utterance = batch["utterance"][idx]
            reference = batch["reference"][idx]
            decision = batch["decision"][idx]
            if mode == "desc":
                input_dict = {
                    "skill" : { "description" : reference},
                    "utterance": utterance
                }
            elif mode == "exemplar":
                input_dict = {
                    "utterance": utterance,
                    "template": reference
                }
            else:
                raise ValueError(f"Mode {mode} is not supported.")

            ins.append(self.prompts[mode](input_dict))
            outs.append(self.label(decision))




